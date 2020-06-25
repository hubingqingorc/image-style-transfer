# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np
import time
import os
import csv
from PIL import Image
from PIL import ImageFile
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataSet(data.Dataset):
    def __init__(self, data_transform=None, is_coco=True):
        self.data_transform = data_transform
        self.img_dir = './content/' if is_coco else './test/'  # content使用coco set/ style使用wikiart set
        self.imgs = self.find_img()
        self.re_size = 256

    def find_img(self):
        imgs = []
        for root, _, files in os.walk(self.img_dir):
            pass
        for file in files:
            imgs.append(root + file)
        return imgs

    def __getitem__(self, index):
        sample = self.imgs[index]
        img = self.proportion_resize(sample)  # 保持x y轴比例进行resize
        if self.data_transform is not None:
            img = self.data_transform(img)
        return img

    def proportion_resize(self, sample):  # 保持x y轴比例进行resize
        img = Image.open(sample)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        w, h = img.size
        ratio = self.re_size / min(w, h)  # 得到缩放比
        new_img = img.resize((round(ratio * w), round(ratio * h)))
        return new_img

    def __len__(self):
        return len(self.imgs)


class FuncConv2d(nn.Module):  # 制作功能卷积，参数由外部指定
    def __init__(self, in_chnls, out_chnls, ksize, stride):
        super(FuncConv2d, self).__init__()
        self.weight = torch.zeros([out_chnls, in_chnls, ksize, ksize]).to(device)
        self.bias = torch.zeros([out_chnls]).to(device)
        self.stride = stride

    def forward(self, x):
        return nn.functional.conv2d(x, self.weight, self.bias, self.stride)


def conv_norm_actv(in_chnls, out_chnls, ksize=3, stride=1, norm=True,
                   relu=True, upsample=None, is_train=False):  # 卷积标准化激活单元
    layers = []
    if upsample:  # 按需上采样
        layers.append(nn.Upsample(scale_factor=upsample, mode='nearest'))
    layers.append(nn.ReflectionPad2d(ksize // 2))  # 映像填充
    if is_train:  # 是否参与训练
        layers.append(nn.Conv2d(in_chnls, out_chnls, kernel_size=ksize, stride=stride))  # 嵌入卷积
    else:
        layers.append(FuncConv2d(in_chnls, out_chnls, ksize, stride))  # 嵌入功能卷积
    if norm:  # 按需标准化
        layers.append(nn.InstanceNorm2d(out_chnls))
    if relu:  # 按需激活
        layers.append(nn.ReLU(inplace=True))
    return layers


class ResidualBlock(nn.Module):  # base article-'Training and investigating residual nets'
    def __init__(self, chnls):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            *conv_norm_actv(chnls, chnls),
            *conv_norm_actv(chnls, chnls, relu=False)
        )

    def forward(self, x):
        return x + self.conv(x)


class ImageTransformNet(nn.Module):
    def __init__(self, base):
        super(ImageTransformNet, self).__init__()
        self.downsample = nn.Sequential(
            *conv_norm_actv(3, base, ksize=9, norm=False, is_train=True),  # 336 * 336
            *conv_norm_actv(base, base * 2, stride=2),  # 168 * 168
            *conv_norm_actv(base * 2, base * 4, stride=2)  # 84 * 84
        )
        self.residual = nn.Sequential(*[ResidualBlock(base * 4) for _ in range(5)])
        self.upsample = nn.Sequential(
            *conv_norm_actv(base * 4, base * 2, upsample=2),
            *conv_norm_actv(base * 2, base, upsample=2),
            *conv_norm_actv(base, 3, ksize=9, norm=False, relu=False, is_train=True)
        )
        self.weight_bias_num = self.para_num()
        self._num = 0

    def forward(self, x):
        x = self.downsample(x)
        x = self.residual(x)
        x = self.upsample(x)
        return x

    def para_num(self):  # 统计网络中功能卷积的参数数量，供给这些功能卷积块参数赋值时使用
        para_num = []

        def find_all_funcconv2d(block):
            for _i in block.children():
                if _i.__class__ == FuncConv2d:
                    weight_num, bias_num = np.prod(_i.weight.size()), _i.bias.size()[0]
                    para_num.append([weight_num, bias_num])
                find_all_funcconv2d(_i)

        find_all_funcconv2d(self)
        return para_num

    def set_weight(self, fc_output):  # 给网络中功能卷积块参数赋值

        def set_all_funcconv2d(block):
            for _i in block.children():
                if _i.__class__ == FuncConv2d:
                    _i.weight = fc_output[self._num][:self.weight_bias_num[self._num][0]].view(_i.weight.size())
                    _i.bias = fc_output[self._num][self.weight_bias_num[self._num][0]:].view(_i.bias.size())
                    self._num += 1
                set_all_funcconv2d(_i)

        set_all_funcconv2d(self)
        self._num = 0


class VGGNet(nn.Module):  # VGG backbone
    def __init__(self):
        super(VGGNet, self).__init__()
        features = list(models.vgg16(pretrained=True).features)[:23]  # 使用预加载的前23层
        self.features = nn.ModuleList(features)
        for p in self.parameters():  # 仅使用VGG网络输出值，故关闭参数梯度计算加快速度
            p.requires_grad = False

    def forward(self, x):
        result = []
        for _i, layer in enumerate(self.features):
            x = layer(x)
            if _i in [3, 8, 15, 22]:
                result.append(x)
        return result


class MetaNet(nn.Module):  # 元网络
    def __init__(self, para_num):
        super(MetaNet, self).__init__()
        self.fc = nn.Linear(1920, 128 * len(para_num))  # 全连接层
        self.len = len(para_num)
        self.fc_layers = []     # 全连接层分支
        for _i in para_num:
            self.fc_layers.append(nn.Linear(128, sum(_i)).to(device))

    def forward(self, x):
        x = F.relu(self.fc(x))  # 计算全连接层结果并激活
        xs = []
        for _j in range(self.len):  # 将全连接层结果切片送入全连接分支
            xs.append(self.fc_layers[_j](x[_j * 128: (_j + 1) * 128]))
        return xs


def mean_std(input_data):  # input = [relu1_2, relu2_2, relu3_3, relu4_3]，计算风格特征值
    mean_std_result = torch.tensor([]).to(device)
    for x in input_data:
        b, c, h, w = x.size()
        x_view = x.view(b, c, -1)
        batch_mean_std = torch.zeros([c * 2]).to(device)
        for i in range(b):
            chnls_mean = torch.mean(x_view[i], dim=1)  # [c]
            chnls_std = torch.std(x_view[i], dim=1)  # [c]
            chnls_mean_std = torch.cat((chnls_mean.unsqueeze(0), chnls_std.unsqueeze(0)), dim=0)  # [2, c]
            chnls_mean_std = chnls_mean_std.t().flatten()  # [2, c]->[c, 2]->[c * 2]
            batch_mean_std += chnls_mean_std
        mean_batch_mean_std = batch_mean_std / b
        mean_std_result = torch.cat((mean_std_result, mean_batch_mean_std), dim=0)
    return mean_std_result


class MultiWorks:
    def __init__(self, load_tsfm_model=None, load_meta_model=None, content_img=None):
        self.load_tsfm_model = load_tsfm_model
        self.load_meta_model = load_meta_model
        self.content_img = content_img

        work = args.work  # 根据输入work对应启动任务
        if work not in ['train', 'transfer', 'finetune']:
            print("args.work should be one of ['train', 'finetune', 'transfer']")
        elif work == 'train':
            self.train()
        elif self.load_tsfm_model is None and self.load_meta_model is None:  # 转化需要模型路径和图片路径
            print('Please input both load_tsfm_model and load_meta_model')
        elif work == 'finetune':
            self.finetune()
        elif work == 'transfer':
            self.transfer()

    def train(self):
        start_time = time.time()
        data_transform = transforms.Compose([transforms.RandomCrop((256, 256)),
                                             transforms.ToTensor()])
        coco_set = DataSet(data_transform=data_transform, is_coco=True)
        wikiart_set = DataSet(data_transform=data_transform, is_coco=False)
        wikiart_set_len = wikiart_set.__len__()
        load_coco = DataLoader(coco_set, batch_size=args.batch_size, shuffle=True)

        print(f"Start Train!  coco_set_len: {coco_set.__len__()}  wikiart_set_batch: {wikiart_set.__len__()}")

        tsfm_model = ImageTransformNet(8).to(device)    # 根据GPU内容，选择base=8
        para_num = tsfm_model.para_num()
        loss_model = VGGNet().to(device).eval()
        meta_model = MetaNet(para_num).to(device)
        optim_paras = []    # 待优化参数
        for _m in [tsfm_model, meta_model]:
            for _p in _m.parameters():
                if _p.requires_grad:
                    optim_paras.append(_p)

        current_lr = args.lr
        optimizer = torch.optim.Adam(optim_paras, lr=current_lr, betas=(0, 0.999), eps=1e-8)    # 优化函数
        criterion = nn.MSELoss(reduction='sum')     # 损失评价函数
        # 采集loss并在最后输出.csv文件
        collect_loss = [['index', 'batch_size', 'lr', 'total_loss', 'content_loss', 'style_loss', 'variation_loss']]

        for i in range(args.epochs):
            for idx, img_content in enumerate(load_coco):
                optimizer.zero_grad()
                if idx % 20 == 0:   # 每20batch, 更新风格图片
                    rdm_idx = random.choice(range(wikiart_set_len))
                    img_style = wikiart_set.__getitem__(rdm_idx)
                    img_style = img_style.unsqueeze(0).to(device)  # [1, 3, 256, 256]
                    t_style = [x.detach() for x in loss_model(img_style)]  # style_img features
                    t_style_m_s = mean_std(t_style)     # 输出风格图片风格特征值
                weight = meta_model(t_style_m_s)    # 风格特征值输入元网络得到图片转换网络功能卷积块参数值
                tsfm_model.set_weight(weight)   # 功能卷积块参数值赋值

                img_content = img_content.to(device)
                y = tsfm_model(img_content)
                y1 = loss_model(y)

                # content loss
                loss_content = criterion(loss_model(img_content)[2].detach(), y1[2])

                # style loss
                y1_mean_std = mean_std(y1)
                loss_style = criterion(y1_mean_std, t_style_m_s)

                # Variation loss
                loss_variation = torch.sum(torch.abs((y[:, :, :-1, 1:] - img_content[:, :, :-1, :-1])) +
                                           torch.abs((y[:, :, 1:, :-1] - img_content[:, :, :-1, :-1])))

                weighted_loss_content = args.weight_content * loss_content
                weighted_loss_style = args.weight_style * loss_style
                weighted_loss_variation = args.weight_variation * loss_variation
                total_loss = weighted_loss_content + weighted_loss_style + weighted_loss_variation

                collect_loss.append([idx, args.batch_size, current_lr, total_loss.item(), weighted_loss_content.item(),
                                     weighted_loss_style.item(), weighted_loss_variation.item()])
                total_loss.backward()
                optimizer.step()

                if idx % 100 == 0:
                    print(f'Epoch: {idx}  Total: {total_loss.item()}  Style: {weighted_loss_style.item()}  '
                          f'Content: {weighted_loss_content.item()}  Variation: {weighted_loss_variation.item()}  '
                          f'Cost_time: {time.time() - start_time}')
                if idx % 1000 == 0:
                    self.interval_plot(y[0], img_content[0], img_style[0])
                if idx == 1:
                    break

        if args.save_model:  # 是否保存模型
            if not os.path.exists(args.save_directory):  # 新建保存文件夹
                os.makedirs(args.save_directory)
            # 模型保存路径
            save_tsfm_model_path = os.path.join(args.save_directory,
                                                time.strftime('%Y%m%d%H%M') + '_tsfm_train_epoch_' + str(i)
                                                + ".pt")
            save_meta_model_path = os.path.join(args.save_directory,
                                                time.strftime('%Y%m%d%H%M') + '_meta_train_epoch_' + str(i)
                                                + ".pt")
            # 训练过程保存路径
            save_loss_path = os.path.join(args.save_directory, time.strftime('%Y%m%d%H%M') + '_train_loss.csv')
            torch.save(tsfm_model.state_dict(), save_tsfm_model_path)
            torch.save(meta_model.state_dict(), save_meta_model_path)
            self.writelist2csv(collect_loss, save_loss_path)  # 写入.csv文件
            print(f'--Save complete!\n--save_tsfm_model_path: {save_tsfm_model_path}\n'
                  f'--save_meta_model_path: {save_meta_model_path}\n--save_loss_path: {save_loss_path}')
        print('Train complete!')

    def finetune(self):
        start_time = time.time()
        data_transform = transforms.Compose([transforms.RandomCrop((256, 256)),
                                             transforms.ToTensor()])
        coco_set = DataSet(data_transform=data_transform, is_coco=True)
        wikiart_set = DataSet(data_transform=data_transform, is_coco=False)
        wikiart_set_len = wikiart_set.__len__()
        load_coco = DataLoader(coco_set, batch_size=args.batch_size, shuffle=True)

        print(f"Start Finetune!  coco_set_len: {coco_set.__len__()}  wikiart_set_batch: {wikiart_set.__len__()}")

        tsfm_model = ImageTransformNet(8).to(device)
        para_num = tsfm_model.para_num()
        tsfm_model.load_state_dict(torch.load(self.load_tsfm_model))  # 模型参数加载

        meta_model = MetaNet(para_num).to(device)
        meta_model.load_state_dict(torch.load(self.load_meta_model))  # 模型参数加载

        loss_model = VGGNet().to(device)

        optim_paras = []
        for _m in [tsfm_model, meta_model]:
            for _p in _m.parameters():
                if _p.requires_grad:
                    optim_paras.append(_p)

        current_lr = args.lr
        optimizer = torch.optim.Adam(optim_paras, lr=current_lr, betas=(0, 0.999), eps=1e-8)
        # optimizer = torch.optim.SGD(tsfm_model.parameters(), lr=current_lr)
        criterion = nn.MSELoss(reduction='sum')
        # 采集loss并在最后输出.csv文件
        collect_loss = [['index', 'batch_size', 'lr', 'total_loss', 'content_loss', 'style_loss', 'variation_loss']]

        for i in range(args.epochs):
            for idx, img_content in enumerate(load_coco):
                optimizer.zero_grad()
                if idx % 20 == 0:
                    rdm_idx = random.choice(range(wikiart_set_len))
                    img_style = wikiart_set.__getitem__(rdm_idx)
                    img_style = img_style.unsqueeze(0).to(device)  # [1, 3, 256, 256]
                    t_style = [x.detach() for x in loss_model(img_style)]  # style_img features
                    t_style_m_s = mean_std(t_style)
                weight = meta_model(t_style_m_s)
                tsfm_model.set_weight(weight)

                img_content = img_content.to(device)
                y = tsfm_model(img_content)
                y1 = loss_model(y)

                # content loss
                loss_content = criterion(loss_model(img_content)[2].detach(), y1[2])
                # style loss
                y1_mean_std = mean_std(y1)
                loss_style = criterion(y1_mean_std, t_style_m_s)

                # Variation loss
                loss_variation = torch.sum(torch.abs((y[:, :, :-1, 1:] - img_content[:, :, :-1, :-1])) +
                                           torch.abs((y[:, :, 1:, :-1] - img_content[:, :, :-1, :-1])))

                weighted_loss_content = args.weight_content * loss_content
                weighted_loss_style = args.weight_style * loss_style
                weighted_loss_variation = args.weight_variation * loss_variation
                total_loss = weighted_loss_content + weighted_loss_style + weighted_loss_variation

                collect_loss.append([idx, args.batch_size, current_lr, total_loss.item(), weighted_loss_content.item(),
                                     weighted_loss_style.item(), weighted_loss_variation.item()])
                total_loss.backward()
                optimizer.step()

                if idx % 100 == 0:
                    print(f'Epoch: {idx}  Total: {total_loss.item()}  Style: {weighted_loss_style.item()}  '
                          f'Content: {weighted_loss_content.item()}  Variation: {weighted_loss_variation.item()}  '
                          f'Cost_time: {time.time() - start_time}')

                if idx % 1000 == 0:
                    self.interval_plot(y[0], img_content[0], img_style[0])

        # 模型保存路径
        save_tsfm_model_path = self.load_tsfm_model[:-3] + '_finetune_' + str(i) + ".pt"
        save_meta_model_path = self.load_meta_model[:-3] + '_finetune_' + str(i) + ".pt"
        # 训练过程保存路径
        save_loss_path = self.load_tsfm_model[:-3] + '_finetune_' + str(i) + "_loss.csv"
        torch.save(tsfm_model.state_dict(), save_tsfm_model_path)
        torch.save(meta_model.state_dict(), save_meta_model_path)
        self.writelist2csv(collect_loss, save_loss_path)  # 写入.csv文件
        print(f'--Save complete!\n--save_tsfm_model_path: {save_tsfm_model_path}\n'
              f'--save_meta_model_path: {save_meta_model_path}\n--save_loss_path: {save_loss_path}')
        print('Finetune complete!')

    def transfer(self):
        tsfm_model = ImageTransformNet(8).to(device)
        para_num = tsfm_model.para_num()
        tsfm_model.load_state_dict(torch.load(self.load_tsfm_model))  # 模型参数加载

        meta_model = MetaNet(para_num).to(device)
        meta_model.load_state_dict(torch.load(self.load_meta_model))  # 模型参数加载

        loss_model = VGGNet().to(device)

        data_transform = transforms.Compose([transforms.RandomCrop((256, 256)),
                                             transforms.ToTensor()])
        coco_set = DataSet(data_transform=data_transform, is_coco=True)
        img_content = coco_set.__getitem__(random.choice(range(coco_set.__len__()))).unsqueeze(0).to(device)
        wikiart_set = DataSet(data_transform=data_transform, is_coco=False)
        img_style = wikiart_set.__getitem__(random.choice(range(wikiart_set.__len__()))).unsqueeze(0).to(device)

        t_style = [x.detach() for x in loss_model(img_style)]  # style_img features
        t_style_m_s = mean_std(t_style)

        weight = meta_model(t_style_m_s)
        tsfm_model.set_weight(weight)

        img_content = img_content.to(device)
        y = tsfm_model(img_content)[0].cpu().detach().numpy().transpose(1, 2, 0).clip(0, 1)
        pic = np.array(np.rint(y * 255), dtype='uint8')
        transfer_img = Image.fromarray(pic)
        transfer_img.save('transfer.jpg')
        print('Transfer complete!')

    @staticmethod
    def writelist2csv(list_data, csv_name):  # 列表写入.csv
        with open(csv_name, "w", newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for one_slice in list_data:
                csv_writer.writerow(one_slice)

    @staticmethod
    def interval_plot(y_hat, img_content, img_style):
        y_hat = y_hat.cpu().detach().numpy().transpose(1, 2, 0)
        img_content = img_content.cpu().detach().numpy().transpose(1, 2, 0)
        img_style = img_style.cpu().detach().numpy().transpose(1, 2, 0)
        plt.subplot(131)
        plt.imshow(y_hat.clip(0, 1))
        plt.title('y')
        plt.subplot(132)
        plt.imshow(img_content)
        plt.title('content')
        plt.subplot(133)
        plt.imshow(img_style)
        plt.title('style')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--weight-content', type=float, default=1, metavar='N',
                        help='the weight of content')
    parser.add_argument('--weight-style', type=float, default=1e5, metavar='N',
                        help='the weight of content')
    parser.add_argument('--weight-variation', type=float, default=1, metavar='N',
                        help='the weight of content')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current Model')
    parser.add_argument('--save-directory', type=str, default='save_model',
                        help='learnt models are saving here')
    parser.add_argument('--work', type=str, default='finetune',  # train, finetune, transfer
                        help='train, finetune, transfer')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    MultiWorks(load_tsfm_model='./save_model/202005201426_tsfm_train_epoch_1_finetune_2_finetune_0_finetune_0.pt',
               load_meta_model='./save_model/202005201426_meta_train_epoch_1_finetune_2_finetune_0_finetune_0.pt')
