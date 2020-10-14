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
from visdom import Visdom

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
        img = self.resize(sample)
        if self.data_transform is not None:
            img = self.data_transform(img)
        return img

    @staticmethod
    def resize(sample):
        img = Image.open(sample)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        new_img = img.resize((args.img_size, args.img_size))
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
        model = models.vgg16(pretrained=False)
        model_load = torch.load(args.vgg_model_path)

        new_model_static_dict = model.state_dict()
        for k in new_model_static_dict.keys():
            if k in model_load.keys():
                new_model_static_dict[k] = model_load[k]
        model.load_state_dict(new_model_static_dict)

        features = list(model.features[:23])  # 使用预加载的前30层
        self.features = nn.ModuleList(features)
        for p in self.parameters():  # 仅使用VGG网络输出值，故关闭参数梯度计算加快速度
            p.requires_grad = False

    def forward(self, x):
        output = []
        for _i, layer in enumerate(self.features):
            x = layer(x)
            if _i in {3, 8, 15, 22}:  # relu1_2, relu2_2, relu3_3, and relu4_3
                output.append(x)
        return output


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

        # coco数据集均值矩阵
        mean_coco_r = torch.full((1, args.img_size, args.img_size), 0.485)
        mean_coco_g = torch.full((1, args.img_size, args.img_size), 0.456)
        mean_coco_b = torch.full((1, args.img_size, args.img_size), 0.406)
        self.mean_coco = torch.cat((mean_coco_r, mean_coco_g, mean_coco_b), dim=0).unsqueeze(0).to(device)
        # coco数据集标准差矩阵
        std_coco_r = torch.full((1, args.img_size, args.img_size), 0.229)
        std_coco_g = torch.full((1, args.img_size, args.img_size), 0.224)
        std_coco_b = torch.full((1, args.img_size, args.img_size), 0.225)
        self.std_coco = torch.cat((std_coco_r, std_coco_g, std_coco_b), dim=0).unsqueeze(0).to(device)

        # coco数据集均值矩阵
        mean_vgg_r = torch.full((1, args.img_size, args.img_size), 0.48501961)
        mean_vgg_g = torch.full((1, args.img_size, args.img_size), 0.45795686)
        mean_vgg_b = torch.full((1, args.img_size, args.img_size), 0.40760392)
        self.mean_vgg = torch.cat((mean_vgg_r, mean_vgg_g, mean_vgg_b), dim=0).unsqueeze(0).to(device)
        # coco数据集标准差矩阵
        std_vgg_r = torch.full((1, args.img_size, args.img_size), 1)
        std_vgg_g = torch.full((1, args.img_size, args.img_size), 1)
        std_vgg_b = torch.full((1, args.img_size, args.img_size), 1)
        self.std_vgg = torch.cat((std_vgg_r, std_vgg_g, std_vgg_b), dim=0).unsqueeze(0).to(device)

        self.transform_input = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.48501961, 0.45795686, 0.40760392],
                                                                        std=[1., 1., 1.]),
                                                   transforms.Lambda(lambda x: x[[2, 1, 0]]),
                                                   transforms.Lambda(lambda x: x.mul(255.))])
        self.transform_output = transforms.Compose([transforms.Lambda(lambda x: x.mul(1. / 255)),
                                                    transforms.Lambda(lambda x: x[[2, 1, 0]]),  # turn to RGB
                                                    transforms.Normalize(
                                                        mean=[-0.48501961, -0.45795686, -0.40760392],
                                                        std=[1, 1, 1])
                                                    ])
        self.transform_to_pil = transforms.Compose([transforms.ToPILImage()])

        self.start_time = time.time()
        self.coco_set = DataSet(data_transform=self.transform_input, is_coco=True)
        self.wikiart_set = DataSet(data_transform=self.transform_input, is_coco=False)
        self.load_coco = DataLoader(self.coco_set, batch_size=args.batch_size, shuffle=True)

        if not os.path.exists(args.save_directory):  # 新建保存文件夹
            os.makedirs(args.save_directory)

        work = args.work  # 根据输入work对应启动任务
        if work not in ['train', 'transfer', 'finetune']:
            print("args.work should be one of ['train', 'finetune', 'transfer']")
        elif work == 'train':
            self.train()
        elif self.load_tsfm_model is None and self.load_meta_model is None:  # 转化需要模型路径和图片路径
            print('Please input load_tsfm_model and load_meta_model')
        elif work == 'finetune':
            self.finetune()
        elif work == 'transfer':
            self.transfer()

    def train(self):
        print(f"Start Train!  "
              f"coco_set_len: {self.coco_set.__len__()}  "
              f"wikiart_set_batch: {self.wikiart_set.__len__()}")

        tsfm_model = ImageTransformNet(args.base).to(device)    # 根据GPU内存，选择base=8
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
        criterion = nn.MSELoss()

        # 采集loss并在最后输出.csv文件
        collect_loss = [['index', 'batch_size', 'lr', 'total_loss', 'content_loss', 'style_loss', 'variation_loss']]
        idx_count = [-1]
        loss_record = []
        cost_time_record = []
        for i in range(args.epochs):
            for idx, img_content in enumerate(self.load_coco):
                optimizer.zero_grad()
                if idx % 30 == 0:   # 每20batch, 更新风格图片
                    rdm_idx = random.choice(range(self.wikiart_set.__len__()))
                    img_style = self.wikiart_set.__getitem__(rdm_idx)
                    img_style = img_style.unsqueeze(0).to(device)  # [1, 3, 256, 256]
                    t_style = [x.detach() for x in loss_model(img_style)]  # style_img features
                    t_style_m_s = mean_std(t_style)     # 输出风格图片风格特征值
                weight = meta_model(t_style_m_s)    # 风格特征值输入元网络得到图片转换网络功能卷积块参数值
                tsfm_model.set_weight(weight)   # 功能卷积块参数值赋值

                img_content = img_content.to(device)

                content = loss_model(img_content)[2].detach()  # 内容特征
                img_content_rgb = img_content[:, (2, 1, 0), :, :] * 1. / 255.
                tsfm_input = (img_content_rgb * self.std_vgg + self.mean_vgg)
                tsfm_input = (tsfm_input - self.mean_coco) / self.std_coco
                y0 = tsfm_model(tsfm_input)
                y1 = (y0 + 1) / 2.
                y2 = (y1 - self.mean_vgg) / self.std_vgg
                y2 = y2[:, (2, 1, 0), :, :] * 255.
                y = loss_model(y2)

                # Content loss
                loss_content = criterion(content, y[2])
                # style loss
                y_mean_std = mean_std(y)
                loss_style = criterion(y_mean_std, t_style_m_s)
                # Variation loss
                loss_variation = torch.sum(torch.pow((y2[:, :, :-1, 1:] - img_content[:, :, :-1, :-1]), 2)) + \
                                 torch.sum(torch.pow((y2[:, :, 1:, :-1] - img_content[:, :, :-1, :-1]), 2))
                # loss_variation = (torch.sum(torch.abs(y2[:, :, :, :-1] - y2[:, :, :, 1:])) +
                #                   torch.sum(torch.abs(y2[:, :, :-1, :] - y2[:, :, 1:, :])))

                weighted_loss_content = args.weight_content * loss_content
                weighted_loss_style = args.weight_style * loss_style
                weighted_loss_variation = args.weight_variation * loss_variation
                total_loss = weighted_loss_content + weighted_loss_style + weighted_loss_variation

                collect_loss.append([idx, args.batch_size, current_lr, total_loss.item(), weighted_loss_content.item(),
                                     weighted_loss_style.item(), weighted_loss_variation.item()])
                total_loss.backward()
                optimizer.step()

                if idx % 1000 == 0:
                    idx_count.append(idx_count[-1] + 1)
                    loss_record.append([total_loss.item(), weighted_loss_content.item(), weighted_loss_style.item(),
                                        weighted_loss_variation.item()])
                    cost_time_record.append(time.time() - self.start_time)
                    vis.line(X=idx_count[1:], Y=loss_record, win='chart1', opts=opts1)
                    vis.line(X=idx_count[1:], Y=cost_time_record, win='chart2', opts=opts2)
                    self.interval_plot(y1[0].clone(),
                                       self.transform_output(img_content[0].clone()),
                                       self.transform_output(img_style[0].clone()))
        self.interval_plot(y1[0].clone(),
                           self.transform_output(img_content[0].clone()),
                           self.transform_output(img_style[0].clone()))

        # 模型保存路径
        save_tsfm_model_path = os.path.join(args.save_directory,
                                            time.strftime('%Y%m%d%H%M') + '_tsfm_train_epoch_' + str(i+1)
                                            + ".pt")
        save_meta_model_path = os.path.join(args.save_directory,
                                            time.strftime('%Y%m%d%H%M') + '_meta_train_epoch_' + str(i+1)
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
        print(f"Start Finetune!  "
              f"coco_set_len: {self.coco_set.__len__()}  "
              f"wikiart_set_batch: {self.wikiart_set.__len__()}")

        tsfm_model = ImageTransformNet(args.base).to(device)
        para_num = tsfm_model.para_num()
        tsfm_model.load_state_dict(torch.load(self.load_tsfm_model))  # 模型参数加载

        meta_model = MetaNet(para_num).to(device)
        meta_model.load_state_dict(torch.load(self.load_meta_model))  # 模型参数加载

        loss_model = VGGNet().to(device).eval()

        optim_paras = []
        for _m in [tsfm_model, meta_model]:
            for _p in _m.parameters():
                if _p.requires_grad:
                    optim_paras.append(_p)

        current_lr = args.lr
        # optimizer = torch.optim.Adam(optim_paras, lr=current_lr, betas=(0, 0.999), eps=1e-8)  # 优化函数
        optimizer = torch.optim.SGD(tsfm_model.parameters(), lr=current_lr*0.00001)
        criterion = nn.MSELoss()
        # 采集loss并在最后输出.csv文件
        collect_loss = [['index', 'batch_size', 'lr', 'total_loss', 'content_loss', 'style_loss', 'variation_loss']]
        idx_count = [-1]
        loss_record = []
        cost_time_record = []

        for i in range(args.epochs):
            for idx, img_content in enumerate(self.load_coco):
                optimizer.zero_grad()
                if idx % 20 == 0:  # 每20batch, 更新风格图片
                    rdm_idx = random.choice(range(self.wikiart_set.__len__()))
                    img_style = self.wikiart_set.__getitem__(rdm_idx)
                    img_style = img_style.unsqueeze(0).to(device)  # [1, 3, 256, 256]
                    t_style = [x.detach() for x in loss_model(img_style)]  # style_img features
                    t_style_m_s = mean_std(t_style)  # 输出风格图片风格特征值
                weight = meta_model(t_style_m_s)  # 风格特征值输入元网络得到图片转换网络功能卷积块参数值
                tsfm_model.set_weight(weight)  # 功能卷积块参数值赋值

                img_content = img_content.to(device)

                content = loss_model(img_content)[2].detach()  # 内容特征
                img_content_rgb = img_content[:, (2, 1, 0), :, :] * 1. / 255.
                tsfm_input = (img_content_rgb * self.std_vgg + self.mean_vgg)
                tsfm_input = (tsfm_input - self.mean_coco) / self.std_coco
                y0 = tsfm_model(tsfm_input)
                y1 = (y0 + 1) / 2.
                y2 = (y1 - self.mean_vgg) / self.std_vgg
                y2 = y2[:, (2, 1, 0), :, :] * 255.
                y = loss_model(y2)

                # Content loss
                loss_content = criterion(content, y[2])
                # style loss
                y_mean_std = mean_std(y)
                loss_style = criterion(y_mean_std, t_style_m_s)
                # Variation loss
                loss_variation = torch.sum(torch.pow((y2[:, :, :-1, 1:] - img_content[:, :, :-1, :-1]), 2)) + \
                                 torch.sum(torch.pow((y2[:, :, 1:, :-1] - img_content[:, :, :-1, :-1]), 2))

                weighted_loss_content = args.weight_content * loss_content
                weighted_loss_style = args.weight_style * loss_style
                weighted_loss_variation = args.weight_variation * loss_variation
                total_loss = weighted_loss_content + weighted_loss_style + weighted_loss_variation

                collect_loss.append([idx, args.batch_size, current_lr, total_loss.item(), weighted_loss_content.item(),
                                     weighted_loss_style.item(), weighted_loss_variation.item()])
                total_loss.backward()
                optimizer.step()

                if idx % 1000 == 0:
                    idx_count.append(idx_count[-1] + 1)
                    loss_record.append([total_loss.item(), weighted_loss_content.item(), weighted_loss_style.item(),
                                        weighted_loss_variation.item()])
                    cost_time_record.append(time.time() - self.start_time)
                    vis.line(X=idx_count[1:], Y=loss_record, win='chart1', opts=opts1)
                    vis.line(X=idx_count[1:], Y=cost_time_record, win='chart2', opts=opts2)
                    self.interval_plot(y1[0].clone(),
                                       self.transform_output(img_content[0].clone()),
                                       self.transform_output(img_style[0].clone()))
            self.interval_plot(y1[0].clone(),
                               self.transform_output(img_content[0].clone()),
                               self.transform_output(img_style[0].clone()))

        # 模型保存路径
        save_tsfm_model_path = self.load_tsfm_model[:-3] + '_finetune_' + str(i+1) + ".pt"
        save_meta_model_path = self.load_meta_model[:-3] + '_finetune_' + str(i+1) + ".pt"
        # 训练过程保存路径
        save_loss_path = self.load_meta_model[:-3] + '_finetune_' + str(i+1) + "_loss.csv"
        torch.save(tsfm_model.state_dict(), save_tsfm_model_path)
        torch.save(meta_model.state_dict(), save_meta_model_path)
        self.writelist2csv(collect_loss, save_loss_path)  # 写入.csv文件
        print(f'--Save complete!\n--save_tsfm_model_path: {save_tsfm_model_path}\n'
              f'--save_meta_model_path: {save_meta_model_path}\n--save_loss_path: {save_loss_path}')
        print('Finetune complete!')

    def transfer(self):
        tsfm_model = ImageTransformNet(args.base).to(device)
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
        plt.imshow(y_hat)
        plt.title('y 10')
        plt.subplot(132)
        plt.imshow(img_content)
        plt.title('content')
        plt.subplot(133)
        plt.imshow(img_style)
        plt.title('style')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--vgg-model-path', type=str, default='./vgg16-00b39a1b.pth',
                        help='vgg model path')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--weight-content', type=float, default=1, metavar='N',
                        help='the weight of content')
    parser.add_argument('--weight-style', type=float, default=50, metavar='N',
                        help='the weight of content')
    parser.add_argument('--weight-variation', type=float, default=1e-5, metavar='N',
                        help='the weight of content')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--img-size', type=int, default=256, metavar='N',
                        help='image size for train (default: 256)')
    parser.add_argument('--base', type=int, default=8, metavar='N',
                        help='feature map base (default: 8)')
    parser.add_argument('--save-directory', type=str, default='save_model',
                        help='learnt models are saving here')
    parser.add_argument('--work', type=str, default='train',  # train, finetune, transfer
                        help='train, finetune, transfer')
    args = parser.parse_args()

    # visdom可视化设置
    vis = Visdom(env="meta network style transfer 20200903")
    assert vis.check_connection()
    opts1 = {
        "title": 'loss with batch count',
        "xlabel": 'batch count',
        "ylabel": 'loss',
        "width": 1000,
        "height": 400,
        "legend": ['total_loss', 'loss_content', 'loss_style', 'loss_variation']
    }
    opts2 = {
        "title": 'cost time with batch count',
        "xlabel": 'batch count',
        "ylabel": 'time in second',
        "width": 1000,
        "height": 400,
        "legend": ['cost time']
    }
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    MultiWorks(load_tsfm_model='./save_model/202010110242_tsfm_train_epoch_10.pt',
               load_meta_model='./save_model/202010110242_meta_train_epoch_10.pt')
    # 202010040011_tsfm_train_epoch_0.pt
