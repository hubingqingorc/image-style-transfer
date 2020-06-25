# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt
import argparse
from collections import namedtuple
import numpy as np
import time
import os
import csv
from PIL import Image


class DataSet(data.Dataset):
    def __init__(self, data_transform=None):
        self.data_transform = data_transform
        self.img_dir = './train2014/'  # coco set dir
        self.imgs = self.find_img()  # get the list of img path
        self.re_size = 256

    def find_img(self):  # find file in folder
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
        if img.mode == 'L':  # 当出现黑白图片时转换mode
            img = img.convert('RGB')
        w, h = img.size
        ratio = self.re_size / min(w, h)  # 得到缩放比
        new_img = img.resize((round(ratio * w), round(ratio * h)))  # 将短的边缩放至self.re_size
        return new_img

    def __len__(self):
        return len(self.imgs)


def conv_norm_actv(in_chnls, out_chnls, ksize=3, stride=1, norm=True, relu=True, upsample=None):    # 卷积标准化激活单元
    layers = []
    if upsample:    # 按需上采样
        layers.append(nn.Upsample(scale_factor=upsample, mode='nearest'))
    layers.append(nn.ReflectionPad2d(ksize // 2))  # 映像填充
    layers.append(nn.Conv2d(in_chnls, out_chnls, kernel_size=ksize, stride=stride))
    if norm:    # 按需标准化
        layers.append(nn.InstanceNorm2d(out_chnls))
    if relu:    # 按需激活
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
            *conv_norm_actv(3, base, ksize=9),  # 256 * 256
            *conv_norm_actv(base, base * 2, stride=2),  # 128 * 128
            *conv_norm_actv(base * 2, base * 4, stride=2)  # 64 * 64
        )
        self.residual = nn.Sequential(*[ResidualBlock(base * 4) for _ in range(5)])  # 64 * 64
        self.upsample = nn.Sequential(
            *conv_norm_actv(base * 4, base * 2, upsample=2),  # 128 * 128
            *conv_norm_actv(base * 2, base, upsample=2),  # 256 * 256
            *conv_norm_actv(base, 3, ksize=9, norm=False, relu=False),  # 256 * 256
            nn.Tanh()
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.residual(x)
        x = self.upsample(x)
        return x


class VGGNet(nn.Module):    # VGG backbone
    def __init__(self):
        super(VGGNet, self).__init__()
        features = list(models.vgg16(pretrained=True).features)[:23]    # 使用预加载的前23层
        self.features = nn.ModuleList(features)
        for p in self.parameters():     # 仅使用VGG网络输出值，故关闭参数梯度计算加快速度
            p.requires_grad = False

    def forward(self, x):
        result = []
        for _i, layer in enumerate(self.features):
            x = layer(x)
            if _i in [3, 8, 15, 22]:
                result.append(x)
        return result


def gram_matrix(x):     # gram 矩阵
    (batch, chnls, h, w) = x.size()
    x = x.view(batch, chnls, h * w)
    x_t12 = x.transpose(1, 2)
    gram = x.bmm(x_t12) / (chnls * h * w)
    return gram


class MultiWorks:
    def __init__(self, model_path=None):
        self.model_path = model_path

        # coco数据集均值矩阵
        mean_r = torch.full((1, 256, 256), 0.485)
        mean_g = torch.full((1, 256, 256), 0.456)
        mean_b = torch.full((1, 256, 256), 0.406)
        self.mean_set = torch.cat((mean_r, mean_g, mean_b), dim=0).unsqueeze(0).to(device)  # [1, 3, 256, 256]
        # coco数据集标准差矩阵
        std_r = torch.full((1, 256, 256), 0.229)
        std_g = torch.full((1, 256, 256), 0.224)
        std_b = torch.full((1, 256, 256), 0.225)
        self.std_set = torch.cat((std_r, std_g, std_b), dim=0).unsqueeze(0).to(device)  # [1, 3, 256, 256]

        work = args.work  # 根据输入work对应启动任务
        if work not in ['train', 'finetune', 'transfer']:
            print("args.work should be one of ['train', 'finetune', 'transfer']")
        elif work == 'train':
            self.train()
        elif self.model_path is None:  # finetune/transfer task need model_path to load model
            print("Please input 'model_path'")
        elif work == 'finetune':
            self.finetune()
        elif work == 'transfer':
            self.transfer()

    def train(self):
        start_time = time.time()
        data_transform = transforms.Compose([transforms.RandomCrop((256, 256)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        data_set = DataSet(data_transform=data_transform)
        load_data = DataLoader(data_set, batch_size=args.batch_size, shuffle=True)
        print(f"Start Train!  data_set_len: {data_set.__len__()}")

        tsfm_model = ImageTransformNet(32).to(device)
        loss_model = VGGNet().to(device).eval()

        current_lr = args.lr
        optimizer = torch.optim.Adam(tsfm_model.parameters(), lr=current_lr, betas=(0.9, 0.999), eps=1e-8)
        criterion = nn.MSELoss(reduction='sum')
        # load style img
        img_style = Image.open(args.style_t)
        img_style = img_style.resize((256, 256))
        img_style = data_transform(img_style).unsqueeze(0).to(device)
        t_style = loss_model(img_style)  # get relu1_2 & relu2_2 & relu3_3 & relu4_3 layers result
        gram_t_style = [gram_matrix(_i).detach() for _i in t_style]  # 获得不需要梯度计算的变量值
        # 采集loss并在最后输出.csv文件
        collect_loss = [['index', 'batch_size', 'lr', 'total_loss', 'content_loss', 'style_loss', 'variation_loss']]

        for i in range(args.epochs):
            for idx, img_content in enumerate(load_data):
                img_content = img_content.to(device)
                optimizer.zero_grad()

                y = (tsfm_model(img_content) + 1) / 2  # scale to (0, 1)
                y_mean_std = (y - self.mean_set) / self.std_set  # 应用与coco数据集同样的标准化
                y1 = loss_model(y_mean_std)

                # Content loss
                loss_content = criterion(loss_model(img_content)[2].detach(), y1[2])
                # Style loss
                loss_style = 0
                for _j in range(len(y1)):
                    loss_style += criterion(gram_matrix(y1[_j]), gram_t_style[_j].expand_as(gram_matrix(y1[_j])))
                # Variation loss
                loss_variation = torch.sum(torch.abs((y_mean_std[:, :, :-1, 1:] - img_content[:, :, :-1, :-1])) +
                                           torch.abs((y_mean_std[:, :, 1:, :-1] - img_content[:, :, :-1, :-1])))

                weighted_loss_content = args.weight_content * loss_content
                weighted_loss_style = args.weight_style * loss_style
                weighted_loss_variation = args.weight_variation * loss_variation
                total_loss = weighted_loss_content + weighted_loss_style + weighted_loss_variation

                collect_loss.append([idx, args.batch_size, current_lr, total_loss.item(), weighted_loss_content.item(),
                                     weighted_loss_style.item(), weighted_loss_variation.item()])
                total_loss.backward()
                optimizer.step()

                if idx % 100 == 0:
                    print(f'batch: {idx}  Total: {total_loss.item()}  Style: {weighted_loss_style.item()}  '
                          f'Content: {weighted_loss_content.item()}  Variation: {weighted_loss_variation.item()}  '
                          f'Cost_time: {time.time() - start_time}')
                if idx % 1000 == 0:
                    self.interval_plot(y_mean_std[0], img_content[0], img_style[0])

        if args.save_model:  # 是否保存模型
            if not os.path.exists(args.save_directory):  # 新建保存文件夹
                os.makedirs(args.save_directory)
            # 模型保存路径
            save_model_path = os.path.join(args.save_directory, time.strftime('%Y%m%d%H%M') + '_train_epoch_'
                                           + str(i) + ".pt")
            # 训练过程保存路径
            save_loss_path = os.path.join(args.save_directory, time.strftime('%Y%m%d%H%M') + '_train_loss.csv')
            torch.save(tsfm_model.state_dict(), save_model_path)
            self.writelist2csv(collect_loss, save_loss_path)  # 写入.csv文件
            print(f'--Save complete!\n--save_model_path: {save_model_path}\n--save_loss_path: {save_loss_path}')
        print('Train complete!')

    def finetune(self):
        start_time = time.time()
        data_transform = transforms.Compose([transforms.RandomCrop((256, 256)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        data_set = DataSet(data_transform=data_transform)
        load_data = DataLoader(data_set, batch_size=args.batch_size, shuffle=True)
        print(f"Start Finetune!  data_set_len: {len(data_set)}")

        tsfm_model = ImageTransformNet(32).to(device)
        tsfm_model.load_state_dict(torch.load(self.model_path))  # 模型参数加载
        loss_model = VGGNet().to(device)

        current_lr = args.lr
        optimizer = torch.optim.Adam(tsfm_model.parameters(), lr=current_lr, betas=(0.9, 0.999), eps=1e-8)
        criterion = nn.MSELoss(reduction='sum')

        # style img
        img_style = Image.open(args.style_t)
        img_style = img_style.resize((256, 256))
        img_style = data_transform(img_style).unsqueeze(0).to(device)
        t_style = loss_model(img_style)
        gram_t_style = [gram_matrix(_i).detach() for _i in t_style]
        # 采集loss并在最后输出.csv文件
        collect_loss = [['index', 'batch_size', 'lr', 'total_loss', 'content_loss', 'style_loss', 'variation_loss']]
        for i in range(args.epochs):
            for idx, img_content in enumerate(load_data):
                img_content = img_content.to(device)
                optimizer.zero_grad()

                y = (tsfm_model(img_content) + 1) / 2  # scale to (0, 1)
                y_mean_std = (y - self.mean_set) / self.std_set  # 应用与coco数据集同样的标准化
                y1 = loss_model(y_mean_std)

                # Content loss
                loss_content = criterion(loss_model(img_content)[2], y1[2])
                # Style loss
                loss_style = 0
                for _j in range(len(y1)):
                    loss_style += criterion(gram_matrix(y1[_j]).detach(), 
                                            gram_t_style[_j].expand_as(gram_matrix(y1[_j])))
                # Variation loss
                loss_variation = torch.sum(torch.abs((y_mean_std[:, :, :-1, 1:] - img_content[:, :, :-1, :-1])) +
                                           torch.abs((y_mean_std[:, :, 1:, :-1] - img_content[:, :, :-1, :-1])))

                weighted_loss_content = args.weight_content * loss_content
                weighted_loss_style = args.weight_style * loss_style
                weighted_loss_variation = args.weight_variation * loss_variation
                total_loss = weighted_loss_content + weighted_loss_style + weighted_loss_variation

                collect_loss.append([idx, args.batch_size, current_lr, total_loss.item(), weighted_loss_content.item(),
                                     weighted_loss_style.item(), weighted_loss_variation.item()])
                total_loss.backward()
                optimizer.step()

                if idx % 100 == 0:
                    print(f'batch: {idx}  Total: {total_loss.item()}  Style: {weighted_loss_style.item()}  '
                          f'Content: {weighted_loss_content.item()}  Variation: {weighted_loss_variation.item()}  '
                          f'Cost_time: {time.time() - start_time}')
                if idx % 1000 == 0:
                    self.interval_plot(y_mean_std[0], img_content[0], img_style[0])

        if args.save_model:  # 是否保存模型
            save_model_path = self.model_path[:-3] + '_finetune_' + str(i) + ".pt"  # 模型保存路径
            save_loss_path = self.model_path[:-3] + '_finetune_' + str(i) + "_loss.csv"  # 训练过程保存路径
            torch.save(tsfm_model.state_dict(), save_model_path)
            self.writelist2csv(collect_loss, save_loss_path)  # 写入.csv文件
            print(f'--Save complete!\n--save_model_path: {save_model_path}\n--save_loss_path: {save_loss_path}')
        print('Finetune complete!')

    def transfer(self):
        tsfm_model = ImageTransformNet(32).to(device)
        tsfm_model.load_state_dict(torch.load(self.model_path))  # 模型参数加载
        data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        print(f"Start Transfer!")
        if not os.path.exists(args.save_transfer_img):  # 新建转换图片保存文件夹
            os.makedirs(args.save_transfer_img)
        for root, dirs, files in os.walk(args.transfer_img_folder):
            pass
        for file in files:
            img = Image.open(root + file)
            if img.mode != 'RGB':  # 转换所有图片模式为RGB
                img = img.convert('RGB')
            w, h = img.size
            new_img = img.resize((256, 256))  # 将短的边缩放至self.re_size
            transform_img = data_transform(new_img)
            img_content = transform_img.to(device).unsqueeze(0).to(device)
            y = ((tsfm_model(img_content) + 1) / 2)[0].cpu().detach().numpy().transpose(1, 2, 0)
            pic = np.array(np.rint(y * 255), dtype='uint8')

            if args.show_transfer_img:
                plt.imshow(pic)
                plt.show()            
            transfer_img = Image.fromarray(pic)
            resize_transfer_img = transfer_img.resize((w, h))
            save_img_path = os.path.join(args.save_transfer_img, file[:-4] + '_transfer' + file[-4:])
            resize_transfer_img.save(save_img_path)
        print('Transfer complete!')

    @staticmethod
    def writelist2csv(list_data, csv_name):  # 列表写入.csv
        with open(csv_name, "w", newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for one_slice in list_data:
                csv_writer.writerow(one_slice)

    def interval_plot(self, y, img_content, img_style):
        y, img_content, img_style = y * self.std_set[0] + self.mean_set[0], img_content * self.std_set[0] + \
                                    self.mean_set[0], img_style * self.std_set[0] + self.mean_set[0]
        y, img_content, img_style = [y.cpu().detach().numpy(), img_content.cpu().detach().numpy(),
                                     img_style.cpu().detach().numpy()]
        y, img_content, img_style = y.transpose(1, 2, 0), img_content.transpose(1, 2, 0), img_style.transpose(1, 2, 0)
        plt.subplot(131)
        plt.imshow(y)
        plt.title('transfer img')
        plt.subplot(132)
        plt.imshow(img_content)
        plt.title('content img')
        plt.subplot(133)
        plt.imshow(img_style)
        plt.title('style img')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--style-t', type=str, default='style_img/style.jpg',
                        help='style image path')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--weight-content', type=float, default=1, metavar='N',
                        help='the weight of content')
    parser.add_argument('--weight-style', type=float, default=5e6, metavar='N',
                        help='the weight of content')
    parser.add_argument('--weight-variation', type=float, default=1, metavar='N',
                        help='the weight of content')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current Model')
    parser.add_argument('--save-directory', type=str, default='save_model',
                        help='learnt models are saving here')
    parser.add_argument('--transfer-img-folder', type=str, default='content_img/',
                        help='transfer image dictionary')
    parser.add_argument('--show-transfer-img', type=bool, default=True,
                        help='show transfer img')
    parser.add_argument('--save-transfer-img', type=str, default='result/',
                        help='transfer img are saving here')
    parser.add_argument('--work', type=str, default='transfer',
                        help='train, finetune, transfer')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    MultiWorks(model_path='save_model/202005190318_train_epoch_1.pt')
