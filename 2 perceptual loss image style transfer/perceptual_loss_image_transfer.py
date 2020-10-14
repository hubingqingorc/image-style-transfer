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
from visdom import Visdom


class DataSet(data.Dataset):
    def __init__(self, data_transform=None):
        self.data_transform = data_transform
        self.img_dir = './train2014/'  # coco set dir
        self.imgs = self.find_img()  # get the list of img path

    def find_img(self):  # find file in folder
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
        if img.mode == 'L':  # 当出现黑白图片时转换mode
            img = img.convert('RGB')
        new_img = img.resize((args.img_size, args.img_size))
        return new_img

    def __len__(self):
        return len(self.imgs)


def conv_norm_actv(in_chnls, out_chnls, ksize=3, stride=1, norm=True, relu=True, upsample=None):  # 卷积标准化激活单元
    layers = []
    if upsample:  # 按需上采样
        layers.append(nn.Upsample(scale_factor=upsample, mode='nearest'))
    layers.append(nn.ReflectionPad2d(ksize // 2))  # 映像填充
    layers.append(nn.Conv2d(in_chnls, out_chnls, kernel_size=ksize, stride=stride))
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
        for m in self.modules():    # 模型参数初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.bias.data.fill_(0)  # 偏差初始为零

    def forward(self, x):
        x = self.downsample(x)
        x = self.residual(x)
        x = self.upsample(x)
        return x


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


def gram_matrix(x):  # gram 矩阵
    (batch, chnls, h, w) = x.size()
    x = x.view(batch, chnls, h * w)
    x_t12 = x.transpose(1, 2)
    gram = x.bmm(x_t12) / (h * w)
    return gram
    

class MultiWorks:
    def __init__(self, model_path=None):
        self.start_time = time.time()
        self.model_path = model_path

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
        self.data_set = DataSet(data_transform=self.transform_input)
        self.load_data = DataLoader(self.data_set, batch_size=args.batch_size, shuffle=True)

        if not os.path.exists(args.save_directory):  # 新建保存文件夹
            os.makedirs(args.save_directory)
        work = args.work  # 根据输入work对应启动任务
        if work not in {'train', 'finetune', 'transfer'}:
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
        print(f"Start Train!  data_set_len: {self.data_set.__len__()}")

        tsfm_model = ImageTransformNet(32).to(device)
        loss_model = VGGNet().to(device).eval()

        current_lr = args.lr
        optimizer = torch.optim.Adam(tsfm_model.parameters(), lr=current_lr, betas=(0.9, 0.999), eps=1e-8)
        criterion = nn.MSELoss()
        weight_style = [1 / n ** 2 for n in [64, 128, 256, 512]]
        # weight_style = [1 / n ** 2 for n in [1, 1, 1, 1]]
        # load style img
        img_style = Image.open(args.style_t)
        img_style = img_style.resize((args.img_size, args.img_size))
        img_style = self.transform_input(img_style).unsqueeze(0).to(device)
        style = loss_model(img_style)  # 风格特征
        gram_target_style = [gram_matrix(_i).detach() for _i in style]  # 风格特征gram矩阵
        # 采集loss并在最后输出.csv文件
        collect_loss = [['index', 'batch_size', 'lr', 'total_loss', 'content_loss', 'style_loss', 'variation_loss']]
        idx_count = [-1]
        loss_record = []
        cost_time_record = []
        for i in range(args.epochs):
            for idx, img_content in enumerate(self.load_data):
                img_content = img_content.to(device)
                optimizer.zero_grad()

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
                # Style loss
                loss_style = 0
                for _j, sub_y in enumerate(y):
                    gram_temp = gram_matrix(sub_y)
                    loss_style += weight_style[_j] * criterion(gram_temp, gram_target_style[_j].expand_as(gram_temp))

                # Variation loss
                loss_variation = torch.sum(torch.pow((y2[:, :, :-1, 1:] - img_content[:, :, :-1, :-1]), 2)) +\
                                 torch.sum(torch.pow((y2[:, :, 1:, :-1] - img_content[:, :, :-1, :-1]), 2))

                weighted_loss_content = args.weight_content * loss_content
                weighted_loss_style = args.weight_style * loss_style
                weighted_loss_variation = args.weight_variation * loss_variation
                total_loss = weighted_loss_content + weighted_loss_style + weighted_loss_variation

                collect_loss.append([idx, args.batch_size, current_lr, total_loss.item(), weighted_loss_content.item(),
                                     weighted_loss_style.item(), weighted_loss_variation.item()])
                total_loss.backward()
                optimizer.step()

                if idx % 500 == 0:
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
        save_model_path = os.path.join(args.save_directory, time.strftime('%Y%m%d%H%M') + '_train_epoch_'
                                       + str(i) + ".pt")
        # 训练过程保存路径
        save_loss_path = os.path.join(args.save_directory, time.strftime('%Y%m%d%H%M') + '_train_loss.csv')
        torch.save(tsfm_model.state_dict(), save_model_path)
        self.writelist2csv(collect_loss, save_loss_path)  # 写入.csv文件
        print(f'--Save complete!\n--save_model_path: {save_model_path}\n--save_loss_path: {save_loss_path}')
        print('Train complete!')

    def finetune(self):
        print(f"Start Finetune!  data_set_len: {self.data_set.__len__()}")
        tsfm_model = ImageTransformNet(32).to(device)
        tsfm_model.load_state_dict(torch.load(self.model_path))  # 模型参数加载
        loss_model = VGGNet().to(device).eval()

        current_lr = args.lr
        optimizer = torch.optim.Adam(tsfm_model.parameters(), lr=current_lr, betas=(0.9, 0.999), eps=1e-8)
        criterion = nn.MSELoss()
        weight_style = [1 / n ** 2 for n in [64., 128., 256., 512.]]

        # style img
        img_style = Image.open(args.style_t)
        img_style = img_style.resize((args.img_size, args.img_size))
        img_style = self.transform_input(img_style).unsqueeze(0).to(device)
        style = loss_model(img_style)  # 风格特征
        gram_target_style = [gram_matrix(_i).detach() for _i in style]  # 风格特征gram矩阵

        # 采集loss并在最后输出.csv文件
        collect_loss = [['index', 'batch_size', 'lr', 'total_loss', 'content_loss', 'style_loss', 'variation_loss']]
        idx_count = [-1]
        loss_record = []
        cost_time_record = []
        for i in range(args.epochs):
            for idx, img_content in enumerate(self.load_data):
                img_content = img_content.to(device)
                optimizer.zero_grad()

                content = loss_model(img_content)[2].detach()  # 内容特征
                y0 = (tsfm_model(img_content) + 1) / 2. * 255.
                y = loss_model(y0)

                # Content loss
                loss_content = criterion(content, y[2])
                # Style loss
                loss_style = 0
                for _j in range(len(y)):
                    gram_temp = gram_matrix(y[_j])
                    loss_style += weight_style[_j] * criterion(gram_temp, gram_target_style[_j].expand_as(gram_temp))
                # Variation loss
                loss_variation = criterion(y0[:, :, :-1, 1:], img_content[:, :, :-1, :-1]) + \
                                 criterion(y0[:, :, 1:, :-1], img_content[:, :, :-1, :-1])

                weighted_loss_content = args.weight_content * loss_content
                weighted_loss_style = args.weight_style * loss_style
                weighted_loss_variation = args.weight_variation * loss_variation
                total_loss = weighted_loss_content + weighted_loss_style + weighted_loss_variation

                collect_loss.append([idx, args.batch_size, current_lr, total_loss.item(), weighted_loss_content.item(),
                                     weighted_loss_style.item(), weighted_loss_variation.item()])
                total_loss.backward()
                optimizer.step()

                if idx % 500 == 0:
                    idx_count.append(idx_count[-1] + 1)
                    loss_record.append([total_loss.item(), weighted_loss_content.item(), weighted_loss_style.item(),
                                        weighted_loss_variation.item()])
                    cost_time_record.append(time.time() - self.start_time)
                    vis.line(X=idx_count[1:], Y=loss_record, win='chart1', opts=opts1)
                    vis.line(X=idx_count[1:], Y=cost_time_record, win='chart2', opts=opts2)
                    self.interval_plot(self.transform_output(y1[0].clone()),
                                       self.transform_output(img_content[0].clone()),
                                       self.transform_output(img_style[0].clone()))
            self.interval_plot(self.transform_output(y1[0].clone()),
                               self.transform_output(img_content[0].clone()),
                               self.transform_output(img_style[0].clone()))
        save_model_path = self.model_path[:-3] + '_finetune_' + str(i) + ".pt"  # 模型保存路径
        save_loss_path = self.model_path[:-3] + '_finetune_' + str(i) + "_loss.csv"  # 训练过程保存路径
        torch.save(tsfm_model.state_dict(), save_model_path)
        self.writelist2csv(collect_loss, save_loss_path)  # 写入.csv文件
        print(f'--Save complete!\n--save_model_path: {save_model_path}\n--save_loss_path: {save_loss_path}')
        print('Finetune complete!')

    def transfer(self):
        print(f"Start Transfer!")
        if not os.path.exists(args.save_transfer_img):  # 新建转换图片保存文件夹
            os.makedirs(args.save_transfer_img)
        for root, dirs, files in os.walk(args.transfer_img_folder):
            pass
        tsfm_model = ImageTransformNet(32).to(device)
        tsfm_model.load_state_dict(torch.load(self.model_path))  # 模型参数加载
        data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        for file in files:
            img = Image.open(root + file)
            if img.mode != 'RGB':  # 转换所有图片模式为RGB
                img = img.convert('RGB')
            w, h = img.size
            new_img = img.resize((args.img_size, args.img_size))  # 将短的边缩放至self.re_size
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
    parser.add_argument('--style-t', type=str, default='style_img/style.jpg',
                        help='style image path')
    parser.add_argument('--batch-size', type=int, default=3, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--weight-content', type=float, default=1e0, metavar='N',
                        help='the weight of content')
    parser.add_argument('--weight-style', type=float, default=1e2, metavar='N',
                        help='the weight of content')
    parser.add_argument('--weight-variation', type=float, default=1e-4, metavar='N',
                        help='the weight of content')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--img-size', type=int, default=256, metavar='N',
                        help='image size for train')
    parser.add_argument('--save-directory', type=str, default='save_model',
                        help='learnt models are saving here')
    parser.add_argument('--transfer-img-folder', type=str, default='transfer_img/',
                        help='transfer image dictionary')
    parser.add_argument('--show-transfer-img', type=bool, default=True,
                        help='show transfer img')
    parser.add_argument('--save-transfer-img', type=str, default='result/',
                        help='transfer img are saving here')
    parser.add_argument('--work', type=str, default='transfer',
                        help='train, finetune, transfer')
    args = parser.parse_args()

    # visdom可视化设置
    vis = Visdom(env="perceptual loss image transfer 20200920")
    assert vis.check_connection()
    opts1 = {
        "title": 'loss with batch count',
        "xlabel": 'batch count',
        "ylabel": 'loss',
        "width": 1000,
        "height": 400,
        "legend": ['total', 'content', 'style', 'variation']
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
    MultiWorks(model_path='save_model/202010012307_train_epoch_1.pt')
