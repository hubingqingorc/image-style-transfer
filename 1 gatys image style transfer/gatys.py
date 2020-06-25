# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
import argparse
import numpy as np
import time
import csv
from PIL import Image
import os
from visdom import Visdom


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


class VGGNet(nn.Module):    # VGG backbone
    def __init__(self):
        super(VGGNet, self).__init__()
        features = list(models.vgg16(pretrained=True).features)[:26]    # 使用预加载的前26层
        self.features = nn.ModuleList(features)
        for p in self.parameters():     # 仅使用VGG网络输出值，故关闭参数梯度计算加快速度
            p.requires_grad = False

    def forward(self, x):
        style = []
        for _i, layer in enumerate(self.features):
            x = layer(x)
            if _i == 20:    # relu_4_2
                content = x
            if _i in [1, 6, 11, 18, 25]:    # relu_1_1/relu_2_1/relu_3_1/relu_4_1/relu_5_1/
                style.append(x)
        return content, style


def gram_matrix(x):     # gram 矩阵
    (batch, chnls, h, w) = x.size()
    x = x.view(batch, chnls, h * w)
    x_t12 = x.transpose(1, 2)
    gram = x.bmm(x_t12) / (chnls * h * w)
    return gram


def transfer():     # 计算转换后图片并存储
    start_time = time.time()    # 开始时间，用于打印用时
    img_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.ToTensor()])
    img_content = Image.open(args.content_target)   # 打开内容图片
    img_content = img_transform(img_content).unsqueeze(0).to(device)    # 转换+变维+送入设备
    img_style = Image.open(args.style_target)   # 打开风格图片
    img_style = img_transform(img_style).unsqueeze(0).to(device)    # 转换+变维+送入设备
    img_input = img_content.clone()     # 拷贝内容图片作为初始输入
    
    loss_model = VGGNet().to(device).eval()

    optimizer = torch.optim.LBFGS([img_input.requires_grad_()], lr=args.lr)     # 优化函数
    criterion = nn.MSELoss(reduction='sum')     # 损失评价函数
    # 采集loss并在最后输出.csv文件
    content = loss_model(img_content)[0].detach()   # 内容特征
    style = loss_model(img_style)[1]    # 风格特征
    gram_target_style = [gram_matrix(_i).detach() for _i in style]  # 风格特征gram矩阵
    epoch_count = []    # 轮次，供visdom显示使用
    loss_record = []    # loss，供visdom显示使用
    cost_time_record = []   # 用时，供visdom显示使用
    for i in range(args.epochs):
        def f():
            optimizer.zero_grad()   # 梯度清零
            y_content, y_style = loss_model(img_input)

            # content loss
            loss_content = criterion(content, y_content)
            # style loss
            gram_y_style = [gram_matrix(_i) for _i in y_style]
            loss_style = 0
            for _j in range(len(gram_y_style)):
                loss_style += criterion(gram_y_style[_j], gram_target_style[_j])

            weighted_loss_content = args.weight_content * loss_content
            weighted_loss_style = args.weight_style * loss_style            
            total_loss = weighted_loss_content + weighted_loss_style
            # visdom显示
            epoch_count.append(i + 1)
            loss_record.append([total_loss.item(), weighted_loss_content.item(), weighted_loss_style.item()])
            cost_time_record.append(time.time() - start_time)
            vis.line(X=epoch_count, Y=loss_record, win='chart1', opts=opts1)
            vis.line(X=epoch_count, Y=cost_time_record, win='chart2', opts=opts2)
            total_loss.backward()
            return total_loss

        optimizer.step(f)
        if i % 1 == 0:     # 绘制中间过程
            interval_plot(img_input, img_content, img_style)
    img_save_path = args.img_save_path + args.style_target.split('/')[-1].split('.')[0] + '+' + \
                    args.content_target.split('/')[-1]
    img = img_input.cpu().detach().numpy()[0].transpose(1, 2, 0).clip(0, 1)
    img = np.array(np.rint(img * 255), dtype='uint8')
    img = Image.fromarray(img)
    img.save(img_save_path)     # 图片存储
    print('Transfer complete!')


def writelist2csv(list_data, csv_name):  # 列表写入.csv
    with open(csv_name, "w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for one_slice in list_data:
            csv_writer.writerow(one_slice)


def interval_plot(y_hat, img_content, img_style):
    y_hat = y_hat.cpu().detach().numpy()[0].transpose(1, 2, 0)
    img_content = img_content.cpu().detach().numpy()[0].transpose(1, 2, 0)
    img_style = img_style.cpu().detach().numpy()[0].transpose(1, 2, 0)
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
    parser.add_argument('--content-target', type=str, default='./input_img/content.jpg',
                        help='content image path')
    parser.add_argument('--style-target', type=str, default='./input_img/style.jpg',
                        help='style image path')
    parser.add_argument('--weight-content', type=float, default=1, metavar='N',
                        help='the weight of content')
    parser.add_argument('--weight-style', type=float, default=10000000, metavar='N',
                        help='the weight of content')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--img-save-path', type=str, default='result_img/',
                        help='transfer img save path')    
    args = parser.parse_args()

    # visdom可视化设置
    vis = Visdom(env="gatys image style transfer")
    assert vis.check_connection()
    opts1 = {
        "title": 'loss of total_loss/content_loss/style_loss with epoch',
        "xlabel": 'epoch',
        "ylabel": 'loss',
        "width": 600,
        "height": 400,
        "legend": ['total_loss', 'content_loss', 'style_loss']
    }
    opts2 = {
        "title": 'cost time with epoch',
        "xlabel": 'epoch',
        "ylabel": 'time in second',
        "width": 400,
        "height": 300,
        "legend": ['cost time']
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transfer()
