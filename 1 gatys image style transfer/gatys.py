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


class VGGNet(nn.Module):  # VGG backbone
    def __init__(self):
        super(VGGNet, self).__init__()
        model = models.vgg19(pretrained=False)
        model_load = torch.load(args.vgg_model_path)

        new_model_static_dict = model.state_dict()
        for k in new_model_static_dict.keys():
            if k in model_load.keys():
                new_model_static_dict[k] = model_load[k]
        model.load_state_dict(new_model_static_dict)
        
        features = list(model.features[:30])  # 使用预加载的前30层
        self.features = nn.ModuleList(features)
        for p in self.parameters():  # 仅使用VGG网络输出值，故关闭参数梯度计算加快速度
            p.requires_grad = False

    def forward(self, x):
        style = []
        for _i, layer in enumerate(self.features):
            x = layer(x)
            if _i == 22:    # relu_4_2
                content = x
            if _i in {1, 6, 11, 20, 29}:    # relu_1_1/relu_2_1/relu_3_1/relu_4_1/relu_5_1/
                style.append(x)
        return content, style


def gram_matrix(x):     # gram 矩阵
    (batch, chnls, h, w) = x.size()
    x = x.view(batch, chnls, h * w)
    x_t12 = x.transpose(1, 2)
    gram = x.bmm(x_t12) / (h * w)
    return gram


def transfer():  # 计算转换后图片并存储
    start_time = time.time()  # 开始时间，用于打印用时
    img_content = Image.open(args.content_target)  # 打开内容图片
    img_style = Image.open(args.style_target)  # 打开风格图片
    s_w, s_h = img_content.size
    s_w, s_h = int(args.scale * s_w), int(args.scale * s_h)
    print(s_w, s_h)
    transform_input = transforms.Compose([transforms.Resize((s_h, s_w)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.48501961, 0.45795686, 0.40760392],
                                                               std=[1., 1., 1.]),
                                          transforms.Lambda(lambda x: x[[2, 1, 0]]),
                                          transforms.Lambda(lambda x: x.mul(255.))])
    transform_output = transforms.Compose([transforms.Lambda(lambda x: x.mul(1. / 255)),
                                           transforms.Lambda(lambda x: x[[2, 1, 0]]),  # turn to RGB
                                           transforms.Normalize(mean=[-0.48501961, -0.45795686, -0.40760392],
                                                                std=[1, 1, 1])
                                           ])
    transform_to_pil = transforms.Compose([transforms.ToPILImage()])

    img_content = transform_input(img_content).unsqueeze(0).to(device)  # 转换+变维+送入设备
    img_style = transform_input(img_style).unsqueeze(0).to(device)  # 转换+变维+送入设备

    img_input = img_content.clone()     # 拷贝内容图片作为初始输入
    # img_input = 255. * (torch.rand(s_w, s_h, 3)).permute(2, 0, 1).unsqueeze(0).to(device)
    loss_model = VGGNet().to(device).eval()

    optimizer = torch.optim.LBFGS([img_input.requires_grad_()])  # 优化函数
    criterion = nn.MSELoss()  # 损失评价函数
    weight_style = [1 / n ** 2 for n in [64, 128, 256, 512, 512]]

    content = loss_model(img_content)[0].detach()  # 内容特征
    style = loss_model(img_style)[1]  # 风格特征
    gram_target_style = [gram_matrix(_i).detach() for _i in style]  # 风格特征gram矩阵

    count = [0]     # 迭代次数，供visdom显示使用
    loss_record = []  # loss，供visdom显示使用
    cost_time_record = []  # 用时，供visdom显示使用

    def f():
        count[0] += 1
        optimizer.zero_grad()  # 梯度清零
        y_content, y_style = loss_model(img_input)

        # content loss
        loss_content = criterion(content, y_content)
        # style loss
        gram_y_style = [gram_matrix(_i) for _i in y_style]
        loss_style = 0
        for _j in range(len(gram_y_style)):
            loss_style += weight_style[_j] * criterion(gram_y_style[_j], gram_target_style[_j])

        weighted_loss_content = args.weight_content * loss_content
        weighted_loss_style = args.weight_style * loss_style
        total_loss = weighted_loss_content + weighted_loss_style
        # visdom显示
        count.append(count[0])
        print(count[0])
        loss_record.append([total_loss.item(), weighted_loss_content.item(), weighted_loss_style.item()])
        cost_time_record.append(time.time() - start_time)
        vis.line(X=count[1:], Y=loss_record, win='chart1', opts=opts1)
        vis.line(X=count[1:], Y=cost_time_record, win='chart2', opts=opts2)
        total_loss.backward()
        return total_loss

    while count[0] < args.iteration:
        if count[0] % 100 == 0:  # 绘制中间过程
            interval_plot(transform_output(img_input.clone()[0]),
                          transform_output(img_content.clone()[0]),
                          transform_output(img_style.clone()[0]))
        optimizer.step(f)
    output = transform_output(img_input.clone()[0])
    interval_plot(output,
                  transform_output(img_content.clone()[0]),
                  transform_output(img_style.clone()[0]))
    output[output > 1] = 1
    output[output < 0] = 0
    img_save_path = args.img_save_path + args.style_target.split('/')[-1].split('.')[0] + '+' + \
                    args.content_target.split('/')[-1]
    img = transform_to_pil(output.cpu().detach())
    img.save(img_save_path)  # 图片存储
    print('Transfer complete!')


def writelist2csv(list_data, csv_name):  # 列表写入.csv
    with open(csv_name, "w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for one_slice in list_data:
            csv_writer.writerow(one_slice)


def interval_plot(y_hat, img_content, img_style):
    y_hat = y_hat.cpu().detach().numpy().transpose(1, 2, 0)
    img_content = img_content.cpu().detach().numpy().transpose(1, 2, 0)
    img_style = img_style.cpu().detach().numpy().transpose(1, 2, 0)
    plt.subplot(131)
    plt.imshow(y_hat.clip(0, 1))
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
    parser.add_argument('--content-target', type=str, default='./input_img/content11.jpg',
                        help='content image path')
    parser.add_argument('--style-target', type=str, default='./input_img/style8.jpg',
                        help='style image path')
    parser.add_argument('--vgg-model-path', type=str, default='./vgg19-d01eb7cb.pth',
                        help='vgg model path')
    parser.add_argument('--weight-content', type=float, default=1e0, metavar='N',
                        help='the weight of content')
    parser.add_argument('--weight-style', type=float, default=1e3, metavar='N',
                        help='the weight of content')
    parser.add_argument('--scale', type=float, default=0.2, metavar='N',
                        help='the ratio to scale content for output')
    parser.add_argument('--iteration', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1, metavar='LR',
                        help='learning rate (default: 1)')
    parser.add_argument('--img-save-path', type=str, default='result_img/',
                        help='transfer img save path')
    args = parser.parse_args()

    # visdom可视化设置
    vis = Visdom(env="gatys image style transfer 20200903 9")
    assert vis.check_connection()
    opts1 = {
        "title": 'loss of total_loss/content_loss/style_loss with epoch',
        "xlabel": 'epoch',
        "ylabel": 'loss',
        "width": 1000,
        "height": 400,
        "legend": ['total_loss', 'content_loss', 'style_loss']
    }
    opts2 = {
        "title": 'cost time with epoch',
        "xlabel": 'epoch',
        "ylabel": 'time in second',
        "width": 1000,
        "height": 400,
        "legend": ['cost time']
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transfer()
