import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import precision_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from FS2K_extract.FS2K_extract.data_utils import FS2KSet, load_checkpoint, save_checkpoint
from FS2K_extract.FS2K_extract.model import my_vgg16, ResNet, resnet18, DenseNet121


def train(hparams):
    device = hparams.device
    seed = hparams.seed
    torch.manual_seed(seed)

    train_model =  hparams.model
    train_anno = hparams.train_file
    test_anno = hparams.test_file
    log_dir = hparams.log_dir
    tagset_file = hparams.tagset_file
    model_file = hparams.model_file
    checkpoint_file = hparams.checkpoint_file

    train_batch_size = hparams.train_batch_size
    test_batch_size = hparams.test_batch_size
    epochs = hparams.epochs

    # 准备数据集
    train_dataset = FS2KSet('datasets/' + hparams.data_type, train_anno)
    # 把数据集装载到DataLoader里
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    # 准备数据集
    test_dataset = FS2KSet('datasets/' + hparams.data_type, test_anno)
    # 把数据集装载到DataLoader里
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=test_batch_size)

    # model
    if train_model == 'vgg16':
        model = my_vgg16().to(device)
    if train_model == 'resnet':
        model = resnet18().to(device)
    if train_model == 'densenet':
        N_CLASSES = 13
        model = DenseNet121(N_CLASSES).to(device)
    print(model)
    torch.cuda.empty_cache()
    checkpoint_dict = {}
    best_f1 = 0.0
    epoch_offset = 0
    # load checkpoint if one exists
    # if os.path.exists(checkpoint_file):
    #     checkpoint_dict = load_checkpoint(checkpoint_file)
    #     # best_f1 = checkpoint_dict['f1']
    #     # epoch_offset = checkpoint_dict['best_epoch'] + 1
    #     model.load_state_dict(torch.load(model_file))
    # else:
    #     checkpoint_dict = {}
    #     best_f1 = 0.0
    #     epoch_offset = 0

    # 损失函数
    criterion = nn.BCELoss().to(device)
    # 使用带有动量的随机梯度下降
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    # 用于存储损失
    loss_list = []

    running_loss = 0.0
    writer = SummaryWriter(os.path.join(log_dir, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        model.train()
        for batch, (X, y) in enumerate(train_loader):
            # 正向传播
            y_pred= model(X.to(device))
            # 计算损失
            loss = criterion(y_pred, y.to(device))
            # 梯度归零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数 卷积核参数 全连接参数
            optimizer.step()
            # 每300次看下损失
            if batch % 300 == 0:
                loss_list.append(loss.data.item())
                print("loss------------", loss.data.item())

        if test_anno:
            model.eval()
            with torch.no_grad():
                y_hat_df = pd.DataFrame(columns=['hair', 'gender', 'earring', 'smile', 'frontal_face', 'hair_color', 'style'])
                y_pred_df = pd.DataFrame(columns=['hair', 'gender', 'earring', 'smile', 'frontal_face', 'hair_color', 'style'])

                # 64batch 测试
                for data in test_loader:
                    X, y = data
                    y_pred = model(X.to(device))
                    # 预测每一个batch的属性结果
                    y_hat_df_temp = pd.DataFrame(columns=['hair', 'gender', 'earring', 'smile', 'frontal_face', 'hair_color', 'style'])
                    y_pred_df_temp = pd.DataFrame(columns=['hair', 'gender', 'earring', 'smile', 'frontal_face', 'hair_color', 'style'])
                    # 二分类标签预测
                    for i in range(5):
                        y_pred[y_pred[:, i] <= 0.5, i] = 0
                        y_pred[y_pred[:, i] > 0.5, i] = 1
                        y_hat_df_temp.iloc[:, i] = list(y[:, i])
                        y_pred_df_temp.iloc[:, i] = list(y_pred[:, i])

                    # 选取概率最大的hair_color 作为 该列的值
                    hari_values_pred = []
                    hari_values_hat = []
                    for i in range(len(y_pred)):
                        hari_values_pred.append(np.argmax(y_pred[i, 5:10].cpu()))
                    for i in range(len(y_pred)):
                        hari_values_hat.append(np.argmax(y[i, 5:10]))

                    y_hat_df_temp.iloc[:, 5] = list(hari_values_hat)
                    y_pred_df_temp.iloc[:, 5] = list(hari_values_pred)

                    # 选取概率最大的style 作为 该列的值
                    style_values_pred = []
                    style_values_hat = []
                    for i in range(len(y_pred)):
                        style_values_pred.append(np.argmax(y_pred[i, 10:13].cpu()))
                    for i in range(len(y_pred)):
                        style_values_hat.append(np.argmax(y[i, 10:13]))

                    y_hat_df_temp.iloc[:, 6] = list(style_values_hat)
                    y_pred_df_temp.iloc[:, 6] = list(style_values_pred)

                    y_hat_df = y_hat_df.append(y_hat_df_temp)
                    y_pred_df = y_pred_df.append(y_pred_df_temp)

                y_hat_df.to_csv("y_hat_df.csv", encoding='utf-8')
                y_pred_df.to_csv("y_pred_df.csv", encoding='utf-8')

                y_hat_df = pd.read_csv("y_hat_df.csv", encoding='utf-8')
                y_pred_df = pd.read_csv("y_pred_df.csv", encoding='utf-8')

                f1_all = []
                precision_all = []
                recall_all = []
                accuracy_all = []
                classprec_df = pd.DataFrame(columns = ['f1', 'precision', 'recall', 'accuracy'] )
                for column in y_pred_df.columns:
                    class_dict = {}
                    f1 = metrics.f1_score(y_hat_df[column], y_pred_df[column], average='weighted')
                    f1_all.append(f1)
                    precision = metrics.precision_score(y_hat_df[column], y_pred_df[column], average='weighted')
                    precision_all.append(precision)
                    recall = metrics.recall_score(y_hat_df[column], y_pred_df[column], average='weighted')
                    recall_all.append(recall)
                    accuracy = metrics.accuracy_score(y_hat_df[column], y_pred_df[column])
                    accuracy_all.append(accuracy)

                    class_dict['f1'] = f1
                    class_dict['precision'] = precision
                    class_dict['recall'] = recall
                    class_dict['accuracy'] = accuracy
                    classprec_df.loc[column] = class_dict
                f1 = sum(f1_all) / len(f1_all)
                precision = sum(precision_all) / len(precision_all)
                recall = sum(recall_all) / len(recall_all)
                accuracy = sum(accuracy_all) / len(accuracy_all)

                writer.add_scalar('Test/f1', f1, epoch)
                writer.add_scalar('Test/precision', precision, epoch)
                writer.add_scalar('Test/recall', recall, epoch)
                writer.add_scalar('Test/accuracy', accuracy, epoch)

                if checkpoint_dict.get('epoch_f1'):
                    checkpoint_dict['epoch_f1'][epoch] = f1
                else:
                    checkpoint_dict['epoch_f1'] = {epoch: f1}
                if f1 > best_f1:
                    best_f1 = f1
                    checkpoint_dict['best_f1'] = best_f1
                    checkpoint_dict['best_epoch'] = epoch
                    torch.save(model.state_dict(), model_file)
                save_checkpoint(checkpoint_dict, checkpoint_file)

    writer.close()

    # 显示损失下降的图像
    plt.plot(np.linspace(0, 1000, len(loss_list)), loss_list)
    plt.show()
