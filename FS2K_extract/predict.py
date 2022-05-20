import os
import re

import numpy as np
import pandas as pd
import torch
import json

from sklearn import metrics
from torch.utils.data import DataLoader

from FS2K_extract.FS2K_extract.hparams import hparams
from FS2K_extract.FS2K_extract.data_utils import FS2KSet
from FS2K_extract.FS2K_extract.model import my_vgg16, ResNet, resnet18, DenseNet121

here = os.path.dirname(os.path.abspath(__file__))

def predict_valid_data(hparams):
    device = hparams.device
    seed = hparams.seed
    torch.manual_seed(seed)

    tagset_file = hparams.tagset_file
    model_file = hparams.model_file
    test_batch_size = hparams.test_batch_size
    test_anno = hparams.test_file

    # 准备数据集
    test_dataset = FS2KSet('datasets/' + hparams.data_type, test_anno)
    # 把数据集装载到DataLoader里
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=test_batch_size)

    if hparams.model == 'vgg16':
        model = my_vgg16().to(device)
    if hparams.model == 'resnet':
        model = resnet18().to(device)
    if hparams.model == 'densenet':
        N_CLASSES = 13
        model = DenseNet121(N_CLASSES).to(device)

    model.load_state_dict(torch.load( model_file ))
    model.eval()

    with torch.no_grad():
        y_hat_df = pd.DataFrame(columns=['hair', 'gender', 'earring', 'smile', 'frontal_face', 'hair_color', 'style'])
        y_pred_df = pd.DataFrame(columns=['hair', 'gender', 'earring', 'smile', 'frontal_face', 'hair_color', 'style'])
        for data in test_loader:
            # 64batch 测试
            X, y = data
            y_pred = model(X.to(device))
            # 预测每一个batch的属性结果
            y_hat_df_temp = pd.DataFrame(columns=['hair', 'gender', 'earring', 'smile', 'frontal_face', 'hair_color', 'style'])
            y_pred_df_temp = pd.DataFrame(columns=['hair', 'gender', 'earring', 'smile', 'frontal_face', 'hair_color', 'style'])
            # 二分类标签预测
            for i in range(5):
                y_pred[y_pred[:, i] <= 0.5, i] = 0
                y_pred[y_pred[:, i] > 0.5, i] = 1
                y_hat_df_temp.iloc[:, i] = list(y[:, i].cpu())
                y_pred_df_temp.iloc[:, i] = list(y_pred[:, i].cpu())

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
        classprec_df.loc['avg'] = {'f1': f1,'precision': precision, 'recall': recall, 'accuracy': accuracy}
        classprec_df.to_csv("test_class_info.csv", encoding='utf-8')


if __name__ == '__main__':
    predict_valid_data(hparams)
