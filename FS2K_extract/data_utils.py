import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from sklearn.preprocessing import OneHotEncoder
# 定义自己的数据集合
import json
import os
from PIL import Image
import numpy as np

train_path = '../datasets/anno_train.json'
test_path = '../datasets/anno_test.json'

# 把数据转成tensor,并遵从正态分布
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def save_checkpoint(checkpoint_dict, file):
    with open(file, 'w', encoding='utf-8') as f_out:
        json.dump(checkpoint_dict, f_out, ensure_ascii=False, indent=2)

def load_checkpoint(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        checkpoint_dict = json.load(f_in)
    return checkpoint_dict

# 自定义数据集
class FS2KSet(Dataset):
    def __init__(self, root, json_path):
        # 所有图片的绝对路径
        with open(json_path) as f:
            row_data = json.load(f)
        # print(row_data)
        # self.imgs = [os.path.join(root, item['image_name']) + '.jpg' for item in row_data]
        self.imgs = [os.path.join(root, item['image_name'].replace('photo', 'sketch').replace('image', 'sketch')) + '.jpg' for item in row_data]

        # one-hot编码
        hair_color_dict = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
        style = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.labels = [[item['hair'], item['gender'], item['earring'], item['smile'], item['frontal_face']] for item in
                       row_data]
        for index, item in enumerate(row_data):
            self.labels[index].extend(hair_color_dict[int(item['hair_color'])])
            self.labels[index].extend(style[int(item['style'])])
        self.transforms = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        # if 'sketch2/' in img_path:
        #     img_path = img_path.replace('jpg', 'png')
        pil_img = Image.open(img_path)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        label = torch.FloatTensor(self.labels[index])
        if data.shape[0] != 3:
            data = data[:3,:,:]
        return data, label

    def __len__(self):
        return len(self.imgs)



if __name__ == '__main__':
    fs = FS2KSet('../datasets/photo', train_path)
    print(fs.__getitem__(0))
