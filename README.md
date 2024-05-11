## 项目概述

​		本项目需要根据`FS2K数据集`进行训练和测试，实现输入一张图片，输入该图片的属性特征信息，提取属性特征包括`hair`（有无头发）、`hair_color`(头发颜色)、`gender`（图像人物性别）、`earring`（是否有耳环）、`smile`（是否微笑）、`frontal_face`(是否歪脖)、`style`（图片风格），详细信息均可通过FS2K的`anno_train.json`和`anno_test.json`获取，本质是一个多标签分类问题。

- 本文探索了三种深度学习模型：VGG16、ResNet18和DenseNet121在该任务下的性能表现。
- 具体实验结果可在github仓库中的experiment_result文件夹获取
- 实验数据集可在datasets文件夹获取

## 处理方案

​        首先对于FS2K数据集用官方的数据划分程序进行划分，之后对划分后的数据进行预处理，统一图片后缀为jpg，之后自定义数据加载类，在数据加载过程中进行标签编码，对图片大小进行统一，并转成tensor，在处理过程中发现存在4个通道的图片，本文采取取前3个通道的方案，之后再对图像进行标准化，可以加快模型的收敛，处理完成的数据作为模型的输入，在深度学习模型方面，首先需要进行模型选择，本文使用了三个模型，分别为VGG16,ResNet121以及DenseNet121，在通过pytorch预训练模型进行加载，并修改模型输出层，输出数量为图片属性特征数，之后在设定模型训练的参数，包括Batch，学习率，epoch等，在每一轮训练完成后，都需要对预测出的特征进行处理，在二分类标签设定概率阈值，多分类标签特征列则进行最大概率类别组合，取预测概率最大的类别作为当前属性的预测结果，每一轮训练都在测试集上进行性能评估，并根据F1指标择优保存模型。训练完成后，在测试集上预测属性提取结果，对每一个属性进行性能评估，最后取平均，得到平均的性能指标。

## Quick start
- 用resnet训练
```shell
   python demo_train.py --model resnet --data_type photo
```
通过指定--model 可切换不同模型，支持vgg、resnet、densenet。指定--data_type 可切换photo或者sketch数据集。
- 预测
```shell
   python demo_predict.py --model resnet --data_type photo
```
参数说明与训练一致。
