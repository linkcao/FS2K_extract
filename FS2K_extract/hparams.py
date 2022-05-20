import os
import argparse

here = os.path.dirname(os.path.abspath(__file__))

default_train_anno = './datasets/anno_train.json'
default_test_file = './datasets/anno_test.json'
default_data_type = 'sketch'

default_output_dir =  './saved_models'
default_log_dir = os.path.join(default_output_dir, 'runs')
default_tagset_dir = './saved_models'
default_model_file = os.path.join(default_output_dir, 'model.bin')
default_checkpoint_file = os.path.join(default_output_dir, 'checkpoint.json')

parser = argparse.ArgumentParser()

# file
parser.add_argument("--train_file", type=str, default=default_train_anno)
parser.add_argument("--test_file", type=str, default=default_test_file)
parser.add_argument("--output_dir", type=str, default=default_output_dir)
parser.add_argument("--log_dir", type=str, default=default_log_dir)
parser.add_argument("--tagset_file", type=str, default=default_tagset_dir)
parser.add_argument("--model_file", type=str, default=default_model_file)
parser.add_argument("--data_type", type=str, default=default_data_type)
parser.add_argument("--checkpoint_file", type=str, default=default_checkpoint_file)

# model
parser.add_argument("--model", type=str, default='resnet')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--test_batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=20)

hparams = parser.parse_args()
