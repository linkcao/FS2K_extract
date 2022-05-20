import os

from FS2K_extract.FS2K_extract.hparams import hparams
from FS2K_extract.FS2K_extract.predict import predict_valid_data

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
here = os.path.dirname(os.path.abspath(__file__))


def main():
    predict_valid_data(hparams)

if __name__ == '__main__':
    main()
