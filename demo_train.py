import os

from FS2K_extract.FS2K_extract.hparams import hparams
from FS2K_extract.FS2K_extract.train import train

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
here = os.path.dirname(os.path.abspath(__file__))


def main():
    train(hparams)

if __name__ == '__main__':
    main()
