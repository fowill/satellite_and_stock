import os

if __name__ == '__main__':
    TRAIN = False
    PREDICT = True

    if TRAIN:
        os.system("python train.py")
    if PREDICT:
        os.system("python predict.py")

