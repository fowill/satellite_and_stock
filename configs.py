import torch

#IMAGE_SIZE
IMAGE_HEIGHT, IMAGE_WIDTH = 480, 480

#Training Directory
TRAIN_IMG_DIR = "./data/training/train_images"
TRAIN_MASK_DIR = "./data/training/train_labels"
TRAIN_REGION_DIR = "./data/training/train_regions"
VAL_IMG_DIR = "./data/training/val_images"
VAL_MASK_DIR = "./data/training/val_labels"
VAL_REGION_DIR = "./data/training/val_regions"
TEST_IMG_DIR = "./data/training/test_images"
TEST_MASK_DIR = "./data/training/test_labels"
TEST_REGION_DIR = "./data/training/test_regions"

TRAINING_MODEL_SAVE_DIR = "./data/training/epoch_models"
TRAINING_RESULT_SAVE_DIR = "./data/training/training_results"

#Training Parameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 1
NUM_WORKERS = 1
PIN_MEMORY = True
LOAD_MODEL = False
K_FOLD_NUM = 10

#Cloud Percentage
CLOUD_PERCENT_PATH = './data/predicting/cloud_percent.csv'

#Predicting Directory
CHECKPOINT_PATH = 'my_checkpoint.pth.tar'
PREDICT_DATA_PATH = './data/predicting/predict_data'
PREDICTED_IMG_PATH = './data/predicting/saved_images'
PREDICTED_PREDS_PATH = './data/predicting/saved_preds'
SUBREGION_PATH = './data/predicting/subregions'







