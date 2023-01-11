import torch
import albumentations as transform
import cv2
import albumentations as transform
from albumentations.pytorch import ToTensorV2

device = "cuda" if torch.cuda.is_available() else "cpu"

bach_size = 1
learning_rate = 0.001
epochs = 10
input_size = 2048
num_layer = 2
num_hidden_size = 1024
class_number = 101

train_transforms = transform.Compose(
    [transform.Resize(width=299, height=299),
        transform.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.8),
        transform.RandomBrightnessContrast(contrast_limit=0.5, brightness_limit=0.5, p=0.2),
        transform.OneOf([
            transform.GaussNoise(p=0.8),
            transform.CLAHE(p=0.8),
            transform.ImageCompression(p=0.8),
            transform.RandomGamma(p=0.8),
            transform.Posterize(p=0.8),
            transform.Blur(p=0.8),
        ], p=1.0),
        transform.OneOf([
            transform.GaussNoise(p=0.8),
            transform.CLAHE(p=0.8),
            transform.ImageCompression(p=0.8),
            transform.RandomGamma(p=0.8),
            transform.Posterize(p=0.8),
            transform.Blur(p=0.8),
        ], p=1.0),
        transform.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.2, border_mode=cv2.BORDER_CONSTANT),
        # transform.Normalize(
        #     mean=[0.5, 0.5, 0.5],
        #     std=[0.5, 0.5, 0.5]
        # ),
        ToTensorV2(),
    ])

test_transforms = transform.Compose([transform.Resize(width=299, height=299), ToTensorV2()])
