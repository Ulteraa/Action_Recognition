import cv2
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import config
import numpy as np
import torchvision
import torch.nn as nn
class Dataset_(Dataset):

    def __init__(self,  train=True, transform=None, n_frame=100):
        super(Dataset_, self).__init__()
        self.model = torchvision.models.inception_v3(pretrained=True).eval()
        self.model.fc = nn.Identity()
        for param in self. model.parameters():
            param.requires_grad = False
        #self.feature_extractor = torch.nn.Sequential(*list(self.model.features)[:-1])
        # self.root_text = root_text
        # self.root_data = root_data
        self.train = train
        self.transform = transform
        self.n_frame = n_frame
        self.dict = {}
        if self.train:
            path = 'address/train.txt'
        else:
            path = 'address/test.txt'
        with open(path, 'r') as file:
            self.ad = list(file.read().split('\n'))
        path_test_label='UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/classInd.txt'
        with open(path_test_label, 'r') as file:
            self.test_label = list(file.read().split('\n'))
            for item in self.test_label:
                sp = list(item.split(' '))
                if len(sp) >= 2:
                    self.dict.update({sp[1]: int(sp[0])})

    def __len__(self):
        return len(self.ad)

    def __getitem__(self, index):
        stack, label = self.get_frames(index)
        if self.transform:
            img_ = config.train_transforms(image=stack[0])
            img_ = img_['image'].unsqueeze(0)
        else:

            img_ = torch.Tensor(stack[0])
            img_ = img_.unsqueeze(0)
        for i in range(1, len(stack)):
            if self.transform:
                image_ = config.train_transforms(image=stack[i])
                image_ = image_['image'].unsqueeze(0)
                #img_ = torch.cat((img_, image_), dim=0)
            else:
                image_ = config.test_transforms(image=stack[i])
                image_ = image_['image'].unsqueeze(0)
            img_ = torch.cat((img_, image_), dim=0)
            # ans = img_[0][:, :, 2]
            # plt.imshow(ans)
            # plt.show()
        img_ = self.model(img_)

        return img_, (label-1)

    def get_frames(self, index):
        adr = self.ad[index]
        adr = adr.split(' ')
        if self.train:
            label = int(adr[1])
        else:
            sp = list(adr[0].split('/'))
            label = self.dict[sp[0]]

        path_ = os.path.join('UCF101/UCF-101', adr[0])
        cam = cv2.VideoCapture(path_)
        stack = []
        if not os.path.exists(path_):
            print('Error: Creating directory of data')
        currentframe = 0
        while (True):
            ret, frame = cam.read()
            if ret:
                 stack.append(frame)
                 currentframe += 1
            else:
                break
        # Release all space and windows once done
        cam.release()
        cv2.destroyAllWindows()
        if len(stack) >= self.n_frame:
            stack = stack[:self.n_frame]
        else:
            for i in range(self.n_frame-len(stack)):
                stack.append(stack[-1])
        return stack, label

    def pre_process(self):
        if self.train:
            path_address = 'UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist'
            adr = os.path.join(path_address, 'train')
            result_text = 'address/train.txt'
        else:
            path_address = 'UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist'
            adr = os.path.join(path_address, 'test')
            result_text = 'address/test.txt'
        list_ = os.listdir(adr)
        # if os.path.exists(result_text):
        #     pass
        # else:
        #     os.makedirs(result_text)
        with open(result_text, 'w') as wr_txt_file:
            for item in list_:
                text_adr = os.path.join(adr, item)
                with open(text_adr, 'r') as file:
                    L = file.read().rstrip()
                    for line in L:
                        if self.train:
                             wr_txt_file.write(line)
                        else:
                            wr_txt_file.write(line)

        wr_txt_file.close()
        file.close()


if __name__=='__main__':
    dataset_ = Dataset_(train=True, transform=True)
    stack = DataLoader(dataset_, batch_size=1, shuffle=False)
    img, label = next(iter(stack))
    print(label)
    print('k')
    # for i in range(100):
    #     image_ = img[0, i, 0,:,:]
    #     plt.imshow(image_)
    #     plt.show()
    #
    #     # print('stop')
    # #dataset_.pre_process(train=False)


