import ml_utils
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
import random

class ModulationConvEncoder(nn.Module):

    def __init__(self, num_classes):
        super(ModulationConvEncoder, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, 3, kernel_size=(3,1), stride=(1,1))
        self.conv2 = nn.Conv2d(3, 6, kernel_size=(3, 1), stride=(1, 1))
        self.conv3 = nn.Conv2d(6, 12, kernel_size=(5, 1), stride=(1, 1))
        self.conv4 = nn.Conv2d(12, 24, kernel_size=(5, 1), stride=(1, 1))
        self.conv5 = nn.Conv2d(24, 48, kernel_size=(10, 1), stride=(1, 1))
        self.conv6 = nn.Conv2d(48, 50, kernel_size=(15, 1), stride=(1, 1))
        self.conv7 = nn.Conv2d(50, 50, kernel_size=(20, 1), stride=(1, 1))

        self.conv8 = nn.Conv2d(50, 25, kernel_size=(3, 2), stride=(1, 1))
        self.conv9 = nn.Conv2d(25, 15, kernel_size=(5, 1), stride=(1, 1))
        self.conv10 = nn.Conv2d(15, 5, kernel_size=(10, 1), stride=(1, 1))
        self.conv11 = nn.Conv2d(5, 3, kernel_size=(10, 1), stride=(1, 1))
        self.conv12 = nn.Conv2d(3, 1, kernel_size=(10, 1), stride=(1, 1))

        decoder_nn = nn.Sequential(
            nn.Linear(in_features=2913, out_features=1000, bias=True),
            nn.Linear(in_features=1000, out_features=500, bias=True),
            nn.Linear(in_features=500, out_features=100, bias=True),
            nn.Linear(in_features=100, out_features=self.num_classes, bias=True),
        )

        self.relu = nn.ReLU()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0001, std=1.0)
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=1.5)

    def forward(self, x):

        out1 = self.relu(self.conv1(x))
        #print(out1.shape)
        out2 = self.relu(self.conv2(out1))
        #print(out2.shape)
        out3 = self.relu(self.conv3(out2))
        #print(out3.shape)
        out4 = self.relu(self.conv4(out3))
        #print(out4.shape)
        out5 = self.relu(self.conv5(out4))
        #print(out5.shape)
        out6 = self.relu(self.conv6(out5))
        #print(out6.shape)
        out7 = self.relu(self.conv7(out6))
        #print(out7.shape)
        out8 = self.relu(self.conv8(out7))
        #print(out8.shape)
        out9 = self.relu(self.conv9(out8))
        #print(out9.shape)
        out10 = self.relu(self.conv10(out9))
        #print(out10.shape)
        out11 = self.relu(self.conv11(out10))
        #print(out11.shape)
        out12 = self.conv12(out11)
        #print(out12.shape)

        flat_out = torch.flatten(out12, 1) # flatten all dimensions except the batch dimension

        return flat_out



# Dataset: https://www.kaggle.com/datasets/dimitrisf/constellation-dataset
class ModulationDataset(Dataset):

    def __init__(self, path='.\dataset\Dataset\Dataset\*.txt', normalize=True, device='cpu'):
        self.modulation_types = ['NO_SIGNAL', 'PAM4', 'PAM16', 'QPSK4', 'PSK8', 'APSK32', 'APSK64', 'QAM16', 'QAM32', 'QAM64']
        self.files = glob.glob(path)
        self.normalize = normalize
        self.num_classes = len(self.modulation_types)
        self.num_samples = 3000
        self.device = device


    def get_and_process_data(self, idx):
        file_name = self.files[idx]
        iq_data = np.genfromtxt(file_name, delimiter=",", skip_header=True)

        if iq_data.shape[0] < 3000:
            class_idx = torch.tensor(self.modulation_types.index('NO_SIGNAL'))
            return torch.zeros(self.num_samples, 2), 'NO_SIGNAL', F.one_hot(class_idx, num_classes=self.num_classes)
        else:
            iq_data = iq_data[0:self.num_samples,:]

        class_name = file_name.split('\\')[-1].split('_')[0]
        class_idx = torch.tensor(self.modulation_types.index(class_name))

        if self.normalize:
            # create scaler
            scaler_1 = MinMaxScaler() # Set range to 0-1
            scaler_2 = StandardScaler() # set mean at 0 with std 1

            scaler_1.fit(iq_data)
            iq_data = scaler_1.transform(iq_data)

            scaler_2.fit(iq_data)
            iq_data = scaler_2.transform(iq_data)

        return iq_data, class_name, class_idx


    def __getitem__(self, idx):

        iq_data, class_name, class_idx = self.get_and_process_data(idx)

        return torch.Tensor(iq_data).to(self.device), class_name, F.one_hot(class_idx, num_classes=self.num_classes).to(self.device)

    def get_two_random_samples(self):

        idx1 = random.randint(0, self.num_classes - 1)
        idx2 = random.randint(0, self.num_classes - 1)

        return self.__getitem__(idx1), self.__getitem__(idx2)

    def __len__(self):
        return len(self.files)


def calc_contrastive_loss(vec1, vec2, y, threshold):

    mse_loss = nn.MSELoss()(vec1, vec2)
    max_of = torch.Tensor([0, threshold - mse_loss.item()])

    return ((1 - y) * mse_loss) + (y * torch.max(max_of))

