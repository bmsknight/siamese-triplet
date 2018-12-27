from torch.utils.data import Dataset
import numpy as np
import torch


class SSDataset(Dataset):

    def __init__(self, root_dir='/home/madshan/PycharmProjects/siameseTest/data', split=0, train=True):
        temp = np.load(root_dir + "/data_" + str(split) + ".npy")
        # x_train, y_train, x_test, y_test = temp[0], temp[1], temp[2], temp[3]
        self.train = train
        if train:
            self.data = torch.Tensor(np.transpose(temp[0], [0, 2, 1]))
            self.labels = torch.Tensor(np.argmax(temp[1],-1)).type(torch.LongTensor)
        else:
            self.data = torch.Tensor(np.transpose(temp[2], [0, 2, 1]))
            self.labels = torch.Tensor(np.argmax(temp[3],-1)).type(torch.LongTensor)

        self.length = self.labels.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = self.data[idx]
        cls = self.labels[idx]

        return data, cls
