import torch
import pandas as pd
import numpy as np
import torch.utils.data as data

from FileUtil.FileIO import LoadTxt, LoadCsv


class Dataset(data.Dataset):
    def __init__(self, data_url, preprocess='Taitanic'):
        self.task = preprocess
        if preprocess == 'Taitanic':
            csv_data = LoadCsv(data_url)
            self.data, self.label = self.preprocess(csv_data)

    def preprocess(self, input, task='Taitanic'):
        if task == 'Taitanic':
            data = input[['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Cabin', 'Embarked']]
            data['Age'] = data['Age'].fillna(data['Age'].mean())
            data['Cabin'] = pd.factorize(data.Cabin)[0]
            data.fillna(0, inplace=True)
            data['Sex'] = [1 if x == 'male' else 0 for x in data.Sex]
            data['p1'] = np.array(data['Pclass'] == 1).astype(np.int32)
            data['p2'] = np.array(data['Pclass'] == 2).astype(np.int32)
            data['p3'] = np.array(data['Pclass'] == 3).astype(np.int32)
            data['e1'] = np.array(data['Embarked'] == 'S').astype(np.int32)
            data['e2'] = np.array(data['Embarked'] == 'C').astype(np.int32)
            data['e3'] = np.array(data['Embarked'] == 'Q').astype(np.int32)
            del data['Pclass']
            del data['Embarked']
            data_np = np.array(data).astype('float32')

            label = input[['Survived']]
            label_np = np.array(label)[:, 0]

        else:
            raise NotImplemented
        return data_np, label_np

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        if self.task == 'Taitanic':
            return self.data.shape[0]
        else:
            raise NotImplemented


def MakeTrainLoader(train_data_url, batch_size):
    dataset = Dataset(train_data_url)
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=1,
                                               pin_memory=True,
                                               drop_last=True)
    return train_loader
