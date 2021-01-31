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
            self.data, self.label = self.preprocess(csv_data, self.task)

    def preprocess(self, input, task):
        if task == 'Taitanic':
            data = input[['Sex', 'Age', 'Name', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked']]

            # feature 1: Sex
            data['Sex'] = data['Sex'].map({'female': 1, 'male': 0}).astype(int)

            # feature 2: Age
            # Age缺失值较多，达到了20%左右，可以考虑删除，也可以进行填充。
            # 这里先进行填充。Age填充时不能直接使用所有数据的mean或者median等进行填充，因为偏差较大，
            # 所以需要进行分析哪些特征会影响Age, 使用Sex，Pclass这二个特征进行填充
            guess_ages = np.zeros((2, 3))
            for i in range(0, 2):
                for j in range(0, 3):
                    temp_data = data[(data['Sex'] == i) & (data['Pclass'] == j + 1)]['Age'].dropna()
                    guess_ages[i][j] = temp_data.mean()
            for i in range(0, 2):
                for j in range(0, 3):
                    data.loc[(data.Age.isnull()) & (data.Sex == i) & (data.Pclass == j + 1), \
                             'Age'] = guess_ages[i, j]
            data.loc[data['Age'] <= 16, 'Age'] = 0
            data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
            data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
            data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
            data.loc[data['Age'] > 64, 'Age'] = 4

            # feature 3: Name -> Title
            data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
            data['Title'] = data['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr',
                                                   'Jonkheer', 'Major', 'Mme', 'Rev'], 'Rare')
            data['Title'] = data['Title'].replace(['Sir'], 'Mr')
            data['Title'] = data['Title'].replace(['Mrs', 'Ms', 'Lady', 'Mlle'], 'Miss')
            title_mapping = {"Master": 1, "Miss": 2, "Mr": 3, "Rare": 4, }
            data['Title'] = data['Title'].map(title_mapping)
            data['Title'] = data['Title'].fillna(0)

            # feature 4: Family Size
            data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

            # feature 5: is Alone
            data['IsAlone'] = 0
            data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

            # feature 6: Embarked
            # 缺失值填充, Embarked（上船的港口,缺失值较少, 行填充)
            freq_port = data['Embarked'].dropna().mode()[0]
            data['Embarked'] = data['Embarked'].fillna(freq_port)
            data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

            # feature 7: FareBand
            data.loc[data['Fare'] < 7.91, 'Fare'] = 0
            data.loc[(data['Fare'] > 7.91) & (data['Fare'] < 14.454), 'Fare'] = 1
            data.loc[(data['Fare'] > 14.454) & (data['Fare'] < 31.0), 'Fare'] = 2
            data.loc[data['Fare'] >= 31.0, 'Fare'] = 3

            # feature 8:  Sex + Age + Class
            data['SAC'] = data.Sex + data.Age + data.Pclass

            del data['Name']
            del data['Fare']
            del data['SibSp']
            del data['Parch']

            data_np = np.array(data).astype('float32')
            label_np = np.array(input[['Survived']]).astype('int64').squeeze()

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
