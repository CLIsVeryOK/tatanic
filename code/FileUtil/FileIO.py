import os
import torch
import numpy as np
import pandas as pd


def LoadTxt(filepath, sep=' '):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as filehandle:
            all_lines = filehandle.readlines()
        result = []
        for line in all_lines:
            result.append(line.strip().split(sep))
        print('data num: ', len(result))
        return result
    else:
        raise Exception('invalid filepath: ', filepath)


def LoadCsv(filepath):
    data_train = pd.read_csv(filepath)
    return data_train


if __name__ == '__main__':
    data_train = LoadCsv('../../data/titanic/train.csv')

    # preprocess
    data = data_train[['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Cabin', 'Embarked']]
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
    temp = np.array(data)

    # show data
    import matplotlib.pyplot as plt
    Survived_0 = data_train.Sex[data_train.Survived == 0].value_counts()
    Survived_1 = data_train.Sex[data_train.Survived == 1].value_counts()
    print('live: ', Survived_1, '\n')
    print('die: ', Survived_0)
    df = pd.DataFrame({u'live': Survived_1, u'die': Survived_0})
    df.plot(kind='bar', stacked=True)
    plt.title(u"survive statistic")
    plt.xlabel(u"harbor")
    plt.ylabel(u"count")
    plt.show()
    print('wait')
