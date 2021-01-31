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
    print('ta miss statistic: ', data_train.isnull().sum() / len(data_train) * 100)

    # preprocess
    # PassengerId：乘客ID
    # Pclass: packet class 船票等级，1st 一等舱 2st 二等舱 3st 三等舱
    # Name: 乘客姓名
    # Sex： 乘客性别
    # Age： 乘客年龄
    # SibSp： 有几个siblings（兄弟姐妹） or spouses（配偶） 在船上
    # parch： 有几个parents or children 在船上
    # ticket: ticket number
    # fare: 船票价格
    # cabin： 船舱号,缺失值达到了78%，所以该特征直接删除吧。
    # embarked： 登船位置 C/Q/S 三个港口
    # 目标：
    # survived: 是否存活
    data = data_train[['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked', 'Survived']]

    # Title
    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    data['Title'] = data['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr', \
                                                         'Jonkheer', 'Major', 'Mme', 'Rev'], 'Rare')
    data['Title'] = data['Title'].replace(['Sir'], 'Mr')
    data['Title'] = data['Title'].replace(['Mrs', 'Ms', 'Lady', 'Mlle'], 'Miss')
    title_mapping = {"Master": 1, "Miss": 2, "Mr": 3, "Rare": 4, }
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'] = data['Title'].fillna(0)

    data.loc[data['Fare'] < 7.91, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] < 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] < 31.0), 'Fare'] = 2
    data.loc[data['Fare'] >= 31.0, 'Fare'] = 3

    data['Fareband'] = pd.qcut(data['Fare'], 4)

    # Sex
    data['Sex'] = data['Sex'].map({'female': 1, 'male': 0}).astype(int)

    # use freq to fill embark NAN
    freq_port = data['Embarked'].dropna().mode()[0]
    data['Embarked'] = data['Embarked'].fillna(freq_port)

    # Age is relate to Sex, Pclass, Embarked
    print(data_train[['Sex', 'Age']].groupby(['Sex'], as_index=False).mean().sort_values(by='Age', ascending=False))
    print(data_train[['Pclass', 'Age']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Age', ascending=False))
    print(data_train[['Embarked', 'Age']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Age', ascending=False))

    guess_ages = np.zeros((2, 3))
    for i in range(0, 2):
        for j in range(0, 3):
            temp_data = data[(data['Sex'] == i) & (data['Pclass'] == j + 1)]['Age'].dropna()
            guess_ages[i][j] = temp_data.mean()

    data['Cabin'] = pd.factorize(data.Cabin)[0]
    data['Sex'] = [1 if x == 'male' else 0 for x in data.Sex]
    data['p1'] = np.array(data['Pclass'] == 1).astype(np.int32)
    data['p2'] = np.array(data['Pclass'] == 2).astype(np.int32)
    data['p3'] = np.array(data['Pclass'] == 3).astype(np.int32)
    data['e1'] = np.array(data['Embarked'] == 'S').astype(np.int32)
    data['e2'] = np.array(data['Embarked'] == 'C').astype(np.int32)
    data['e3'] = np.array(data['Embarked'] == 'Q').astype(np.int32)
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
