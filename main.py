#coding=utf8
import numpy as np
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns
import re

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,:]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

# 加载停用词表
def load_stop_words(filePath):
    return []

stop_words = load_stop_words('./stop_words.txt')

def read_data(filePath):
    data = pd.read_excel(filePath, names=['text', 'label'], header=None)
    data['labels'] = data['label'].apply(literal_eval)
    # data.info()
    # print(data.sample(5))
    label_dict = {}
    labels_list = data['labels'].values
    for label_list in labels_list:
        for label in label_list:
            if label not in label_dict:
                label_dict[label] = 1
            else:
                label_dict[label] += 1
    label_df = pd.DataFrame(list(label_dict.items()), columns=['label', 'count']).sort_values(by='count', axis=0, ascending=False)
    # // 标签分布图
    # print(label_df.head(10))
    # label_df.plot(x='label', y='count', kind='bar', legend=False, grid=True, figsize=(10, 6))
    # plt.title("label count")
    # plt.ylabel('count')
    # plt.xlabel('label')
    # plt.show()
    # // 标签数量分布图
    # tagCount = data['labels'].apply(lambda x : len(x))
    # x = tagCount.value_counts()
    # plt.figure(figsize=(8,5))
    # ax = sns.barplot(x.index, x.values)
    # plt.title("label number")
    # plt.ylabel('number')
    # plt.xlabel('label')
    # plt.show()
    # 文本长度分布
    # lens = data.text.str.len()
    # print(lens.head())
    # print(lens.shape)
    # lens.hist(bins=30, figsize=(10, 6), grid=False)
    # plt.show()
    # // 查看空值的数量
    # print(data.text.isnull().sum())
    # print(data.label.isnull().sum())
    return data

def text_filter(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    return ' '.join([w for w in text.split(' ') if w not in stop_words])

def preprocess(data):
    pass


if __name__ == "__main__":
    train = read_data('./train.xlsx')
    train_final = [text_filter(str(w)) for w in train['text']]
    print(train_final[:10])