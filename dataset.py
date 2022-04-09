

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold

from tokenizer import *
import pandas as pd
import torch
import re



class News_Dataset(Dataset):
    def __init__(self,tokenized_dataset,label,train=True):
        self.text= tokenized_dataset
        self.labels = label.reset_index()
        self.train = train

    def __getitem__(self,idx):
        item={}
        item['input_ids']=self.text['input_ids'][idx]
        if self.train:
            item['label'] = torch.tensor(self.labels['target'][idx], dtype=torch.long)
        else:
            item['label'] = torch.tensor(0, dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)




def preprocess(dataset):
    for i in range(len(dataset)):
        text = dataset['text'][i]
        #review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"\n\]\[\>]', '',text) #@%*=()/+ 와 같은 문장부호 제거
        #review = re.sub(r'\d+','', review)#숫자 제거
        review = text.lower() #소문자 변환
        review = re.sub(r'\s+', ' ', review) #extra space 제거
        review = re.sub(r'<[^>]+>','',review) #Html tags 제거
        review = re.sub(r'\s+', ' ', review) #spaces 제거
        review = re.sub(r"^\s+", '', review) #space from start 제거
        review = re.sub(r'\s+$', '', review) #space from the end 제거
        review = re.sub(r'_', ' ', review) #space from the end 제거
        dataset['text'][i]=review
    return dataset



def load_data(datadir):
    dataset = pd.read_csv(datadir)
    dataset = preprocess(dataset)
    return dataset

def tokenized_dataset(dataset,tokenizer):

    tokenized_text = tokenizer(list(dataset['text']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=700,
        add_special_tokens=True,
        )

    return tokenized_text

def split_data(dataset):
    split = StratifiedShuffleSplit(test_size=0.2, random_state=42)
    for train_index, dev_index in split.split(dataset,dataset['target']):
        train_dataset = dataset.loc[train_index]
        dev_dataset = dataset.loc[dev_index]
        return train_dataset, dev_dataset



