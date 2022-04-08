from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tokenizer import *
import pandas as pd
import torch




class News_Dataset(Dataset):
    def __init__(self,tokenized_dataset,label):
        self.dataset = dataset
        self.text= tokenized_dataset
        

    def __getitem__(self,idx):
        sentence = self.tokenized_dataset[i]
        label = self.label[i]
        return sentence,label

    def __len__(self):
        return len(tokenized_dataset)




def preprocess(dataset):
    for i in range(len(dataset)):
        text = dataset['text'][i]
        review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"\n\]\[\>]', '',text) #@%*=()/+ 와 같은 문장부호 제거
        review = re.sub(r'\d+','', review)#숫자 제거
        review = review.lower() #소문자 변환
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

def tokenize_dataset(dataset):
    pass

def split_data(dataset):
    pass



