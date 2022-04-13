

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold
from tokenizers import BertWordPieceTokenizer
import pandas as pd
import torch
import re
import json


class News_Dataset(Dataset):
    def __init__(self,tokenized_dataset,label,train=True):
        self.text= tokenized_dataset
        self.train = train
        self.labels=label

    def __getitem__(self,idx):
        item = {key: torch.LongTensor(val[idx]) for key, val in self.text.items()}
        if self.train:
            item['labels'] = torch.LongTensor([self.labels[idx]])
        else:
            item['labels'] = torch.LongTensor(0)
        return item

    def __len__(self):
        return len(self.labels)


def preprocess(dataset):
    #f=open('/opt/ml/news_classification/corpus.txt','w')
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
        #f.write(review)
    #f.close()
        
    return dataset

def tokenizing(corpus_dir,tokenizer,vocab_size):
    
    tokenizer.train(
        files = [corpus_dir],
        vocab_size = vocab_size,
        min_frequency = 10,
        limit_alphabet = 1000,
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        show_progress = True,
        wordpieces_prefix = "##",
    )
    
    tokenizer.save('/opt/ml/news_classification/vocab')
    vocab_file = '/opt/ml/news_classification/vocabs.txt'
    vocab_path = '/opt/ml/news_classification/vocab'
    f = open(vocab_file,'w',encoding='utf-8')
    with open(vocab_path) as json_file:
        json_data = json.load(json_file)
        for item in json_data["model"]["vocab"].keys():
            f.write(item+'\n')
        f.close()


def load_data(datadir):
    dataset = pd.read_csv(datadir)
    dataset = preprocess(dataset)
    return dataset

def tokenized_dataset(dataset,tokenizer,max_len):
    tokens=[]
    token_type_ids=[]
    attention_masks=[]
    for i in range(len(dataset)):
        encodings=tokenizer.encode('[CLS]',dataset['text'][i])
        token=encodings.ids
        if len(token) > max_len:
            token = token[:max_len]
            token_type_id=encodings.type_ids[:max_len]
            attention_mask=encodings.attention_mask[:max_len]
        else:
            token = token + [0]*(max_len-len(token))
            token_type_id=encodings.type_ids+ [0]*(max_len-len(token))
            attention_mask=encodings.attention_mask+ [0]*(max_len-len(token))
        tokens.append(token)
        token_type_ids.append(token_type_id)
        attention_masks.append(attention_mask)
    
    return {'input_ids':torch.LongTensor(tokens)}

def split_data(dataset):
    split = StratifiedShuffleSplit(test_size=0.2, random_state=42)
    for train_index, dev_index in split.split(dataset,dataset['target']):
        train_dataset = dataset.loc[train_index]
        dev_dataset = dataset.loc[dev_index]
        return train_dataset, dev_dataset



