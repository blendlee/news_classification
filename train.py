from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, set_seed
from tokenizers import BertWordPieceTokenizer
from transformers import ElectraModel, ElectraConfig,ElectraTokenizer,ElectraForSequenceClassification, AutoConfig
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from model import *
from dataset import *
import wandb
import sklearn
import numpy as np
import pandas as pd
import torch

wandb.init(project="news_classification", entity="blendlee")


def compute_metrics(pred):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.
    return {
        'accuracy': acc,
    }




def train():
    set_seed(42)

    datadir='/opt/ml/news_classification/data/train.csv'

    vocab_size=5000
    max_len=256


    ##########Tokenizer 정의##########
    #Custom tokenizer
    #tokenizer = BertWordPieceTokenizer()
    #tokenizing('/opt/ml/news_classification/corpus.txt',BertWordPieceTokenizer(),vocab_size)

    #Electra tokenizer
    #tokenizer = ElectraTokenizer('/opt/ml/news_classification/vocabs.txt')
    #tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
    tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
    #tokenizing('/opt/ml/news_classification/corpus.txt',BertWordPieceTokenizer(),vocab_size)

    
    dataset = load_data(datadir)
    train_data,val_data = split_data(dataset)


    ##########모델 정의##########
    #Custom Model
    #model = Classifier(vocab_size)
    #huggingface=False

    #Electra Model
    #model_config = ElectraConfig()
    #model_config.num_labels=20
    #model = ElectraForSequenceClassification(config=model_config)
    #huggingface=True

    #bert-base Model
    model_config = AutoConfig.from_pretrained("google/electra-small-discriminator")
    model =ElectraClassifier(model_config)

    huggingface=True

    tokenized_train,train_label = tokenized_dataset(train_data,tokenizer,max_len,huggingface=huggingface)
    tokenized_val,val_label = tokenized_dataset(val_data,tokenizer,max_len,huggingface=huggingface)
    
    
    train_dataset = News_Dataset(tokenized_train,train_label,train=True)
    val_dataset = News_Dataset(tokenized_val,val_label,train=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)

 
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        save_total_limit=5,              # number of total save model.
        save_steps=500,                 # model saving step.
        num_train_epochs=500,              # total number of training epochs
        learning_rate=5e-4,               # learning_rate
        per_device_train_batch_size=64,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.00001,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=500,              # log saving step.
        evaluation_strategy='epoch',
        save_strategy='epoch', # evaluation strategy to adopt during training
                                    # `no`: No evaluation during training.
                                    # `steps`: Evaluate every `eval_steps`.
                                    # `epoch`: Evaluate every end of epoch.
        eval_steps = 100,            # evaluation step.
        load_best_model_at_end = True, 
        report_to='wandb'
        )
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
        compute_metrics=compute_metrics         # define metrics function
        )

        # train model
    trainer.train()


    torch.save(model.state_dict(), '/opt/ml/news_classification/best_model/best_model.pt')



def main():
    train()

if __name__ == '__main__':
    main()