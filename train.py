from transformers import  Trainer, TrainingArguments, set_seed
from transformers import BertTokenizer
from model import *
from dataset import *
from tokenizer import *
import wandb
import sklearn
import numpy as np
import pandas as pd
import torch

wandb.init(project="news_classification", entity="blendlee")

def klue_re_micro_f1(preds, labels):
   
    label_indices = list(range(20))
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0


def compute_metrics(pred):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)


    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

    return {
        'micro f1 score': f1,
        'accuracy': acc,
    }




def train():
    set_seed(42)

    datadir='/opt/ml/Daycon/dataset/train.csv'

    tokenizer =BertTokenizer('/opt/ml/Daycon/vocabs.txt')
    model = LSTMClassifier(700,768)

    dataset = load_data(datadir)
    train_data,val_data = split_data(dataset)
    
    train_label = train_data['target']
    val_label = val_data['target']
 
    tokenized_train = tokenized_dataset(train_data,tokenizer)
    tokenized_val = tokenized_dataset(val_data,tokenizer)
    
    
    train_dataset = News_Dataset(tokenized_train,train_label)
    val_dataset = News_Dataset(tokenized_val,val_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)


    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        save_total_limit=5,              # number of total save model.
        save_steps=500,                 # model saving step.
        num_train_epochs=10,              # total number of training epochs
        learning_rate=5e-5,               # learning_rate
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
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


    torch.save(model.state_dict(), '/opt/ml/Daycon/best_model')



def main():
    train()

if __name__ == '__main__':
    main()