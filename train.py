from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, set_seed
from tokenizers import BertWordPieceTokenizer
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
    tokenizer = BertWordPieceTokenizer()
    tokenizing('/opt/ml/news_classification/corpus.txt',tokenizer,vocab_size)
    
    dataset = load_data(datadir)
    train_data,val_data = split_data(dataset)
    
    train_data = train_data.reset_index()
    val_data = val_data.reset_index()
    
    train_text = train_data['text']
    train_label = train_data['target']
    
    val_text = val_data['text']
    val_label = val_data['target']
 
    max_len=500
    tokenized_train = tokenized_dataset(train_data,tokenizer,max_len)
    tokenized_val = tokenized_dataset(val_data,tokenizer,max_len)
    
    
    train_dataset = News_Dataset(tokenized_train,train_label)
    val_dataset = News_Dataset(tokenized_val,val_label)

    #모델 정의
    #MODEL_NAME = "bert-base-uncased"
    #model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    #model_config.num_labels = 20
    #model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    
    
    model = Classifier(vocab_size)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)

    # epochs=20

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=128,
    #     num_workers=5,
    #     shuffle=True,
    #     pin_memory=use_cuda,
    #     drop_last=True,
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=128,
    #     num_workers=5,
    #     shuffle=True,
    #     pin_memory=use_cuda,
    #     drop_last=True,
    # )

    # criterion=nn.CrossEntropyLoss()

    # for epoch in epochs:
    #     model.train()
    #     loss_value = 0
    #     matches = 0
    #     for idx, train_batch in enumerate(train_loader):
    #         inputs, labels = train_batch
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)
    #         optimizer.zero_grad()
    #         logits = model(inputs)
    #         preds = torch.argmax(logits, dim=-1)
            

    #         loss = criterion(preds.view(-1,20), label.view(-1))

    #         loss.backward()
    #         optimizer.step()
    #         loss_value += loss.item()
    #         matches += (preds == labels).sum().item()
        #       if (idx + 1) % args.log_interval == 0:
        #           train_loss = loss_value / args.log_interval
        #           train_acc = matches / args.batch_size / args.log_interval
        #           current_lr = get_lr(optimizer)
        #           print(
        #             f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
        #             f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
        #         )
        #           logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
        #           logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

        #           loss_value = 0
        #       matches = 0

        # scheduler.step()



    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        save_total_limit=5,              # number of total save model.
        save_steps=500,                 # model saving step.
        num_train_epochs=500,              # total number of training epochs
        learning_rate=5e-2,               # learning_rate
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=8,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.0001,               # strength of weight decay
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