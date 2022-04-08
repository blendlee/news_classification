from transformers import  Trainer, TrainingArguments
from model import *
from dataset import *
from tokenizer import *



def train():
    set_seed(42)

    datadir='/opt/ml/Daycon/dataset/train.csv'

    tokenizer = NewsTokenizer()
    model = BertClassifier()


    dataset = load_data(datadir)

    train_data,val_data = split_data(dataset)

    train_dataset = News_Dataset()
    val_dataset = News_Dataset()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)


    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        save_total_limit=5,              # number of total save model.
        save_steps=500,                 # model saving step.
        num_train_epochs=4,              # total number of training epochs
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
        train_dataset=RE_train_dataset,         # training dataset
        eval_dataset=RE_dev_dataset,             # evaluation dataset
        compute_metrics=compute_metrics         # define metrics function
        )

        # train model
    trainer.train()
        if num_splits == 1:
            best_dir = f'./best_model'
        else: best_dir = f'./best_model/{fold}_best_model'

    os.makedirs(best_dir, exist_ok=True)
    torch.save(model.state_dict(), PATH)



def main():
    train()

if __name__ == '__main__':
    main()