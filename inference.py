
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, BigBirdModel, AutoConfig
import torch
from torch.utils.data import DataLoader
from load_data_for_R import *
from model_for_R import *
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F


def inference(model, tokenized_sent, device):
    dataloader = DataLoader(tokenized_sent, batch_size=32, shuffle=False)
    model.eval()
    pred = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                input_ids = data['input_ids'].to(device),
                #labels = data['labels'].to(device)
            )
        logits = outputs[1]
        result = np.argmax(logit)
        pred.append(result)

    return pred



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

tokenizer = BertWordPieceTokenizer()
tokenizing('/opt/ml/news_classification/corpus.txt',tokenizer)

model = Classifier()

test_data = load_data('/opt/ml/news_classification/data/test.csv')

tokenized_test = tokenized_dataset(test_data,tokenizer,500)
#test_label=
dataset = News_Dataset(tokenized_test,test_label,train=False)
model.load_state_dict(torch.load('/opt/ml/news_classification/best_model/best_model.pt'))

pred = inference(model, RE_dataset, device)

#probs=[]
#for fold in range(1,6):
#    model.load_state_dict(torch.load(f'./best_model/{fold}_best_model/pytorch_model.bin'))
#    model.to(device)
#    output_pred, output_prob = inference(model, RE_dataset, device)
#    probs.append(output_prob)
# prob=sum(probs)/5


test = pd.read_csv('/opt/ml/news_classification/data/test.csv')
test_id = test['id'].to_list()

output = pd.DataFrame({'id':test_id, 'target':pred})

output.to_csv('/opt/ml/news_classification/submission/submission.csv', index=False)

print('Finish!!!!!!!!!')