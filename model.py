
import torch
import torch.nn as nn



class LSTMClassifier(nn.Module):
    def __init__(self,input_dim,hiddin_dim):
        super(LSTMClassifier, self).__init__()

        self.labels=20
        self.input_dim = input_dim
        self.hidden_dim = hiddin_dim
        self.lstm = nn.LSTM(input_size=self.input_dim,hidden_size=self.hidden_dim,num_layers=2,dropout=0.2)
        self.fclayer = nn.Linear(self.hidden_dim*2, self.labels)

    def forward(self,input_ids,labels):

        hidden, (last_hidden, last_cell) = self.lstm(input_ids)
        cat_hidden= torch.cat((last_hidden[0], last_hidden[1]), dim= 1)
        logits= self.fclayer(cat_hidden)
        outputs = (logits,) + outputs[2:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.labels), labels.view(-1))
        outputs = (loss,) + outputs

        return outputs
