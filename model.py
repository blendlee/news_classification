
import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self,vocab_size):
        super(Classifier, self).__init__()
        self.embedding=nn.Embedding(num_embeddings = 35000,embedding_dim=256)
        self.lstm1=nn.LSTM(input_size=256,hidden_size=128,bidirectional=True,num_layers=1)
        self.lstm2=nn.LSTM(input_size=256,hidden_size=128,bidirectional=True,num_layers=1)
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.seq=nn.Sequential(nn.Linear(1,1024),
                  nn.Dropout(0.25),
                  nn.Linear(1024,512),
                  nn.Dropout(0.25),
                  nn.Linear(512,256),
                  nn.Dropout(0.25),
                  nn.Linear(256,128),
                  nn.Dropout(0.25),
                  nn.Linear(128,64),
                  nn.Dropout(0.25),
                  nn.Linear(64,20),
                  nn.Softmax(dim=2))
        
        
    def forward(self,input_ids,labels):
        output=self.embedding(input_ids)
        output=self.lstm1(output)[0]
        output=self.lstm2(output)[0]
        output=self.maxpool(output)
        logits=self.seq(output)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 20), labels.view(-1))

        outputs = (loss,logits)

        return outputs
