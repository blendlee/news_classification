from transformers import ElectraPreTrainedModel,AutoModel,AutoConfig
import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self,vocab_size):
        super(Classifier, self).__init__()
        self.embedding=nn.Embedding(num_embeddings = vocab_size,embedding_dim=256)
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
        output1=self.lstm2(output)[0]
        output=self.maxpool(output1)
        logits=self.seq(output)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 20), labels.view(-1))

        outputs = (loss,logits)

        return outputs

class ElectraClassifier(ElectraPreTrainedModel):
    def __init__(self,config):
        super(ElectraClassifier,self).__init__(config)
        self.model = AutoModel.from_pretrained("google/electra-small-discriminator")
        self.model_config = config
        self.model_config.num_labels = 20
        self.num_labels = 20
        self.hidden_dim = self.model_config.hidden_size
        self.lstm= nn.LSTM(input_size= self.hidden_dim, hidden_size= self.hidden_dim, num_layers= 2, dropout= 0.2,
                           batch_first= True, bidirectional= True)
        self.fc= nn.Linear(self.hidden_dim*2, self.model_config.num_labels)
    
    def forward(self,input_ids, attention_mask,token_type_ids,labels):

        outputs = self.model(
            input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids
        )
        sequence_output = outputs.last_hidden_state

        hidden, (last_hidden, last_cell)= self.lstm(sequence_output)
        cat_hidden= torch.cat((last_hidden[0], last_hidden[1]), dim= 1)
        logits= self.fc(cat_hidden)

        outputs = (logits,) + outputs[2:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        outputs = (loss,) + outputs

        return outputs