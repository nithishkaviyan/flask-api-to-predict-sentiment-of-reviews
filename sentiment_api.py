from flask import request, Flask, jsonify

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

import numpy as np
import nltk
nltk.download('punkt')


##LSTM model
class StateLSTM(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(StateLSTM,self).__init__()
        
        self.lstm=nn.LSTMCell(in_dim,out_dim)
        self.out_dim=out_dim
        
        self.a=None
        self.c=None
        
    def reset_state(self):
        self.a=None
        self.c=None
        
    def forward(self,x):
        batch=x.data.size()[0]
        if (self.a is None):
            state_size=[batch,self.out_dim]
            self.c=Variable(torch.zeros(state_size)).cuda()
            self.a=Variable(torch.zeros(state_size)).cuda()
            
        self.a,self.c=self.lstm(x,(self.a,self.c))
            
        return self.a

class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout,self).__init__()
        self.d=None
        
    def reset_state(self):
        self.d=None
                
    def forward(self,x,dropout=0.5,train=True):
        if (train==False):
            return x
        if(self.d is None):
            self.d=x.data.new(x.size()).bernoulli_(1-dropout)
        mask=Variable(self.d, requires_grad=False)/(1-dropout)
            
        return mask*x

class Sentiment(nn.Module):
    def __init__(self,vocab_size,hidden_units):
        super(Sentiment,self).__init__()
        self.embedding=nn.Embedding(vocab_size,hidden_units)
        
        self.lstm1=StateLSTM(hidden_units,hidden_units)
        self.bn_lstm1=nn.BatchNorm1d(hidden_units)
        self.lstm1_do=LockedDropout()
        
        self.l1=nn.Linear(hidden_units,1)
        self.loss=nn.BCEWithLogitsLoss()
        
    def reset_state(self):
        self.lstm1.reset_state()
        self.lstm1_do.reset_state()
        
    def forward(self,x,y,train=True):
        embed = self.embedding(x)
        
        steps=embed.shape[1]
        
        self.reset_state()
        
        output=[]
        for i in range(steps):
            a=self.lstm1(embed[:,i,:])
            a=self.bn_lstm1(a)
            a=self.lstm1_do(a,train=train)
            
            output.append(a)
            
        output=torch.stack(output)
        output=output.permute(1,2,0)
        
        max_pool=nn.MaxPool1d(steps)
        a=max_pool(output)
        a=a.view(a.size(0),-1)
        
        a=self.l1(a)
            
        return self.loss(a[:,0],y), a[:,0]



##BOW model
class BOW_model(nn.Module):
    def __init__(self,vocab_size,hidden_units):
        super(BOW_model,self).__init__()
        self.embedding=nn.Embedding(vocab_size,hidden_units)
        self.l1=nn.Linear(hidden_units,hidden_units)
        self.l1_bn=nn.BatchNorm1d(hidden_units)
        self.l2_do=nn.Dropout(p=0.5)
        self.l2=nn.Linear(hidden_units,1)
        
        self.loss=nn.BCEWithLogitsLoss()
        
    def forward(self,inp,labels):
        bow_embed=[]
        for i in range(len(inp)):
            lookup_tensor=Variable(torch.LongTensor(inp[i])).cuda()
            embed=self.embedding(lookup_tensor)
            embed=embed.mean(dim=0)
            bow_embed.append(embed)
        bow_embed=torch.stack(bow_embed,dim=0)
            
        x=F.relu(self.l1_bn(self.l1(bow_embed)))
        x=self.l2(self.l2_do(x))
        
        return self.loss(x[:,0],labels), x[:,0]




##CNN model
class CNN_model(nn.Module):
  def __init__(self,vocab_size,hidden_units,in_dim,nc=5,s=1,p=0,f=3):
    super(CNN_model,self).__init__()
    self.embed=nn.Embedding(vocab_size,hidden_units)
    self.conv1=nn.Conv2d(1,nc,(f,hidden_units),stride=s,padding=p)
    self.maxpool=nn.MaxPool1d(in_dim)
    self.do=nn.Dropout(p=0.2)
    self.l1=nn.Linear(nc,1)
    self.loss=nn.BCEWithLogitsLoss()

    
  def forward(self,x,y):
    embed=self.embed(x)
    embed=embed.view(-1,1,embed.shape[1],embed.shape[2])
    x=self.conv1(embed)
    x=x.squeeze(-1)
    x=self.maxpool(x)
    x=x.view(-1,x.shape[1]*x.shape[2])
    x=self.do(x)
    x=self.l1(x)
    loss=self.loss(x[:,0],y)
        
    return loss, x[:,0]


##Function to compute predicted sentiment and accuracy

def pred_acc(pred_score, targ,sent):
  if sent:
    pred = (pred_score>=0)
    truth = (targ>=0.5)
    acc = pred.eq(truth).sum().cpu().item()

    return ({"predicted sentiment":pred.cpu().tolist(),"Accuracy":acc/len(pred)*100})

  else:
    pred = (pred_score>=0).cpu().tolist()

    return ({"predicted sentiment" : pred})


app = Flask(__name__)

@app.route('/sentiment',methods=['POST'])
def sentiment_predict():
  json_ = request.json ##Reads the input given in json format (Output will be a dictionary)
  text = []


  ##Input format is in a dictionary format with key 'reviews' containing a list containing text
  for i in json_['reviews']:
    text.append(nltk.word_tokenize(i.lower().replace('\n',' ')))
     

  ##Convert text to id based on word_to_id dictionary
  text_id = [[word_to_id.get(i,-1)+1 for i in x] for x in text]
  text_id = [[0 if i > vocab_size else i for i in x] for x in text_id]
  
  ##Input to the LSTM model
  x_inp = np.zeros((len(text_id),sequence_length),dtype=np.int)
  
  for n,i in enumerate(text_id):
    if (len(i) <= sequence_length):
      x_inp[n,:len(i)] = i
    else:
      x_inp[n] = i[:sequence_length]
  
  data = Variable(torch.LongTensor(x_inp)).cuda()

  if 'sentiment' not in json_.keys():
    target = Variable(torch.FloatTensor(np.zeros(len(x_inp)))).cuda()
  else:
    target = Variable(torch.FloatTensor(np.array(json_['sentiment']))).cuda()
  
  #optimizer.zero_grad()
  
  with torch.no_grad():
    loss,score = lstm_model(data,target,train=False)

  lstm_out = pred_acc(score,target, 'sentiment' in json_.keys())

  ##Input to CNN model
  x_inp2 = np.zeros((len(text_id),sequence_length_cnn),dtype=np.int)

  for n,i in enumerate(text_id):
    if (len(i) <= sequence_length_cnn):
      x_inp2[n,:len(i)] = i
    else:
      x_inp2[n] = i[:sequence_length_cnn]
    
  data_cnn = Variable(torch.LongTensor(x_inp2)).cuda()

  if 'sentiment' not in json_.keys():
    target_cnn = Variable(torch.FloatTensor(np.zeros(len(x_inp2)))).cuda()
  else:
    target_cnn = Variable(torch.FloatTensor(np.array(json_['sentiment']))).cuda()

  with torch.no_grad():
    loss2,score2 = cnn_model(data_cnn,target_cnn)  
  
  cnn_out = pred_acc(score2,target_cnn, 'sentiment' in json_.keys())

  ##Input to BOW model
  x_inp3 = list(text_id)
  
  if 'sentiment' not in json_.keys():
    target_bow = Variable(torch.FloatTensor(np.zeros(len(x_inp3)))).cuda()
  else:
    target_bow = Variable(torch.FloatTensor(np.array(json_['sentiment']))).cuda()  

  with torch.no_grad():
    loss3,score3 = bow_model(x_inp3,target_bow)  
  
  bow_out = pred_acc(score3,target_bow, 'sentiment' in json_.keys())

  return jsonify({"LSTM model":lstm_out, "CNN model":cnn_out, "BOW model":bow_out})


  '''
  if 'sentiment' in json_.keys():  
    pred = (score>=0)
    truth = (target>=0.5)
    acc = pred.eq(truth).sum().cpu().item()    

    return jsonify({"LSTM model":{"predicted sentiment":pred.cpu().tolist(),"Accuracy":acc/len(x_inp)*100}})

  else:
    pred = (score>=0).cpu().tolist()

    return jsonify({"predicted sentiment" : pred})
  '''


if __name__ == '__main__':
  
  ##Load saved LSTM model
  lstm_model = torch.load('yelp_lstm.pth')
  lstm_model.cuda()
  lstm_model.eval()
  
  ##Load saved BOW model
  bow_model = torch.load('yelp_bow.pth')
  bow_model.cuda()
  bow_model.eval() 

  ##Load saved CNN model
  cnn_model = torch.load('yelp_cnn.pth')
  cnn_model.cuda()
  cnn_model.eval() 
    

  ##Load id to word dictionary
  id_to_word = np.load('yelp_dictionary.npy')
  
  ##Create word to id dictionary from  id_to_word
  word_to_id = {word:n for n,word in enumerate(id_to_word)}
  
  ##Load vocabulary size variable
  vocab_size = np.load('yelp_vocab_size.npy')
  
  ##Load sequence length
  #sequence_length = np.load('yelp_sequence_length.npy')
  sequence_length = 500

  sequence_length_cnn = np.load('yelp_cnn_sequence_length.npy')
  

  port = 1996
  app.run(port = port, debug = True)