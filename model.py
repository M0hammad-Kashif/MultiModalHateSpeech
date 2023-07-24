from imports import *

XLNET_MODEL_NAME = 'xlnet-base-cased'
xlnet_tokenizer = TOKENIZER = transformers.XLNetTokenizer.from_pretrained(
        XLNET_MODEL_NAME, 
        do_lower_case=True
    )

BERT_MODEL_NAME = 'bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)




class ReBert(nn.Module):

    def __init__(self,bert,xlnet):

      super(ReBert, self).__init__()

      self.bert = bert.to(device)
      self.xlnet  = xlnet.to(device)
      self.mobilenet = models.mobilenet_v3_small(pretrained=True)
      self.mobilenet.classifier[3] = nn.Linear(in_features=1024, out_features=512, bias=True)
      

      # dropout layer
      self.dropout = nn.Dropout(0.1)

      # relu activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      self.fc1 = nn.Linear(768,512)
      self.fc0 = nn.Linear(768,512)

      # dense layer 2 (Output layer)

      self.fc2 = nn.Linear(512*3,128)  
      self.fc3 = nn.Linear(128,2)

      #softmax activation function
      self.softmax = nn.LogSoftmax(dim=1)

      #image model


    #define the forward pass
    def forward(self, sent_id0, mask0,sent_id1, mask1, image):
      
      

    #   embeddings form xlnet
      x1 = self.xlnet(sent_id1, attention_mask=mask1)
      x1 = x1[0]
      x1 = torch.mean(x1, 1)
      
      x1 = self.relu(self.fc1(x1))
      
    #   embeddings from bert
      x0 = self.bert(sent_id0, attention_mask=mask0)
      x0 = self.relu(self.fc0(x0.pooler_output))
      
      image = image.permute(0,3,1,2)
      x2 = self.relu(self.mobilenet(image))

      x = self.fc2(torch.concat((x0,x1,x2),1))
      x = self.relu(x)
      
      x = self.fc3(x)
      x = self.relu(x)
      return x
  
  

xlnet = transformers.XLNetModel.from_pretrained("xlnet-base-cased")
xlnet.vocab_size = 512
for param in xlnet.parameters():
    param.requires_grad = False
    
bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
bert.vocab_size = 512
for param in bert.parameters():
    param.requires_grad = False
