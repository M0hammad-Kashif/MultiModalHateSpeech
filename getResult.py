import matplotlib.pyplot as plt
import pandas as pd
from imports import *
from imports import *
from model import ReBert , bert
from data import *
import json

df = pd.read_csv('stA_test.csv')

text = df['text'].tolist()
filenames = df['filename'].tolist()


class CustomTextDataset(Dataset):
        def __init__(self, X ,y ,toknizer , max_token_len=512 , test =False):
                self.X = X
                self.tokenizer = tokenizer
                self.max_token_len = max_token_len
                self.y = y 
                self.test = test

        def __len__(self):
                return len(self.X)

        def __getitem__(self, idx):
                      
            try: 
                    filename = self.y[idx].split('/')[-1].split('.jpg')[0]
                    extracted_text = unidecode(self.X[idx])

    

                    encodings = self.tokenizer.encode_plus(
                            text = extracted_text,
                            add_special_tokens = True,
                            max_length = self.max_token_len,
                            return_token_type_ids = False,
                            padding="max_length",
                            truncation=True,
                            return_attention_mask = True,
                            return_tensors='pt',
                            # is_split_into_words=True
                    )

                    return dict(
                            text = extracted_text,
                            input_ids = encodings['input_ids'].flatten(),
                            attention_mask = encodings['attention_mask'].flatten(),
                            filename = filename

                            
                    )
            except:
                    extracted_text = unidecode(self.X[idx])
                    # label = self.y[idx]
                    print(extract_text)
                    print(f'failed at {idx}')
 
                
                


testset = CustomTextDataset(X=text,y=filenames,toknizer=tokenizer,max_token_len=512,test=True)



test_dataloader = DataLoader(testset,batch_size=64,shuffle=False)


def test_model(model,dataloader):

    model = model.to(device)
    
    model.load_state_dict(torch.load('best.pth'))

    fileid, predictions = [], []

    model.eval()
    with torch.no_grad():

        loop = tqdm(dataloader, total=len(dataloader),desc='Test')

        for b , data in enumerate(loop):

            filename = data['filename']
            mask = data['attention_mask'].to(device)
            ids = data['input_ids'].to(device)


            out = model(ids,mask)

            


            predictions.extend(F.softmax(out, 1).cpu().detach().numpy())
            fileid.extend(filename)
            







    predictions = np.array(predictions)
    predicted_labels = predictions.argmax(1)
    predicted_labels = list(predicted_labels)
    predicted_labels = [ int(x) for x in predicted_labels]
    
    pred = ['prediction']*len(predicted_labels)
    index = ['index']*len(fileid)

    result = dict(zip(fileid,predicted_labels))
    with open('resultSubmission.json', 'w') as f:
        json.dump(result, f)
    


model = ReBert(bert).to(device)

test_model(model,test_dataloader)

