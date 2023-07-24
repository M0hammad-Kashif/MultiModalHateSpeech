from imports import *
from model import bert_tokenizer , xlnet_tokenizer
from utils import *

df = pd.read_csv('multiModalTrainingSetA.csv')
# df = df.dropna()
# X = df['text'].astype(str).values.tolist()
# X = df['text'].tolist()
X = df.drop('label',axis=1)
# print(X)
y = df['label'].tolist()
# print(type(y))
				
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

X_train_text = X_train['text'].tolist()
X_test_text = X_test['text'].tolist()

X_train_filename = X_train['filename'].tolist()
X_test_filename = X_test['filename'].tolist()

srcPath = 'subTaskA/'

class CustomTextDataset(Dataset):
        def __init__(self, X_text,X_image ,y ,tokenizer0,tokenizer1 , max_token_len=512):
                self.X = X_text
                self.images = X_image
                self.tokenizer1 = tokenizer1
                self.tokenizer0 = tokenizer0
                self.max_token_len = max_token_len
                self.y = y 

        def __len__(self):
                return len(self.images)

        def __getitem__(self, idx):
                try: 
                        extracted_text = unidecode(self.X[idx])
                        label = self.y[idx]   
                        image = cv2.imread(srcPath + str(self.images[idx]))
                        # print('image opened')
                        image = cv2.resize(image,(224,224))
                        image = image/255.0
                        
                        encodings0 = self.tokenizer0.encode_plus(
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

                        encodings1 = self.tokenizer1.encode_plus(
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

                        return [ dict(
                                text = extracted_text,
                                input_ids = encodings0['input_ids'].flatten(),
                                attention_mask = encodings0['attention_mask'].flatten(),
                                label = torch.tensor(float(label)),
                                image = image
                                
                        ) ,
                        dict(
                                text = extracted_text,
                                input_ids = encodings1['input_ids'].flatten(),
                                attention_mask = encodings1['attention_mask'].flatten(),
                                label = torch.tensor(float(label))
                                
                        ) ]
                except:
                        extracted_text = unidecode(self.X[idx])
                        # label = self.y[idx]
                        print(extract_text)
                        print(f'failed at {idx}')
                


trainset = CustomTextDataset(X_train_text,X_train_filename,y_train,bert_tokenizer,xlnet_tokenizer,512)
testset = CustomTextDataset(X_test_text,X_train_filename,y_test,bert_tokenizer,xlnet_tokenizer,512)


train_dataloader = DataLoader(trainset,batch_size=64,shuffle=True)
valid_dataloader = DataLoader(testset,batch_size=64,shuffle=False)


# for b, data in enumerate(train_dataloader):
#         print(data[0]['image'].shape)