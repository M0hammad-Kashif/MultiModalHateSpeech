from imports import *
from model import tokenizer
from utils import *

hate_images = glob('subTaskA/Hate Speech/**.jpg')
nohate_images = glob('subTaskA/No Hate Speech/**.jpg')

shuffle(hate_images)
shuffle(nohate_images)


train_hate = hate_images[:round(len(hate_images)*.80)]
train_nohate = nohate_images[:round(len(nohate_images)*.80)]


test_hate = hate_images[round(len(hate_images)*.80):]
test_nohate = nohate_images[round(len(nohate_images)*.80):]


train_images = train_hate + train_nohate
test_images = test_hate + test_nohate

lbl_0 = 0.0
lbl_1 = 0.0

for pth in train_images:

    rd = pth.split('/')[1]
    if rd == 'Hate Speech':
        lbl_1 +=1
    else:
        lbl_0 +=1
        
w0 = lbl_0 / (lbl_0 + lbl_1)
w1 = lbl_1 / (lbl_0 + lbl_1)

wts = torch.tensor([w0,w1]).to(device)




img_size = 128
aug= A.Compose([
            A.Resize(img_size,img_size),

            A.Normalize(mean=(0), std=(1)),
            ToTensorV2(p=1.0),
        ], p=1.0)


class CustomTextDataset(Dataset):



        def __init__(self, imageDirectory , tokenizer , transform, max_token_len=128):


                self.imagePath = imageDirectory
                self.tokenizer = tokenizer
                self.max_token_len = max_token_len
                self.transform = transform


        def __len__(self):
                return len(self.imagePath)

        def __getitem__(self, idx):
                filePath = self.imagePath


                image = Image.open(filePath[idx]) # this is PIL image

                image = np.array(image) # (H,W,C)

                image = self.transform(image=image)['image']


                extracted_text = extract_text(filePath[idx])

                label = 0
                rd = filePath[idx].split('/')[1]
                if rd == 'Hate Speech':
                        label = 1

                encodings = self.tokenizer.encode_plus(
                        text = extracted_text,
                        add_special_tokens = True,
                        max_length = self.max_token_len,
                        return_token_type_ids = False,
                        padding="max_length",
                        truncation=True,
                        return_attention_mask = True,
                        return_tensors='pt'
                )

                return dict(
                        text = extracted_text,
                        input_ids = encodings['input_ids'].flatten(),
                        attention_mask = encodings['attention_mask'].flatten(),
                        label = torch.tensor(float(label)),
                        image= image
                )
                


trainset = CustomTextDataset(train_images,tokenizer,aug,256)
testset = CustomTextDataset(test_images,tokenizer,aug,256)


train_dataloader = DataLoader(trainset,batch_size=16,shuffle=True)
valid_dataloader = DataLoader(testset,batch_size=16,shuffle=False)