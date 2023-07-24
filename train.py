from imports import *
from model import ReBert , bert , xlnet
# from model2 import ReBert , xlent
from data import *

model = ReBert(bert,xlnet).to(device)

criterion = nn.CrossEntropyLoss(reduction='sum')

optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4,weight_decay=3e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)



def train_one_epoch(model,dataloader):
    model = model.to(device)
    model.train()



    train_loss = 0
    train_acc = 0

    actuals, predictions = [], []

    loop = tqdm(dataloader, total=len(dataloader),desc='Train')

    for b , data in enumerate(loop):
        image = data[0]['image'].to(device)
        image = image.float()
        mask0 = data[0]['attention_mask'].to(device)
        ids0 = data[0]['input_ids'].to(device)
        
        mask1 = data[1]['attention_mask'].to(device)
        ids1 = data[1]['input_ids'].to(device)
        
        
        labels = data[0]['label'].type(torch.LongTensor).to(device)

        # print(labels.shape)

        out = model(ids0,mask0,ids1,mask1,image)

        # print(out.shape)
        # pred = torch.sigmoid(out)
        # pred = torch.round(pred)

        cur_train_loss = criterion(out, labels)
        # cur_train_acc = (pred == labels).sum().item() / labels.shape[0]

        actuals.extend(labels.cpu().numpy().astype(int))
        predictions.extend(F.softmax(out, 1).cpu().detach().numpy())


        cur_train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        train_loss += cur_train_loss.item()




    scheduler.step()

    predictions = np.array(predictions)
    predicted_labels = predictions.argmax(1)
    accuracy = (predicted_labels == actuals).mean()

    return train_loss/len(dataloader) , accuracy





def valid_one_epoch(model,dataloader):

    model = model.to(device)

    val_loss = 0
    val_acc = 0
    actuals, predictions = [], []

    model.eval()
    with torch.no_grad():

        loop = tqdm(dataloader, total=len(dataloader),desc='Valid')

        for b , data in enumerate(loop):

            image = data[0]['image'].to(device)
            image = image.float()
            mask = data['attention_mask'].to(device)
            ids = data['input_ids'].to(device)
            labels = data['label'].type(torch.LongTensor).to(device)

            out = model(ids,mask)

            # pred = torch.sigmoid(out)
            # pred = torch.round(pred)

            actuals.extend(labels.cpu().numpy().astype(int))
            predictions.extend(F.softmax(out, 1).cpu().detach().numpy())


            cur_valid_loss = criterion(out, labels)
            val_loss += cur_valid_loss.item()

            # val_acc += (pred == labels).sum().item() / labels.shape[0]

    predictions = np.array(predictions)
    predicted_labels = predictions.argmax(1)
    accuracy = (predicted_labels == actuals).mean()

    return val_loss/len(dataloader) ,accuracy


NUM_EPOCHS = 50
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_acc = 0.0


for epoch in range(NUM_EPOCHS):

    train_loss , train_acc = train_one_epoch(model=model, dataloader=train_dataloader)
    val_loss , val_acc = valid_one_epoch(model=model, dataloader=valid_dataloader)

    print(f"\n Epoch:{epoch + 1} / {NUM_EPOCHS},train loss:{train_loss:.5f}, train acc: {train_acc:.5f}, valid loss:{val_loss:.5f}, valid acc:{val_acc:.5f}")


    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    if val_acc > best_acc:
      torch.save(model.state_dict(),'best_xlent.pth')


df = pd.DataFrame.from_dict(history)
df.to_csv(r'report.csv', index = False, header=True)
