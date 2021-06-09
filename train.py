import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader


from data.data_loader import PianoDataset
from models.pixelcnn import PixelCNN


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=int, default=3e-3)
opt = parser.parse_args()
print(opt)

train = PianoDataset(opt.data_dir)
train_loader = DataLoader(train, batch_size=opt.batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PixelCNN().to(device)

optimizer = optim.Adam(model.parameters(), lr=opt.lr)
criterion = nn.CrossEntropyLoss()

loss_overall = []

for epoch in range(opt.epochs):
    step = 0
    loss_ = 0
    for images in tqdm(train_loader):
        
        images = images.to(device)
        target = images[:,0,:,:].long().to(device)
        
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


        loss_+=loss
        step+=1

    print('Epoch:'+str(epoch)+'\t'+ str(step) +'\t Iterations Complete \t'+'loss: ', loss.item()/1000.0)
    loss_overall.append(loss_/1000.0)
    loss_=0
    print('Epoch ' + str(epoch) + ' Over!')
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    print("Saving Checkpoint!")
    if(epoch==opt.epochs-1):
        torch.save(net.state_dict(), 'checkpoints/Model_Checkpoint_'+'Last'+'.pt')
    else:
        torch.save(net.state_dict(), 'checkpoints/Model_Checkpoint_'+str(epoch)+'.pt')
    print('Checkpoint Saved')
