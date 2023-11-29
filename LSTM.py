import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMModel(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size):
        super(LSTMModel,self).__init__()
        self.lstm=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True).float()
        self.fc=nn.Linear(hidden_size,output_size)

    def forward(self,x,lengths):
        # pack
        x_packed=pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False)
        out_packed, _=self.lstm(x_packed)

        # unpack
        out,_=pad_packed_sequence(out_packed,batch_first=True)

        # output
        out=self.fc(out[:,-1,:])
        return out


def train(dataloader,model,loss_fn,optimizer):
    model.train()
    for batch,(X,y,length) in enumerate(dataloader):
        pred=model(X,length)
        loss=loss_fn(pred,y)
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        print(f'loss:{loss:>7f}')


def test(dataloader,model,loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy:{(100 * correct):>0.1f}%, Avg loss:{test_loss:>8f} \n")


def run(trainloader,testloader):
    epoch=20
    model = LSTMModel(input_size=38, hidden_size=50, num_layers=2, output_size=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for i in range(epoch):
        print(f"Epoch{i + 1}\n-----------------------------")
        train(trainloader,model,criterion,optimizer)
        test(testloader,model,criterion)
    print('done')

