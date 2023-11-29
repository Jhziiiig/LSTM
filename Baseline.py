import os
import torch
import LSTM
import Process
import numpy as np
import pandas as pd
from hmmlearn import hmm
from torch.utils.data import DataLoader
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


def main(model):
    batch_size=24
    train_data=Process.Dataloader('train')
    test_data=Process.Dataloader('test')
    train_dataloader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
    test_dataloader=DataLoader(test_data,batch_size=batch_size)

    if model=='LSTM':
        print('LSTM\n---------------------------------------------------------')
        LSTM.run(train_dataloader,test_dataloader)


if __name__=='__main__':
    main('LSTM')