import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

# Train data and test data
offset = 20
cookActivities = {"cairo.txt": {"Other": offset,
                            "Work": offset + 1,
                            "Take_medicine": offset + 2,
                            "Sleep": offset + 3,
                            "Leave_Home": offset + 4,
                            "Eat": offset + 5,
                            "Bed_to_toilet": offset + 6,
                            "Bathing": offset + 7,
                            "Enter_home": offset + 8,
                            "Personal_hygiene": offset + 9,
                            "Relax": offset + 10,
                            "Cook": offset + 11},
                  "kyoto7.txt": {"Other": offset,
                             "Work": offset + 1,
                             "Sleep": offset + 2,
                             "Relax": offset + 3,
                             "Personal_hygiene": offset + 4,
                             "Cook": offset + 5,
                             "Bed_to_toilet": offset + 6,
                             "Bathing": offset + 7,
                             "Eat": offset + 8,
                             "Take_medicine": offset + 9,
                             "Enter_home": offset + 10,
                             "Leave_home": offset + 11},
                  "kyoto8.txt": {"Other": offset,
                             "Bathing": offset + 1,
                             "Cook": offset + 2,
                             "Sleep": offset + 3,
                             "Work": offset + 4,
                             "Bed_to_toilet": offset + 5,
                             "Personal_hygiene": offset + 6,
                             "Relax": offset + 7,
                             "Eat": offset + 8,
                             "Take_medicine": offset + 9,
                             "Enter_home": offset + 10,
                             "Leave_home": offset + 11},
                  "kyoto11.txt": {"Other": offset,
                              "Work": offset + 1,
                              "Sleep": offset + 2,
                              "Relax": offset + 3,
                              "Personal_hygiene": offset + 4,
                              "Leave_Home": offset + 5,
                              "Enter_home": offset + 6,
                              "Eat": offset + 7,
                              "Cook": offset + 8,
                              "Bed_to_toilet": offset + 9,
                              "Bathing": offset + 10,
                              "Take_medicine": offset + 11},
                  "milan.txt": {"Other": offset,
                            "Work": offset + 1,
                            "Take_medicine": offset + 2,
                            "Sleep": offset + 3,
                            "Relax": offset + 4,
                            "Leave_Home": offset + 5,
                            "Eat": offset + 6,
                            "Cook": offset + 7,
                            "Bed_to_toilet": offset + 8,
                            "Bathing": offset + 9,
                            "Enter_home": offset + 10,
                            "Personal_hygiene": offset + 11},
                  }


def split():
    path='milan'
    for root,dirs,files in os.walk(path):
        for file in files:
            filepath=os.path.join(root,file)
            Index1=file.find('_')
            Index2=file.find('.')
            key=file[(Index1+1):Index2]
            data=pd.read_pickle(filepath)

            num=len(data)
            trainpath='train//'+key+'.pkl'
            if os.path.exists(trainpath):
                print(f'{key} is Done')
            else:
                with open(trainpath,'wb') as f:
                    pd.to_pickle(data[:int(num*0.8)],f)

            testpath='test//'+key+'.pkl'
            if os.path.exists(testpath):
                print(f'{key} is Done')
            else:
                with open(testpath,'wb') as f:
                    pd.to_pickle(data[int(num*0.8):],f)


class Dataloader(Dataset):
    def __init__(self,data_dir):
        self.data_dir=data_dir
        self.paths=[]
        self.labels=[]

        for folder in os.listdir(data_dir):
            activity_path=os.path.join(data_dir,folder)
            if os.path.exists(activity_path):
                index=folder.find('.')
                label=folder[:index]
                self.labels.append(label)
                self.paths.append(activity_path)
            else:
                print(f'{activity_path} not exists.')

        num_encoder=LabelEncoder()
        self.num_labels=num_encoder.fit_transform(self.labels)

    def __len__(self):
        length=0
        for folder in self.paths:
            data=pd.read_pickle(folder)
            length += len(data)
        return length

    def __getitem__(self, index):
        # Get accumulative length of each pkl file.
        length=[]
        for folder in self.paths:
            data = pd.read_pickle(folder)
            length.append(len(data))
        num=len(length)
        for i in range(1,num):
            length[i]=length[i]+length[i-1]

        # find item by index
        location=0
        for i in range(num):
            if index+1<=length[i] and i!=0:
                location=i
                index=index-length[i-1]
                break
            elif i==0 and index+1<=length[0]:
                location=0
                break
        data=pd.read_pickle(self.paths[location])
        label=self.num_labels
        count=0
        for row in data[index]:
            if row[0] == 0:
               break
            count += 1
        return data[index].iloc[:,4:].values,label[location],count



