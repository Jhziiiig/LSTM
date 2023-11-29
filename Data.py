import os
import re
import math
import warnings
import numpy as np
import pandas as pd
#from decimal import Decimal, getcontext
from datetime import datetime

offset = 20
max_lenght = 2000
warnings.simplefilter(action='ignore', category=FutureWarning)
# cookActivities字典储存动作的某种排序
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
# mappingActivities字典储存某一个子行为对应的Activity
mappingActivities = {"cairo.txt": {"": "Other",
                               "R1 wake": "Other",
                               "R2 wake": "Other",
                               "Night wandering": "Other",
                               "R1 work in office": "Work",
                               "Laundry": "Work",
                               "R2 take medicine": "Take_medicine",
                               "R1 sleep": "Sleep",
                               "R2 sleep": "Sleep",
                               "Leave home": "Leave_Home",
                               "Breakfast": "Eat",
                               "Dinner": "Eat",
                               "Lunch": "Eat",
                               "Bed to toilet": "Bed_to_toilet"},
                     "kyoto7.txt": {"R1_Bed_to_Toilet": "Bed_to_toilet",
                                "R2_Bed_to_Toilet": "Bed_to_toilet",
                                "Meal_Preparation": "Cook",
                                "R1_Personal_Hygiene": "Personal_hygiene",
                                "R2_Personal_Hygiene": "Personal_hygiene",
                                "Watch_TV": "Relax",
                                "R1_Sleep": "Sleep",
                                "R2_Sleep": "Sleep",
                                "Clean": "Work",
                                "R1_Work": "Work",
                                "R2_Work": "Work",
                                "Study": "Other",
                                "Wash_Bathtub": "Other",
                                "": "Other"},
                     "kyoto8.txt": {"R1_shower": "Bathing",
                                "R2_shower": "Bathing",
                                "Bed_toilet_transition": "Other",
                                "Cooking": "Cook",
                                "R1_sleep": "Sleep",
                                "R2_sleep": "Sleep",
                                "Cleaning": "Work",
                                "R1_work": "Work",
                                "R2_work": "Work",
                                "": "Other",
                                "Grooming": "Other",
                                "R1_wakeup": "Other",
                                "R2_wakeup": "Other"},
                     "kyoto11.txt": {"": "Other",
                                 "R1_Wandering_in_room": "Other",
                                 "R2_Wandering_in_room": "Other",
                                 "R1_Work": "Work",
                                 "R2_Work": "Work",
                                 "R1_Housekeeping": "Work",
                                 "R1_Sleeping_Not_in_Bed": "Sleep",
                                 "R2_Sleeping_Not_in_Bed": "Sleep",
                                 "R1_Sleep": "Sleep",
                                 "R2_Sleep": "Sleep",
                                 "R1_Watch_TV": "Relax",
                                 "R2_Watch_TV": "Relax",
                                 "R1_Personal_Hygiene": "Personal_hygiene",
                                 "R2_Personal_Hygiene": "Personal_hygiene",
                                 "R1_Leave_Home": "Leave_Home",
                                 "R2_Leave_Home": "Leave_Home",

                                 "R1_Enter_Home": "Enter_home",
                                 "R2_Enter_Home": "Enter_home",
                                 "R1_Eating": "Eat",
                                 "R2_Eating": "Eat",
                                 "R1_Meal_Preparation": "Cook",
                                 "R2_Meal_Preparation": "Cook",
                                 "R1_Bed_Toilet_Transition": "Bed_to_toilet",
                                 "R2_Bed_Toilet_Transition": "Bed_to_toilet",
                                 "R1_Bathing": "Bathing",
                                 "R2_Bathing": "Bathing"},
                     "milan.txt": {"": "Other",
                               "Master_Bedroom_Activity": "Other",
                               "Meditate": "Other",
                               "Chores": "Work",
                               "Desk_Activity": "Work",
                               "Morning_Meds": "Take_medicine",
                               "Eve_Meds": "Take_medicine",
                               "Sleep": "Sleep",
                               "Read": "Relax",
                               "Watch_TV": "Relax",
                               "Leave_Home": "Leave_Home",
                               "Dining_Rm_Activity": "Eat",
                               "Kitchen_Activity": "Cook",
                               "Bed_to_Toilet": "Bed_to_toilet",
                               "Master_Bathroom": "Bathing",
                               "Guest_Bathroom": "Bathing"},
                     }

datasets = ["../dataset/cairo.txt", "../dataset/kyoto7.txt", "../dataset/kyoto8.txt", "../dataset/kyoto11.txt", "../dataset/milan.txt"]
datasetsNames = [i.split('/')[-1] for i in datasets] # 得到如cairo.txt的名字

def load(filename):
    # dataset fields
    timestamp=[]
    sensors=[]
    Signal=[]
    activities=[]

    activity='' # empty

    with open(filename, 'rb') as features:
        database=features.readlines()
        for i,line in enumerate(database):
            f_info=line.decode().split()
            try:
                # timestamp
                if not ('.' in str(np.array(f_info[0]))+str(np.array(f_info[1]))):
                    f_info[1]=f_info[1]+'.000000'
                timestamp.append(datetime.strptime(str(np.array(f_info[0]))+str(np.array(f_info[1])),'%Y-%m-%d%H:%M:%S.%f'))
                # sensors
                sensors.append(str(np.array(f_info[2])))
                # Signal
                Signal.append(str(np.array(f_info[3])))
                # activities
                if len(f_info)==4: # if activity exists
                    activities.append(activity)
                else:
                    des=str(''.join(np.array(f_info[4:])))
                    if 'begin' in des:
                        activity=re.sub('begin','',des)
                        if activity[-1]==' ':
                            activity=activity[:-1]
                        activities.append(activity)
                    if 'end' in des:
                        activities.append(activity)
                        activity=''
            except IndexError:
                print(i,line)
    features.close()

    # convert sub_activities to activities
    for i in range(0,len(activities)):
        activities[i]=mappingActivities[filename][activities[i]]
    dataset={'activities':activities, 'timestamp': timestamp, 'sensors':sensors, 'Signal':Signal}
    dataset= pd.DataFrame(dataset)
    return dataset


# split the dataset by activities
def split(dataset):
    primary_split=[]
    # first level
    xx=[] # temporary store the data from the same activity
    count=0
    for i in range(1,dataset.shape[0]):
        if dataset.iloc[i,0]==dataset.iloc[i-1,0] and i!=dataset.shape[0]-1:
            xx.append(dataset.iloc[i].tolist())
        elif i==dataset.shape[0]-1:
            count += 1
            xx.append(dataset.iloc[i].tolist())
            primary_split.append(xx)
            xx = []
        else:
            if xx!=[]:
                primary_split.append(xx)
                xx = []
                count += 1
    # second level
    activity=sorted(set(dataset['activities']))
    final_split={}
    for i in activity:
        final_split[i]=[]
    for i in range(0,count):
        final_split[primary_split[i][0][0]].append(primary_split[i])
    return final_split
    # final_split is a dictionary with activity key and list of metrix value. The metrix value contains
# Generate features


def longitudinal_features(filename,final_split,sensors):
    sensor = sorted(set(sensors))

    for key,value in final_split.items():
        for m in range(0,len(value)): # m is the order of matrix
            # Initialize
            rows=len(value[m])
            colname=[sensorid for sensorid in sensor]

            SensorCount = pd.DataFrame(index=range(rows),columns=colname)
            SensorCount = SensorCount.fillna(0)
            EventPause = []
            SensorPause = []

            for i in range(len(value[m])): # i is row
                # Event Pause
                if i==0:
                    EventPause.append(0)
                else:
                    EventPause.append((value[m][i][1]-value[m][i-1][1]).total_seconds()*1000) # millisecond

                # Sensor Pause
                sensor_id=value[m][i][2]
                if i!=0:
                    for j in reversed(range(0,i)):
                        if value[m][j][2]==sensor_id:
                            SensorPause.append((value[m][i][1]-value[m][j][1]).total_seconds()*1000)
                            break
                        if j==0:
                            SensorPause.append(0)
                else:
                    SensorPause.append(0)

                # Sensor Count
                if i!=0:
                    SensorCount.iloc[[i]]=SensorCount.iloc[[i-1]]
                    SensorCount.loc[i, value[m][i][2]] += 1
                else:
                    SensorCount.loc[i, value[m][i][2]]=1
            finalDict={}
            for i in range(len(final_split[key][m][0])):
                finalDict[str(i)]=[row[i] for row in final_split[key][m]]

            final=pd.DataFrame(finalDict)

            EventPauseDict={'EventPause':EventPause}
            EventPause=pd.DataFrame(EventPauseDict)

            SensorPauseDict={'SensorPause':SensorPause}
            SensorPause=pd.DataFrame(SensorPauseDict)

            final_split[key][m] = pd.concat([final, EventPause, SensorPause, SensorCount], axis=1)
    return  final_split


def cross_sectional_features(final_split,sensors):
    # support= number of activity triggered this sensor / total number of activity
    sensors=list(sorted(set(sensors)))
    for key,value in final_split.items():
        support = {}
        # Initialize
        support={sensorid:1 for sensorid in sensors}

        # Calculate
        for j in range(len(sensors)): # sensor
            for i in range(len(value)): # matrix
                support[sensors[j]]+=((1 if value[i].iloc[-1,j+6]>0 else 0)/len(value))

        # Integrate
        support_key = list(support.keys())
        support_value = []
        for n in support_key:
            support_value.append(support[n])

        support = pd.DataFrame({'Sensors': support_key, 'Support': support_value})

        for i in range(0,len(value)):
            final_split[key][i].rename(columns={'0': 'Activity', '1': 'Timestamp', '2': 'Sensors', '3': 'Signal'},
                                       inplace=True)
            final_split[key][i]=pd.merge(final_split[key][i],support,how='left',on='Sensors')

    # Probability of event number= the number of activity reach a certain event number / total number of activity
    for key,value in final_split.items():
        L=[]
        prob_event_num=[]
        length=len(final_split[key])
        for i in range(0,length):
            L.append(len(final_split[key][i]))
        minL=min(L)
        maxL=max(L)
        for i in range(0,minL+1):
            prob_event_num.append(1)
        for i in range(minL+1,maxL):
            count=0
            for j in range(0,length):
                if len(final_split[key][j])>=i:
                    count+=1
            prob_event_num.append(count/length)
        prob_event_numDict={'Prob_Event_Num':prob_event_num}
        prob_event_num=pd.DataFrame(prob_event_numDict)
        for i in range(0,length):
            final_split[key][i]=pd.concat([final_split[key][i],prob_event_num.iloc[:final_split[key][i].shape[0]]],axis=1)

    # pad
    max_rows=max(len(matrix) for name,matrix_list in final_split.items() for matrix in matrix_list)
    for key,value in final_split.items():
        for i in range(len(value)):
            # padding
            dataframe=pd.DataFrame(index=range(final_split[key][i].shape[0],max_rows),columns=final_split[key][i].columns)
            dataframe=dataframe.fillna(0)
            final_split[key][i]=pd.concat([final_split[key][i],dataframe],axis=0)

    # Probability of event time: Normal distribution
    for key,value in final_split.items():
        rows_num = len(value[0])
        ptime_ej = []
        k = len(value)
        for j in range(rows_num): # j is the row
            time_ej = []
            prob=[]

            for matrix in value:
                if matrix['Timestamp'][j]!=0 and j!=0:
                    time_ej.append((matrix['Timestamp'][j]-matrix['Timestamp'][0]).total_seconds()*1000) # millisecond
                else:
                    time_ej.append(0)

            mean_time=np.mean(time_ej)
            var_time=np.var(time_ej)

            for i in range(k):
                if var_time!=0:
                    prob.append(
                        math.exp(-(time_ej[i] - mean_time) ** 2 / (2 * var_time)) / (math.sqrt(var_time * 2 * math.pi)))
                else:
                    prob.append(1)
            ptime_ej.append(prob)
        ptime_ej=pd.DataFrame(ptime_ej)

        for i in range(k):
            final_split[key][i]=pd.concat([final_split[key][i],ptime_ej.iloc[:,i]],axis=1)
            final_split[key][i].rename(columns={i:'Prob_of_Event_Time'},inplace=True)
    '''
    # poisson probability of sensor count
    for key, value in final_split.items():  # different key
        col_mean=[]
        y=len(value[0])
        x=len(value)
        for j in range(y):  # different rows
            psensor_count = []

            for i in range(x):  # different matrix
                psensor_count.append(value[i].iloc[j, 6:-3])

            psensor_count = pd.DataFrame(psensor_count)
            col_mean.append(np.round(np.mean(psensor_count, axis=0),4).tolist())
        col_mean=pd.DataFrame(col_mean,columns=['p'+x for x in sensors])

        for i in range(x):
            final_split[key][i]=pd.concat([final_split[key][i],col_mean],axis=1)
    '''
    return final_split


'''
    for key,value in final_split.items(): # different key
        for j in range(len(value[0])): # different rows
            psensor_count=[]

            for i in range(len(value)): # different matrix
                psensor_count.append(value[i].iloc[j,6:-3])

            psensor_count = pd.DataFrame(psensor_count)
            col_mean=np.mean(psensor_count,axis=0)
            length=len(col_mean)
            
            for k in range(length): # kth columns
                for h in range(psensor_count.shape[0]): # hth rows
                    if psensor_count.iloc[h,k]>100:
                        getcontext().prec = 174  # 设置精度
                        psensor_count.iloc[h,k]=Decimal(math.pow(col_mean[k],psensor_count.iloc[h,k]))*Decimal(math.exp((-1)*col_mean[k]))/Decimal(math.factorial(int(psensor_count.iloc[h,k])))
                    else:
                        psensor_count.iloc[h,k]=math.pow(col_mean[k],psensor_count.iloc[h,k])*math.exp((-1)*col_mean[k])/math.factorial(int(psensor_count.iloc[h,k]))
            k = len(final_split[key][0].columns[6:-3])

            for i in range(psensor_count.shape[0]):
                if j==0:
                    dataframe = pd.DataFrame(index=range(final_split[key][i].shape[0]),
                                             columns=['p'+x for x in final_split[key][i].columns[6:-3]])
                    dataframe = dataframe.fillna(0)
                    final_split[key][i] = pd.concat([final_split[key][i], dataframe], axis=0)
                    final_split[key][i] = final_split[key][i].fillna(0)

                final_split[key][i].iloc[j,-k:]=psensor_count.iloc[i,:].values
    return final_split
'''


def getdata_step1(filename):
    datafile = filename.split('.')[0] + '_1.pkl'
    if os.path.exists(datafile):
        with open(datafile, 'rb') as f:
            data_load = pd.read_pickle(f)

        return data_load
    else:
        dataset=load(filename)
        final=split(dataset)
        final = longitudinal_features(filename, final, dataset['sensors'])
        with open(datafile, 'wb') as f:
            pd.to_pickle(final, f)

        return final


def getdata_step2(filename,final_step1):
    datafile = filename.split('.')[0] + '_2.pkl'
    if os.path.exists(datafile):
        with open(datafile, 'rb') as f:
            data_load = pd.read_pickle(f)

        return data_load
    else:
        dataset = load(filename)
        final = cross_sectional_features(final_step1, dataset['sensors'])
        with open(datafile, 'wb') as f:
            pd.to_pickle(final, f)

        return final


def getdata(filename):
    final=getdata_step1(filename)
    print('Step 1 has been done')
    final=getdata_step2(filename,final)
    print("Step 2 has been done")
    return final


def normal_and_splitting(file):
    for key,value in file.items():
        # normalization
        num=len(file[key])
        max_row_num=0
        min_row_num=100000
        for i in range(num):
            if file[key][i].shape[0]>max_row_num:
                max_row_num=file[key][i].shape[0]
            if file[key][i].shape[0]<min_row_num:
                min_row_num=file[key][i].shape[0]
            for j in ['EventPause','SensorPause','Prob_of_Event_Time']:
                mean=file[key][i][j].mean()
                std=file[key][i][j].std()
                file[key][i][j]=(file[key][i][j]-mean)/std
        '''
        # check the length of each activitiy
        print(f'The number of {key}:{num}')
        print(f'The max length of {key}:{max_row_num}')
        print(f'The min length of {key}:{min_row_num}')
        '''
        # To save memory,we break the file into several parts based on the activity type.
        datapath = f'milan\\milan_{key}.pkl'
        if os.path.exists(datapath):
            print('Done')
        else:
            with open(f'milan_{key}.pkl','wb') as f:
                pd.to_pickle(file[key],f)




final=getdata('milan.txt')
normal_and_splitting(final)
