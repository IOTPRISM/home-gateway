import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd
import time 
import os
import pathlib
from torch import optim
from torch.autograd import Variable 
from pathlib import Path
from random import randint


directory = r'$yourpath'
patient_ID = ['2zbyXzYNKPwiPtjaA2L64o.npy','3hY7Mp7u9YPo1xMARSxLhc.npy','4h1dAuzg9rdrhyojwxUS26.npy']
colums1 = ['time','end time']
colums2 = ['time','end time','Reading 1']
colums3 = ['time','end time','Reading 1','Reading 2']
door_key = 0
switch = 0
pir_key = 0
hidden_size = 2

def Data_Loader(patient):
    sensors_lst = []
    column_names = []
    for subdirectory in os.scandir(directory):
        lock = False
        for patients in os.scandir(subdirectory):
            if str(pathlib.Path(patients)).endswith(patient_ID[0]):
                if lock == False:
                    column_names.append(str(os.path.basename(subdirectory)))
                    lock == True                                              
                sensors = np.load(pathlib.Path(patients))
                sensors_lst.append(sensors)
    sensors_pd = pd.DataFrame([sensors_lst],columns=column_names)
    dataframe_lst = []
    df = pd.DataFrame()
    sensor_num = 0
    Sensor_ID = column_names[sensor_num]
    print(Sensor_ID)
    for i in range(len(sensors_pd[Sensor_ID].loc[0])):
        dataframe_lst.append(sensors_pd[Sensor_ID].loc[0][i])
    if len(sensors_pd[Sensor_ID].loc[0][0]) == 2: 
        df = pd.DataFrame(dataframe_lst,columns = colums1)
        df = df.drop("end time", axis=1)
    elif len(sensors_pd[Sensor_ID].loc[0][0]) == 3:    
        df = pd.DataFrame(dataframe_lst,columns = colums2)
        df = df.drop("end time", axis=1)
    elif len(sensors_pd[Sensor_ID].loc[0][0]) == 4:    
        df = pd.DataFrame(dataframe_lst,columns = colums3)
        df = df.drop("end time", axis=1)
    df['labels'] = 0
    anomaly_lst = []
    col_name = df.columns
    col_len = len(df.columns)
    return df


def PIR_anomaly(self,n,id):
    global pir_time
    global pir_key
    ranint = randint(0,19) # there are 20 locations
    if pir_key == 0:
        pir_time = self
        s = [str(integer) for integer in pir_time]
        a_string = "".join(s)
        pir_time = int(a_string)
        pir_key = 1
    if pir_key == 1:
        pir_time = pir_time + 60
        return [pir_time] + [ranint]*(n - 2) + [1]

        
def anomaly_insertion(n,type,dataframe,id):
    df = dataframe
    anomaly_lst = []
    col_name = df.columns
    col_len = len(df.columns)
    start_t = dataframe['time'].iloc[0]
    end_t = dataframe['time'].iloc[-1]
    for i in range(n):
        random_timestamp = random.randint(start_t,end_t)
        if type == 'zero':
            zp_anomaly_lst = zero_anomaly([random_timestamp],col_len)
        if type == 'random':
            zp_anomaly_lst = random_anomaly([random_timestamp],col_len)
        if type == 'door':
            zp_anomaly_lst = door_anomaly([random_timestamp],col_len,id)
        if type == 'PIR':
            zp_anomaly_lst = PIR_anomaly([random_timestamp],col_len,id)
        anomaly_lst.append(zp_anomaly_lst)
    anomaly_df = pd.DataFrame(anomaly_lst,columns=col_name)
    concat_df = pd.concat([dataframe,anomaly_df],ignore_index=True)
    concat_df = concat_df.sort_values('time')
    concat_df = concat_df.reset_index(drop=True)
    return  concat_df


dataT1 = Data_Loader(patient_ID[0])
dataT2 = Data_Loader(patient_ID[1])
df1 = Data_Loader(patient_ID[2])

dataV = anomaly_insertion(120,'PIR',df1,id=1)

dataT1 = dataT1.astype('float32')
dataT2 = dataT2.astype('float32')
dataV = dataV.astype('float32')

train_data = [dataT1,dataT2]
train_data = pd.concat(train_data)
valid_data = dataV

X_trainD = train_data.loc[:, train_data.columns != 'labels']
X_valD = valid_data.loc[:, valid_data.columns != 'labels']
Y_trainD = train_data.loc[:,'labels']
Y_valD = valid_data.loc[:,'labels']

X_train = torch.tensor(X_trainD.values).float()
X_val =torch.tensor(X_valD.values).float()
Y_train = torch.tensor(Y_trainD.values).float()
Y_val = torch.tensor(Y_valD.values).float()


class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes = num_classes 
        self.num_layers = num_layers 
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.seq_length = seq_length 
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                        num_layers=num_layers, batch_first=True) 
        self.lstm2 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                        num_layers=num_layers, batch_first=True)
        self.fc_1 =  nn.Linear(hidden_size, 128) 
        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=0.3)
    def forward(self,x):
        x = x.unsqueeze(0)
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) 
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) 
        output, (h1, c1) = self.lstm1(x, (h_0, c_0))
        output, (h2, c2) = self.lstm1(x, (h1, c1))
        hn = h2.view(-1, self.hidden_size) 
        out = self.relu(hn)
        out = self.dropout(out)
        out = self.fc_1(out) 
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc(out) 
        out = self.relu(out) 
        return out

model = LSTM(2,hidden_size,4,1,2880)
optimizer = torch.optim.SGD(model.parameters(), lr = 3e-3,momentum=0.9)
criterion = nn.L1Loss()
batch_size =1
n_epochs =1
permutation_train = torch.randperm(X_train.size()[0])
permutation_val = torch.randperm(X_val.size()[0])
anomalies = []
mae_history = []
history = []

def Train_AD():
    for epoch in range(n_epochs):
        accuracy = 0
        accuracy_history = []
        timer_plot = []
        for i in range(1,X_train.size()[0], batch_size):
            optimizer.zero_grad()
            indices = permutation_train[i:i+batch_size]
            batch_x, batch_y = X_train[indices], Y_train[indices]
            outputs = model.forward(batch_x)
            loss = criterion(outputs,batch_y)
            mae_history.append(loss)
            loss.backward()
            optimizer.step()
            print(f"-> Training for hidden size = "+str(hidden_size)+":   "+str(i)+"/"+str(X_train.size()[0]), end = "\r")        
        reconstruction_error_threshold = max(mae_history)
        print("the reconstruction error is:" + str(reconstruction_error_threshold))
        for a in range(1,X_val.size()[0], batch_size):
            start_time = time.time() 
            indicesV = permutation_val[a:a+batch_size]
            batch_x_val, batch_y_val = X_val[indicesV], Y_val[indicesV]
            output = model.forward(batch_x_val)
            loss = criterion(output,batch_y_val)
            history.append(loss)
            if loss > reconstruction_error_threshold:
                anomalies.append(batch_x_val)
            if(output == batch_y_val):
                accuracy = accuracy +1
                accuracy_history.append(accuracy/a)
            time_elapsed = time.time() - start_time
            timer_plot.append(1000*time_elapsed/len(X_val))
            print(f"-> Testing for hidden size = "+str(hidden_size)+":     "+str(a)+"/"+str(X_val.size()[0]), end = "\r")
        print(f"Finished for Hidden Size = "+str(hidden_size)+"                                             ")
    print(len(anomalies))
    return anomalies

anomalies,accuracy = Train_AD()
print(len(anomalies)) # for the program to work efficiently, the number of anomalies should be the same as the ones inserted.
print(anomalies) # double check that they are indeed the correct ones.
