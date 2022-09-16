class Net(nn.Module):
    global size_mat
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(size_mat, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)        
        self.fc4 = nn.Linear(512, 2)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def Train_AD(time_h,time_m,val_h,val_m,pat):
    X_train,Y_train,X_val,Y_val,train_data,valid_data,anom_lst,Sensor_ID,n = Data_Preprocessing(time_h,time_m,val_h,val_m,pat)
    global size_mat
    size_mat = len(X_train[0])
    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr = 3e-5)
    criterion = nn.L1Loss()
    batch_size = 1
    n_epochs = 10
    permutation_train = torch.randperm(X_train.size()[0])
    permutation_val = torch.randperm(X_val.size()[0])
    anomalies = []
    valid_loss = []
    train_loss = []
    inference_time = []
    for epoch in range(n_epochs):
        train_loss = []
        for i in range(0,len(train_data), batch_size):
            optimizer.zero_grad()
            indices = permutation_train[i:i+batch_size]
            batch_x, batch_y = X_train[indices], Y_train[indices]
            outputs = model.forward(batch_x)
            loss = criterion(outputs,batch_y)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        threshold = mean(train_loss)

    for a in range(0,X_val.size()[0], batch_size):
        start_time = time.time() 
        indicesV = permutation_val[a:a+batch_size]
        batch_x_val, batch_y_val = X_val[indicesV], Y_val[indicesV]
        output = model.forward(batch_x_val)
        pred = torch.max(output,1)[1]
        loss = criterion(output,batch_y_val)
        valid_loss.append(loss.item())
        if loss.item() >= threshold:
            anomalies.append(batch_x_val)
        time_elapsed = time.time() - start_time
        inference_time.append(1000*time_elapsed)   
    return anomalies,train_loss,valid_loss,Sensor_ID,n,size_mat,mean(inference_time)

def Main(time_h,time_m,val_h,val_m,pat):
    anomalies,train_loss,valid_loss,Sensor_ID,n,size_mat,inf_t = Train_AD(time_h,time_m,val_h,val_m,pat)
    counter = 0
    if len(anomalies) == 0:
        counter = 0
    else:
        for i in range(len(anomalies)):
            x = anomalies[i].numpy()
            if x[0][size_mat-1]==1.0: # the anomalies are created with a time difference of 1.0, this can be used to detect them.
                counter = counter+1
    correct = counter
    fake_positive = len(anomalies) - correct
    correct = correct/n
    accuracy = ((correct*n)+(n-fake_positive))/(n*2)
    print('')
    print(str(Sensor_ID)+', For training window: ' +str(time_h)+ ':'+str(time_m)+', For validation window: '+str(val_h)+':'+str(val_m))
    print('Correctly Identified Anomalies (fraction) : ' + str(correct))
    print('Fake Positives: '+ str(fake_positive))
    print('Accuracy: '+str(accuracy))
    print('Average Inference time: '+ str(inf_t))

    return Sensor_ID,accuracy,inf_t


train_time = [[24,0],[12,0],[3,0],[0,45],[0,30],[0,15]]# time window format: [hours,minutes]
valid_time= [[24,0],[12,0],[3,0],[0,45],[0,30],[0,15]]
sensors_lst = []
periods_lst = []
correct_lst = []
false_lst = []
total_results = []
repetitions = 1
for patient in range(len(patient_ID)):
    try:
        for sensor in range(22):
            global sensor_num
            sensor_num = sensor
            try:
                for c in range(len(valid_time)):
                    for i in range(len(train_time)):
                        count_r = 0
                        for b in range(repetitions):
                            print('repetition: '+ str(b))
                            try:
                                Sensor_ID,accuracy,inf_t = Main(train_time[i][0],train_time[i][1],valid_time[c][0],valid_time[c][1],patient)
                                count_r = count_r +1
                                total_results.append([Sensor_ID,train_time[i][0],train_time[i][1],valid_time[c][0],valid_time[c][1],accuracy,inf_t])
                            except: pass
            except: pass
    except: pass

print(total_results)
