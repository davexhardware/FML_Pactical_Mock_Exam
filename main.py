#!/usr/bin/env python
import pandas as pd
import numpy as np
import torch.cuda
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch import nn,optim,cuda
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"
class NeuralNetwork(nn.Module):
    def __init__(self,input_size,hidden_sizes,output_size):
        super().__init__()
        self.input_size=input_size
        self.output_size=output_size
        self.hidden_layers=[]
        self.h_layers=[]
        for i in range(len(hidden_sizes)):
          if(i==0):
            self.h_layers.append(nn.Linear(input_size,hidden_sizes[0]))
          else:
            self.h_layers.append(nn.Linear(hidden_sizes[i-1],hidden_sizes[i]))
          self.add_module(f"layer{i}",self.h_layers[-1])
        self.regressor=nn.Linear(hidden_sizes[-1],output_size)
    def forward(self,X):
        for layer in self.h_layers:
          X=layer(X)
        return self.regressor(X)
if __name__=="__main__":
    filename="ds_salaries new.csv"
    categorical_features=['work_year', 'experience_level', 'employment_type', 'job_title', 'employee_residence',
    'remote_ratio', 'company_location', 'company_size']
    dataset=pd.read_csv(filename).drop(['salary','salary_currency'],axis=1).drop_duplicates()
    factor=1.5
    print(dataset.shape)
    for col in dataset.columns:
        if col not in categorical_features:
            X=dataset[col].dropna(axis=0)
            q1=np.quantile(X.values,q=0.25)
            q3=np.quantile(X.values,q=0.75)
            iqr=q3-q1
            lower_bound=q1-factor*iqr
            upper_bound=q3+factor*iqr
            lower_mask=X<lower_bound
            upper_mask=X>upper_bound
            X[lower_mask|upper_mask]=np.nan
            x_mean=np.mean(X,axis=0)
            X[lower_mask|upper_mask]=x_mean
            dataset[col]=X
    # Setting up the dataset: normalize y, make the categorical features of X into numerical ones (by get_dummies)
    # and then split and normalize also X
    y=dataset['salary_in_usd']
    i=1
    X=pd.get_dummies(dataset,columns=categorical_features)
    scaler=StandardScaler()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    X_train_std=scaler.fit_transform(X_train)
    X_test_std=scaler.transform(X_test)
    y_mean=np.mean(y_train)
    y_std=np.std(y_train)
    y_train_std=np.array((y_train-y_mean)/y_std)
    y_test_std=np.array((y_test-y_mean)/y_std)
    y=np.array((y-y_mean)/y_std)
    # Set the parameters
    input_size=X.shape[1]
    hidden_sizes=[5,3]
    # Create the method, define the loss function and the optimizer (tells your model how to update its internal parameters to best lower the loss)
    X_train_std=torch.from_numpy(X_train_std).float().to(device)
    X_test_std=torch.from_numpy(X_test_std).float().to(device)
    y_train_std=torch.from_numpy(y_train_std).float().to(device)
    y_test_std=torch.from_numpy(y_test_std).float().to(device)
    final_test_loss={}
    final_distances={}
    for lri in [0.1,0.01,0.005,0.001,0.0001]:
      for l2 in [1,0.8,0.5,0.1,0.01]:
        model=NeuralNetwork(input_size,hidden_sizes,1).to(device)
        loss_func=nn.MSELoss()
        weight_updater=optim.SGD(model.parameters(),lr=lri,weight_decay=l2) # start with a standard lr
        #For plotting
        loss_history=[]
        test_loss_history=[]
        epochs=3000
        for epoch in range(epochs):
            model.train()
            y_pred=model(X_train_std)
            # calculate the loss for the iteration
            loss=loss_func(y_pred,y_train_std)
            # compute loss gradients for the parameters
            loss.backward()
            # update the parameters on the computed gradients
            weight_updater.step()
            # In PyTorch, for example, when you perform backpropagation to compute
            # the gradients of the loss, these gradients accumulate by default through the epochs.
            # It's a common practice to zero them using this line to avoid interference from previous iterations.
            weight_updater.zero_grad()
            # we inform the model that we are evaluating
            model.eval()
            with torch.inference_mode():
                test_pred=model(X_test_std)
                test_loss=loss_func(test_pred,y_test_std)
                test_loss_history.append(test_loss.to('cpu'))
            loss_history.append(loss.to('cpu'))
        with torch.no_grad():
          plt.figure(i)
          i+=1
          #is 1 a good loss(?)
          plt.plot(range(epochs),loss_history,'r-',test_loss_history,'b--')
          plt.legend(['training_loss','test_loss'])
          plt.title(label=f'lr:{lri}, l2:{l2}')
          fin_dist=min(np.subtract(test_loss_history[-20:],loss_history[-20:]))
          final_distances[(lri,l2)]=fin_dist
          print('minimum asintotic distance:',fin_dist)
          final_test_loss[(lri,l2)]=min(test_loss_history)
          plt.show()
    minimum=min(final_distances,key=final_distances.get)
    # Obv the couple of (learning rate, normalization parameter) that
    # has the minimum distance between the train and the test losses will
    # be the less overfitted model
    print('Minimum training-testing loss distance with parameters: ',minimum)
    minimum=min(final_test_loss,key=final_test_loss.get)
    print('Minimum test loss with parameters: ',minimum)