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

filename="ds_salaries new.csv"
categorical_features=['work_year', 'experience_level', 'employment_type', 'job_title', 'employee_residence',
'remote_ratio', 'company_location', 'company_size']
dataset=pd.read_csv(filename).drop(['salary','salary_currency'],axis=1)
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
y_mean=np.mean(y)
y_std=np.std(y)
y=np.array((y-y_mean)/y_std)
X=pd.get_dummies(dataset,columns=categorical_features)
scaler=StandardScaler()
X_std=scaler.fit_transform(X)
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
# Set the parameters
input_size=X.shape[1]
hidden_sizes=[5,3]
# Create the method, define the loss function and the optimizer (tells your model how to update its internal parameters to best lower the loss)
model=NeuralNetwork(input_size,hidden_sizes,1).to(device)
loss_func=nn.MSELoss()
weight_updater=optim.SGD(model.parameters(),lr=0.01) # start with a standard lr
X_train_std=torch.from_numpy(X_train).float().to(device)
X_test_std=torch.from_numpy(X_test).float().to(device)
Y_train_std=torch.from_numpy(Y_train).float().to(device)
Y_test_std=torch.from_numpy(Y_test).float().to(device)
#For plotting
loss_history=[]
test_loss_history=[]
epochs=5000
for epoch in range(epochs):
    model.train()
    y_pred=model(X_train_std)
    # calculate the loss for the iteration
    loss=loss_func(y_pred,Y_train_std)
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
        test_loss=loss_func(test_pred,Y_test_std)
        test_loss_history.append(test_loss.to('cpu'))
    loss_history.append(loss.to('cpu'))
with torch.no_grad():
  f1=plt.figure(1)
  plt.plot(range(epochs),loss_history,'r-',test_loss_history,'b--')
  plt.show()
"""
params = {
    'optimizer__lr': [0.001, 0.01, 0.005, 0.0005],
    'max_epochs': list(range(1000, 5000, 500)),
    'optimizer': [optim.SGD, optim.RMSprop, optim.Adagrad],
    'module__input_size':[input_size],
    'module__hidden_sizes':[hidden_sizes],
    'module__output_size':[1]
}
net=NeuralNetRegressor(module=NeuralNetwork,
                       max_epochs=500,
                       criterion=nn.MSELoss())
grid = GridSearchCV(estimator=net, param_grid=params, n_jobs=-1, cv=3)
grid_result=grid.fit(X_std,y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))"""