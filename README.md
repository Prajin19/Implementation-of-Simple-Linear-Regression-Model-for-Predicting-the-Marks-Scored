# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Prajin S 
RegisterNumber: 212223230151 
*/
```
```Python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df=pd.read_csv("student_scores.csv")
print(df.head())
print(df.tail())
df.info()
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,-1].values
print(y)
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=1/3,random_state=1)
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)
mse=mean_squared_error(Y_test,Y_pred)
mae=mean_absolute_error(Y_test,Y_pred)
rms=np.sqrt(mse)
print('MSE = ',mse)
print('MAE = ',mae)
print('RMS = ',rms)
plt.scatter(X_test,Y_test,color="violet")
plt.plot(X_test,Y_pred)
plt.title("Test Set (H vs S)")
plt.xlabel("Scores")
plt.ylabel("Hours")
plt.show()
```

## Output:
![image](https://github.com/user-attachments/assets/07b2fff7-cd96-4224-84fa-7d5a79f7123c)
![image](https://github.com/user-attachments/assets/108fe67c-0a97-4404-9c47-fb63b8b01250)
![image](https://github.com/user-attachments/assets/3e0b922a-fba9-4a79-baf1-36d2de425f3a)
![image](https://github.com/user-attachments/assets/2019a25f-64d5-463a-9ada-ed88ba07e245)
![image](https://github.com/user-attachments/assets/8edb99f6-2bec-490b-b11d-298dbeab225c)
![image](https://github.com/user-attachments/assets/6d1e4dce-7e30-4190-bc63-3ddd597e9b22)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
