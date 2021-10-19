
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

path ='E:\\برامج\\DATA.txt'
data= pd.read_csv(path,header=None,names=('Population','Profit'))
#data.columns=['Population','Profit'] this alterative names=('Population','Profit')
print(data.head(10))

x=data.iloc[:,0].values.reshape(-1,1)
y=data.iloc[:,1].values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in 10,000s")
plt.show()

from sklearn.linear_model import LinearRegression
LR=LinearRegression()

LR.fit(x, y)
LR.intercept_
LR.coef_

y_pred=LR.predict(x)

plt.scatter(x, y, c='y')
plt.plot(x,y_pred)
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in 10,000s")
plt.savefig("graph.png")
plt.show()

#model Evaluation

"""import sklearn.metrics as mc

print(mc.mean_absolute_error(y,y_pred))

"""
