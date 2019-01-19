import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy import stats

data= pd.read_csv('dataset.csv')
#print(data.head(10))

X=data.iloc[:, 1].values
Y=data.iloc[:, 2].values
Z=data.iloc[:, 0].values

print(X)
print(Y)

plt.scatter(X, Y)
#plt.show()
plt.savefig("C:/Users/Atharva/Desktop/My stuff/Study Material/BE LP3/Machine Learning/Assignment 1/scat.png", format="png")

slope, intercept, r_value, p_value, std_err= stats.linregress(X, Y)

print("r value of BMI attr: ",r_value**2)

def predict(x):
	return slope*x + intercept



fitline= predict(X)

plt.scatter(X, Y)
plt.plot(X, fitline, c='r')
plt.savefig("C:/Users/Atharva/Desktop/My stuff/Study Material/BE LP3/Machine Learning/Assignment 1/reg.png", format="png")

print("Equation of line with X=BMI is: y=",slope,"x +","(",intercept,")")

plt.scatter(Z, Y)
plt.savefig("C:/Users/Atharva/Desktop/My stuff/Study Material/BE LP3/Machine Learning/Assignment 1/scat_hour.png")
h_slope, h_intercept, h_r_value, h_p_value, h_std_err= stats.linregress(Z, Y)
print("r value of hours attr: ",h_r_value**2)

def predict_h(x):
	return h_slope*x + h_intercept

h_fitline=predict_h(Z)
plt.scatter(Z, Y)
plt.plot(Z, h_fitline, c='b')
plt.savefig("C:/Users/Atharva/Desktop/My stuff/Study Material/BE LP3/Machine Learning/Assignment 1/reg_hour.png")
print("Equation of line with X=Hours driving is: y=",h_slope,"x +","(",h_intercept,")")