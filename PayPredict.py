import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('salary_data.csv')

dataset.head()
x = dataset.iloc[:, 0]
y = dataset.iloc[:, 1]
print(x)
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train)
print(y_train)

y_train

import random
class Linear_Regression:

    
    def __init__(self, xtrain,ytrain, learning_rate=0.01, epoch=30):
        self.learning_rate = learning_rate
        self.theta0 = random.random()
        self.theta1 = random.random()
        self.epoch = epoch
        
        self.x = xtrain
        self.y = ytrain
        self.m = len(self.x)
        #self.history = []

    def display(self):
        plt.scatter(self.x, self.y, color='blue', label='Training data')
        xsamples=np.linspace(self.x.min(),self.x.max(),100)
        ysample_pred = self.h(xsamples)
        plt.plot(xsamples, ysample_pred, color='red', label='Hypothesis')
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        #plt.legend()
        plt.title("Linear Regression Fit")
        plt.show()

    def display_line_eq(self):
        print("Hypothesis Function: h(x) =",self.theta0," +", self.theta1,"x")

    def h(self, x_val):
        return self.theta0 + self.theta1 * x_val

    #def compute_cost(self):
    #    return np.sum((self.h(self.x) - self.y) ** 2) / (2 * self.m)

    def Training_by_Gradient_Decent(self):
        for epoch in range(self.epoch):
            temp0 = self.theta0 - self.learning_rate * (1/self.m) * np.sum(self.h(self.x)-self.y)
            temp1 = self.theta1 - self.learning_rate * (1/self.m) * np.sum((self.h(self.x)-self.y)*self.x)

            self.theta0 = temp0
            self.theta1 = temp1

            #cost = self.compute_cost()
            #self.history.append(cost)

            #print(f"Epoch {epoch}: Cost = {cost:.4f}")
            self.display_line_eq()
            self.display()

        print("Training complete.")

    def evaluate(self,xtest,ytest):
        y_pred = self.h(xtest)
        ss_total = np.sum((ytest - np.mean(ytest)) ** 2)
        ss_res = np.sum((ytest - y_pred) ** 2)
        r2_score = 1 - (ss_res / ss_total)
        print("R² Score: {r2_score:.4f}")
        return r2_score

obj=Linear_Regression(x_train,y_train)
obj.Training_by_Gradient_Decent()
obj.evaluate(x_test,y_test) 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('salary_data.csv')

dataset.head()
x = dataset.iloc[:, 0]
y = dataset.iloc[:, 1]
print(x)
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train)
print(y_train)

y_train

import random
class Linear_Regression:

    
    def __init__(self, xtrain,ytrain, learning_rate=0.01, epoch=30):
        self.learning_rate = learning_rate
        self.theta0 = random.random()
        self.theta1 = random.random()
        self.epoch = epoch
        
        self.x = xtrain
        self.y = ytrain
        self.m = len(self.x)
        #self.history = []

    def display(self):
        plt.scatter(self.x, self.y, color='blue', label='Training data')
        xsamples=np.linspace(self.x.min(),self.x.max(),100)
        ysample_pred = self.h(xsamples)
        plt.plot(xsamples, ysample_pred, color='red', label='Hypothesis')
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        #plt.legend()
        plt.title("Linear Regression Fit")
        plt.show()

    def display_line_eq(self):
        print("Hypothesis Function: h(x) =",self.theta0," +", self.theta1,"x")

    def h(self, x_val):
        return self.theta0 + self.theta1 * x_val

    #def compute_cost(self):
    #    return np.sum((self.h(self.x) - self.y) ** 2) / (2 * self.m)

    def Training_by_Gradient_Decent(self):
        for epoch in range(self.epoch):
            temp0 = self.theta0 - self.learning_rate * (1/self.m) * np.sum(self.h(self.x)-self.y)
            temp1 = self.theta1 - self.learning_rate * (1/self.m) * np.sum((self.h(self.x)-self.y)*self.x)

            self.theta0 = temp0
            self.theta1 = temp1

            #cost = self.compute_cost()
            #self.history.append(cost)

            #print(f"Epoch {epoch}: Cost = {cost:.4f}")
            self.display_line_eq()
            self.display()

        print("Training complete.")

    def evaluate(self,xtest,ytest):
        y_pred = self.h(xtest)
        ss_total = np.sum((ytest - np.mean(ytest)) ** 2)
        ss_res = np.sum((ytest - y_pred) ** 2)
        r2_score = 1 - (ss_res / ss_total)
        print("R² Score: {r2_score:.4f}")
        return r2_score

obj=Linear_Regression(x_train,y_train)
obj.Training_by_Gradient_Decent()
obj.evaluate(x_test,y_test) 

