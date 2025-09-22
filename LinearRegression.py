
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

## Download latest version
#path = kagglehub.dataset_download("aartisonigara/study-hours-vs-exam-scores")
#
#print("Path to dataset files:", path) 
# 


data =pd.read_csv("/home/neuro/.cache/kagglehub/datasets/aartisonigara/study-hours-vs-exam-scores/versions/1/student_scores.csv")

# inputs 
x = data[["Hours"]].values
# outputs
y = data[["Scores"]].values


print("x hours  : ", x)
print("y scores : ", y)


def loss_function(m , b , x , y) : 
    total_error = 0 
    for i in range(len(x)) : 
        total_error += (y[i] - (m * x[i] + b)) ** 2
    total_error = total_error / len(x)  
    return total_error


def gradient_descent (m_current , b_current , x , y , L) : 
    n = len(x)
    Dem = 0 
    Deb = 0 
    for i in range(n) : 
        Dem += (-2 / n) * (x[i] * (y[i] - (m_current * x[i] + b_current )))
        Deb += (-2 / n) * ((y[i] - (m_current * x[i] + b_current )))

    m = m_current - (L * Dem) 
    b = b_current - (L * Deb)
    return m , b 

m = 0 
b = 0 
L = 0.01
iteraiton = 300

for i in range(iteraiton) : 
    m , b=  gradient_descent(m , b ,x , y , L)
    
# Plot the actual data points
plt.scatter(x, y, color="red", label="Data Points")
# Plot the regression line
plt.plot(x, m * x + b, color="blue", label="Trendline")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.title("Hours of Study vs Scores")
plt.legend()
plt.show()
