import pandas as pd
import numpy as np
import requests
import io
from sklearn.neighbors import KNeighborsClassifier

"""
At the end of the code the accuracy is printed to the console with 10% of error rate: If f(5) = 100 in the training set 
guesses provides 90 <  f(5) < 110 will be accepted as true since knn works with labels rather than linear numbers.

The function get_y_val() can be used to get any guessed y value for a given x (I needs to be called after model is implemented)

"""

# returns the predicted y value for given knn model and given x
def get_y_val(knn_model,x_value):
    return knn_model.predict(np.array(x_value).reshape(-1,1))[0]

# Downloading the csv file from your GitHub account
# take the data and convert it to numpy array -> then reshape the numpy array
url = "https://raw.githubusercontent.com/boragungoren-portakalteknoloji/ATILIM-ECON484-Spring2022/main/Homework%20Assignments/HW3%20-%20First%20KNN%20Model/Training%20Data%20with%20Outliers.csv"
download = requests.get(url).content
df = pd.read_csv(io.StringIO(download.decode('utf-8')))
x_data = np.array(df["X"].tolist()).reshape(-1,1)
y_data = np.array(df["Y"].tolist()).reshape(-1,1)
train_list = df.values.tolist()

url = "https://raw.githubusercontent.com/boragungoren-portakalteknoloji/ATILIM-ECON484-Spring2022/main/Homework%20Assignments/HW3%20-%20First%20KNN%20Model/Validation%20Data.csv"
download = requests.get(url).content
df = pd.read_csv(io.StringIO(download.decode('utf-8')), sep=";")
x_val = np.array(df["X"].tolist()).reshape(-1,1)
y_val = np.array(df["Y"].tolist()).reshape(-1,1)
validation_list = df.values.tolist()

# I trained the knn algorithm in a loop where k can be 1-10
# I get the best accuracy when k is 7 therefore I chose 7 for k
model = KNeighborsClassifier(n_neighbors=7)

# Train the model using the training sets
model.fit(x_data,y_data.ravel())
# Predicted Output
predicted = model.predict(x_val)
# YOU CAN ONLY CALL get_y_val() function after this point
# ex : get_y_val(model,x_value)
print("Guesssed f(5) is : ".format(model,5))
error_rate = 0.1
count = 0
for i in range(len(predicted)):
    if (y_val[i] + y_val[i] * error_rate) > predicted[i] > (y_val[i] - y_val[i] * error_rate):
        count +=1

print("Accuracy is {}".format(count / len(predicted)))
