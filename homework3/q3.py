from openpyxl import load_workbook
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

# read data from excel file then convert it to format that we can train the model
# same as q2
book = load_workbook("Milkshake.xlsx")
sheet = book.active
rows = sheet.rows
header = [cell.value for cell in next(rows)] # names of the rows

data = []
for row in rows:
    arr = []
    for name,val in zip(header,row):
        arr.append(val.value)
    data.append(arr)

# create data and labels
np_data = np.array(data)
labels = [element[7] for element in np_data]
np_data = np.delete(np_data,7,1)
np_data = np.delete(np_data,0,1)
np_labels = np.array(labels)

# to find the best k-fold try from 2-fold to 8-fold
for i in range(2, 9):
    k_fold = KFold(n_splits=i,shuffle=False)
    accuracies = []
    for train_index, test_index in k_fold.split(np_data):
        # create data and labels according to indexes given by KFold
        x_train = np.array([np_data[a] for a in train_index])
        x_test = np.array([np_labels[a] for a in train_index])
        y_train = np.array([np_data[a] for a in test_index])
        y_test = np.array([np_labels[a] for a in test_index])
        # train model and get predictions
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(x_train, x_test)
        predicted = model.predict(y_train)
        true = 0
        # calculate the accuracies
        for j in range(len(predicted)):
            if predicted[j] == y_test[j]:
                true +=1
        accuracy = true/len(predicted)
        accuracies.append(accuracy)
    # print the average accuracy for all folds
    # Example : average of [0:2](test) - [2:6] , [0:4] - [4:6](test) , [0:2]+[4:6] - [2:4](test)
    print("Average accuracy is : {:.2f}% for {} fold validation".format((sum(accuracies)/len(accuracies))*100,i))

""""
We get the best performance with 5 fold cross validation in our example

"""

