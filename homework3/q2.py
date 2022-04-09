from openpyxl import load_workbook
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# read data from excel file then convert it to format that we can train the model
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

"""
We use some of the training data as test data
Take first 7 rows for test data
"""
test_data = np_data[0:7]
test_labels = np_labels[0:7]

# knn operations from the link given in the hw description
model = KNeighborsClassifier(n_neighbors=7)
model.fit(np_data, np_labels)
predicted = model.predict(test_data)

# calculate the accuracy
count = 0
for i in range(len(test_data)):
    if test_labels[i] == predicted[i]:
        count +=1
print("Accuracy is : {:.2f}% for my test set".format((count / len(test_labels))*100))
