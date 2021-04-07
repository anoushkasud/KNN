
####importing libraries####
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

####reading data####
data = pd.read_csv('C:\\Users\\HP\\Desktop\\datasci\\train.csv\\train.csv')

#print(data.head())  #prints top 5 rows#
#print(data.tail())  #prints last 5 rows#
#print(data['label'].value_counts()) #returns a series object counting all the unique values#

####seprating dependent and independent variables####
X = data.drop(['label'],axis = 1)
y = data['label']
#print(X.shape,y.shape)

####training data####

#impoting train test split function
from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(X,y,test_size = 0.3, random_state = 10)
#print(train_x.shape,test_x.shape,train_y.shape,test_y.shape)


#in this case we don't need to scale data
###for scaling 
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScale r()
# scaler.fit(train_x)
# train_x = scaler.transform(train_x)
# test_x = scaler.transform(test_x)


#importing knnclassifier and metrics
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn import metrics

#defining elbow curve
def elbow(k):

    #initiating an empty list
    error_test = []

    #training list over every value of k
    for i in k:
        #creating an instance for knn
        clf = knn(n_neighbors=i)
        clf.fit(train_x,train_y)
        #appending errors calculated using predict to the emply list 
        tmp = clf.predict(test_x)
        tmp = metrics.accuracy_score(tmp,test_y)
        error = 1-tmp
        error_test.append(error)
    return error_test

#defining range of k
k = range(1,10)

#calling the elbow function
test = elbow(k)

#plotting the curves
plt.plot(k, test)
plt.xlabel('K Neighbors')
plt.ylabel('Test error')
plt.title('Elbow curve for test')
plt.show()

# range(1,20) gives k=3 with test error <0.034 # #lowest#

#getting the value of k corresponding lowest value of error 
m={}
for i in range(1,10):
    m[i]=np.interp(i,k,test)

val=1
for j in range(1,10):
    if(val>m[j]):
        val=m[j]
        num=j

#creating an instance for knn
clf = knn(n_neighbors=num) #found before
clf.fit(train_x,train_y)

pred = clf.predict(test_x)

from sklearn.metrics import classification_report
print(classification_report(test_y, pred))