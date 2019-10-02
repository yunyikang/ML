import pandas as pd  
import numpy as np

#implement perceptron algorithm
tr_data = pd.read_csv('data/1/train_1_5.csv', header=None,names=['sym','intensity','y'])
ts_data = pd.read_csv('data/1/test_1_5.csv', header=None,names=['sym','intensity','y'])
tithe = np.array([0,0])
#print('Number of training data is {}'.format(tr_data.shape[0]))
#print('Number of test data is {}'.format(ts_data.shape[0]))

#training
"""
for i in range(30):
    h = np.dot([tr_data['sym'][i],tr_data['intensity'][i]], tithe)
    #error when h is < 0, so update
    if h < 0:
        print(h)

        tithe = tithe + tr_data['y'][i] * np.array([tr_data['sym'][i],tr_data['intensity'][i]])

"""
for i in range(10):
    for j in range(tr_data.shape[0]):
        e = np.dot([tr_data['sym'][j],tr_data['intensity'][j]], tithe) * tr_data['y'][j]

        #error when h is < 0, so update
        if e <= 0:
            tithe = tithe + tr_data['y'][j] * np.array([tr_data['sym'][j],tr_data['intensity'][j]])

        
#testing
r = 0
for i in range(ts_data.shape[0]):
    e = np.dot([ts_data['sym'][i],ts_data['intensity'][i]], tithe) * ts_data['y'][i]
    
    if e <= 0:
        r = r + 1
print(r)
r = r / ts_data.size
print(r)

