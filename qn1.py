import pandas as pd  
import numpy as np

#implement perceptron algorithm
tr_data = pd.read_csv('data/1/train_1_5.csv', header=None,names=['sym','intensity','y'])
ts_data = pd.read_csv('data/1/test_1_5.csv', header=None,names=['sym','intensity','y'])
tithe = np.array([0,0])
#training
print('Number of training data is {}'.format(tr_data.shape[0]))
"""
print(tr_data.head())
for i in range(3):
    h = np.dot([tr_data['sym'][i],tr_data['intensity'][i]], tithe) * tr_data['y'][i]
    print('tithe is {} and h is {}'.format(tithe, h))


    if h <= 0:
            tithe = tithe + tr_data['y'][i] * np.array([tr_data['sym'][i],tr_data['intensity'][i]])
"""
for i in range(5):
    for j in range(tr_data.shape[0]):
        h = np.dot([tr_data['sym'][j],tr_data['intensity'][j]], tithe) * tr_data['y'][j]

        #update
        if h <= 0:
            tithe = tithe + tr_data['y'][j] * np.array([tr_data['sym'][j],tr_data['intensity'][j]])
            print(tithe)

        
#testing
loss = 0
for i in range(ts_data.shape[0]):
    h = np.dot([ts_data['sym'][i],ts_data['intensity'][i]], tithe) * ts_data['y'][i]
    if h <= 0:

        loss = loss + 1
print('Number of test data is {}'.format(ts_data.shape[0]))
print(loss)
loss = loss / ts_data.size
print(loss)

"""
h1 = -0.725767*-0.811273+0.022763*0.035524
print(h1)
print(-0.811273+-0.725767,0.022763+0.035524)
"""