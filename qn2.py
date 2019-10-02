import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt

x = pd.read_csv('data/2/hw1x.dat', header=None,names='x')
y = pd.read_csv('data/2/hw1y.dat', header=None,names='y')
n = x.shape[0]
z = x.apply(lambda x: x + 1)

b = z.transpose().dot(y) / n
A = z.transpose().dot(z) / n
tithe = b['y'][0]/A['x'][0]

tithe_x = z.apply(lambda x: x*tithe)

plt.plot(z, y, 'ro')

#using least square loss get empirical risk
r = 0
for i in range(x.shape[0]):
  loss = (y['y'][i] - z['x'][i]*tithe)**2 * 0.5
  r = r + loss
r = r /n
print(r)

