import h5py
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
import numpy as np


filename = '/Users/rf/Desktop/AI/DeepLearning/Open_AI/Q-Learning-for-Trading-master/weights/201810022126-dqn.h5'
filename1='/Users/rf/Desktop/AI/DeepLearning/Open_AI/Q-Learning-for-Trading-master/portfolio_val/201810022126-test.p'
f = h5py.File(filename, 'r')

pickle_off = open(filename1,"rb")
emp = pickle.load(pickle_off)
print(emp)
pickle_off1 = open(filename1,"rb")
emp1 = pickle.load(pickle_off1, encoding='bytes')
emp2=pd.DataFrame(emp1)
print(emp2.head(10))
emp2.plot()
plt.show()

# Calculate the simple average of the data
mean = [np.mean(emp2)]*len(emp2)
print(mean)
mean=pd.DataFrame(mean)
mean.plot()
plt.show()
# List all weights
#dset = f['dense_1']
#print(dset.name)
#print(list(f.keys()))

# Get the data
#data = list(f[a_group_key])


