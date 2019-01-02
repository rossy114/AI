import pickle

infile = open(/Users/rf/Desktop/AI/DeepLearning/Open_AI/Q-Learning-for-Trading-master/weights/201810022126-dqn.h5,'rb')
new_dict = pickle.load(infile)
infile.close()