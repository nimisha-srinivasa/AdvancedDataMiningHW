import numpy as np
import time
import matplotlib.pyplot as plt
import os
from data_utils import load_CIFAR10
from neural_net import TwoLayerNet


def split_strategy(Xtr, Ytr):
	num_samples = Xtr.shape[0]

	Xval = Xtr[num_samples*0.8:]
	Yval = Ytr[num_samples*0.8:]
	Xtr = Xtr[:num_samples*0.8]
	Ytr = Ytr[:num_samples*0.8]

	return Xtr, Ytr, Xval, Yval

inputsize = 32*32*3
outputsize = 10

##Load Dataset
dir = os.path.dirname(__file__)
rootname = os.path.join(dir, 'dataset/cifar-10-batches-py')
Xtr, Ytr, Xte, Yte = load_CIFAR10(rootname)
Xtr = Xtr.reshape(50000,3072)
Xte = Xte.reshape(10000,3072)
Xtr, Ytr, Xval, Yval = split_strategy(Xtr, Ytr)

#define the hyper parameters
hiddenlayer_size_arg = 500
batch_size_arg = 2000
num_iters_arg =1000
learning_rate_arg =0.002
learning_rate_decay_arg =0.98
reg_arg=1e-5
verbose = False
'''
###### uncomment the following section to print the value of parameters ######
print "params values:"
print "hiddenlayer_size="+str(hiddenlayer_size_arg)
print "batch_size="+str(batch_size_arg)
print "num_iters="+str(num_iters_arg)
print "learning_rate"+str(learning_rate_arg)
print "learning_rate_decay="+str(learning_rate_decay_arg)
print "regularisation="+str(reg_arg)
print "\n\n\n"
'''

nnet = TwoLayerNet(inputsize, hiddenlayer_size_arg, outputsize)
start_time = time.clock()
train_result = 	nnet.train(Xtr, Ytr, Xval, Yval, learning_rate=learning_rate_arg, 
			   	learning_rate_decay=learning_rate_decay_arg,
            	reg=reg_arg, num_iters=num_iters_arg,
            	batch_size=batch_size_arg, verbose=verbose)
end_time = time.clock()
print "first loss:"+str(train_result['loss_history'][0])+" last loss:"+str(train_result['loss_history'][-1])
#print "loss:\n\n"+str(train_result['loss_history'])
print "Training accuracy:"+str(nnet.accuracy(Xtr, Ytr))
print "Validation accuracy:"+str(nnet.accuracy(Xval, Yval))
print "Testing accuracy:"+str(nnet.accuracy(Xte, Yte))
time_taken = end_time-start_time
print "Time taken for training is "+str(time_taken)
print "\n\n"


'''
###### uncomment the following section to see the effect of different parameters ######

accuracy_history = []
timeset=[]
parameter_set = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2]
for i in parameter_set:
	learning_rate_arg = i
	print "hiddenlayer_size="+str(learning_rate_arg)
	nnet = TwoLayerNet(inputsize, hiddenlayer_size_arg, outputsize)
	start_time = time.clock()
	train_result = 	nnet.train(Xtr, Ytr, Xval, Yval, learning_rate=learning_rate_arg, 
				   	learning_rate_decay=learning_rate_decay_arg,
	            	reg=reg_arg, num_iters=num_iters_arg,
	            	batch_size=batch_size_arg, verbose=verbose)
	end_time = time.clock()
	print "first loss:"+str(train_result['loss_history'][0])+" last loss:"+str(train_result['loss_history'][-1])
	#print "loss:\n\n"+str(train_result['loss_history'])
	print "Training accuracy:"+str(nnet.accuracy(Xtr, Ytr))
	print "Validation accuracy:"+str(nnet.accuracy(Xval, Yval))
	test_accuracy = nnet.accuracy(Xte, Yte)
	accuracy_history.append(test_accuracy)
	print "Testing accuracy:"+str(test_accuracy)
	time_taken = end_time-start_time
	timeset.append(round(time_taken, 2))
	print "Time taken for training is "+str(time_taken)
	print "\n\n"

plt.ylabel("Accuracy")
plt.xlabel("Learning rate")
plt.plot(parameter_set, accuracy_history, marker='o')
plt.savefig("D:\UCSB_projects\AdvDataMining\hw1\\result_pics\\test.png")   
plt.close() 

plt.ylabel("Time taken")
plt.xlabel("Learning rate")
plt.plot(parameter_set, timeset, marker='o')
plt.savefig("D:\UCSB_projects\AdvDataMining\hw1\\result_pics\\test2.png")   
plt.close() 
'''
