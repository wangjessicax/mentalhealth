import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import glob
import os

def loader():

	file = open("data.txt","r") 
	contents=file.read()
		
		

		#print(contents)
		#print(contents)
		#creates array with each data set 
	train_totalList=list()
	for x in range(0, 30000):
		train_totalList.append(contents[x*2033:(x+1)*2033])
	return(train_totalList)
	torch.tensor(train_totalList)
		#creates second array with specifically depression data which is in column 113 
	train_depList=list()
	for x in range(0,30000):
		train_depList.append(contents[(x*2033)+113])
	#print(depList)
	torch.tensor(train_depList)

def test_loader():
	file = open("data.txt","r") 
	contents=file.read()
		
		

		#print(contents)
		#print(contents)
		#creates array with each data set 
	test_totalList=list()
	for x in range(0, 30346):
		test_totalList.append(contents[x*2033:(x+1)*2033])
	torch.tensor(test_totalList)
		#creates second array with specifically depression data which is in column 113 
	test_depList=list()
	for x in range(0,30346):
		test_depList.append(contents[(x*2033)+113])
	#print(depList)
	torch.tensor(test_depList)



