# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import os
from model.model_ops import *
from model.mymodel import *

datasets_file_path = os.path.dirname(os.path.dirname(os.getcwd()))+'/datasets/cifar-10-batches-bin'

model1 = MyModel()
#model1.train(datasets_file_path,False)
#model1.train(datasets_file_path,True)
#model1.depthwise_test(datasets_file_path)
model1.gen_test(datasets_file_path)
