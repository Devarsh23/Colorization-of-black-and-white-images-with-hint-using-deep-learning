from model import PaintsTensorFlowRefinedModel
import hyperparam
import numpy as np 
import pandas as pd
import os
import tensorflow as tf
import cv2
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import time


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)

model_refined = PaintsTensorFlowRefinedModel.PaintsTensorFlowTrain(batch_size = 4)
imagepath = ''  # path to real test image
linepath = ''  # path to line test image
prediction = model_refined.convert_to_pred(imagepath,linepath)

plt.figure()

#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(1,3) 

# use the created array to output your multiple images. In this case I have stacked 4 images vertically
axarr[0].imshow(cv2.cvtColor(cv2.imread(imagepath), cv2.COLOR_BGR2RGB))
axarr[1].imshow(cv2.imread(linepath))
axarr[2].imshow(prediction[0])