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
model_refined.training(image_path = '/Data/Color_image',line_path = '/Data/Line_image', loadEpochs=5)  #number of epochs 
model_refined.save_model(saved_path = '/saved_model/PaintsTensorFlowRefinedModel.h5')