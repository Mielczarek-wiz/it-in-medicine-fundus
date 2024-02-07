import numpy as np
import skimage as ski
import warnings
from skimage import io
from skimage.filters import gaussian, unsharp_mask, sobel
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
import skimage.morphology as mp
from skimage import feature, exposure
import cv2
from sklearn.neighbors import KNeighborsClassifier
from skimage.transform import resize
from imblearn.under_sampling import RandomUnderSampler
import pickle
import time
warnings.simplefilter(action='ignore', category=FutureWarning)
import random

def RescaleImage(frame, scale = 0.20):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def geo_mean(sensitivity, specificity):
    return (sensitivity * specificity) ** (1.0 / 2)
	
	
#funkcja zwraca finalny wynik w postaci statystyk: macierz pomyłek, trafność (accuracy), czułość (sensitivity), swoistość (specificity)
def statistical_analysis(binary_mask, expert_mask):
    MIN = np.percentile(expert_mask, 0.0)
    MAX = np.percentile(expert_mask, 100.0)
    expert_mask = (expert_mask - MIN) / (MAX - MIN)
    fail_matrix = np.zeros((expert_mask.shape[0], expert_mask.shape[1]))
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for y in range(len(binary_mask)):
        for x in range(len(binary_mask[y])):
            if(binary_mask[y][x] != expert_mask[y][x]):
                fail_matrix[y][x] = 1.0
            if(binary_mask[y][x] == 1 and expert_mask[y][x] == 1):
                true_positive += 1
            if(binary_mask[y][x] == 0 and expert_mask[y][x] == 0):
                true_negative += 1
            if(binary_mask[y][x] == 1 and expert_mask[y][x] == 0):
                false_positive += 1
            if(binary_mask[y][x] == 0 and expert_mask[y][x] == 1):
                false_negative += 1
    number_of_pixels = expert_mask.shape[0] * expert_mask.shape[1]
    print('===========================')
	
    print(f"Loss: {np.sum(fail_matrix)/number_of_pixels}")
    print(f"TP: {true_positive/number_of_pixels}")
    print(f"TN: {true_negative/number_of_pixels}")
    print(f"FP: {false_positive/number_of_pixels}")
    print(f"FN: {false_negative/number_of_pixels}")
    accuracy = (true_negative + true_positive) / number_of_pixels
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    geo = geo_mean(sensitivity, specificity)
    
    print(f"Accuracy: {accuracy}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"Geometric mean: {geo}")
    print('===========================')