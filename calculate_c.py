import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
import json
import pandas as pd
from shutil import copy2
from numpy import linalg as LA
import pickle
import scipy.stats as stats
from PIL import Image
from utils import utils
from multiprocessing import Pool


def get_classes(csv_file):
    df = pd.read_csv(csv_file)
    classes = []
    for f in df['name']:
        classes.append(f)
    return classes


def addRep(image, label, height, width):
    for h in range(height):
        for w in range(width):
            if label[h,w,0] == k:
                class_representative[h,w] += image[h,w]
                cnt[h,w] += 1



def getCk(in_dir, label_dir, csv_file):
    classes = get_classes(csv_file)
    input_images = sorted(os.listdir(in_dir))

    height, width, _ = utils.load_image(in_dir+"/"+input_images[0]).shape

    #height = 1
    #width = 1

    for k in range(len(classes)):
        cnt = np.zeros((height, width))
        class_representative = np.zeros((height, width, 3))
        for f in input_images:
            print("Reading " + f)
            image_path = in_dir + "/" + f
            label_path = label_dir + "/" + f
            image = cv2.imread(image_path,-1)
            label = cv2.imread(label_path,-1)
            for h in range(height):
                for w in range(width):
                    if label[h,w,0] == k:
                        class_representative[h,w] += image[h,w]
                        cnt[h,w] += 1

        for h in range(height):
            for w in range(width):
                if cnt[h,w] > 0:
                    class_representative[h,w] = class_representative[h,w] / cnt[h,w]
        img = Image.fromarray(class_representative.astype(np.uint8))
        img.save("C"+str(k)+".png")
        print("saved...")


if __name__ == '__main__':
    in_dir = "Images/"
    label_dir = "Labels/"
    csv_file = "class_dict.csv"
    print("here")
    getCk(in_dir, label_dir, csv_file)
