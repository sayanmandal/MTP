import os
import cv2
import numpy as np
from shutil import copy2
import pandas as pd

def load_dic(classes):
    dic = {}
    cnt = 0
    for c in classes:
        dic[cnt] = c
        cnt += 1
    return dic



def load_image(image_name):
    image = cv2.imread(image_name)
    #print(image[511,511])
    return image



def calc_set(pixelX, pixelY, _class, dir, output_dir, image_dir):
    files = sorted(os.listdir(dir))
    for f in files:
        image = load_image(dir+f)
        #print(image[pixelX,pixelY][2])
        if image[pixelX,pixelY][2] == _class:
            copy2(image_dir+f,output_dir)
            print("copied")
    print("done")


def image_classes(dir, image_dir, dic):

    files = sorted(os.listdir(dir))

    for f in files:
        image = load_image(dir+f)
        i_shape = image.shape
        classes = set()
        for i in range(i_shape[0]):
            for j in range(i_shape[1]):
                classes.add(dic[image[i,j][2]])
        print("file "+ f+".png has:")
        print(classes)
        for c in classes:
            copy2(image_dir+f, c)
    print("done")


def get_classes(csv_file):
    df = pd.read_csv(csv_file)
    classes = []
    for f in df['name']:
        classes.append(f)
    return classes


if __name__ == '__main__':
    '''
    in_dir = "Labels/"
    out_dir = "Vehicles/"
    image_dir = "Images/"
    calc_set(511, 511, 10, in_dir, out_dir, image_dir)
    '''
    csv_file = "class_dict.csv"
    classes = get_classes(csv_file)
    print(classes)
    dic = load_dic(classes)

    in_dir = "Labels/"
    image_dir = "Images/"

    image_classes(in_dir, image_dir, dic)

    '''
    for c in classes:
        os.mkdir(c)
        print("Directory "+c +" created")
    '''
