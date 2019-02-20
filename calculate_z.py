import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
import json
import pandas as pd
from shutil import copy2
from numpy import linalg as LA


from utils import utils, helpers
from builders import model_builder




def get_classes(csv_file):
    df = pd.read_csv(csv_file)
    classes = []
    for f in df['name']:
        classes.append(f)
    return classes




def load_model(args, csv_file):
    print("Retrieving dataset information ...")
    class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
    class_names_string = ""
    for class_name in class_names_list:
        if not class_name == class_names_list[-1]:
            class_names_string = class_names_string + class_name + ", "
        else:
            class_names_string = class_names_string + class_name

    num_classes = len(label_values)

	# Initializing network
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess=tf.Session(config=config)


    net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
    net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])

    network, _ = model_builder.build_model(args.model, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width, crop_height=args.crop_height, is_training=False)

	#network=tf.nn.softmax(logits=network, axis = 3)
	#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output))
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output))
    sess.run(tf.global_variables_initializer())
	#checkpoint_path = "checkpoints/latest_model_" + args.model + "_" + args.dataset + ".ckpt"
    checkpoint_path = "checkpoints/latest_model_" + args.model + "_" + args.dataset + ".ckpt"
	# checkpoint_path = "DeepLab_V3_Github_pretrained_Models\deeplabv3_cityscapes_train\latest_model_" + args.model + "_" + args.dataset + ".ckpt"
    print('Loading model checkpoint weights ...')
    saver=tf.train.Saver(max_to_keep=1000)
    saver.restore(sess, checkpoint_path)

    print(network)



    #net_input = tf.placeholder(tf.float32,shape=[None,None,3])

    layer_A, layer_B = network[0], network[1]





    classes = get_classes(csv_file)

    columns = ['Class', 'Z_minus_Z_bar']

    data = []

    for c in classes:
        cnt = 0
        in_dir = c + "/"
        files = sorted(os.listdir(in_dir))
        Result_Z = np.zeros((128*128*256,1))
        for f in files:
            print("File: "+f)
            cnt += 1
            input_image = np.expand_dims(np.float32(utils.load_image(in_dir+f)), axis = 0)/255.0
            #input_image = input_image.reshape((1,input_image.shape[0],input_image.shape[1],input_image.shape[2]))
            #print(input_image.dtype)
            #print(input_image)
            layer_A_output, layer_B_output = sess.run([layer_A, layer_B], feed_dict={net_input: input_image})

            layer_B_output = layer_B_output.reshape((-1,1))
            Result_Z = np.add(Result_Z, layer_B_output)

        Result_Z = Result_Z / (cnt*1.0)
        print(Result_Z)

        norms = np.zeros((cnt,1))
        idx = 0
        for f in files:
            temp = []
            input_image = np.expand_dims(np.float32(utils.load_image(in_dir+f)), axis = 0)/255.0
            layer_A_output, layer_B_output = sess.run([layer_A, layer_B], feed_dict={net_input: input_image})

            layer_B_output = layer_B_output.reshape((-1,1))
            norms[idx,0] = np.square(LA.norm(layer_B_output - Result_Z))
            temp.append(c)
            temp.append(norms[idx,0])
            data.append(temp)
            idx += 1

        #print(norms)
        #print(np.argmin(norms, axis = 1)[0])
        #filename = files[np.argmin(norms, axis = 0)[0]]
        #out_dir = c + "_Center/"
        #os.mkdir(out_dir)
        #copy2(in_dir+filename, out_dir)
        #print("copying " + filename + " done")
    df = pd.DataFrame(data, columns = columns)
    df.to_csv("Data.csv")
    print("CSV done..")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
    parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
    parser.add_argument('--model', type=str, default="DeepLabV3_plus",help='The model you are using')
    parser.add_argument('--dataset', type=str, default="Carla", required=False, help='The dataset you are using')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = main()
    csv_file = "class_dict.csv"
    load_model(args, csv_file)
