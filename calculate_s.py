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

    #print(network)

    #net_input = tf.placeholder(tf.float32,shape=[None,None,3])

    layer_A, layer_B = network[0], network[1]

    classes = get_classes(csv_file)

    ScA = np.zeros((len(classes), len(classes)))
    ScB = np.zeros((len(classes), len(classes)))

    A_shape = 128*128*256
    B_shape = 128*128*256

    A_scores = np.zeros((A_shape, len(classes)))
    B_scores = np.zeros((B_shape, len(classes)))

    idx = 0
    for c in classes:
        in_dir = c + "_Center/"
        files = sorted(os.listdir(in_dir))
        for f in files:
            print("File: " + f)
            input_image = np.expand_dims(np.float32(utils.load_image(in_dir+f)), axis = 0)/255.0
            #input_image = input_image.reshape((1,input_image.shape[0],input_image.shape[1],input_image.shape[2]))
            #print(input_image.dtype)
            #print(input_image)
            layer_A_output, layer_B_output = sess.run([layer_A, layer_B], feed_dict={net_input: input_image})

            layer_A_output = layer_A_output.reshape((-1,1))
            layer_B_output = layer_B_output.reshape((-1,1))

            A_scores[:, idx] = layer_A_output[:,0]
            B_scores[:, idx] = layer_B_output[:,0]

            break
        idx += 1
    #print(A_scores)
    #print(A_scores.shape)

    #calculate ScA
    for i in range(ScA.shape[0]):
        for j in range(ScA.shape[1]):
            vecA = A_scores[:, i].reshape((A_shape, 1))
            vecB = A_scores[:, j].reshape((A_shape, 1))
            ScA[i, j] = np.square(LA.norm(vecA - vecB))


    #calculate ScB
    for i in range(ScB.shape[0]):
        for j in range(ScB.shape[1]):
            vecA = B_scores[:, i].reshape((B_shape, 1))
            vecB = B_scores[:, j].reshape((B_shape, 1))
            ScB[i, j] = np.square(LA.norm(vecA - vecB))

    return A_scores, B_scores, ScA, ScB


def calculate_diff_S(ScA, ScB):
    return (ScA - ScB)


def get_scores(incoming_dir, A_scores, B_scores, ScA, ScB, csv_file, ScA_B, lmbda):
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

    A_shape = 128*128*256
    B_shape = 128*128*256

    classes = get_classes(csv_file)

    #print(network)

    #net_input = tf.placeholder(tf.float32,shape=[None,None,3])

    layer_A, layer_B, net = network[0], network[1], network[2]

    files = sorted(os.listdir(incoming_dir))

    print(net)

    scores = []

    for f in files:
        print("File: " + f)
        input_image = np.expand_dims(np.float32(utils.load_image(incoming_dir+f)), axis = 0)/255.0
        #input_image = input_image.reshape((1,input_image.shape[0],input_image.shape[1],input_image.shape[2]))
        #print(input_image.dtype)
        #print(input_image)
        layer_A_output, layer_B_output, net_output = sess.run([layer_A, layer_B, net], feed_dict={net_input: input_image})

        layer_A_output = layer_A_output.reshape((-1,1))
        layer_B_output = layer_B_output.reshape((-1,1))

        SxA = np.zeros((len(classes), 1))
        SxB = np.zeros((len(classes), 1))

        idx = 0
        for c in range(len(classes)):
            vecA = A_scores[:, idx].reshape(A_shape, 1)
            vecB = B_scores[:, idx].reshape(B_shape, 1)

            SxA[idx, :] = np.square(LA.norm(layer_A_output - vecA))
            SxB[idx, :] = np.square(LA.norm(layer_B_output - vecB))

            idx += 1

        net_output = np.array(net_output[0,:,:,:])
        net_output = helpers.reverse_one_hot(net_output)

        #print(net_output.shape)
        alphas = helpers.get_alpha(net_output)
        #print(alphas)
        SxA_B = calculate_diff_S(SxA, SxB)
        SxA_B_cap = np.zeros((len(classes), 1))
        #print(ScA_B)
        for i in range(len(classes)):
            vecB = ScA_B[:,i].reshape((len(classes), 1))
            #print(vecB)
            #print(alphas[:,i])
            #print(vecB)
            SxA_B_cap += alphas[:,i] * vecB
        #break
        #print(SxA_B_cap)
        tau, _ = stats.kendalltau(SxA_B, SxA_B_cap)
        #print(tau)
        distinctiveness = (1-tau)/2.0
        uncertainty = 0.0
        for i in range(len(alphas)):
            uncertainty += alphas[:,i] * (1-alphas[:,i])
        uncertainty = -1.0 * uncertainty

        score = (1-lmbda) * distinctiveness + lmbda * uncertainty
        print(score[0])
        scores.append(score[0])

    return scores







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
    #A_scores, B_scores, ScA, ScB = load_model(args, csv_file)
    '''
    with open("A_scores.pickle","wb") as pkl_out:
        pickle.dump(A_scores, pkl_out)

    with open("B_scores.pickle","wb") as pkl_out:
        pickle.dump(B_scores, pkl_out)

    with open("ScA.pickle","wb") as pkl_out:
        pickle.dump(ScA, pkl_out)

    with open("ScB.pickle","wb") as pkl_out:
        pickle.dump(ScB, pkl_out)

    print("Pickling done...")
    '''

    with open("A_scores.pickle","rb") as pkl_in:
        A_scores = pickle.load(pkl_in)

    with open("B_scores.pickle","rb") as pkl_in:
        B_scores = pickle.load(pkl_in)

    with open("ScA.pickle","rb") as pkl_in:
        ScA = pickle.load(pkl_in)

    with open("ScB.pickle","rb") as pkl_in:
        ScB = pickle.load(pkl_in)

    #print(ScA)
    #print(ScB)
    ScA_B = calculate_diff_S(ScA, ScB)
    incoming_dir = "incoming_set/"
    scores = get_scores(incoming_dir, A_scores, B_scores, ScA, ScB, csv_file, ScA_B, 0.7)
    print(scores)
