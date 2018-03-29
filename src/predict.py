import tensorflow as tf
import numpy as np
import argparse
from PIL import Image
import config
import math
import json
from .model import CRNN
from .utils import sparse_tuple_from, to_seq_len, resize_image, label_to_array,labels_to_string
import cv2
import xlrd
import time
import os
from scipy.misc import imread, imresize,imshow
import matplotlib.pyplot as plt

def label2string(decode,data_dict):
    str=""
    for key,value in data_dict.items():
        if value==decode:
            str=key
            break
    return str

def loaddict():
    xlsxpath = '..//index//words_index.xlsx'
    file = xlrd.open_workbook(xlsxpath)
    try:
        sh = file.sheet_by_name(u"Sheet1")
    except:
        print("no sheet")
    dict = {}
    for i in range(0, 6847):
        char = sh.cell_value(i, 0)
        index = int(sh.cell(i, 1).value)
        dict[char] = index - 1
    return dict



def load_img(path):
    image_path = path
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
    h,w,c=img.shape
    im_scale=float(32)/float(h)
    img = cv2.resize(img, (int(w*im_scale), 32))
    tmp=img
    img = np.asarray(img, dtype=np.float32)
    img = img / 255.0
    train = img
    train = np.reshape(train, (1, 32,int(w*im_scale) , 3))
    width=int(w*im_scale)
    seq_len=math.ceil((math.ceil((width-1)/2)-2)/2)-2
    return train,tmp,seq_len

def load_graph():
    num_class=7000
    with tf.Graph().as_default():
        inputs = tf.placeholder(tf.float32, [1, 32, None, 3])
        # The CRNN
        crnn = CRNN(inputs)
        # The length of the sequence
        seq_len = tf.placeholder(tf.int32, [None], name='seq_len')
        logits = tf.reshape(crnn, [-1, 512])  # (batchsizex(width/32))*512
        W = tf.Variable(tf.truncated_normal([512, num_class], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0., shape=[num_class]), name="b")
        logits = tf.matmul(logits, W) + b
        logits = tf.reshape(logits, [1, -1, num_class])
        logits = tf.transpose(logits, (1, 0, 2))
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
        saver = tf.train.import_meta_graph('..//model_1226/model.ckpt.meta')
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state("../model_1226/")
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("load success")
            else:
                print("no such file")
                return
            graph = tf.get_default_graph()
            print(graph)

def main():
    """
        Runs the model on an picture of a line of text
    """
    print('===========load dict===========')
    # with open("..//index//data_2.json", 'r') as f:
    #     data_dict = json.load(f)
    data_dict=loaddict()
    with tf.Graph().as_default():
        inputs = tf.placeholder(tf.float32, [1, 32, None, 3])
        # The CRNN
        crnn = CRNN(inputs)
       # The length of the sequence
        seq_len = tf.placeholder(tf.int32, [None], name='seq_len')
        logits = tf.reshape(crnn, [-1, 512])  # (batchsizex(width/32))*512
        W = tf.Variable(tf.truncated_normal([512, config.NUM_CLASSES], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0., shape=[config.NUM_CLASSES]), name="b")
        logits = tf.matmul(logits, W) + b
        logits = tf.reshape(logits, [1, -1, config.NUM_CLASSES])
        # Final layer, the output of the BLSTM
        logits = tf.transpose(logits, (1, 0, 2))
        # The decoded answer
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len,merge_repeated=False)
        # The error rate
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state("../model_1215/")
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("load success")
            else:
                print("no such file")
                return
            count=0
            for root ,dirnames,imgnames in os.walk("..//hxhbwords//"):
                for dirname in dirnames:
                    for _,_,filenames in os.walk(os.path.join(root,dirname)):
                        for file in filenames:
                            imgname=os.path.join(root,dirname,file)
                            train,image,seq_=load_img(imgname)
                            print("real label",imgname)
                            start_time = time.time()
                            decoded_val,log= sess.run(
                               [decoded,logits],
                               {
                                   inputs: train,
                                   seq_len: [seq_],
                               }
                            )
                            result=""
                            # index_res=[]
                            # for sublog in log:
                            #     index_res.append(np.argsort(sublog[0])[::-1][0])
                            # print(index_res)
                            # print(decoded_val[0][1])
                            for number in decoded_val[0][1]:
                                tmp=label2string(number,data_dict)
                                if isinstance(tmp,float):
                                    tmp=str(int(tmp))
                                result+=tmp
                            print("predict label",result)
                            tmpname=os.path.splitext(imgname)
                            os.rename(imgname,tmpname[0]+"_AI_"+result+".jpg")
                            print(count)
                            count+=1
                            # cv2.imshow("1",image)
                            # cv2.waitKey()
if __name__ == '__main__':
    # main()
    load_graph()
