import tensorflow as tf
import numpy as np
import argparse
from PIL import Image
import config
import math
import json
from model import CRNN
from utils import sparse_tuple_from, to_seq_len, resize_image, label_to_array,labels_to_string
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
    for i in range(0, 6866):
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
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, beam_width=1,top_paths=1,merge_repeated=False)
        # The error rate
        saver = tf.train.Saver(tf.global_variables())
        # config_gpu = tf.ConfigProto()
        # config_gpu.gpu_options.allow_growth = True
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            ckpt = tf.train.get_checkpoint_state("../model_0328/")
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("load success")
            else:
                print("no such file")

                return
            for root ,_,imgnames in os.walk("C://Users//jacob//Desktop//切割词条//"):
                start_time = time.time()
                for imgname in imgnames:
                    train,image,seq_=load_img(root+imgname)
                    print("real label",imgname)
                    try:
                        decoded_val= sess.run(
                           [decoded],
                           {
                               inputs: train,
                               seq_len: [seq_],
                           }
                        )
                    except:
                        raise "error"
                        continue
                    result=""
                    for number in decoded_val[0][0][1]:
                        tmp=label2string(number,data_dict)
                        if isinstance(tmp,float):
                            tmp=str(int(tmp))
                        result+=tmp
                    print("predict label",result)
                    cv2.imshow("1",image)
                    cv2.waitKey()
                print(time.time() - start_time)
            coord.request_stop()  # queue需要关闭，否则报错
            coord.join(threads)
if __name__ == '__main__':
    main()




