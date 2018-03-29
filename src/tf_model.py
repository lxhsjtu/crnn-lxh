import tensorflow as tf
import numpy as np
import argparse
from PIL import Image
import math
import json
from model import CRNN
import cv2
import xlrd
import time
import os

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
    for i in range(0, 6852):
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


def load_model(model_path):
    with tf.Graph().as_default():
        inputs = tf.placeholder(tf.float32, [1, 32, None, 3])
        crnn = CRNN(inputs)
        seq_len = tf.placeholder(tf.int32, [None], name='seq_len')
        logits = tf.reshape(crnn, [-1, 512])  # (batchsizex(width/32))*512
        W = tf.Variable(tf.truncated_normal([512, 7000], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0., shape=[7000]), name="b")
        logits = tf.matmul(logits, W) + b
        logits = tf.reshape(logits, [1, -1, 7000],name="reshape_log")
        logits = tf.transpose(logits, (1, 0, 2),name="final_log")
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, top_paths=1, merge_repeated=False)
        saver = tf.train.Saver(tf.global_variables())
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("load success")
        else:
            print("no such file")
            return
        return sess,decoded,inputs,seq_len

def predict(sess,decoded,inputs,seq_len,img_name):
    data_dict=loaddict()
    train, image, seq_ = load_img(imgname)
    print("real label", imgname)
    try:
        decoded_val = sess.run([decoded],{inputs: train,seq_len: [seq_],})
    except:
        raise "error"
        return
    result = ""
    for number in decoded_val[0][0][1]:
        tmp = label2string(number, data_dict)
        if isinstance(tmp, float):
            tmp = str(int(tmp))
        result += tmp
    print("predict label", result)

def close_sess(sess):
    sess.close()

if __name__ == '__main__':
    sess,decoded,inputs,seq_len=load_model("../model_1226/")
    imgname="..//test_data//1435.png"
    predict(sess,decoded,inputs,seq_len,imgname)
    close_sess(sess)