import os
import sys
import time
import numpy as np
import tensorflow as tf
import xlrd
import config
import cv2
import gc
import codecs
from model import CRNN
from utils import sparse_tuple_from, to_seq_len, resize_image, label_to_array
import json
def create_ground_truth(label):
    """
        Create our ground truth by replacing each char by its index in the CHAR_VECTOR
    """

    return [config.CHAR_VECTOR.index(l) for l in label.split('_')[0]]

def ground_truth_to_word(ground_truth):
    """
        Return the word string based on the input ground_truth
    """

    return ''.join([config.CHAR_VECTOR[i] for i in ground_truth])



def load_data(folders):
    """
        Load all the images in the folder
    """
    examples = []
    count=0
    for folder in folders:
        for f in os.listdir(folder):
            count+=1
            print(count)
            arr, initial_len = resize_image(
                os.path.join(folder, f)
            )
            tmp = f[9:-4]
            tmp=tmp.replace("_","*")
            examples.append(
                (
                    arr,
                    tmp,
                    initial_len
                )
            )
    return examples

def load_data_big(folders):
    """
        Load all the images in the folder
    """
    examples = []
    count=0
    for folder in folders:
        filapath,filename=os.path.split(folder)
        count+=1
        print(count)
        arr, initial_len = resize_image(
            folder
        )
        tmp = filename[9:-4]
        tmp=tmp.replace("_","*")
        tmp = tmp.replace("~", "/")
        tmp = tmp.replace('?', '？')
        tmp = tmp.replace(',', '，')
        tmp = tmp.replace('×', '*')
        tmp = tmp.replace('！', '!')
        tmp = tmp.replace('＞', '>')
        tmp = tmp.replace('＜', '<')
        examples.append(
            (
                arr,
                tmp,
                initial_len
            )
        )
    return examples

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

def main(args):
    print('===========load dict===========')
    data_dict=loaddict()
    iteration_count = 1000
    batch_size = 64
    batch_image=400000
    log_save_dir = "..//model//"
    restore=True

    # The training data
    print('==============load data=============')
    imagefiles=[]
    with codecs.open("image_path.txt",'r',encoding='utf-8') as file:
        line = file.readline()
        while line:
            imagefiles.append(line.strip())
            line = file.readline()
        file.close()
    # data= load_data(data_dir)
    # print('data size:', len(data))

    # perm=np.arange(len(data))
    # np.random.shuffle(perm)
    # data=np.asarray(data)
    # train_data=data[perm]
    # test_data = data[int(len(data) * 0.10):]
    graph = tf.Graph()
    with graph.as_default():
        inputs = tf.placeholder(tf.float32, [batch_size, 32, None, 3],name='inputs')
        # The CRNN
        crnn = CRNN(inputs)
        # Our target output
        targets = tf.sparse_placeholder(tf.int32, name='targets')
        # The length of the sequence
        seq_len = tf.placeholder(tf.int32, [None], name='seq_len')
        logits = tf.reshape(crnn, [-1, 512])   #(batchsize x 37) x 512
        W = tf.Variable(tf.truncated_normal([512, config.NUM_CLASSES], stddev=0.1,dtype=tf.float32), name="W")
        b = tf.Variable(tf.constant(0., shape=[config.NUM_CLASSES],dtype = tf.float32), name="b")
        print(logits.get_shape())
        logits = tf.matmul(logits, W) + b
        print(logits.get_shape())
        logits = tf.reshape(logits, [batch_size, -1, config.NUM_CLASSES])        # batch_size x 36 x NUM_CLASSES
        print(logits.get_shape())
        # Final layer, the output of the BLSTM
        logits = tf.transpose(logits, (1, 0, 2))                                  #36 x batch_size x NUM_CLASSES
        global_step = tf.Variable(0, trainable=False)
        # Loss and cost calculation
        loss = tf.nn.ctc_loss(targets, logits, seq_len)
        cost = tf.reduce_mean(loss)
        # learning_rate = tf.train.exponential_decay(0.1,
        #                                            global_step,
        #                                            5000,
        #                                            0.1, staircase=True)
        # Training step
        # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(cost)
        # optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost,global_step=global_step)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001).minimize(loss=cost, global_step=global_step)
        # The decoded answer
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len)
        # The error rate
        seq_dis = tf.reduce_mean(tf.
                                 edit_distance(tf.cast(decoded[0], tf.int32), targets))
    config_gpu = tf.ConfigProto()
    config_gpu.gpu_options.allow_growth=True
    with tf.Session(graph=graph,config=config_gpu) as sess:
        # tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        # Train
        if restore:
            print('=============load model============')
            ckpt = tf.train.get_checkpoint_state("../model/")
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("load success")
            else:
                print("no such file")
                return
        print('============begin training=============')
        for it in range(0, iteration_count):
            i=0
            for iter in range(1+(len(imagefiles)//batch_image)):
                imagepath=imagefiles[iter*batch_image:(iter+1)*batch_image]
                train_data=load_data_big(imagepath)
                for b in [train_data[x*batch_size:x*batch_size + batch_size] for x in range(0, int(len(train_data) / batch_size))]:
                    start_time = time.time()
                    in_data, labels, data_seq_len = zip(*b)
                    # print(data_seq_len)
                    data_targets = np.asarray([label_to_array(lbl, data_dict) for lbl in labels])
                    data_targets = sparse_tuple_from(data_targets)
                    # print(np.shape(data_targets[0]))
                    # print(np.shape(data_targets[1]))
                    # print(np.shape(data_targets[2]))
                    # print(data_targets[0])
                    # print(data_targets[1])
                    # print(data_targets[2])
                    data_shape = np.shape(in_data)
                    in_data = np.reshape(in_data, (data_shape[0], data_shape[1], data_shape[2], 3))
                    costacc, _ = sess.run(
                        [cost,optimizer],
                        {
                            inputs: in_data,
                            targets:data_targets,
                            seq_len: data_seq_len,
                        }
                    )
                    i+=1
                    print('epoch:{}/1000,cost={},iter={},time={}'.format(it,costacc,i,time.time()-start_time))
                del train_data
                gc.collect()
                print("complete 40W images")
                if (it % 1 == 0):
                    checkpoint_path = os.path.join(log_save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path)
                # iter_avg_cost += (np.sum(cost_val) / batch_size) / (int(len(train_data) / batch_size))
            print("complete one epoch")

if __name__=='__main__':
    main(sys.argv)


