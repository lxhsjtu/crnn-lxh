import numpy as np
import cv2
from scipy.misc import imread, imresize
def resize_image(image):
    """
        Resize an image to the "good" input size
        固定高度为32像素，宽度为160,图像为灰度模式，所有图像的像素值都归一化到0-1之间.
    """
    # data=np.zeros((32,320),dtype=np.float32)
    img=cv2.imdecode(np.fromfile(image, dtype=np.uint8), 1)
    # img=cv2.resize(img,(int(w*float(32/36)),32))
    # if img.shape[1]>900:
    img = cv2.resize(img, (160, 32))
    img=np.asarray(img,dtype=np.float32)
    img=img/255.0
    # data[:,0:img.shape[1]]=img
    # im = cv2.imread(image, 0).astype(np.float32) / 255.0
    # im = cv2.resize(im, (160, 32))
    # im=np.asarray(im,dtype=np.float32)
    return img, 37
def sparse_tuple_from(sequences, dtype=np.int32):
    """
        Inspired (copied) from https://github.com/igormq/ctc_tensorflow_example/blob/master/utils.py
    """
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), [i for i in range(len(seq))]))
        values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)
    return indices, values, shape

def to_seq_len(inputs, max_len):
    return np.ones(np.shape(inputs)[0]) * max_len

def labels_to_string(labels, word_string):
    result = ""
    for i in labels:
        result += word_string[i] if i != 0 else '-'
    return result

def label_to_array(label, dict):
    index=[]
    number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for x in label.strip():
        if dict.get(x)!=None:
            index.append(int(dict[x]))
            continue
        if x in number:
            index.append(int(dict[int(x)]))
    return index