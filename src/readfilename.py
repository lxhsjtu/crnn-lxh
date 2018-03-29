import os
import sys
import time
import numpy as np
import codecs
import tensorflow as tf
import xlrd
import cv2

def cal_len(str,dict):
    index = []
    number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for x in str.strip():
        if dict.get(x) != None:
            index.append(int(dict[x]))
            continue
        if x in number:
            index.append(int(dict[int(x)]))
    return len(index)

def load_data(folders,dict):
    """
        Load all the images in the folder
    """
    examples = []
    count=0
    pathstring=[]
    for folder in folders:
        for f in os.listdir(folder):
            count+=1
            print(count)
            path=folder+f
            try:
                char_len=cal_len(f[9:-4],dict)
                img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
                h,w,c=img.shape
                if h<w and w>=30 and char_len<37 and char_len>2:
                    pathstring.append(path.strip())
            except:
                continue
    return pathstring

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
    data_dir = ["..//..//切割词条//"]
    dict = loaddict()
    imagename=load_data(data_dir,dict)
    perm = np.arange(len(imagename))
    np.random.shuffle(perm)
    imagename = np.asarray(imagename)
    train_data = imagename[perm]
    print(len(train_data))
    print(train_data[0:20])
    with codecs.open("image_path.txt",'w',encoding='utf-8') as f:
        for name in train_data:
            f.write(name)
            f.write('\n')
        f.close()
if __name__=='__main__':
    main(sys.argv)
    # imagefiles=[]
    # with codecs.open("imagename.txt",'r',encoding='utf-8') as file:
    #     line = file.readline()
    #     while line:
    #         imagefiles.append(line.strip())
    #         line = file.readline()
    # img = cv2.imdecode(np.fromfile('..//..//切割词条//51069834_只做全新原装正品.png', dtype=np.uint8), 1)
    # print(img.shape)
