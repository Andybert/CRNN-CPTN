import os, sys
from os import path
import lmdb  # install lmdb by "pip install lmdb"
import cv2
import numpy as np
#import PIL
#from PIL import Image


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)


def createDataset(outputPath, imagePathList, labelList, lexiconList, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert (len(imagePathList) == len(labelList))
    nSamples = len(labelList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(0, nSamples):
        imagePath = imagePathList[i]
        label = lexiconList[int(labelList[i])]
        print(imagePath, label)
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        '''    
        img = Image.open(imagePath, 'rb')
        img = img.resize((100, 32), Image.ANTIALIAS)
        img.save(imagePath)
        '''
        #     print(img.size)
        #       img=Image.open(imagePath,'r')
        #        print(img.size)
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
            # print(imageBin.size)
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label

        #if lexiconList:
        #    lexiconKey = 'lexicon-%09d' % cnt
        #    cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def mine(file_path_data,output):
    lex = []
    ls = []
    img = []
    file_path = path.relpath(file_path_data)
    with open(file_path, 'r') as f:
        for line in f:
            splitLine = line.split(' ')
            ls.append(splitLine[1].strip())
            img.append('/home/ahmed/Downloads/mnt/syn/max/90kDICT32px'+splitLine[0][1:])


            # print(ls)
            #    print(img)
            # if count==100000:
            #  break
    file_path_lexicon = path.relpath("/home/ahmed/Downloads/mnt/lexicon.txt")
    with open(file_path_lexicon, 'r') as f:
        for line in f:
            lex.append(line.strip())
            #   print(lex[int(ls[10])])
    createDataset(output, img, ls, lex, checkValid=True)


if __name__ == '__main__':
    #mine('/home/ahmed/Downloads/mnt/annotation_train.txt','/home/ahmed/Downloads/mnt/data/train/')
    #mine('/home/ahmed/Downloads/mnt/annotation_val.txt',  '/home/ahmed/Downloads/mnt/data/valid/')
     #mine('/home/ahmed/Downloads/mnt/annotation_test.txt',  '/home/ahmed/Downloads/mnt/data/test/')
'''
import os
import lmdb  # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import glob

p_train="/home/ahmed/Downloads/training_data/train_scaled/"
p_valid="/home/ahmed/Downloads/training_data/valid_scaled/"
path = '/home/ahmed/Downloads/training_data/'
path_train = 'train_scaled/'
path_valid = 'valid_scaled/'
path_output_train = '/home/ahmed/Downloads/training_data/output/train/'
path_output_valid = '/home/ahmed/Downloads/training_data/output/valid/'
# Train
os.chdir(path + path_train)
images_train = glob.glob("*.jpg")
left_train, labels_train, right_train = list(zip(*[os.path.splitext(x)[0].split('_')
                                                   for x in images_train]))



# Valid
os.chdir(path+path_valid)
images_valid = glob.glob("*.jpg")
left_test,labels_valid,right_valid = list(zip(*[os.path.splitext(x)[0].split('_')
                                     for x in images_valid]))



def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k,v)



def createDataset(outputPath, images_train, labels_train,input_path, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert (len(images_train) == len(labels_train))
    nSamples = len(images_train)

    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = images_train[i]
        label = labels_train[i]
        if not os.path.exists(input_path + imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(input_path+ imagePath, 'rb') as f:
            imageBin = f.read()

        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    createDataset(path_output_train, images_train, labels_train,p_train)
    createDataset(path_output_valid, images_valid, labels_valid, p_valid)
'''
