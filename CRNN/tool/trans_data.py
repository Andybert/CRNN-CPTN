import os
import lmdb  # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import glob

# real_path='/home/ahmed/Downloads/sample/train/'
real_path = '/home/ahmed/Downloads/mnt/fine_tune/'
path = '/home/ahmed/Downloads/mnt/fine_tune/'
path_train = 'train/'
path_test = 'valid/'
path_output_train = '/home/ahmed/Downloads/mnt/fine_tune/train/'
path_output_test='/home/ahmed/Downloads/mnt/fine_tune/valid/'
path_train_2='/home/ahmed/Downloads/mnt/fine_tune/train/'
path_test_2='/home/ahmed/Downloads/mnt/fine_tune/valid/'

# Train
os.chdir(path + path_train)
images_train = glob.glob("*.jpg")
left_train, labels_train, right_train = list(zip(*[os.path.splitext(x)[0].split('_')
                                                   for x in images_train]))



#Test
os.chdir(path+path_test)
images_test = glob.glob("*.jpg")
left_test,labels_test,right_test = list(zip(*[os.path.splitext(x)[0].split('_')
                                     for x in images_test]))



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


def createDataset(outputPath, path,images_train, labels_train, lexiconList=None, checkValid=True):
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
        if not os.path.exists(path + imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(path + imagePath, 'rb') as f:
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
    createDataset(path_output_train, path_train_2,images_train, labels_train)
    createDataset(path_output_test, path_test_2,images_test, labels_test)