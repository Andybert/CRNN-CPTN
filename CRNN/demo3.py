
from __future__ import division
import cv2
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn
#import timey
from datetime import datetime
import glob
import os
import pandas as pd



df = pd.read_csv('/home/ahmed/Pictures/cogedis/cogedis_words_3/test_digit.csv')
#df['crnn']=None
df = df.astype(str)
#test_path='/home/ahmed/Downloads/crnn.pytorch-master/data/IIITS5K/'

test_path='/home/ahmed/Pictures/cogedis/cogedis_words_3/'
#model_path = '/home/ahmed/Downloads/crnn.pytorch-master/data/crnn.pth'
model_path='/home/ahmed/Pictures/model/s_digit_77.81/netCRNN_64_13.pth'
#img_path = '/home/ahmed/Downloads/crnn.pytorch-master/data/demo.png'
#img_path = '/home/ahmed/Downloads/crnn.pytorch-master/data/logo.png'
#img_path = '/home/ahmed/Downloads/crnn.pytorch-master/data/ShareImg.png'
#img_path = '/home/ahmed/Downloads/crnn.pytorch-master/data/char.png'
#img_path = '/home/ahmed/Downloads/crnn.pytorch-master/data/awe.jpg'

#digits

#img_path = '/home/ahmed/Downloads/crnn.pytorch-master/data/digits3.png'

#alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
alphabet="0123456789"
#alphabet = '0123456789'

model = crnn.CRNN(32, 1, 11,256, 1).cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))

os.chdir(test_path)
images_names=glob.glob("*.png")
i=0
#t = datetime.now()
p=0
n=0
#crnn_predicted=[]
df['crnn_pred']=None
H=0
set_img = set([x.rsplit('.', 1)[0] for x in images_names])
for img in images_names:
    if img.rsplit('.',1)[0] in df.id.values:
        image = Image.open(img).convert('L')
        id, ext = img.rsplit('.', 1)
        # start = time.time()
        a = datetime.now()

        image = transformer(image).cuda()
        image = image.view(1, *image.size())
        image = Variable(image)

        model.eval()
        preds = model(image)

        _, preds = preds.max(2)
        preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        idx = df[df['id'] == id].index.values
        df.loc[df.index[idx], 'crnn_pred'] = sim_pred
        # df.iloc[idx, df.columns.get_loc(4)] = sim_pred
        # crnn_predicted.append(sim_pred)
        # end = time.time()
        b = datetime.now()
        c = b - a
        print('%-20s => %-20s' % (raw_pred, sim_pred), img)
        if sim_pred == df.loc[df.id == id, 'manual_raw_value'].item():
            H += 1

        # print(end - start)
        print(c.total_seconds(), " seconds")
        print(c.total_seconds() * 1000, " miliseconds")
        '''
        if (sim_pred+'.png') == img:
            p +=1
        '''
        # print(c)
        i += 1
        n += c.total_seconds()


#print(df.columns.tolist())
#df = df.astype(str)
#print(df.head())
#df.to_csv('/home/ahmed/Pictures/cogedis/cogedis_words_large/words_new_2.csv',index=False,encoding='utf-8')

#csv_input['crnn_predict'] = crnn_predicted
#csv_input.to_csv('/home/ahmed/Pictures/bytel/bytel_words/words_new.csv', index=False)
#k = datetime.now()
#z=k-t

#print('number of images',i)
#print(z.total_seconds(), " estimated total time in second")
#print(z.total_seconds()*1000, " estimated total time in millisecond")
#print(p, " sur ", i , " images bien reconnu")
df.to_csv('/home/ahmed/Pictures/cogedis/cogedis_words_3/test_predicted_digit_only.csv',sep=',')
print("total images ", i)
print("correct pre ",H)
print ("accuracy", (H/i)*100)
print(n,"estimated time for prediciton")

'''
image = Image.open(img_path).convert('L')


#start = time.time()
a = datetime.now()

image = transformer(image).cuda()
image = image.view(1, *image.size())
image = Variable(image)

model.eval()
preds = model(image)

_, preds = preds.max(2)
preds = preds.squeeze(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
#end = time.time()
b = datetime.now()
c = b - a
print('%-20s => %-20s' % (raw_pred, sim_pred))
#print(end - start)
print(c.total_seconds())
print(c)
'''

