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

# csv_input = pd.read_csv('/home/ahmed/Pictures/bytel/bytel_words/words.csv')

#test_path='/home/ahmed/Downloads/crnn.pytorch-master/data/IIITS5K/'

#model_path = '/home/ahmed/crnn/data/crnn.pth'
model_path = './data/crnn.pth'
#img_path = '/home/ahmed/crnn/data/demo.png'
#img_path='/home/ahmed/Pictures/cogedis/2-total.png'
img_path = './data/demo.png'

#digits

#img_path = '/home/ahmed/Downloads/crnn.pytorch-master/data/digits3.png'

alphabet = '0123456789abcdefghijklmnopqrstuvwxyz,.'
#alphabet = '0123456789'

model = crnn.CRNN(32, 1, 37, 256, 1)

print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))

os.chdir(test_path)
images_names = glob.glob("*.png")
i = 0
#t = datetime.now()
p = 0
n = 0
crnn_predicted = []
for img in images_names:
    image = Image.open(img).convert('L')

    # start = time.time()
    a = datetime.now()

    image = transformer(image)
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
    crnn_predicted.append(sim_pred)
    # end = time.time()
    b = datetime.now()
    c = b - a
    print('%-20s => %-20s' % (raw_pred, sim_pred), img)
    # print(end - start)
    print(c.total_seconds(), " seconds")
    print(c.total_seconds() * 1000, " miliseconds")
    '''
    if (sim_pred+'.png') == img:
        p +=1
    '''
    #print(c)
    i += 1
    n += c.total_seconds()
# csv_input['crnn_predict'] = crnn_predicted
# csv_input.to_csv('/home/ahmed/Pictures/bytel/bytel_words/words_new.csv', index=False)
#k = datetime.now()
#z=k-t

#print('number of images',i)
#print(z.total_seconds(), " estimated total time in second")
#print(z.total_seconds()*1000, " estimated total time in millisecond")
#print(p, " sur ", i , " images bien reconnu")
print("total images ", i)
print(n, "estimated time")
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
