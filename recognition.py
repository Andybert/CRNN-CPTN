import torch
from torch.autograd import Variable
import sys
sys.path.append("CRNN")
import utils
import dataset
from PIL import Image
import time

import models.crnn as crnn


def crnnloader():
    #model_path = '/home/ahmed/crnn/data/crnn.pth'
    model_path = './CRNN/data/crnn.pth'
    #img_path = '/home/ahmed/crnn/data/demo.png'
    #img_path='/home/ahmed/Pictures/cogedis/2-total.png'
    img_path = './CRNN/data/demo.png'
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

    model = crnn.CRNN(32, 1, 37, 256, 1)
    print('loading pretrained model from %s' % model_path)
    model.load_state_dict(torch.load(model_path))

    converter = utils.strLabelConverter(alphabet)
    return model, converter


def recognize(model, converter, image):
    transformer = dataset.resizeNormalize((100, 32))
    image = Image.fromarray(image)
    image = transformer(image)
    image = image.view(1, *image.size())
    image = Variable(image)
    t = time.time()
    model.eval()
    preds = model(image)
    cost = time.time() - t
    _, preds = preds.max(2, keepdim=True)
    preds = preds.squeeze(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    return sim_pred, cost
