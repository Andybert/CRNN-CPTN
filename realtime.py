import sys
import argparse
import recognition
import detection
import cv2
import numpy as np
from os import system

parser = argparse.ArgumentParser()
parser.add_argument("echo")
args = parser.parse_args()

img = cv2.imread(args.echo)
net, sess = detection.detectorload()
model, converter = recognition.crnnloader()
cropedimage = detection.ctpn(sess, net, img, 1)
print("\n" * 50)

text = []
print(len(cropedimage))
for image in cropedimage:
    buf = cv2.resize(image.copy(),
                     (int(image.shape[1] * 2), int(image.shape[0] * 2)))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    text_sim, cost = recognition.recognize(model, converter, image)
    print('Text: ' + text_sim)

    print('                       Recognition Time: ' + str(np.round(cost, 4)))
    print("\n")
    text.append(text_sim)
    cv2.imshow('My window', buf)
    cv2.waitKey(50)
    system('say ' + text_sim)
    cv2.waitKey(2500)
