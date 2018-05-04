from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys, cv2
import glob
import shutil
sys.path.append("TextDetection")
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(
        im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def crop_image(img, boxes, scale):
    crop = []
    for box in boxes:
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(
                box[3] - box[0]) < 5:
            continue
        if box[8] >= 0.9:
            color = (0, 255, 0)
        elif box[8] >= 0.8:
            color = (255, 0, 0)

        min_x = min(int(box[0]), int(box[2]), int(box[4]), int(box[6]))
        min_y = min(int(box[1]), int(box[3]), int(box[5]), int(box[7]))
        max_x = max(int(box[0]), int(box[2]), int(box[4]), int(box[6]))
        max_y = max(int(box[1]), int(box[3]), int(box[5]), int(box[7]))

        crop_img = img[min_y:max_y, min_x:max_x]
        crop.append(crop_img)
        # print(crop_img.shape)
        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(1000)
    return crop


def draw_boxes(img, boxes, scale):
    lineType = 2
    for box in boxes:
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(
                box[3] - box[0]) < 5:
            continue
        if box[8] >= 0.9:
            color = (0, 255, 0)
        elif box[8] >= 0.8:
            color = (255, 0, 0)
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                 color, 2)
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])),
                 color, 2)
        cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])),
                 color, 2)
        cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])),
                 color, 2)
    cv2.imshow('My window', img)
    cv2.moveWindow("My window", 50, 200)
    cv2.waitKey(7500)


def ctpn(sess, net, frame, draw):
    # timer = Timer()
    # timer.tic()

    img, scale = resize_im(
        frame, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    scores, boxes = test_ctpn(sess, net, img)

    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    buf = img.copy()
    crop = crop_image(buf, boxes, scale)

    # timer.toc()
    if draw is 1:
        draw_boxes(img, boxes, scale)
    return crop


def detectorload():
    cfg_from_file('TextDetection/ctpn/text.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()
    try:
        ckpt = tf.train.get_checkpoint_state('TextDetection/checkpoints/')
        print(
            'Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = test_ctpn(sess, net, im)
    return net, sess

    net, sess = detectorload()
    ctpn(sess, net, im_name)
