# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from tool.utils import *
from tool.darknet2pytorch import Darknet
import argparse
import os
from os import listdir
from os.path import isfile, join

"""hyper parameters"""
use_cuda = False
num_classes = 80
if num_classes == 20:
    namesfile = 'data/voc.names'
elif num_classes == 80:
    namesfile = 'data/coco.names'
else:
    namesfile = 'data/x.names'


def detect(cfgfile, weightfile, imgfile):
    m = Darknet(cfgfile)

    # m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    img = Image.open(imgfile).convert('RGB')
    sized = img.resize((m.width, m.height))

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, num_classes, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    class_names = load_class_names(namesfile)
    file_txt_out = imgfile.split('/')[-1][:-4]
    
    os.makedirs("out_txt", exist_ok=True)
    save_boxes_txt(img, boxes, f'out_txt/{file_txt_out}.txt', class_names)
    os.makedirs("out_img", exist_ok=True)
    plot_boxes(img, boxes, f'out_img/{file_txt_out}.jpg', class_names)


def detect_imges(cfgfile, weightfile, imgfile_list=['data/dog.jpg', 'data/giraffe.jpg']):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    imges = []
    imges_list = []
    for imgfile in imgfile_list:
        img = Image.open(imgfile).convert('RGB')
        imges_list.append(img)
        sized = img.resize((m.width, m.height))
        imges.append(np.expand_dims(np.array(sized), axis=0))
    
    images = np.concatenate(imges, 0)
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, images, 0.5, num_classes, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))
    
    class_names = load_class_names(namesfile)
    os.makedirs("out_img", exist_ok=True)
    for i,(img,box) in enumerate(zip(imges_list,boxes)):
        file_txt_out = imgfile_list[i]
        print(i)
        print(file_txt_out)
        continue
        save_boxes_txt(img, box, '{}/{}.txt'.format('out_txt', file_txt_out), class_names)
        plot_boxes(img, box, '{}/predictions{}.jpg'.format("out_img",i), class_names)


def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, m.num_classes, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)


def detect_cv2_camera(cfgfile, weightfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("./test.mp4")
    cap.set(3, 1280)
    cap.set(4, 720)
    print("Starting the YOLO loop...")

    while True:
        ret, img = cap.read()
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        start = time.time()
        boxes = do_detect(m, sized, 0.5, num_classes, 0.4, use_cuda)
        finish = time.time()
        print('Predicted in %f seconds.' % (finish - start))

        class_names = load_class_names(namesfile)
        result_img = plot_boxes_cv2(img, boxes, savename=None, class_names=class_names)

        cv2.imshow('Yolo demo', result_img)
        cv2.waitKey(1)

    cap.release()


def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, m.num_classes, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./checkpoints/Yolov4_epoch1.pth',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-imgfile', type=str,
                        default='./data/mscoco2017/train2017/190109_180343_00154162.jpg',
                        help='path of your image file.', dest='imgfile')
    parser.add_argument('-source', type=str, 
                        default='/content/PennFudanPed/PNGImages',
                        help='path to folder image', dest='source')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    if (os.path.isdir(args.source)==True):
      # onlyfiles = [join(args.source, f) for f in listdir(args.source) if isfile(join(args.source, f))]
      for f in sorted(listdir(args.source)):
        if isfile(join(args.source, f)):
          detect(args.cfgfile, args.weightfile, join(args.source, f))
      # detect_imges(args.cfgfile, args.weightfile, onlyfiles)
    exit(0)
    if args.imgfile:
        detect(args.cfgfile, args.weightfile, args.imgfile)
        # detect_imges(args.cfgfile, args.weightfile)
        # detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
        # detect_skimage(args.cfgfile, args.weightfile, args.imgfile)
    else:
        detect_cv2_camera(args.cfgfile, args.weightfile)

