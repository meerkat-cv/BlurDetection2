#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import json
import argparse
import logging

import cv2
import numpy

from blur_detection.detection import estimate_blur
from blur_detection.detection import fix_image_size
from blur_detection.detection import pretty_blur_map
from matplotlib import pyplot as mp
import matplotlib

def find_images(input_dir):
    extensions = [".jpg", ".png", ".jpeg"]

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                yield os.path.join(root, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run blur detection on a single image')
    parser.add_argument('-i', '--input_dir', dest="input_dir", type=str, required=True, help="directory of images")
    parser.add_argument('-s', '--save_path', dest='save_path', type=str, required=True, help="path to save output")
    # parameters
    parser.add_argument("-t", "--threshold", dest='threshold', type=float, default=100.0, help="blurry threshold")
    parser.add_argument("-f", "--fix_size", dest="fix_size", help="fix the image size", action="store_true")
    # options
    parser.add_argument("-v", "--verbose", dest='verbose', help='set logging level to debug', action="store_true")
    parser.add_argument("-d", "--display", dest='display', help='display images', action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    results = []
    image_score = []

    input_dirs = ['/home/gfickel/meerkat/clients/axon/docs/Non-blurry_Customer_photos', \
                  '/home/gfickel/meerkat/clients/axon/docs/Blurred_Customer_photo/']

    # input_dirs = ['/home/gfickel/meerkat/clients/axon/docs/Non-blurry ID document images/', \
    #               '/home/gfickel/meerkat/clients/axon/docs/Blurred ID document images/']

    mp.xlim(0,1)
    mp.ylim(0,600)


    for curr_dir in input_dirs:
        scores = []
        img_score = []
        for input_path in find_images(curr_dir):
            try:
                input_image = cv2.imread(input_path)

                if args.fix_size:
                    input_image = fix_image_size(input_image)

                blur_map, score, blurry = estimate_blur(input_image)
                scores.append(score)
                img_score.append((input_path, score))
            except Exception as e:
                print(e)
                pass

        results.append(scores)
        image_score.append(img_score)
        print('Final: ', str(numpy.mean(scores)))


    mp.xlim(0,1)
    mp.ylim(0,600)

    for dist in results[0]:
        x, y = numpy.random.rand(), dist
        pts1 = mp.scatter(x, y, color=[0.5,0.1,0.1], s=60)

    for dist in results[1]:
        x, y = numpy.random.rand(), dist
        pts2 = mp.scatter(x, y, color=[0.1,0.1,0.5], s=100)


    num_pts = len(results[0])+len(results[1])
    acc = []
    for thresh in range(200):
        correct = 0
        for dist in results[0]:
            if dist > thresh: correct += 1
        for dist in results[1]:
            if dist < thresh: correct += 1
        acc.append(correct/num_pts)

    best_idx = numpy.argmax(acc)
    print('acc: ', acc[best_idx])

    for im_score in image_score[0]:
        if im_score[1] < best_idx: os.system('cp {} bad_non_blur/.'.format(im_score[0]))
    for im_score in image_score[1]:
        if im_score[1] > best_idx: os.system('cp {} bad_blur/.'.format(im_score[0]))

    mp.legend([pts2, pts1], ['Blur', 'Non Blur'], loc='upper left')
    mp.plot([0,1], [best_idx,best_idx], color=[0.1,0.5,0.1])
    mp.show()