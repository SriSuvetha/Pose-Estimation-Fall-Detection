#!/usr/bin/env python3

import argparse
import logging
import time
from pprint import pprint
import cv2
import numpy as np
import sys
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import os
logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=str, default=0)
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--save_video',type=bool,default=False, 
                        help='To write output video. default name file_name_output.avi')
    args = parser.parse_args()
    
    
    
    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(0)
    if(args.camera == '0'):
        file_write_name = 'camera_0'
    else:
        pass
        
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    count = 0
    y1 = [0,0]
    frame = 0
    while True:
        ret_val, image = cam.read()
        i =1
        count+=1
        if count % 11 == 0:
            continue
        
        if not ret_val:
            break
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        for human in humans:
            for i in range(len(humans)):
                try:
                    a = human.body_parts[0]   
                    x = a.x*image.shape[1]
                    y = a.y*image.shape[0]   
                    y1.append(y)   
                except:
                    pass

                if ((y - y1[-2]) > 25):  
                    cv2.putText(image, "Fall Detected", (20,50), cv2.FONT_HERSHEY_COMPLEX, 2.5, (0,0,255), 
                        2, 11)
                    print("Fall detected",i+1, count)
        
      
        cv2.imshow('Fall detection Estimation', image)
        fps_time = time.time()
        if(frame == 0) and (args.save_video):  
            out = cv2.VideoWriter(file_write_name+'_output.avi',cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                    20,(image.shape[1],image.shape[0]))
            out.write(image)
        k = cv2.waitKey(1)
    
        if k == ord('q'):
            break
        
        

    cv2.destroyAllWindows()
