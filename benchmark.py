"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import time
import socket
import json
import cv2

import logging as log

from argparse import ArgumentParser
from inference import Network

import numpy as np
import tensorflow as tf

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

prev_count = 0
total_count = 0

duration = 0
avg_duration = 0

enter_time = 0.0
person_in_frame = False

def update_count(count, cur_time):
    global prev_count, total_count, duration, avg_duration, enter_time, person_in_frame
    
    send_update = False
    
    if count > prev_count and not person_in_frame:
        person_in_frame = True
        total_count += 1
        enter_time = cur_time

    if person_in_frame:
        duration = int(cur_time - enter_time)

    if count < prev_count and person_in_frame:
        if duration < 2: # false trigger
            return
        person_in_frame = False
        avg_duration += int((duration - avg_duration) / total_count)
        send_update = True

    prev_count = count
    return send_update

def draw_boxes_tf(frame, result, width, height, prob_threshold):
    # Output shape is (boxes[0], scores[0], num_detection)
    boxes = np.squeeze(result[0])
    scores = np.squeeze(result[1])
    num_detection = int(result[2])
    if not num_detection:
        return frame, 0
    count = 0
    for idx in range(0, num_detection):
        box = boxes[idx]
        conf = scores[idx]
        if conf >= prob_threshold:
            count = 1
            xmin = int(box[1] * width)
            ymin = int(box[0] * height)
            xmax = int(box[3] * width)
            ymax = int(box[2] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        return frame, count

def draw_boxes(frame, result, width, height, prob_threshold):
    count = 0
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= prob_threshold:
            count = 1
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
    return frame, count

def get_model(model_path):
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph

def tensor_detect(cap, out, width, height, args):
    model_graph = get_model(args.model)

    global total_count, duration, avg_duration

    image_tensor = model_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = model_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = model_graph.get_tensor_by_name('detection_scores:0')
    num_detection = model_graph.get_tensor_by_name('num_detections:0')

    with tf.Session(graph=model_graph) as sess:
        net_input_shape = (300, 300) # TODO: get from model

        while cap.isOpened():

            flag, frame = cap.read()
            if not flag:
                break

            frame = cv2.resize(frame, net_input_shape)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            result = sess.run([detection_boxes, detection_scores, num_detection],
                            feed_dict={image_tensor: np.expand_dims(frame, axis=0)})

            frame, current_count = draw_boxes_tf(frame, result, *net_input_shape, float(args.prob_threshold))
            send_update = update_count(current_count, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            if send_update:
                print(current_count, total_count, duration)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (width, height))

            out.write(frame)
#             cv2.imwrite('out.png', frame)

def infer_on_stream(cap, out, width, height, args):
    infer_network = Network()
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()
    
    while cap.isOpened():

        flag, frame = cap.read()
        if not flag:
            break

        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        infer_network.exec_net(p_frame)

        if infer_network.wait() == 0:

            result = infer_network.get_output()
            
            frame, current_count = draw_boxes(frame, result, width, height, float(args.prob_threshold))
            send_update = update_count(current_count, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            if send_update:
                print(current_count, total_count, duration)

        out.write(frame)
#         cv2.imwrite('out.png', frame)

def pipeline(handler, args, output):
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    width = int(cap.get(3))
    height = int(cap.get(4))

    out = cv2.VideoWriter(output, 0x00000021, cap.get(cv2.CAP_PROP_FPS), (width, height))

    handler(cap, out, width, height, args)

    cap.release()
    out.release()

def main():
    args = build_argparser().parse_args()
    args.model = args.model + '.pb'

    start_time = time.time()
    pipeline(tensor_detect, args, 'out-tensorflow.mp4')
    print("Tensorflow\n\t time: ", time.time()-start_time, " sec")
    
    args = build_argparser().parse_args()
    args.model = args.model + '.xml'
    
    start_time = time.time()
    pipeline(infer_on_stream, args, 'out-inference.mp4')
    print("Inference Engine\n\t time: ", time.time()-start_time, " sec")

if __name__ == '__main__':
    main()
