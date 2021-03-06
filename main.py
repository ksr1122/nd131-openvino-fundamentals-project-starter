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
import imghdr

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

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

def connect_mqtt():
    ### Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

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

def draw_boxes(frame, result, width, height, prob_threshold):
    '''
    Draw bounding boxes onto the frame.
    '''
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

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    ### Handle the input stream ###
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    # Input validation
    if not cap.isOpened():
        print("Please provide a valid image/video file")
        exit(1)

    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()

    # Grab the shape of the input
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a video writer for the output video
#     out = cv2.VideoWriter('out.mp4', 0x00000021, cap.get(cv2.CAP_PROP_FPS), (width, height))

    global total_count, avg_duration

    ### Loop until stream is over ###
    while cap.isOpened():

        ### Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break

        ### Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### Start asynchronous inference for specified request ###
        infer_network.exec_net(p_frame)

        ### Wait for the result ###
        if infer_network.wait() == 0:

            ### Get the results of the inference request ###
            result = infer_network.get_output()

            ### Extract any desired stats from the results ###
            frame, current_count = draw_boxes(frame, result, width, height, float(prob_threshold))

            ### Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            send_update = update_count(current_count, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            client.publish("person", json.dumps({"count": current_count, "total": total_count}))
            if send_update:
                client.publish("person/duration", json.dumps({"duration": duration}))

        ### Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()
#         out.write(frame)

        ### Write an output image if `single_image_mode` ###
        ext = imghdr.what(args.input)
        if ext:
            cv2.imwrite('out.' + ext, frame)

    cap.release()
#     out.release()
    

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)

    # Disconnect from MQTT server
    client.disconnect()


if __name__ == '__main__':
    main()
