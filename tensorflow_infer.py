# -*- coding:utf-8 -*-
import cv2
import time
import argparse

import numpy as np
from PIL import Image
#from keras.models import model_from_json
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from load_model.tensorflow_loader import load_tf_model, tf_inference
from sort import Sort
from sound import play_sound, stop_thread
from face_recognition_library import *
import os, datetime

class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = datetime.datetime.now()
        self._end = datetime.datetime.now()
        self._numFrames = 0

    def start(self):
        # start the timer   
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1
        self._end = datetime.datetime.now()

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


sess, graph = load_tf_model('models/face_mask_detection.pb')
# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}

def inference(image,
              conf_thresh=0.5,
              iou_thresh=0.4,
              target_shape=(160, 160),
              draw_result=True,
              show_result=True,
              return_result=False
              ):
    '''
    Main function of detection inference
    :param image: 3D numpy array of image
    :param conf_thresh: the min threshold of classification probabity.
    :param iou_thresh: the IOU threshold of NMS
    :param target_shape: the model input size.
    :param draw_result: whether to daw bounding box to the image.
    :param show_result: whether to display the image.
    :return:
    '''
    # image = np.copy(image)
    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0  # 归一化到0~1
    image_exp = np.expand_dims(image_np, axis=0)
    y_bboxes_output, y_cls_output = tf_inference(sess, graph, image_exp)
    # print(image_exp.shape)

    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )

    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)
        # print(image.shape)
        # print(bbox)

        if draw_result:
            if class_id == 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            # cv2.rectangle(image_resized, (int(bbox[0]*260), int(bbox[1]*260)),
            #         (int(xmax*260), int(ymax*260)), color, 2)
            cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
        # output_info.append([class_id, conf, xmin, ymin, xmax, ymax])
        output_info.append((class_id, conf, xmin, ymin, xmax, ymax))

    
    # print(output_info)

    if return_result:
        return image
        # return image_resized

    if show_result:
        Image.fromarray(image).show()
    return output_info


def run_on_video(video_path, output_video_name, conf_thresh):
    cap = cv2.VideoCapture(video_path)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # writer = cv2.VideoWriter(output_video_name, fourcc, int(fps), (int(width), int(height)))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened():
        raise ValueError("Video open failed.")
        return
    status = True
    idx = 0
    while status:
        start_stamp = time.time()
        status, img_raw = cap.read()
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        read_frame_stamp = time.time()
        if (status):
            inference(img_raw,
                      conf_thresh,
                      iou_thresh=0.5,
                      target_shape=(260, 260),
                      draw_result=True,
                      show_result=False)
            cv2.imshow('image', img_raw[:, :, ::-1])
            cv2.waitKey(1)
            inference_stamp = time.time()
            # writer.write(img_raw)
            write_frame_stamp = time.time()
            idx += 1
            print("%d of %d" % (idx, total_frames))
            print("read_frame:%f, infer time:%f, write time:%f" % (read_frame_stamp - start_stamp,
                                                                   inference_stamp - read_frame_stamp,
                                                                   write_frame_stamp - inference_stamp))
    # writer.release()

# tracking 
tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.1)
ids = []

saved_ids = []

log_dir = 'log'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if __name__ == "__main__":

    # cam = cv2.VideoCapture(2)
    cam = cv2.VideoCapture(0)

    fps = FPS()

    while True:
        det = []
        ret, img = cam.read()
        if ret == False:
            print('Can not read video stream')
        
        img = cv2.flip(img, 1)
        frame_cp = np.copy(img)

        # img = inference(img, show_result=False, return_result=True, target_shape=(260,260))
        output_info = inference(img, show_result=False, draw_result=False, return_result=False, target_shape=(260,260))
        # print(output_info)
        for (lb, conf, x1, y1, x2, y2) in output_info:
            # print((lb, conf, x1, y1, x2, y2))
            det.append((x1, y1, x2, y2,1, lb))

        predict = tracker.update(np.array(det))

        for pre in predict:
            # print(pre)
            x1, y1, x2, y2, id, lb=int(pre[0]),int(pre[1]),int(pre[2]),int(pre[3]),int(pre[4]), int(pre[5])
            # print('predict: ', (x1, y1, x2, y2))
            if id not in ids:
                ids.append(id)

            if lb == 0:      
                cv2.putText(frame_cp, 'mask', (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0))
                # cv2.putText(frame_cp, str(id), (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX ,  
                #     1, (0,255,0) , 2, cv2.LINE_AA) 
                cv2.rectangle(frame_cp,(x1,y1) ,(x2,y2), (0,255,0), 2)
            else: #not wearing mask
                #save log
                #face recognition
                #ting ting ting ting
                new_detect = False
                if id not in saved_ids: #if not save this people before
                    # face recognition
                    top = int(y1 / 4)
                    right = int(x2 / 4)
                    bottom = int(y2 / 4)
                    left = int(x1/ 4)

                    face_location = (top, right, bottom, left) #for recognition
                    name, face_dis = recognize_with_location(img, face_location)
                    # name = 'unknown'
                    
                    print('Save to: %s \t Name: %s \t Face_dis: %.2f' % (log_dir, name, face_dis))
                    saved_ids.append(id)
                    new_detect = True

                    #got some exception with thread sound here
                    play_sound()
                
                cv2.rectangle(frame_cp,(x1,y1) ,(x2,y2), (0,0,255), 2)

                cv2.putText(frame_cp, 'no mask', (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255))
                # cv2.putText(frame_cp, str(id), (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX ,  
                #         1, (0,0,255) , 2, cv2.LINE_AA) 

                if new_detect:
                    cv2.imwrite(log_dir + '/' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '_' + name + '.jpg', frame_cp)
               
        fps.update()
        cv2.putText(frame_cp,'FPS: ' + str(round(fps.fps())), (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # print(len(tracker.trackers))
        for trk in tracker.trackers:
            color = trk.color
            # print(len(trk.history))
            if len(trk.history) > 20:
                for i in range(1,20):
                    location = trk.history[-i]
                    x1, y1, x2, y2 = location.reshape(4)
                    x_centroid = int((x1+x2) / 2)
                    y_centroid = int((y1+y2) / 2)
                    cv2.circle(frame_cp, (x_centroid, y_centroid), 2, color, 2)
            else:
                for location in trk.history:
                    x1, y1, x2, y2 = location.reshape(4)
                    x_centroid = int((x1+x2) / 2)
                    y_centroid = int((y1+y2) / 2)
                    cv2.circle(frame_cp, (x_centroid, y_centroid), 2, color, 2)
                # print(location.reshape(4))
            # print(trk.history)
            
            # if (len(trk.history) > 0):
            #     x1, y1, x2, y2 = trk.history[0]
            #     # print(trk.history[0].shape)
            #     print(x1,y1)
        
        cv2.imshow('webcam', frame_cp)
        key = cv2.waitKey(1)

        if key == 27:
            # stop_thread = True
            # play_sound()
            break
        elif key == ord('p'):
            cv2.waitKey(0)
    cv2.destroyAllWindows()
    