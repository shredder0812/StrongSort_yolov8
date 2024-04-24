import torch
import numpy as np
import cv2
from time import perf_counter
from ultralytics import YOLO
import os
import yaml 
from easydict import EasyDict as edict
from pathlib import Path

import supervision as sv
from strongsort.strong_sort import StrongSORT

from util import YamlParser

SAVE_VIDEO = True
TRACKER = "strongsort"

class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness =3, text_thickness= 3, text_scale= 1.5 )

        reid_weights = Path("weights/osnet_x0_25_msmt17.pt")

        tracker_config = "strongsort/configs/strong_sort.yaml"
        cfg = YamlParser()
        cfg.merge_from_file(tracker_config)
        self.tracker = StrongSORT(
            reid_weights,
            torch.device(self.device),
            False,
            max_dist = cfg.strong_sort.max_dist,
            max_iou_distance = cfg.strong_sort.max_iou_distance,
            max_age = cfg.strong_sort.max_age,
            #max_unmatched_preds = cfg.strong_sort.max_unmatched_preds,
            n_init = cfg.strong_sort.n_init,
            nn_budget = cfg.strong_sort.nn_budget,
            mc_lambda = cfg.strong_sort.mc_lambda,
            ema_alpha = cfg.strong_sort.ema_alpha,
        )


    def load_model(self):
        model = YOLO("visdrone_s.pt")
        model.fuse()
        return model
    def predict(self, frame):
        results = self.model(frame)
        return results
    
    def draw_results(self, frame, results):
        xyxys = []
        confidences = []
        class_ids = []
        detections = []
        boxes = []
        
        for result in results:
            class_id = result.boxes.cls.cpu().numpy().astype(int)
            if len(class_id) == 0:
                continue
            if len(class_id)>1:
                class_id = class_id[0]
            if class_id == 0:
                xyxys.append(result.boxes.xyxy.cpu().numpy())
                confidences.append(result.boxes.conf.cpu().numpy())
                class_ids.append(class_id.reshape(1))
                boxes.append(result.boxes)
                detections = sv.Detections(
                    xyxy = result.boxes.xyxy.cpu().numpy(),
                    confidence = result.boxes.conf.cpu().numpy(),
                    class_id=result.boxes.cls.cpu().numpy().astype(int), 
                )
            self.labels = []
            #print(detections)
            for _, m, confidence, class_id, tracker_id, d in detections:
                #print(_,  confidence, class_id, tracker_id)
                self.labels.append(f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}")
                
            #self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                           #for _, m, confidence, class_id, tracker_id,d in detections]
                           #zip(detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id) if tracker_id is not None]

        #print(self.labels)
        frame = self.box_annotator.annotate(scene = frame, detections= detections, labels = self.labels)
        return frame, boxes
    

    
    def __call__(self):
        cap = cv2.VideoCapture("visdrone_126_0001.mp4")
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2688)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1512)

        if SAVE_VIDEO:
            outputvid = cv2.VideoWriter('result_traching.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (2688, 1512))

        tracker = self.tracker

        if hasattr(tracker, 'model'):
            if hasattr(tracker.model, 'warmup'):
                tracker.model.warmup()
        
        outputs = [None]
        curr_frames, prev_frames = None, None

        detection_start_time = 0
        detection_end_time = 0
        tracking_start_time = 0
        tracking_end_time = 0

        while True:
            start_time = perf_counter()
            ret, frame = cap.read()

            assert ret
            detection_start_time = perf_counter()
            results = self.predict(frame)
            detection_end_time = perf_counter()
            #print(results)
            frame, _ = self.draw_results(frame, results)

            if hasattr(tracker, 'tracker') and hasattr(tracker.tracker, 'camera_update'):
                if prev_frames is not None and curr_frames is not None:
                    tracker.tracker.camera_update(prev_frames, curr_frames)

            for result in results:
                #print(result[0])
                #print(frame)
                outputs[0] = tracker.update(result, frame)
                for i, (output) in enumerate(outputs[0]):
                    bbox = output[0:4]
                    tracked_id = output[4]
                    #cls = output[5]
                    #conf = output[6]
                    top_left = (int(bbox[-2]-100), int(bbox[1]))
                    cv2.putText(frame, f"ID: {tracked_id}", top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
                
            tracking_end_time = perf_counter()

            detection_time = detection_end_time - detection_start_time
            tracking_time = tracking_end_time - tracking_start_time

            detection_fps = 1 / detection_time
            tracking_fps = 1 / tracking_time



            cv2.putText(frame, f'Detection FPS: {detection_fps}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            cv2.putText(frame, f'Tracking FPS: {tracking_fps}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

            cv2.imshow('YOLOv8 Detection', frame)

            if SAVE_VIDEO:
                outputvid.write(frame)
            if cv2.waitKey(5) & 0xFF ==27:
                break

        if SAVE_VIDEO:
            outputvid.release()
        cap.release()
        cv2.destroyAllWindows()

detector = ObjectDetection(capture_index=0)
detector()


