from ultralytics import YOLO
from pathlib import Path
from time import perf_counter, time
import cv2
import numpy as np
import torch
from boxmot import StrongSORT
from pathlib import Path
import sys
from datetime import datetime, timedelta
from boxmot import StrongSORT
from pathlib import Path

from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.motion.cmc import get_cmc_method
from boxmot.trackers.strongsort.sort.detection import Detection
from boxmot.trackers.strongsort.sort.tracker import Tracker
from boxmot.utils.matching import NearestNeighborDistanceMetric
from boxmot.utils.ops import xyxy2tlwh
from boxmot.utils import PerClassDecorator


class StrongSORT(object):
    def __init__(
        self,
        model_weights,
        device,
        fp16,
        per_class=False,
        max_dist=0.2,
        max_iou_dist=0.7,
        max_age=30,
        n_init=1,
        nn_budget=100,
        mc_lambda=0.995,
        ema_alpha=0.9,
    ):
        self.max_dist=0.95,
        self.max_iou_dist=0.95,
        self.max_age=300,
        self.per_class = per_class
        rab = ReidAutoBackend(
            weights=model_weights, device=device, half=fp16
        )
        self.model = rab.get_backend()
        self.tracker = Tracker(
            metric=NearestNeighborDistanceMetric("cosine", max_dist, nn_budget),
            max_iou_dist=max_iou_dist,
            max_age=max_age,
            n_init=n_init,
            mc_lambda=mc_lambda,
            ema_alpha=ema_alpha,
        )
        self.cmc = get_cmc_method('ecc')()

    @PerClassDecorator
    def update(self, dets, img, embs=None):
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img' input format '{type(img)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        xyxy = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]
        det_ind = dets[:, 6]

        if len(self.tracker.tracks) >= 1:
            warp_matrix = self.cmc.apply(img, xyxy)
            for track in self.tracker.tracks:
                track.camera_update(warp_matrix)

        # extract appearance information for each detection
        if embs is not None:
            features = embs
        else:
            features = self.model.get_features(xyxy, img)

        tlwh = xyxy2tlwh(xyxy)
        detections = [
            Detection(box, conf, cls, det_ind, feat) for
            box, conf, cls, det_ind, feat in
            zip(tlwh, confs, clss, det_ind, features)
        ]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update >= 1:
                continue

            x1, y1, x2, y2 = track.to_tlbr()

            id = track.id
            conf = track.conf
            cls = track.cls
            det_ind = track.det_ind

            outputs.append(
                np.concatenate(([x1, y1, x2, y2], [id], [conf], [cls], [det_ind])).reshape(1, -1)
            )
        if len(outputs) > 0:
            return np.concatenate(outputs)
        return np.array([])

test_vid = "230411BVK107 - Trim.mp4"
model_weights = "5-class-model.pt"

model_name_dict = {
    "model_yolo/thucquan.pt": '_TQ',
    "model_yolo/daday.pt": '_DD',
    "model_yolo/htt.pt": '_HTT'
}

input_video_name = 'CS201'

model_classes_dict = {
    "5-class-model.pt": ['Viem thuc quan', 'Viem da day' ,'Ung thu thuc quan', 'Ung thu da day', 'Loet HTT'],
}

model_classes = model_classes_dict.get(model_weights, ['polyp', 'esophagael cancer'])

print("Input Video Name:", input_video_name)
print("Model Classes:", model_classes)


class Colors:
    def __init__(self, num_colors=80):
        self.num_colors = num_colors
        self.color_palette = self.generate_color_palette()


    def generate_color_palette(self):
        hsv_palette = np.zeros((self.num_colors, 1, 3), dtype=np.uint8)
        hsv_palette[:, 0, 0] = np.linspace(0, 180, self.num_colors, endpoint=False)
        hsv_palette[:, :, 1:] = 255
        bgr_palette = cv2.cvtColor(hsv_palette, cv2.COLOR_HSV2BGR)
        return bgr_palette.reshape(-1, 3)

    def __call__(self, class_id):
        color = tuple(map(int, self.color_palette[class_id]))
        return color

class ObjectDetection:
    def __init__(self, model_weights="yolov8s.pt", capture_index=0, min_temporal_threshold=0, max_temporal_threshold=0, iou_threshold=0.2, use_frame_id=False):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.model = self.load_model(model_weights)
        self.classes = self.model.names
        self.classes = model_classes
        self.colors = Colors(len(self.classes))
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.capture_index = capture_index
        self.cap = self.load_capture()
        reid_weights = Path("osnet_x0_25_endocv_30.pt")
        self.tracker = StrongSORT(reid_weights,
                                  torch.device(self.device),
                                  fp16 = False,
                                  max_dist=0.95,
                                  max_iou_dist=0.95,
                                  max_age=300
                                  )
        self.min_temporal_threshold = min_temporal_threshold
        self.max_temporal_threshold = max_temporal_threshold
        self.iou_threshold = iou_threshold
        self.use_frame_id = use_frame_id
        self.labels = {}
        self.saved_images = {}
        self.last_detected_frame = None

    def load_model(self, weights):
        model = YOLO(weights)
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame, stream=True, verbose=False, conf=0.5, line_width=1)
        return results

    def _frame_idx_to_hmsf(self, frame_id: int):
        """convert to hmsf timestamp by given frame idx and fps"""
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        assert self.video_fps
        base = datetime.strptime('00:00:00.000000', '%H:%M:%S.%f')
        delta = timedelta(seconds=frame_id/self.video_fps)
        return (base + delta).strftime('%H:%M:%S.%f')

    def _frame_idx_to_hms(self, frame_id: int):
        """convert to hms timestamp by given frame idx and fps"""
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        assert self.video_fps
        base = datetime.strptime('00:00:00', '%H:%M:%S')
        delta = timedelta(seconds=frame_id//self.video_fps)
        return (base + delta).strftime('%H:%M:%S')

    def load_capture(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        video_name = "tracking_" + input_video_name + ".mp4"
        self.writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        return cap

    def write_seqinfo_ini(self, seq_name, seq_length, frame_rate, im_width, im_height, im_ext, im_dir):
        with open("seqinfo.ini", "w") as f:
            f.write("[Sequence]\n")
            f.write(f"name={seq_name}\n")
            f.write(f"imDir={im_dir}\n")  # Thay thế bằng thư mục chứa ảnh nếu cần
            f.write(f"frameRate={frame_rate}\n")
            f.write(f"seqLength={seq_length}\n")
            f.write(f"imWidth={im_width}\n")
            f.write(f"imHeight={im_height}\n")
            f.write(f"imExt={im_ext}\n")

    def calculate_iou(self, box1, box2):
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        # Calculate areas of each bounding box
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # Calculate union area
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area

        return iou

    def update_track_id(self, current_tracks, previous_tracks):
        updated_tracks = []
        for current_track in current_tracks:
            min_distance = float('inf')
            matching_track_id = None
            for previous_track in previous_tracks:
                if current_track[6] != previous_track[6]:
                    continue  # Skip tracks of different classes
                iou = self.calculate_iou(current_track[:4], previous_track[:4])
                #print(iou, self.iou_threshold)
                if iou > self.iou_threshold:
                    if self.use_frame_id:
                        time_diff = abs(current_track[3] - previous_track[3])
                        if time_diff < min_distance:
                            min_distance = time_diff
                            matching_track_id = previous_track[4]
                    else:
                        time_diff = abs(current_track[1] - previous_track[1])
                        if time_diff < min_distance:
                            min_distance = time_diff
                            matching_track_id = previous_track[4]

            if matching_track_id is not None:
                current_track[4] = matching_track_id
            updated_tracks.append(current_track)
        return updated_tracks

    def draw_tracks(self, frame, tracks, txt_file, overlap_threshold=0.5):
        seq_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        im_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        im_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_id = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))-1
        timestamp_hms = self._frame_idx_to_hms(frame_id)
        timestamp_hmsf = self._frame_idx_to_hmsf(frame_id)
        null_notes = "Tracking"
        labels_dict = {}
        for track in tracks:
            x1, y1, x2, y2 = int(track[0]), int(track[1]), int(track[2]), int(track[3])
            id = int(track[4])
            conf = round(track[5], 2)
            class_id = int(track[6])
            class_name = self.classes[class_id]
            cv2.rectangle(frame, (x1,y1), (x2, y2), self.colors(class_id), 5)
            # self.save_first_detected_frame(frame, track)
            # Write result to txt file
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            scale_height = frame.shape[0]
            scale_width = frame.shape[1]
            # Update label if the object ID is new or changed
            if id not in self.labels:
                self.labels[id] = class_name
            self.save_first_detected_frame(frame, track)
            txt_file.write(f"{timestamp_hms},{timestamp_hmsf},{frame_id},{frame_rate},{class_name},{id},{id},{null_notes},{frame.shape[0]},{frame.shape[1]},{scale_height},{scale_width},{x1},{y1},{x2},{y2},{center_x},{center_y}\n")

        return frame

    def display_labels(self, frame, tracks):
        # Tạo một từ điển để lưu trữ các nhãn đã được gán
        frame_id = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))-1
        labels_dict = {}
        last_detection_times = {}
        previous_label_colors = {}

        # Lặp qua các tracks và cập nhật từ điển labels_dict
        for track in tracks:
            id = int(track[4])
            class_id = int(track[6])
            class_name = self.classes[class_id]
            labels_dict[id] = class_name
        # Hiển thị nhãn trên khung hình
        for id, label in self.labels.items():
            # label = f'{self.labels[id]}, ID: {id}'
            if id in labels_dict:
                # Nếu đối tượng có trong tracks, hiển thị nhãn mới
                self.labels[id] = labels_dict[id]
                class_id = int(track[6])
                label_color = self.colors(class_id)
                previous_label_colors[id] = label_color
                last_detection_times[id] = time()  # Lưu màu của nhãn mới
                label = f'{self.labels[id]}, ID: {id}'
            else:
                # Nếu không phát hiện được đối tượng trong frame, sử dụng màu của nhãn trước đó
                label_color = previous_label_colors.get(id, (255, 255, 255))

            self.labels = {}

            # Hiển thị nhãn trên khung hình
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 5)
            label_x = frame.shape[1] - w - 20
            label_y = 50 + h
            cv2.rectangle(frame, (label_x, label_y - h - 15), (label_x + w + 10,label_y + 10), (0, 0, 0), -1)
            cv2.putText(frame, label, (label_x + 5, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, label_color, 3)

        return frame

    def save_first_detected_frame(self, frame, track):
        x1, y1, x2, y2 = int(track[0]), int(track[1]), int(track[2]), int(track[3])
        id = int(track[4])
        class_id = int(track[6])
        key = (id, class_id)

        if hasattr(self, 'last_saved_key') and self.last_saved_key != key:
            # Clear the saved images if there is a change in class or id
            self.saved_images.clear()

        if key not in self.saved_images:
            object_img = frame[y1:y2, x1:x2]
            height, width = object_img.shape[:2]
            #print(height, width)
            if height > 0:
              aspect_ratio = width / height
              new_width = 300
              if aspect_ratio == 0:
                new_height = 300
              else:
                new_height = int(new_width / aspect_ratio)

              if new_height > 980:
                  new_height = 980
                  new_width = int(new_height * aspect_ratio)

              resized_img = cv2.resize(object_img, (new_width, new_height))

            if height <= 0:
              resized_img = cv2.resize(object_img, (300, height))

            self.saved_images[key] = resized_img
            self.last_saved_key = key

    def draw_saved_images(self, frame):
        for (id, class_id), img in self.saved_images.items():
            x_offset = 1600
            y_offset = 100
            y_end = y_offset + img.shape[0]
            x_end = x_offset + img.shape[1]
            #cv2.rectangle(frame, (1600, 100), (x_end, 1080), (0, 0, 0), -1)
            frame[y_offset:y_end, x_offset:x_end] = img
            #print(img.shape)
        return frame


    def __call__(self):
        tracker = self.tracker

        # Lấy thông tin từ video kết quả
        seq_name = "StrongSort"
        im_dir = "img1"
        seq_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        im_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        im_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        im_ext = ".jpg"  # Phần mở rộng của ảnh

        # Ghi thông tin vào file seqinfo.ini
        self.write_seqinfo_ini(seq_name, seq_length, frame_rate, im_width, im_height, im_ext, im_dir)

        # Mở file txt để ghi kết quả
        with open("tracking_result.txt", "w") as txt_file:
            txt_file.write("timestamp_hms,timestamp_hmsf,frame_idx,fps,object_cls,object_idx,object_id,notes,frame_height,frame_width,scale_height,scale_width,x1,y1,x2,y2,center_x,center_y\n")
            previous_tracks = []
            while True:
                start_time = perf_counter()
                ret, frame = self.cap.read()
                if not ret:
                    break

                label = "Unknown"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 5)
                label_x = frame.shape[1] - w - 20
                label_y = 50 + h
                cv2.rectangle(frame, (label_x, label_y - h - 15), (label_x + w + 10, label_y + 10), (0, 0, 0), -1)
                cv2.putText(frame, label, (label_x + 5, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                detections = self.predict(frame)
                for dets in detections:
                    tracks = tracker.update(dets.boxes.data.to("cpu").numpy(), frame)
                    if len(tracks.shape) == 2 and tracks.shape[1] == 8:
                        if len(previous_tracks) > 0:
                            tracks = self.update_track_id(tracks, previous_tracks)
                        frame = self.draw_tracks(frame, tracks, txt_file)
                        previous_tracks = tracks
                self.display_labels(frame, tracks)
                self.draw_saved_images(frame)
                end_time = perf_counter()
                # fps = 1 / np.round(end_time - start_time, 2)
                # cv2.rectangle(frame, (0, 30), (220, 80), (255, 255, 255), -1)
                # cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), self.font, 1.5, (0, 255, 0), 5)
                self.writer.write(frame)
                cv2.imshow('frame', frame)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            self.cap.release()
            self.writer.release()
            cv2.destroyAllWindows()

detector = ObjectDetection(model_weights, test_vid)
#detector = ObjectDetection(model_weights, capture_index=0)
detector()