"""
자율주행 메인 프로그램 (최종 수정)
1. YOLO 최적화: 4프레임마다 1번 실행
2. YOLO 단일 박스: 확률 가장 높은 1개만 검출 (NMS 적용)
3. 신호등 전용 감도: --traffic-threshold 옵션으로 별도 조절 가능
4. LiDAR 비상 정지: 전방 30cm 이내 정지
"""

import cv2
import time
import glob
import signal
import sys
import threading
import math
from pathlib import Path
from serial import Serial
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from ultralytics import YOLO

# ROS 라이브러리 체크
try:
    import rospy
    from sensor_msgs.msg import LaserScan
    ROS_AVAILABLE = True
except ImportError:
    print("경고: ROS 라이브러리를 찾을 수 없습니다. LiDAR 기능이 비활성화됩니다.")
    ROS_AVAILABLE = False

from model import SteeringModel


class MotorController:
    """아두이노와 시리얼 통신하여 모터 제어"""
    def __init__(self, port='/dev/arduino', baudrate=115200):
        self.ser = None
        ports_to_try = [port, '/dev/ttyUSB0', '/dev/ttyUSB1']
        for p in ports_to_try:
            try:
                self.ser = Serial(p, baudrate, timeout=0.5)
                print(f"모터 컨트롤러 연결 성공: {p}")
                break
            except: continue
        
        if self.ser is None: print("모터 연결 실패. (시뮬레이션 모드로 동작)")
        time.sleep(1)

    def create_command(self, steering, speed):
        STX = 0xEA; ETX = 0x03; Length = 0x03
        dummy1 = 0x00; dummy2 = 0x00
        Checksum = ((~(Length + steering + speed + dummy1 + dummy2)) & 0xFF) + 1
        return bytearray([STX, Length, steering, speed, dummy1, dummy2, Checksum, ETX])

    def send_command(self, steering, speed):
        if self.ser: self.ser.write(self.create_command(steering, speed))

    def stop(self): self.send_command(90, 90)
    def close(self): self.stop(); self.ser.close() if self.ser else None


class ImageSource:
    """이미지 소스 추상화"""
    def __init__(self, source_type='file', file_pattern='sample/*.jpg', camera_id=0):
        self.source_type = source_type
        if source_type == 'file':
            self.image_files = sorted(glob.glob(file_pattern))
            if not self.image_files: raise ValueError(f"이미지 없음: {file_pattern}")
            self.current_index = 0; self.loop = True
            print(f"이미지 파일 {len(self.image_files)}개 로드됨")
        elif source_type == 'camera':
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened(): raise ValueError(f"카메라 {camera_id} 연결 실패")
            print(f"카메라 {camera_id} 연결됨")

    def read_frame(self):
        if self.source_type == 'file':
            if self.current_index >= len(self.image_files):
                if self.loop: self.current_index = 0
                else: return None
            img = cv2.imread(self.image_files[self.current_index])
            self.current_index += 1
            if img is None: return self.read_frame()
            return img
        elif self.source_type == 'camera':
            ret, frame = self.cap.read()
            return frame if ret else None

    def release(self):
        if self.source_type == 'camera': self.cap.release()


class AutoDrive:
    """자율주행 메인 클래스"""
    def __init__(self, use_motor=True, show_debug=True, speed=100,
                 model_path='weights/best_model.pth', steering_gain=1.0,
                 record_video=False, output_path='output/drive_recording.mp4',
                 yolo_model_path='weights/yolo.pt', yolo_labels_path='weights/yolo_label.txt',
                 bbox_size_threshold=0.15, traffic_threshold=0.05, use_yolo=True):
        
        self.use_motor = use_motor
        self.show_debug = show_debug
        self.default_speed = speed
        self.speed = speed
        self.running = True
        self.record_video = record_video
        self.output_path = output_path
        self.video_writer = None
        self.steering_gain = steering_gain
        self.use_yolo = use_yolo
        
        # [설정] 일반 물체 인식 크기 vs 신호등 인식 크기 분리
        self.bbox_size_threshold = bbox_size_threshold
        self.traffic_threshold = traffic_threshold

        # 상태 변수
        self.is_camera_stopped = False
        self.is_lidar_stopped = False
        self.stop_reason = ""
        self.detected_objects = []
        self.front_distance = 999.0

        # 각도 설정
        self.angle_min = 45.0; self.angle_max = 135.0; self.angle_neutral = 90.0
        self.resize_w = 320; self.resize_h = 180; self.crop_y0 = 120

        # AI 모델 로드
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()

        # YOLO 로드
        if self.use_yolo:
            self.yolo_model = self._load_yolo_model(yolo_model_path)
            self.yolo_labels = self._load_yolo_labels(yolo_labels_path)
        else: self.yolo_model = None

        # 모터 연결
        if self.use_motor:
            self.motor = MotorController()
            if self.motor.ser is None: self.use_motor = False

        # ROS (LiDAR) 설정
        if ROS_AVAILABLE:
            try:
                rospy.init_node('auto_drive_node', anonymous=True, disable_signals=True)
                rospy.Subscriber('/scan', LaserScan, self._lidar_callback, queue_size=1)
                print("LiDAR 연결 성공 (/scan)")
            except: print("ROS 초기화 실패")

        signal.signal(signal.SIGINT, self._signal_handler)

    def _lidar_callback(self, data):
        # LiDAR 데이터 처리 (전방 30cm 감지)
        center_deg = 180; width_deg = 40; min_dist = 200.0
        for i, r in enumerate(data.ranges):
            dist = r * 100
            if math.isinf(dist) or math.isnan(dist) or dist == 0: continue
            angle = math.degrees(data.angle_min + i * data.angle_increment)
            if angle < 0: angle += 360
            
            diff = abs(angle - center_deg)
            if diff > 180: diff = 360 - diff
            if diff <= width_deg / 2: min_dist = min(min_dist, dist)

        self.front_distance = min_dist
        
        # 히스테리시스 적용 (떨림 방지)
        if self.front_distance < 30.0: self.is_lidar_stopped = True
        elif self.front_distance > 35.0: self.is_lidar_stopped = False

    @torch.no_grad()
    def _load_model(self, path):
        model = SteeringModel().to(self.device)
        try: state = torch.load(path, map_location=self.device, weights_only=True)
        except: state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state: sd = state["state_dict"]
        elif isinstance(state, dict) and "model_state" in state: sd = state["model_state"]
        else: sd = state
        model.load_state_dict(sd)
        model.eval()
        return model

    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize((self.resize_h, self.resize_w)),
            transforms.Lambda(lambda img: img.crop((0, self.crop_y0, self.resize_w, self.resize_h))),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def _load_yolo_model(self, path):
        print(f"YOLO 로드: {path}")
        return YOLO(path)

    def _load_yolo_labels(self, path):
        labels = {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if '→' in line:
                        p = line.split('→')
                        if len(p) == 2: labels[int(p[0])] = p[1].strip()
            print(f"YOLO 레이블 {len(labels)}개 로드")
        except: pass
        return labels

    def _clamp_angle(self, angle):
        return max(self.angle_min, min(float(angle), self.angle_max))

    @torch.no_grad()
    def _predict_steering(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        raw_angle = self.model(tensor).item()
        offset = (raw_angle - self.angle_neutral) * self.steering_gain
        return self._clamp_angle(self.angle_neutral + offset)

    def _detect_objects(self, frame):
        if not self.use_yolo or self.yolo_model is None: return []

        # [요청 1] agnostic_nms=True: 중복 박스 제거
        results = self.yolo_model(frame, verbose=False, agnostic_nms=True, conf=0.40)
        candidates = []

        if results:
            for box in results[0].boxes:
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.yolo_labels.get(cls + 1, f"Class_{cls}")
                
                ratio = ((x2-x1)*(y2-y1)) / (frame.shape[0]*frame.shape[1])

                # [요청 2] 신호등은 별도 설정값(traffic_threshold) 사용
                traffic_lights = ['Red Light', 'Green Light', 'Yellow Light', 'Yellow light']
                threshold = self.traffic_threshold if label in traffic_lights else self.bbox_size_threshold
                
                if ratio >= threshold:
                    candidates.append({'label':label, 'bbox':(int(x1),int(y1),int(x2),int(y2)), 'conf':conf, 'ratio':ratio})

        # [요청 1] 신뢰도(conf)가 가장 높은 1개만 반환
        if candidates:
            candidates.sort(key=lambda x: x['conf'], reverse=True)
            return candidates[:1]
        
        return []

    def _update_driving_state(self, objects):
        self.detected_objects = objects
        stop_list = ['Person', 'Red Light', 'Stop Sign', 'Yellow Light', 'Yellow light']
        stop_objs = [o['label'] for o in objects if o['label'] in stop_list]
        self.is_camera_stopped = bool(stop_objs)
        self.stop_reason = ", ".join(stop_objs)
        
        if not self.is_camera_stopped:
            if any(o['label'] == 'Speed Limit_80' for o in objects): self.speed = 102
            elif any(o['label'] == 'Speed Limit_40' for o in objects): self.speed = 100
            else: self.speed = self.default_speed

    def _draw_debug_overlay(self, frame, angle):
        debug = frame.copy()
        for o in self.detected_objects:
            x1,y1,x2,y2 = o['bbox']
            is_danger = o['label'] in ['Person','Red Light','Stop Sign','Yellow Light', 'Yellow light']
            color = (0,0,255) if is_danger else (0,255,0)
            cv2.rectangle(debug, (x1,y1), (x2,y2), color, 2)
            cv2.putText(debug, f"{o['label']}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        status = "LiDAR STOP" if self.is_lidar_stopped else (f"STOP: {self.stop_reason}" if self.is_camera_stopped else "DRIVING")
        color = (0,0,255) if "STOP" in status else (0,255,0)
        
        cv2.putText(debug, f"Steer: {angle:.1f}", (10,30), 1, 0.7, (0,255,0), 2)
        cv2.putText(debug, f"Speed: {self.speed}", (10,60), 1, 0.7, (0,255,0), 2)
        cv2.putText(debug, f"LiDAR: {self.front_distance:.1f}cm", (10,90), 1, 0.7, (255,255,0), 2)
        cv2.putText(debug, status, (10,120), 1, 0.7, color, 2)
        return debug

    def _signal_handler(self, signum, frame):
        self.running = False

    def _init_video_writer(self, frame_shape):
        if self.record_video and self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.output_path, fourcc, 10, (frame_shape[1], frame_shape[0]))

    def run(self, image_source):
        print(">>> 자율주행 시작")
        print(f"> 설정: 일반객체크기={self.bbox_size_threshold}, 신호등크기={self.traffic_threshold}")
        frame_count = 0
        loop_count = 0 

        try:
            while self.running and (not ROS_AVAILABLE or not rospy.is_shutdown()):
                frame = image_source.read_frame()
                if frame is None: break

                # [최적화] 4프레임마다 1번만 YOLO 실행
                if self.use_yolo and loop_count % 4 == 0:
                    detected_objects = self._detect_objects(frame)
                    self._update_driving_state(detected_objects)
                else:
                    pass # 이전 감지 상태 유지 시 깜빡일 수 있으므로 비움

                # 조향은 매 프레임 계산
                steering_angle = self._predict_steering(frame)
                
                debug_image = self._draw_debug_overlay(frame, steering_angle)

                if frame_count == 0 and self.record_video: self._init_video_writer(debug_image.shape)

                # [기능 3] LiDAR 우선 정지
                final_speed = 90 if self.is_lidar_stopped or self.is_camera_stopped else self.speed

                if self.use_motor: self.motor.send_command(int(steering_angle), final_speed)

                if self.record_video and self.video_writer: self.video_writer.write(debug_image)
                if self.show_debug:
                    cv2.imshow('Auto Drive', debug_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                
                if image_source.source_type == 'file': time.sleep(0.1)
                frame_count += 1
                loop_count += 1

        finally:
            self._cleanup(image_source)

    def _cleanup(self, image_source):
        print("\n정리 중...")
        if self.use_motor: self.motor.close()
        if self.video_writer: self.video_writer.release()
        image_source.release()
        cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='file')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--no-motor', action='store_true')
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--speed', type=int, default=100)
    parser.add_argument('--model', type=str, default='weights/best_model.pth')
    parser.add_argument('--steering-gain', type=float, default=1.0)
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--output', type=str, default='output/drive_recording.mp4')
    parser.add_argument('--yolo-model', type=str, default='weights/yolo.pt')
    parser.add_argument('--yolo-labels', type=str, default='weights/yolo_label.txt')
    parser.add_argument('--no-yolo', action='store_true')
        
    # [요청하신 부분] bbox 조절 인자들
    parser.add_argument('--bbox-threshold', type=float, default=0.01, help='일반 물체 인식 최소 크기 (기본 1%)')
    parser.add_argument('--traffic-threshold', type=float, default=0.005, help='신호등 인식 최소 크기 (기본 0.5%)')

    args = parser.parse_args()

    image_source = ImageSource(source_type=args.source, camera_id=args.camera)

    auto_drive = AutoDrive(
        use_motor=not args.no_motor,
        show_debug=not args.no_display,
        speed=args.speed,
        model_path=args.model,
        steering_gain=args.steering_gain,
        record_video=args.record,
        output_path=args.output,
        yolo_model_path=args.yolo_model,
        yolo_labels_path=args.yolo_labels,
        bbox_size_threshold=args.bbox_threshold,
        traffic_threshold=args.traffic_threshold, # [추가됨]
        use_yolo=not args.no_yolo
    )

    auto_drive.run(image_source)


if __name__ == "__main__":
    main()