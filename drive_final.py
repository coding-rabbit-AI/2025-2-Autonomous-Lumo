"""
자율주행 시스템 메인 모듈

이 모듈은 다음 기능들을 통합하여 자율주행을 수행합니다:
- AI 기반 조향 예측 (SteeringModel)
- YOLO 기반 객체 인식 (신호등, 표지판, 사람 등)
- LiDAR 기반 전방 장애물 감지
- 아두이노 모터 제어

주요 구성요소:
    MotorController: 아두이노와 시리얼 통신하여 모터 제어
    ImageSource: 다양한 입력 소스 지원 (파일, 카메라, 비디오)
    AutoDrive: 자율주행 메인 로직 통합
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
from queue import Queue, Empty

# ROS(LiDAR) 라이브러리 선택적 로드
try:
    import rospy
    from sensor_msgs.msg import LaserScan
    ROS_AVAILABLE = True
except ImportError:
    print("경고: ROS 라이브러리를 찾을 수 없습니다. LiDAR 기능이 비활성화됩니다.")
    ROS_AVAILABLE = False

from model import SteeringModel


class MotorController:
    """
    아두이노와 시리얼 통신을 통한 모터 제어 클래스

    조향과 속도 명령을 아두이노로 전송합니다.
    통신 프로토콜: STX-Length-Steering-Speed-Dummy-Checksum-ETX

    Attributes:
        ser: 시리얼 포트 객체 (연결 실패 시 None)
    """

    def __init__(self, port='/dev/arduino', baudrate=115200):
        """
        시리얼 포트 연결 초기화

        Args:
            port: 기본 시리얼 포트 경로
            baudrate: 통신 속도 (기본 115200)
        """
        self.ser = None
        ports_to_try = [port, '/dev/ttyUSB0', '/dev/ttyUSB1']

        # 여러 포트를 순차적으로 시도
        for p in ports_to_try:
            try:
                self.ser = Serial(p, baudrate, timeout=0.5)
                print(f"모터 컨트롤러 연결 성공: {p}")
                break
            except:
                continue

        if self.ser is None:
            print("모터 연결 실패. (시뮬레이션 모드로 동작)")
        time.sleep(1)

    def create_command(self, steering, speed):
        """
        모터 제어 명령 바이트 배열 생성

        Args:
            steering: 조향 각도 (45-135, 중립 90)
            speed: 속도 값 (90=정지, 100=전진)

        Returns:
            bytearray: 8바이트 명령 패킷
        """
        STX = 0xEA
        ETX = 0x03
        Length = 0x03
        dummy1 = 0x00
        dummy2 = 0x00
        Checksum = ((~(Length + steering + speed + dummy1 + dummy2)) & 0xFF) + 1
        return bytearray([STX, Length, steering, speed, dummy1, dummy2, Checksum, ETX])

    def send_command(self, steering, speed):
        """조향 및 속도 명령 전송"""
        if self.ser:
            self.ser.write(self.create_command(steering, speed))

    def stop(self):
        """모터 정지 (중립 위치)"""
        self.send_command(90, 90)

    def close(self):
        """모터 정지 후 시리얼 포트 닫기"""
        self.stop()
        if self.ser:
            self.ser.close()


class ImageSource:
    """
    다양한 입력 소스를 통합 처리하는 클래스

    지원 소스:
        - file: 이미지 파일 시퀀스 (*.jpg)
        - camera: 웹캠 실시간 입력
        - video: 비디오 파일 재생

    Attributes:
        source_type: 'file', 'camera', 'video' 중 하나
        cap: VideoCapture 객체 (camera/video 모드)
        image_files: 이미지 파일 경로 리스트 (file 모드)
        current_index: 현재 읽는 이미지 인덱스 (file 모드)
    """

    def __init__(self, source_type='file', file_pattern='sample/*.jpg', camera_id=0, video_path=None):
        """
        입력 소스 초기화

        Args:
            source_type: 입력 소스 타입
            file_pattern: 이미지 파일 패턴 (file 모드)
            camera_id: 카메라 디바이스 ID (camera 모드)
            video_path: 비디오 파일 경로 (video 모드)
        """
        self.source_type = source_type

        if source_type == 'file':
            self.image_files = sorted(glob.glob(file_pattern))
            if not self.image_files:
                raise ValueError(f"이미지 없음: {file_pattern}")
            self.current_index = 0
            self.loop = True
            print(f"이미지 파일 {len(self.image_files)}개 로드됨")

        elif source_type == 'camera':
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                raise ValueError(f"카메라 {camera_id} 연결 실패")
            print(f"카메라 {camera_id} 연결됨")

        elif source_type == 'video':
            if not video_path:
                raise ValueError("비디오 경로가 지정되지 않았습니다.")
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise ValueError(f"비디오 파일 열기 실패: {video_path}")
            print(f"비디오 파일 로드됨: {video_path}")

    def read_frame(self):
        """
        다음 프레임 읽기

        Returns:
            numpy.ndarray: BGR 이미지 프레임 (실패 시 None)
        """
        if self.source_type == 'file':
            if self.current_index >= len(self.image_files):
                if self.loop:
                    self.current_index = 0
                else:
                    return None
            img = cv2.imread(self.image_files[self.current_index])
            self.current_index += 1
            if img is None:
                return self.read_frame()  # 손상된 이미지 건너뛰기
            return img

        elif self.source_type == 'camera':
            ret, frame = self.cap.read()
            return frame if ret else None

        elif self.source_type == 'video':
            ret, frame = self.cap.read()
            if not ret:
                # 비디오 끝 도달 시 처음으로 되감기
                print("비디오 반복 재생")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            return frame if ret else None

    def release(self):
        """리소스 해제"""
        if self.source_type in ['camera', 'video']:
            self.cap.release()


class AutoDrive:
    """
    자율주행 시스템 통합 클래스

    AI 조향 예측, YOLO 객체 인식, LiDAR 센서, 모터 제어를 통합하여
    자율주행을 수행합니다.

    주요 기능:
        - AI 모델 기반 조향 각도 예측
        - YOLO를 통한 실시간 객체 감지 (비동기 처리)
        - 신호등/표지판/보행자 인식 및 정지 로직
        - LiDAR 기반 전방 장애물 감지
        - 아두이노 모터 제어

    Attributes:
        is_camera_stopped: 카메라 감지 기반 정지 상태
        is_lidar_stopped: LiDAR 감지 기반 정지 상태
        detected_objects: YOLO 감지 객체 리스트
        front_distance: LiDAR 전방 거리 (cm)
        speed: 현재 속도 설정값
    """

    def __init__(self, use_motor=True, show_debug=True, speed=100,
                 model_path='weights/best_model.pth', steering_gain=1.0,
                 record_video=False, output_path='output/drive_recording.mp4',
                 yolo_model_path='weights/yolo.pt', yolo_labels_path='weights/yolo_label.txt',
                 bbox_size_threshold=0.15, traffic_threshold=0.05, use_yolo=True):
        """
        자율주행 시스템 초기화

        Args:
            use_motor: 모터 제어 활성화 여부
            show_debug: 디버그 화면 표시 여부
            speed: 기본 속도 (90=정지, 100=전진)
            model_path: AI 조향 모델 경로
            steering_gain: 조향 민감도 (기본 1.0)
            record_video: 비디오 녹화 여부
            output_path: 녹화 파일 저장 경로
            yolo_model_path: YOLO 모델 경로
            yolo_labels_path: YOLO 레이블 파일 경로
            bbox_size_threshold: 일반 객체 인식 최소 크기 비율
            traffic_threshold: 신호등 인식 최소 크기 비율
            use_yolo: YOLO 사용 여부
        """
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

        # 객체 인식 크기 임계값 (일반 객체 vs 신호등)
        self.bbox_size_threshold = bbox_size_threshold
        self.traffic_threshold = traffic_threshold

        # 주행 상태 변수
        self.is_camera_stopped = False
        self.is_lidar_stopped = False
        self.stop_reason = ""
        self.detected_objects = []
        self.front_distance = 999.0

        # 적색 신호등 5초 정지 제어 (1회만)
        self.red_light_stop_start_time = None
        self.red_light_stop_duration = 5.0
        self.red_light_stopped_once = False

        # YOLO 비동기 처리 (메인 루프 blocking 방지)
        self.frame_queue = Queue(maxsize=2)
        self.yolo_lock = threading.Lock()
        self.yolo_thread = None

        # 조향 각도 설정
        self.angle_min = 45.0
        self.angle_max = 135.0
        self.angle_neutral = 90.0
        self.resize_w = 320
        self.resize_h = 180
        self.crop_y0 = 120

        # AI 모델 로드
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()

        # YOLO 모델 로드
        if self.use_yolo:
            self.yolo_model = self._load_yolo_model(yolo_model_path)
            self.yolo_labels = self._load_yolo_labels(yolo_labels_path)
        else:
            self.yolo_model = None

        # 모터 컨트롤러 연결
        if self.use_motor:
            self.motor = MotorController()
            if self.motor.ser is None:
                self.use_motor = False

        # ROS (LiDAR) 초기화
        if ROS_AVAILABLE:
            try:
                rospy.init_node('auto_drive_node', anonymous=True, disable_signals=True)
                rospy.Subscriber('/scan', LaserScan, self._lidar_callback, queue_size=1)
                print("LiDAR 연결 성공 (/scan)")
            except:
                print("ROS 초기화 실패")

        signal.signal(signal.SIGINT, self._signal_handler)

    def _lidar_callback(self, data):
        """
        LiDAR 센서 데이터 콜백 함수

        전방 중심 ±20도 범위의 최소 거리를 계산하여 장애물 감지
        히스테리시스(30cm/35cm)를 적용해 떨림 방지

        Args:
            data: LaserScan 메시지 (ROS)
        """
        center_deg = 180
        width_deg = 40
        min_dist = 200.0

        for i, r in enumerate(data.ranges):
            dist = r * 100  # 미터 -> 센티미터 변환
            if math.isinf(dist) or math.isnan(dist) or dist == 0:
                continue

            angle = math.degrees(data.angle_min + i * data.angle_increment)
            if angle < 0:
                angle += 360

            # 전방 중심 범위 내에 있는지 확인
            diff = abs(angle - center_deg)
            if diff > 180:
                diff = 360 - diff
            if diff <= width_deg / 2:
                min_dist = min(min_dist, dist)

        self.front_distance = min_dist

        # 히스테리시스: 30cm 이하 정지, 35cm 이상 해제
        if self.front_distance < 30.0:
            self.is_lidar_stopped = True
        elif self.front_distance > 35.0:
            self.is_lidar_stopped = False

    @torch.no_grad()
    def _load_model(self, path):
        """AI 조향 모델 로드"""
        model = SteeringModel().to(self.device)
        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
        except:
            state = torch.load(path, map_location=self.device)

        if isinstance(state, dict) and "state_dict" in state:
            sd = state["state_dict"]
        elif isinstance(state, dict) and "model_state" in state:
            sd = state["model_state"]
        else:
            sd = state

        model.load_state_dict(sd)
        model.eval()
        return model

    def _get_transform(self):
        """이미지 전처리 변환 파이프라인 생성"""
        return transforms.Compose([
            transforms.Resize((self.resize_h, self.resize_w)),
            transforms.Lambda(lambda img: img.crop((0, self.crop_y0, self.resize_w, self.resize_h))),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def _load_yolo_model(self, path):
        """YOLO 객체 인식 모델 로드"""
        print(f"YOLO 로드: {path}")
        return YOLO(path)

    def _load_yolo_labels(self, path):
        """YOLO 레이블 파일 로드 (형식: ID→Label)"""
        labels = {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if '→' in line:
                        p = line.split('→')
                        if len(p) == 2:
                            labels[int(p[0])] = p[1].strip()
            print(f"YOLO 레이블 {len(labels)}개 로드")
        except:
            pass
        return labels

    def _clamp_angle(self, angle):
        """조향 각도를 허용 범위(45-135)로 제한"""
        return max(self.angle_min, min(float(angle), self.angle_max))

    @torch.no_grad()
    def _predict_steering(self, frame):
        """
        AI 모델을 사용한 조향 각도 예측

        Args:
            frame: 입력 이미지 (BGR)

        Returns:
            float: 예측된 조향 각도 (45-135)
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        raw_angle = self.model(tensor).item()
        offset = (raw_angle - self.angle_neutral) * self.steering_gain
        return self._clamp_angle(self.angle_neutral + offset)

    def _detect_objects(self, frame):
        """
        YOLO를 사용한 객체 감지

        중복 제거 후 신뢰도가 가장 높은 1개 객체만 반환
        신호등은 별도 크기 임계값 적용

        Args:
            frame: 입력 이미지

        Returns:
            list: 감지된 객체 정보 [{label, bbox, conf, ratio}, ...]
        """
        if not self.use_yolo or self.yolo_model is None:
            return []

        # YOLO 추론 (agnostic_nms로 중복 박스 제거)
        results = self.yolo_model(frame, verbose=False, agnostic_nms=True, conf=0.40)
        candidates = []

        if results:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.yolo_labels.get(cls + 1, f"Class_{cls}")

                # 바운딩 박스 크기 비율 계산
                ratio = ((x2 - x1) * (y2 - y1)) / (frame.shape[0] * frame.shape[1])

                # 신호등은 별도 임계값 적용
                traffic_lights = ['Red Light', 'Green Light', 'Yellow Light', 'Yellow light']
                threshold = self.traffic_threshold if label in traffic_lights else self.bbox_size_threshold

                if ratio >= threshold:
                    candidates.append({
                        'label': label,
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'conf': conf,
                        'ratio': ratio
                    })

        # 신뢰도가 가장 높은 1개만 반환
        if candidates:
            candidates.sort(key=lambda x: x['conf'], reverse=True)
            return candidates[:1]

        return []

    def _yolo_worker_thread(self):
        """
        YOLO 비동기 처리 워커 스레드

        메인 루프와 독립적으로 YOLO 추론을 실행하여
        조향 예측에 영향을 주지 않도록 함
        """
        print("YOLO 스레드 시작")
        while self.running:
            try:
                # 큐에서 프레임 가져오기 (0.1초 대기)
                frame = self.frame_queue.get(timeout=0.1)

                # YOLO 객체 감지
                detected_objects = self._detect_objects(frame)

                # Thread-safe 결과 업데이트
                with self.yolo_lock:
                    self._update_driving_state(detected_objects)

            except Empty:
                continue
            except Exception as e:
                print(f"YOLO 스레드 오류: {e}")

        print("YOLO 스레드 종료")

    def _update_driving_state(self, objects):
        """
        감지된 객체 기반 주행 상태 업데이트

        정지 조건:
            - Red Light: 최초 1회만 5초 정지
            - Person, Stop Sign, Yellow Light: 즉시 정지

        속도 조절:
            - Speed Limit_80: 속도 102
            - Speed Limit_40: 속도 100

        Args:
            objects: YOLO 감지 객체 리스트
        """
        self.detected_objects = objects

        # 적색 신호등 5초 정지 (1회만)
        has_red_light = any(o['label'] == 'Red Light' for o in objects)
        red_light_should_stop = False

        if has_red_light and not self.red_light_stopped_once:
            if self.red_light_stop_start_time is None:
                # 타이머 시작
                self.red_light_stop_start_time = time.time()
                red_light_should_stop = True
            else:
                # 5초 경과 확인
                elapsed = time.time() - self.red_light_stop_start_time
                if elapsed < self.red_light_stop_duration:
                    red_light_should_stop = True
                else:
                    # 5초 경과 후 플래그 설정
                    self.red_light_stopped_once = True
                    red_light_should_stop = False

        # 기타 정지 조건
        other_stop_list = ['Person', 'Stop Sign', 'Yellow Light', 'Yellow light']
        other_stop_objs = [o['label'] for o in objects if o['label'] in other_stop_list]

        # 최종 정지 판단
        self.is_camera_stopped = red_light_should_stop or bool(other_stop_objs)

        # 정지 이유 문자열 생성
        stop_reasons = []
        if red_light_should_stop:
            remaining = self.red_light_stop_duration - (time.time() - self.red_light_stop_start_time)
            stop_reasons.append(f"Red Light ({remaining:.1f}s)")
        stop_reasons.extend(other_stop_objs)
        self.stop_reason = ", ".join(stop_reasons)

        # 속도 제한 표지판 처리
        if not self.is_camera_stopped:
            if any(o['label'] == 'Speed Limit_80' for o in objects):
                self.speed = 102
            elif any(o['label'] == 'Speed Limit_40' for o in objects):
                self.speed = 100
            else:
                self.speed = self.default_speed

    def _draw_debug_overlay(self, frame, angle):
        """
        디버그 정보가 표시된 이미지 생성

        Args:
            frame: 원본 프레임
            angle: 조향 각도

        Returns:
            numpy.ndarray: 디버그 정보가 오버레이된 이미지
        """
        debug = frame.copy()

        # Thread-safe 데이터 읽기
        with self.yolo_lock:
            detected_objects_copy = self.detected_objects.copy()
            is_camera_stopped = self.is_camera_stopped
            stop_reason = self.stop_reason

        # 감지된 객체 바운딩 박스 그리기
        for o in detected_objects_copy:
            x1, y1, x2, y2 = o['bbox']
            is_danger = o['label'] in ['Person', 'Red Light', 'Stop Sign', 'Yellow Light', 'Yellow light']
            color = (0, 0, 255) if is_danger else (0, 255, 0)
            cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
            cv2.putText(debug, f"{o['label']}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 주행 상태 텍스트 생성
        if self.is_lidar_stopped:
            status = "LiDAR STOP"
        elif is_camera_stopped:
            status = f"STOP: {stop_reason}"
        else:
            status = "DRIVING"

        color = (0, 0, 255) if "STOP" in status else (0, 255, 0)

        # 정보 표시
        cv2.putText(debug, f"Steer: {angle:.1f}", (10, 30), 1, 0.7, (0, 255, 0), 2)
        cv2.putText(debug, f"Speed: {self.speed}", (10, 60), 1, 0.7, (0, 255, 0), 2)
        cv2.putText(debug, f"LiDAR: {self.front_distance:.1f}cm", (10, 90), 1, 0.7, (255, 255, 0), 2)
        cv2.putText(debug, status, (10, 120), 1, 0.7, color, 2)

        return debug

    def _signal_handler(self, signum, frame):
        """시그널 핸들러 (Ctrl+C 감지)"""
        self.running = False

    def _init_video_writer(self, frame_shape):
        """비디오 녹화 초기화"""
        if self.record_video and self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.output_path, fourcc, 10, (frame_shape[1], frame_shape[0])
            )

    def run(self, image_source):
        """
        자율주행 메인 루프

        동작 흐름:
            1. YOLO 워커 스레드 시작 (비동기 객체 감지)
            2. 매 프레임마다:
                - AI 조향 각도 예측 (blocking 없음)
                - 4프레임마다 YOLO 큐에 프레임 추가
                - LiDAR/카메라 정지 조건 확인
                - 모터 제어 명령 전송
                - 디버그 화면 표시 및 녹화

        Args:
            image_source: ImageSource 객체
        """
        print(">>> 자율주행 시작")
        print(f"> 설정: 일반객체크기={self.bbox_size_threshold}, 신호등크기={self.traffic_threshold}")

        # YOLO 비동기 처리 스레드 시작
        if self.use_yolo and self.yolo_model is not None:
            self.yolo_thread = threading.Thread(target=self._yolo_worker_thread, daemon=True)
            self.yolo_thread.start()

        frame_count = 0
        loop_count = 0

        try:
            while self.running and (not ROS_AVAILABLE or not rospy.is_shutdown()):
                frame = image_source.read_frame()
                if frame is None:
                    break

                # 4프레임마다 YOLO 큐에 프레임 추가 (성능 최적화)
                if self.use_yolo and loop_count % 4 == 0:
                    try:
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()  # 오래된 프레임 제거
                            except Empty:
                                pass
                        self.frame_queue.put_nowait(frame.copy())
                    except:
                        pass

                # AI 조향 예측 (매 프레임)
                steering_angle = self._predict_steering(frame)

                # 디버그 오버레이 생성
                debug_image = self._draw_debug_overlay(frame, steering_angle)

                if frame_count == 0 and self.record_video:
                    self._init_video_writer(debug_image.shape)

                # 정지 조건 확인 (LiDAR 우선)
                with self.yolo_lock:
                    is_camera_stopped = self.is_camera_stopped

                final_speed = 90 if self.is_lidar_stopped or is_camera_stopped else self.speed

                # 모터 제어
                if self.use_motor:
                    self.motor.send_command(int(steering_angle), final_speed)

                # 녹화 및 화면 표시
                if self.record_video and self.video_writer:
                    self.video_writer.write(debug_image)
                if self.show_debug:
                    cv2.imshow('Auto Drive', debug_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if image_source.source_type == 'file':
                    time.sleep(0.1)

                frame_count += 1
                loop_count += 1

        finally:
            self._cleanup(image_source)

    def _cleanup(self, image_source):
        """리소스 정리 및 종료 처리"""
        print("\n정리 중...")
        self.running = False

        # YOLO 스레드 종료 대기
        if self.yolo_thread and self.yolo_thread.is_alive():
            print("YOLO 스레드 종료 대기...")
            self.yolo_thread.join(timeout=2.0)

        if self.use_motor:
            self.motor.close()
        if self.video_writer:
            self.video_writer.release()

        image_source.release()
        cv2.destroyAllWindows()


def main():
    """
    자율주행 프로그램 진입점

    명령행 인자 처리 및 AutoDrive 인스턴스 실행

    주요 인자:
        --source: 입력 소스 (file/camera/video)
        --speed: 기본 속도 (90=정지, 100=전진)
        --model: AI 조향 모델 경로
        --yolo-model: YOLO 모델 경로
        --bbox-threshold: 일반 객체 인식 최소 크기 비율
        --traffic-threshold: 신호등 인식 최소 크기 비율
        --no-motor: 모터 제어 비활성화 (시뮬레이션 모드)
        --record: 비디오 녹화 활성화
    """
    import argparse
    parser = argparse.ArgumentParser(description='AI 기반 자율주행 시스템')

    # 입력 소스 설정
    parser.add_argument('--source', type=str, default='file',
                        help="입력 소스: 'file', 'camera', 'video'")
    parser.add_argument('--camera', type=int, default=0,
                        help='카메라 디바이스 ID')
    parser.add_argument('--video-file', type=str, default=None,
                        help='비디오 파일 경로 (source=video 시 필수)')

    # 주행 설정
    parser.add_argument('--no-motor', action='store_true',
                        help='모터 제어 비활성화')
    parser.add_argument('--speed', type=int, default=100,
                        help='기본 속도 (90=정지, 100=전진)')
    parser.add_argument('--steering-gain', type=float, default=1.0,
                        help='조향 민감도 (기본 1.0)')

    # 모델 설정
    parser.add_argument('--model', type=str, default='weights/best_model.pth',
                        help='AI 조향 모델 경로')
    parser.add_argument('--yolo-model', type=str, default='weights/yolo.pt',
                        help='YOLO 모델 경로')
    parser.add_argument('--yolo-labels', type=str, default='weights/yolo_label.txt',
                        help='YOLO 레이블 파일 경로')
    parser.add_argument('--no-yolo', action='store_true',
                        help='YOLO 비활성화')

    # 객체 인식 임계값
    parser.add_argument('--bbox-threshold', type=float, default=0.01,
                        help='일반 물체 인식 최소 크기 비율 (기본 1%%)')
    parser.add_argument('--traffic-threshold', type=float, default=0.005,
                        help='신호등 인식 최소 크기 비율 (기본 0.5%%)')

    # 디버그 및 녹화
    parser.add_argument('--no-display', action='store_true',
                        help='디버그 화면 비활성화')
    parser.add_argument('--record', action='store_true',
                        help='주행 영상 녹화')
    parser.add_argument('--output', type=str, default='output/drive_recording.mp4',
                        help='녹화 파일 저장 경로')

    args = parser.parse_args()

    # 입력 소스 초기화
    image_source = ImageSource(
        source_type=args.source,
        camera_id=args.camera,
        video_path=args.video_file
    )

    # 자율주행 시스템 초기화
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
        traffic_threshold=args.traffic_threshold,
        use_yolo=not args.no_yolo
    )

    # 자율주행 실행
    auto_drive.run(image_source)


if __name__ == "__main__":
    main()