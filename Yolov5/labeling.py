import torch
import cv2
import os
import numpy as np
from tqdm import tqdm

#################################
# YOLO를 이용한 데이터 라벨링 자동화 스크립트
################################

IMAGE_DIR = "C:\\Users\\sungb\\Downloads\\251123_traffic-images\\traffic_images\\train\\images"    # 라벨링할 사진 폴더
LABEL_DIR = "C:\\Users\\sungb\\Downloads\\251123_traffic-images\\traffic_images\\train\\labels"       # 라벨 저장할 폴더
CONF_THRES = 0.5                   # 정확도 50% 이상만 인정
IGNORE_BOTTOM_RATIO = 0.85         # 화면 하단 15% (0.85~1.0)에 있는 건 '손'으로 보고 무시
# ==========================================

def is_red_or_green(img_roi):
    """
    신호등 영역(ROI)을 잘라내서 빨간불인지 초록불인지 판별하는 함수
    Return: 1(Red), 2(Green), or None(판별불가)
    """
    hsv = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)
    
    # 1. 빨간색 범위 정의 (HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2
    
    # 2. 초록색 범위 정의 (HSV)
    lower_green = np.array([35, 70, 50])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # 픽셀 수 세기
    red_pixels = cv2.countNonZero(mask_red)
    green_pixels = cv2.countNonZero(mask_green)
    total_pixels = img_roi.shape[0] * img_roi.shape[1]
    
    if total_pixels == 0: return None

    # 비율 계산 (빨강이나 초록이 전체의 10% 이상일 때만 인정)
    red_ratio = red_pixels / total_pixels
    green_ratio = green_pixels / total_pixels
    
    if red_ratio > 0.1 and red_ratio > green_ratio:
        return 1  # Red Light (User ID: 1)
    elif green_ratio > 0.1 and green_ratio > red_ratio:
        return 2  # Green Light (User ID: 2)
    
    return None # 색깔이 안 보이면(꺼진 신호등 등) 무시

def run_auto_labeling():
    os.makedirs(LABEL_DIR, exist_ok=True)

    # 1. 고성능 모델 로드 (yolov5x)
    print("라벨링")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
    
    # COCO 클래스 기준: 0(사람), 9(신호등), 11(정지표지판)
    model.classes = [0, 9, 11] 

    # 이미지 목록 가져오기
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"총 {len(image_files)}장 처리 시작!")

    count = 0
    
    for img_file in tqdm(image_files):
        img_path = os.path.join(IMAGE_DIR, img_file)
        
        # 이미지 읽기 (OpenCV)
        img0 = cv2.imread(img_path)
        if img0 is None: continue
        h, w, _ = img0.shape

        # YOLO 추론
        results = model(img_path)
        
        detections = []
        
        # 감지된 물체 분석
        if len(results.xywhn[0]) > 0:
            for *xywh, conf, cls in results.xywhn[0]:
                if conf < CONF_THRES: continue
                
                # [필터링 1] 손(Hand) 제거: 중심점 y좌표가 화면 하단에 있으면 무시
                y_center = xywh[1].item()
                if y_center > IGNORE_BOTTOM_RATIO:
                    continue

                coco_class = int(cls)
                user_class = -1 # 초기화
                
                # [매핑 로직] COCO ID -> User ID 변환
                if coco_class == 0:     # COCO Person
                    user_class = 0      # -> User Person
                
                elif coco_class == 11:  # COCO Stop Sign
                    user_class = 3      # -> User Stop Sign
                
                elif coco_class == 9:   # COCO Traffic Light
                    # [색상 분석] 좌표를 픽셀 단위로 변환해서 자르기
                    x_c, y_c, bbox_w, bbox_h = xywh
                    x1 = int((x_c - bbox_w / 2) * w)
                    y1 = int((y_c - bbox_h / 2) * h)
                    x2 = int((x_c + bbox_w / 2) * w)
                    y2 = int((y_c + bbox_h / 2) * h)
                    
                    # 이미지 범위 벗어나지 않게 클램핑
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    roi = img0[y1:y2, x1:x2]
                    
                    if roi.size > 0:
                        color_id = is_red_or_green(roi)
                        if color_id is not None:
                            user_class = color_id # 1(Red) or 2(Green)
                        else:
                            continue # 색깔 구분 안 되면 저장 안 함 (선택사항)
                    else:
                        continue

                # 유효한 클래스면 리스트에 추가
                if user_class != -1:
                    line = f"{user_class} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}\n"
                    detections.append(line)

        # txt 파일 쓰기
        txt_path = os.path.join(LABEL_DIR, os.path.splitext(img_file)[0] + ".txt")
        with open(txt_path, 'w') as f:
            f.writelines(detections)
            
        if len(detections) > 0:
            count += 1

    print(f"\n 완료! {count}개 이미지 라벨링 끝.")
    print(f" 'labelImg'를 켜서 확인해보세요. (Speed Limit은 직접 추가해야 함)")

if __name__ == "__main__":
    run_auto_labeling()