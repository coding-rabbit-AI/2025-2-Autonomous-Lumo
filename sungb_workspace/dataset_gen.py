import cv2
import numpy as np
import math
import os
import csv

# ==========================================
# [설정 구역]
VIDEO_PATH = "drive_data.mp4"  # 파일명 확인하세요!
SAVE_FOLDER = "dataset"
STRIDE = 1                           # 1이면 모든 프레임 저장 (데이터 최대화)
CUTOFF_SECONDS = 54                  # ★ 54초까지만 쓰고 자르기!
# ==========================================

def process_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"에러: {VIDEO_PATH} 파일을 찾을 수 없습니다.")
        return

    # 동영상의 초당 프레임 수(FPS) 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 54초가 몇 번째 프레임인지 계산 (예: 30fps * 54초 = 1620프레임)
    max_frame_index = int(fps * CUTOFF_SECONDS)

    print(f"영상 정보: FPS={fps:.2f}, 컷오프 시간={CUTOFF_SECONDS}초 (약 {max_frame_index} 프레임)")

    img_save_path = os.path.join(SAVE_FOLDER, "resize")
    os.makedirs(img_save_path, exist_ok=True)

    csv_path = os.path.join(SAVE_FOLDER, "labels.csv")
    
    # 파일 열기 ('w'모드: 기존 내용 지우고 새로 씀)
    f = open(csv_path, 'w', newline='') 
    wr = csv.writer(f)
    wr.writerow(["filename", "steering"])

    print(f"작업 시작... 초 지점에서 자동으로 멈춥니다)")
    saved_count = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # ★ 54초(계산된 프레임 수)를 넘어가면 반복문 종료!
        if frame_count > max_frame_index:
            print(f"{CUTOFF_SECONDS}초에 도달하여 작업을 종료합니다.")
            break

        if frame_count % STRIDE == 0:
            # 1. 전처리 (리사이즈 -> 흑백 -> 하단 자르기 -> 차선 검출)
            frame_resize = cv2.resize(frame, (320, 180))
            gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
            height, width = frame_resize.shape[:2]
            roi_height = height // 10
            roi_y = height - roi_height
            
            line_row = gray[roi_y:roi_y + 1, :]
            _, thresholded = cv2.threshold(line_row, 100, 255, cv2.THRESH_BINARY_INV)
            nonzero_x = np.nonzero(thresholded)[1]

            if len(nonzero_x) > 0:
                left_x = nonzero_x[0]
                right_x = nonzero_x[-1]
                center_x = (left_x + right_x) // 2

                # 2. 각도 계산
                angle = (math.degrees(math.atan(((320.0 - 2 * center_x) * 0.65) / 280))) * 3 
                # 각도 계산 후 저장 # (320 - 2*center_x) : 중심으로부터 얼마나 치우쳤는지 계산 atan으로 각도 변환 후 보정 계수 0.65, 3배 확대
                angle_deg = int(90 - angle)
                angle_deg = max(45, min(135, angle_deg)) # 범위 제한

                # --- [저장 1: 원본] ---
                fname_org = f"frame_{frame_count:06d}_org.jpg"
                cv2.imwrite(os.path.join(img_save_path, fname_org), frame_resize)
                wr.writerow([fname_org, angle_deg])
                saved_count += 1

                # --- [저장 2: 좌우 반전 (데이터 2배 뻥튀기)] ---
                frame_flip = cv2.flip(frame_resize, 1) 
                angle_flip = 180 - angle_deg 
                
                fname_flip = f"frame_{frame_count:06d}_flip.jpg"
                cv2.imwrite(os.path.join(img_save_path, fname_flip), frame_flip)
                wr.writerow([fname_flip, angle_flip])
                saved_count += 1

        frame_count += 1

    cap.release()
    f.close()
    print(f"✅ 완료! 총 {saved_count}장의 데이터가 생성되었습니다.")

if __name__ == '__main__':
    process_video()