import cv2
from ultralytics import YOLO
import time
import argparse

###########################################
# 욜로 모델 테스트용 스크립트
# 학습한 모델로 웹캠 또는 동영상에서 실시간 탐지 수행
###########################################
def test_yolo():
    # 1. 설정 파싱
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best.pt', help='학습된 모델 파일 경로')
    parser.add_argument('--source', type=str, default='0', help='0: 웹캠, 또는 동영상 파일 경로(test.mp4)')
    parser.add_argument('--conf', type=float, default=0.5, help='감지 임계값 (0.5 이상만 표시)')
    opt = parser.parse_args()

    # 2. 모델 로드
    print(f" 모델 로딩 중: {opt.weights}...")
    try:
        model = YOLO(opt.weights)
    except Exception as e:
        print(f"모델 로드 실패! 파일이 있는지 확인하세요. 에러: {e}")
        return

    # 3. 입력 소스 설정 (웹캠 or 동영상)
    source = opt.source
    if source.isnumeric():
        source = int(source) # 웹캠 번호(0)일 경우 숫자로 변환
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"카메라/영상을 열 수 없습니다: {source}")
        return

    # 젯슨 나노용 해상도 조절 (웹캠일 때만)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"YOLO 탐지 테스트 시작 (종료: 'q')")

    # FPS 계산용 변수
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("영상 종료")
            break

        # ------------------------------------------------------
        # [핵심] YOLO 추론 및 시각화
        # ------------------------------------------------------
        # conf: 확신도 설정, imgsz: 입력 이미지 크기 (작을수록 빠름)
        results = model.predict(frame, conf=opt.conf, imgsz=320, verbose=False)
        
        # 결과 그리기 
        annotated_frame = results[0].plot()

        # ------------------------------------------------------
        # FPS 표시 (성능 확인용)
        # ------------------------------------------------------
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 화면 출력
        cv2.imshow("YOLOv5 Detection Test", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_yolo()