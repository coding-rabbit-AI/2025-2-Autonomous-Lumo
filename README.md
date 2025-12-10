# 2025-2-Autonomous-Lumo
2025년 가을학기 수업 '자율주행과 인공지능' 팀 Lumo repository


Yolov5n 결과는 Yolov5\yolov5_result\seed_1 안에 있습니다.


2025-2-AUTONOMOUS-LUMO/
│
├── 📜 drive_final.py          # [메인] 자율주행 통합 실행 코드
├── 📜 README.md               # 프로젝트 설명
│
├── 📂 CNN_Model/              
│   ├── config.py              # 설정 값
│   ├── model.py               # 모델 아키텍처
│   ├── train.py               # 학습 실행
│   ├── inference.py           # 동영상 테스트
│   ├── utils.py               # 유틸리티 함수
│   ├── dataset_gen.py         # 동영상 분할, 데이터 라벨링 파일
│   └── model_test.py          # 모델 테스트
│
└── 📂 Yolov5/                 
    ├── yolo5_setting.yaml     # 욜로 설정 파일
    ├── yolo_train.py          # 첫번쨰 yolo train 
    ├── yolo_train_v2.py       # 두번째 yolo train 
    ├── yolo_opencv.py         # YOLO 모델 테스트용 파일
    └── labeling.py            # 데이터 자동 라벨링 도구