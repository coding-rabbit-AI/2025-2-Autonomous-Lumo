# 🚗 End-to-End Self-Driving Steering Prediction

# **모델 path: (sungb_workspace\CNN_Model\train\exp2)**
## 0. Project Overview (개요)
본 프로젝트는 단일 카메라 영상(`drive_data.mp4`)을 입력받아 **자동차의 조향각(Steering Angle)을 예측하는 CNN 모델**을 구축하는 것을 목표로 합니다.

기존의 수동 라벨링 방식(사람이 직접 각도를 기록하는 방식) 대신, **Computer Vision (OpenCV) 알고리즘을 활용한 Auto-Labeling 시스템**을 구축하여 학습 데이터를 자동으로 생성했습니다. 이를 통해 라벨링 비용을 최소화하면서도, 도로의 차선을 인식하고 주행 경로를 판단하는 End-to-End 자율주행 모델을 구현했습니다.


---

## 1. Data Preprocessing (데이터 전처리)

학습 데이터는 주행 영상에서 프레임을 추출하고, 모델 학습에 최적화된 형태로 가공되었습니다.

### 1.1 Frame Extraction & Auto-Labeling
- **Source**: `drive_data.mp4` (전방 주행 영상)
- **Sampling**: `STRIDE=1` 설정을 통해 영상의 **모든 프레임**을 손실 없이 추출하여 데이터의 연속성을 확보했습니다.
- **Auto-Labeling (Self-Supervised)**:
  - OpenCV의 `Thresholding` 및 `Histogram` 분석을 통해 차선의 위치를 검출.
  - 차선 중심점과 화면 중앙의 편차를 계산하여 **조향각(Steering Angle)** 라벨을 자동 생성.
  - 유효하지 않은 데이터(정차 중이거나 차선 미검출 구간)를 제거하기 위해 **54초(Cutoff)** 지점에서 데이터 생성을 종료했습니다.
- **Resizing**: 원본 영상을 **320x180 (RGB)** 크기로 리사이즈하여 연산 효율성을 높였습니다.

### 1.2 Data Augmentation (데이터 증강)
데이터 부족 문제를 해결하고 모델의 일반화 성능을 높이기 위해 다음과 같은 증강 기법을 적용했습니다.
- **Horizontal Flip (좌우 반전)**:
  - 주행 데이터 특성상 좌회전/우회전 데이터의 불균형이 발생할 수 있음.
  - 이미지를 좌우로 반전시키고, 라벨(조향각) 또한 대칭 변환(`180 - angle`)하여 데이터 양을 **2배로 증강**했습니다.
  - 이를 통해 모델이 편향되지 않고 양방향 조향 능력을 학습하도록 유도했습니다.

### 1.3 Data information
- Datasize: **320x180 (RGB)**,
- Date num : 2162
- Data directory/데이터 경로 : sungb_workspace/CNN_Modle/dataset
---

## 2. CNN Architecture (모델 구조)

NVIDIA의 PilotNet 등 자율주행에 널리 쓰이는 구조를 참고하여 경량화된 CNN 모델을 설계했습니다.

### 2.1 Model Layers
모델은 **특징 추출(Feature Extraction)**을 위한 Convolutional Layer와 **회귀(Regression)**를 위한 Fully Connected Layer로 구성됩니다.

* **Input**: `(3, 60, 320)` - 전처리 단계에서 이미지 하단 60픽셀(ROI)만 크롭하여 입력.
* **Convolutional Layers (5 Layers)**:
    1.  `Conv2d(3 → 24, kernel=5, stride=2)` + ReLU
    2.  `Conv2d(24 → 36, kernel=5, stride=2)` + ReLU
    3.  `Conv2d(36 → 48, kernel=5, stride=2)` + ReLU
    4.  `Conv2d(48 → 64, kernel=3)` + ReLU
    5.  `Conv2d(64 → 64, kernel=3)` + ReLU
* **Fully Connected Layers (4 Layers)**:
    * `Flatten` → `Linear(100)` → `Linear(50)` → `Linear(10)` → `Output(1)`

### 2.2 Training Parameters
* **Epochs**: 50 (충분한 수렴을 위해 설정)
* **Batch Size**: 32 (Drop Last 적용으로 안정성 확보)
* **Learning Rate**: `1e-3` (Adam Optimizer)
* **Loss Function**: MSELoss (Mean Squared Error)

---

## 3. Evaluation (평가 및 결과)

### 3.1 Evaluation Methodology
학습된 모델의 성능을 검증하기 위해 전체 데이터의 **20%를 Test Set**으로 분리하여 평가를 진행했습니다. 평가 지표로는 회귀 문제에 적합한 MAE, R2 Score, 그리고 허용 오차 기반의 Custom Accuracy를 사용했습니다.

### 3.2 Quantitative Results (정량적 결과)

| Metric | Score | Description |
| :--- | :--- | :--- |
| **MAE (Mean Absolute Error)** | **2.07°** | 예측값과 정답 간의 평균 오차가 약 2도 내외로, 매우 정밀한 예측 성능을 보임. |
| **R2 Score** | **0.98** | 데이터에 대한 모델의 설명력이 98%로, OpenCV 알고리즘의 주행 로직을 완벽에 가깝게 모방함. |
| **Accuracy (±5°)** | **93.5%** | 오차 5도 이내를 정답으로 간주했을 때, 93.5%의 정확도를 달성. |

### 3.3 Qualitative Analysis
* **Loss Convergence**: 학습 초기 `Loss: 7000+`에서 시작하여 50 Epoch 후 `Loss: 16.5`까지 안정적으로 수렴하였습니다.
* **Inference**: 실제 주행 영상 테스트 결과, 직선 주행뿐만 아니라 곡선 구간에서도 차선을 이탈하지 않고 부드러운 조향각을 예측하는 것을 확인했습니다. (sungb_workspace\CNN_Model\inference\exp2\result.mp4)

---

© 2025 Autonomous Driving Project.