import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm
import torchvision.transforms as transforms

# 기존 프로젝트 파일들 임포트 (파일 위치가 같아야 함)
from model import SteeringModel
from utils import SteeringDataset
import config

MODEL_PATH = "train/exp2/best_model.pth"      # 학습된 모델 경로
NEW_TEST_CSV = "test_dataset_1000/labels.csv"     # 새로운 데이터의 정답(라벨) 파일
NEW_TEST_IMG_DIR = "test_dataset_1000/resize/"            # 새로운 이미지가 들어있는 폴더
# ==================================================

def evaluate_new_data():
    # 0. 경로 
    if not os.path.exists(NEW_TEST_CSV):
        print(f"Error: 데이터 파일(을 찾을 수 없습니다.")
        return

    # 1. 데이터 전처리 
    def crop_bottom(img):
        img = img.resize((config.RESIZE_WIDTH, config.RESIZE_HEIGHT))
        return img.crop((0, 120, 320, 180))

    transform = transforms.Compose([
        transforms.Lambda(crop_bottom),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    print(f"데이터 로딩")
    
    # 데이터 전체를 데이터 셋으로 활용
    test_set = SteeringDataset(NEW_TEST_CSV, NEW_TEST_IMG_DIR, transform)
    # 데이터 로더 생성
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0, drop_last=False)
    # 2. 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SteeringModel().to(device)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: 모델 파일없음")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print(f"평가 시작")

    # 3. 예측 시작
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, angles in tqdm(test_loader):
            imgs = imgs.to(device)
            
            # 모델 예측
            outputs = model(imgs).squeeze()
            
            # 결과 저장
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(angles.numpy())

    # numpy 배열로 변환
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 4. 지표 계산 및 성적표 출력
    mae = mean_absolute_error(all_labels, all_preds) # 평균 오차 (MAE)
    r2 = r2_score(all_labels, all_preds) # r2 점수
    
    diff = np.abs(all_preds - all_labels)
    acc_5deg = np.mean(diff <= 5.0) * 100 #오차 5도
    acc_10deg = np.mean(diff <= 10.0) * 100 #오차 10도

    print("\n" + "="*40)
    print(f" 평가")
    print("="*40)
    print(f"1. 평균 오차 (MAE)       : {mae:.4f} 도")
    print(f"   (실제 핸들 각도와 평균 {mae:.2f}도 차이)")
    print("-" * 40)
    print(f"2. 운전 싱크로율 (R2)    : {r2:.4f}")
    print(f"   (최대 1.0, 높을수록 사람과 유사함)")
    print("-" * 40)
    print(f"3. 정확도 (Accuracy)")
    print(f"   - 오차 5도 이내 성공률 : {acc_5deg:.2f}%")
    print(f"   - 오차 10도 이내 성공률: {acc_10deg:.2f}%")
    print("="*40)
    
if __name__ == "__main__":
    try:
        evaluate_new_data()
    except Exception as e:
        print(f"\n 에러 발생: {e}")