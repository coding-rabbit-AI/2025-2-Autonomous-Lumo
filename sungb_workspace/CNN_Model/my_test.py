import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm

from model import SteeringModel
from utils import SteeringDataset
import config
from config import *
import torchvision.transforms as transforms

# --- 설정 ---
MODEL_PATH = "train/exp2/best_model.pth"  # 모델 경로 확인!

# ------------

def calculate_metrics():
    # 1. 데이터셋 준비 (학습 때와 동일하게)
    def crop_bottom(img):
        img = img.resize((config.RESIZE_WIDTH, config.RESIZE_HEIGHT))
        return img.crop((0, 120, 320, 180))

    transform = transforms.Compose([
        transforms.Lambda(crop_bottom),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = SteeringDataset(LABELS_CSV, DATASET_DIR, transform)
    
    # 테스트 데이터만 분리 (전체의 20%)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_set = random_split(dataset, [train_size, test_size])
    
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0, drop_last=True)

    # 2. 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SteeringModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print(f"📊 모델 평가 중... (테스트 데이터: {len(test_set)}개)")

    # 3. 예측 시작
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, angles in tqdm(test_loader):
            imgs = imgs.to(device)
            outputs = model(imgs).squeeze()
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(angles.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 4. 지표 계산
    # (1) MAE: 평균적으로 몇 도 틀렸는지
    mae = mean_absolute_error(all_labels, all_preds)
    
    # (2) R2 Score: 데이터의 경향을 얼마나 잘 설명하는지 (1.0이 만점)
    r2 = r2_score(all_labels, all_preds)
    
    # (3) Custom Accuracy: 오차 5도 이내면 '정답'으로 인정
    diff = np.abs(all_preds - all_labels)
    acc_5deg = np.mean(diff <= 5.0) * 100  # 5도 이내
    acc_10deg = np.mean(diff <= 10.0) * 100 # 10도 이내

    print("\n" + "="*30)
    print(f"   🚗 AI Driver 성적표   ")
    print("="*30)
    print(f"1. 평균 오차 (MAE)      : {mae:.2f} 도")
    print(f"   (평균적으로 {mae:.1f}도 정도 빗나감)")
    print("-" * 30)
    print(f"2. 운전 싱크로율 (R2)   : {r2:.2f} / 1.0")
    print(f"   (1.0에 가까울수록 사람과 똑같음)")
    print("-" * 30)
    print(f"3. 정확도 (Accuracy)    ")
    print(f"   - 5도 이내 합격률    : {acc_5deg:.1f}%")
    print(f"   - 10도 이내 합격률   : {acc_10deg:.1f}%")
    print("="*30)

if __name__ == "__main__":
    try:
        calculate_metrics()
    except Exception as e:
        print(f"에러 발생: {e}")
        print("팁: pip install scikit-learn 을 했는지 확인하세요!")