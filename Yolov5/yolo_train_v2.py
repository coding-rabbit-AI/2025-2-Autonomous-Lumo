from ultralytics import YOLO
import torch
##############################################
# 다중 시드로 YOLOv5s 모델 학습 스크립트
# 파라미터 관련 설정은 주석 참고
##############################################
def run_training():
    # 학습할 시드 목록 (필요시 늘릴것)
    SEEDS = [1] 

    # 결과를 저장할 딕셔너리
    results_metrics = {} 

    print("YOLO 모델 학습")

    for seed in SEEDS:
        try:
            print(f"\n===== Training Seed {seed} =====")
            
            # 1. 시드 고정 
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            # 2. 모델 로드
            model = YOLO("yolov5nu.pt") 
            
            # 3. 모델 학습 
            train_results = model.train(
                # [기본 설정]
                data=r"C:\Users\sungb\Documents\lumo\2025-2-Autonomous-Lumo\Yolov5\yolo5_setting.yaml",
                epochs=100,
                imgsz=640,
                batch=16,
                device=0,
                seed=seed,
                workers=0,
                
                # [Augmentation]
                
                # 1. 색상/혼합 (신호등 보호)
                mixup=0.0,       #  끄기 (색 섞임 방지)
                hsv_h=0.0,       #  끄기 (빨강->초록 변조 방지)
                hsv_s=0.6,       #  켜기 (조명 밝기 변화 대응)
                hsv_v=0.4,       #  켜기 (그림자/역광 대응)
                
                # 2. 기하학적 변형 (트랙 특성 반영)
                degrees=0.0,     #  끄기 (평지 트랙이므로 회전 불필요)
                translate=0.1,   #  켜기 (차가 트랙 중앙에서 약간 벗어날 때 대비)
                scale=0.5,       #  켜기 (멀리 있는 신호등 ~ 가까운 신호등)
                shear=0.0,       #  끄기 (이미지 찌그러트리기 불필요)
                perspective=0.0, #  끄기 (원근 왜곡 불필요)
                
                # 3. 방향/배경 
                fliplr=0.0,      #  끄기 (신호등 좌우 반전 절대 금지)
                mosaic=1.0,      #  필수 (배경 암기 방지, 작은 물체 인식 향상)
                
                # 4. 과적합 방지
                dropout=0.1,     
                freeze=10    # (Transfer Learning) 앞단 10개 층을 얼려서 적은 데이터 효율 극대화
            )
            
            # --- 학습 성공 후 메트릭 저장 ---
            metrics = getattr(train_results, 'results_dict', None) or getattr(train_results, 'metrics', {})
            results_metrics[seed] = metrics
            print(f" Seed {seed} training completed successfully.")

        except Exception as e:
            # --- 학습 실패 시 ---
            print(f" ERROR: Seed {seed} training failed!")
            print(f"   Error details: {e}")
            print("   Skipping to the next seed...")

    print("\n🏁 All seed training attempts finished.")

# 메인 실행 보호
if __name__ == '__main__':
    # GPU 캐시 정리 (메모리 확보)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    run_training()