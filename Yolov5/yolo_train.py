from ultralytics import YOLO
import torch
#############################################
# 다중 시드로 YOLOv5s 모델 학습 스크립트
#############################################
def run_training():
    # 학습할 시드 목록
    SEEDS = [1] 

    # 결과를 저장할 딕셔너리
    results_metrics = {} 

    print("Starting multi-seed training for YOLOv5s with exception handling...")

    for seed in SEEDS:
        try:
            print(f"\n===== Training Seed {seed} =====")
            # 시드 고정 (재현성을 위해)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            # YOLOv5n 모델 로드
            model = YOLO("yolov5nu.pt") # YOLOvnu 사전학습 모델 사용
            
            # 모델 학습
            train_results = model.train(
                data="C:\\Users\\sungb\\Documents\\lumo\\2025-2-Autonomous-Lumo\\Yolov5\\yolo5_setting.yaml",
                epochs=100,
                imgsz=640,
                batch=16,
                device=0,
                seed=seed,
                deterministic=True,
                project="yolov5_multi_test",
                name=f"seed_{seed}",
                workers=0 
            )
            
            # --- 2. 학습 성공 시 ---
            # 메트릭 저장 (Ultralytics 버전에 따라 속성 이름이 다를 수 있어 getattr 사용)
            metrics = getattr(train_results, 'results_dict', None) or getattr(train_results, 'metrics', {})
            results_metrics[seed] = metrics
            print(f" Seed {seed} training completed successfully.")

        except Exception as e:
            # --- 3. 학습 실패 시 ---
            print(f" ERROR: Seed {seed} training failed!")
            print(f"   Error details: {e}")
            print("   Skipping to the next seed...")

    print("\n🏁 All seed training attempts finished.")

# [핵심 수정 2] 윈도우 필수: 메인 실행 블록 보호
if __name__ == '__main__':
    run_training()