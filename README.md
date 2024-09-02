# Setup
```
pip install -e .
pip install torch torchvison pyyaml
datasets의 yaml의 train, vaild, test 절대 경로로 수정
```
# Folder Structure
```
datasets
├── KITTI-3 
│   ├── test 
│   ├── train
│   ├── valid 
└───└── data.yaml              # 학습 및 평가을 위한 yaml 파일

yolov8_pruning
├── dataset                    
├── bigmodel                   # big model
├── compress                   # 압축에 필요한 코드 구현
├── ultralytics                # for yolov8
├── weights                    # 학습된 모델 weight 폴더
└── checkpoints                # 학습 시 모델 저장을 위한 폴더
```

# Usage
## before model training
```
python train_baseline_model.py --name before_big --bs 64 --epoch 300 --device 0 --data ../datasets/Nextchip/Nextchip.yaml
```
### Hyperparameters
- ```--model_pt``` : pretrained weight file or 학습 재개를 위한 모델 weight file (default='./weights/yolov8n.pt')
- ```--bs``` : Batch Size (default=64)
- ```--epoch``` : fine-funing epoch (default=150)
- ```--name``` : 학습 시 결과 및 model 저장할 폴더 이름. (default=test ex. checkpoints/test에 저장)
- ```--resume``` : 학습 재개 여부
- ```--data``` : 학습에 사용될  data yaml file (default='coco.yaml')
- ```--device``` : 학습 및 평가에 사용할 device 지정(ex.single GPU (--device 0), multiple GPUs (--device 0,1), CPU (--device cpu))  (default=0)

## Pruning and finetuning
```
python prune_finetune.py --bmodel big_model_path.pt --compress_ratio 0.6 --prune_type H --method L1 --epoch 300 --device 0 --bs 64 --name prune
```
### Hyperparameters
- ```--bmodel``` : 압축할 모델 weight file
- ```--compress_ratio``` : pruning 시 big model 압축 비율 (default=0.6)
- ```--compress_type``` : pruning 할 big model 구성 요소 선택 (default=H)
- ```--method``` : pruning method 선택 (default=H)
- ```--epoch``` : fine-funing epoch (default=150)
- ```--device``` : 학습 및 평가에 사용할 device 지정(ex.single GPU (--device 0), multiple GPUs (--device 0,1), CPU (--device cpu))  (default=0)
- ```--name``` : 학습 시 결과 및 model 저장할 폴더 이름. (default=test ex. checkpoints/test에 저장)
- ```--bs``` : Batch Size (default=64)
- ```--resume_path``` : 중단된 compress model의 학습 재개를 위한 last weight 파일

## Evaluation
```
python eval.py --model < weight.pt > --data < data.yaml >
```
### Hyperparameters
- ```--model``` : 평가할 모델 종류 weight.pt 파일
- ```--data```  : 평가할 dataset yaml 파일


