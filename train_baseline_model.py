from ultralytics import YOLO
import torch
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='baseline')
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--model_pt', type=str, default='./weights/yolov8n.pt')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--data', type=str, default="./datasets/KITTI-3/data.yaml")
parser.add_argument("--device", type=str, default='0,1')

args = parser.parse_args()

if torch.cuda.current_device() == 0:
    wandb.init(project='before_big', config=args)

model = YOLO(args.model_pt)
if not args.resume:
    model.train(data=args.data, epochs=args.epoch, imgsz=640, device=args.device, name=args.name,
                batch=args.bs, workers=8, save_period=5, project='checkpoints')
else:
    model.train(data=args.data, epochs=args.epoch, imgsz=640, device=args.device, name=args.name,
                batch=args.bs, workers=8, save_period=5, project='checkpoints', resume=args.resume)
wandb.finish()