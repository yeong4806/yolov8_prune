import sys
sys.path.append("/home/yjlee/ultralytics")
from ultralytics import YOLO
import argparse

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='./weights/yolov8n.pt')
    parser.add_argument('--data', type=str, default="./datasets/KITTI-3/data.yaml")

    args = parser.parse_args()

    model = YOLO(args.model)

    metrics = model.val(data=args.data)
    print('mAP50:', metrics.box.map50)
