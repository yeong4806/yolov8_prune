from ultralytics import YOLO
from compress.Compress import PruneHandler as ph
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bmodel', type=str)
parser.add_argument('--compress_ratio', type=float, default='0.5')
parser.add_argument('--prune_type', type=str, default='H', help="H or B or ALL")
parser.add_argument('--method', type=str, default='L2', help="L1 or L2 or GM")
parser.add_argument('--cfg_output_path', type=str, default='test')
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--name', type=str, default='test')
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--resume_path', type=str)
parser.add_argument("--device", type=str, default='0,1')

args = parser.parse_args()
if not args.resume_path:
    bigmodel = YOLO(args.bmodel)
    ph = ph(bigmodel, args.compress_ratio, args.method, args.cfg_output_path, args.prune_type)
    cmodel = ph.compress_yolov8()

    cmodel.train(data="/home/yjlee/yolov8_prune/datasets/KITTI-3/data.yaml", epochs=args.epoch, imgsz=640, resume=False,
                 device=args.device, name=args.name, batch=args.bs, workers=8,
                 project='checkpoints')
else:
    cmodel = YOLO(args.resume_path)
    cmodel.train(data="/home/yjlee/yolov8_prune/datasets/KITTI-3/data.yaml", epochs=50, imgsz=640, device=args.device, name=args.name, batch=args.bs, workers=8, resume=True, project='checkoints')
