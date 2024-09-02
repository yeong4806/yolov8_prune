# !pip install roboflow

from roboflow import Roboflow

rf = Roboflow(api_key="qxYCCz31crhy7OnPlUNl")
project = rf.workspace("sebastian-krauss").project("kitti-9amcz")
version = project.version(3)
dataset = version.download("yolov8")
