import torch 
from ultralytics import YOLO 
import wandb

load = False
exp_id = 'exp1'

version = 'v8'

if version == 'v1':
    print('Please, check to modify ultralytics/nn/modules/head/Detect')
    print('for TinyissimoYOLOv1.3 small and big change')
    print('line 36 to: self.reg_max=16')
    exit(1)

device = torch.device("cuda")
if load:
    model_name = f'./results/{exp_id}/weights/last.pt'
    model = YOLO(model_name) 
else:
    model_name = f"./ultralytics/cfg/models/tinyissimo/tinyissimo-{version}.yaml"
    model = YOLO(model_name) 

img_size = 224
input_size = (1, 1, img_size, img_size)  
 
# Train
model.train(data="coco.yaml",  project="results", name="exp", optimizer='SGD',  imgsz=img_size,  epochs=1000,  batch=512)

# Export
model.export(format="onnx", project="results", name="exp", imgsz=[img_size,img_size]) 
