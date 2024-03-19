import torch 
from ultralytics import YOLO 
#from ultralytics.yolo.utils.get_model_info import get_model_info 

device = torch.device("cuda")
model_name = "./ultralytics/cfg/models/tinyissimo/tinyissimo-v8.yaml"
model = YOLO(model_name) 

img_size = 256
input_size = (1, 1, img_size, img_size)  
 

# Size
# get_model_info(model.model.model)
 
# Train
model.train(data="RATS.yaml",  project="results", name="exp", optimizer='SGD',  imgsz=img_size,  epochs=100,  batch=64)

# Evaluate
results = model(imgsz=[img_size,img_size], max_det=3, show_labels=True,  project="./", source="/datasets/pbonazzi/img_rats/val/images",   conf=0.3,  save=True)

model.export(format="onnx", project="results", name="exp", imgsz=[img_size,img_size]) 