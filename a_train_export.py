import torch 
from ultralytics import YOLO 

device = torch.device("cuda")
model_name = "./ultralytics/cfg/models/tinyissimo/tinyissimo-v8.yaml"
model = YOLO(model_name) 

img_size = 256
input_size = (1, 1, img_size, img_size)  
 
# Train
model.train(data="RATS.yaml",  project="results", name="exp", optimizer='SGD',  imgsz=img_size,  epochs=100,  batch=64)

# Export
model.export(format="onnx", project="results", name="exp", imgsz=[img_size,img_size]) 
