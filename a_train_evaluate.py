import torch 
from src import YOLO
            
def get_model_info(model):

    # Memory Footprint
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Model size: {:.3f}MB'.format(size_all_mb))

    # Number of layers
    num_layers = sum(1 for _ in model.modules()) - 1
    print(f"Number of layers: {num_layers}")

    # Number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

device = torch.device("cuda")
model_name = "./src/models/tinyissimo/tinyissimo-v8.yaml"
model = YOLO(model_name) 

img_size = 256
input_size = (1, 3, img_size, img_size)  
 

# Size
get_model_info(model.model.model)

# Train
model.train(data="VOC.yaml",  project="IOTDI2024", name=model_name,optimizer='SGD',  imgsz=img_size,  epochs=1,  batch=64)

# Evaluate
results = model(imgsz=[img_size,img_size], max_det=5, show_labels=True,  project="./", source="./test_dataset/26.png",   conf=0.25,  save=True)
model.export(format="onnx", project="IOTDI2024", name=model_name,imgsz=[img_size,img_size]) 