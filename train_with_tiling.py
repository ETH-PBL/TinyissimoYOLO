
from ultralytics import YOLO

from ultralytics.nn.tasks import get_size
from ultralytics.utils.offline_tiling import Tiler

import wandb
wandb.init()
run = wandb.init(
    # Set the project where this run will be logged
    project="ultralytics-test",
    )

from ultralytics import YOLO

# Tiling
tiler = Tiler('tiling_config.yaml')
tiler.get_split_dataset()

# Load a model
model = YOLO('tinyissimoyolo.yaml')

# Train the model
model.train(data='CARPK_tiling.yaml',
            imgsz=256,
            epochs=1, 
            batch=64,
            single_cls=True)

# Size
get_size(model.model.model)

# Count the number of layers
num_layers = sum(1 for _ in model.model.model.modules()) - 1
print(f"Number of layers: {num_layers}")

# Count the number of parameters
num_params = sum(p.numel() for p in model.model.model.parameters())
print(f"Number of parameters: {num_params}")

model.export(format="onnx",imgsz=[256,256], opset=12)  # export the model to ONNX format