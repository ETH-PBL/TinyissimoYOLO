import torch 
from ultralytics import YOLO 
import os

device = torch.device("cuda")
model_name = "./ultralytics/cfg/models/tinyissimo/tinyissimo-v8.yaml" 
model = YOLO(model_name).load("results/exp5/weights/best.pt")
 
output_dir = "result_images_VOC_200"
os.makedirs(output_dir, exist_ok=True)

results = model([f"gap9_tests/{i}.png" for i in range(1, 28)])

i = 0
for result in results:
    output_path = os.path.join(output_dir, "result" + str(i) + ".png")
    result.save(filename=output_path)  # save to disk
    i += 1

print(f"Results saved in {output_dir}")