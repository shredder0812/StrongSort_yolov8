from ultralytics import YOLO
import torch

model = torch.load('5-class-model.pt')

print(model)
