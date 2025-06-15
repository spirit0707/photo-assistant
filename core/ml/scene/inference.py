import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import os

PLACES365_WEIGHTS = os.path.join('data', 'resnet18_places365.pth.tar')
PLACES365_CLASSES = os.path.join('data', 'categories_places.txt')

with open(PLACES365_CLASSES) as f:
    PLACES365_SCENE_CLASSES = [line.strip().split(' ')[0][3:] for line in f.readlines()]

_places365_model = None
def get_places365_model():
    global _places365_model
    if _places365_model is None:
        model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
        checkpoint = torch.load(PLACES365_WEIGHTS, map_location='cpu')
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.fc = nn.Linear(model.fc.in_features, 365)
        model.load_state_dict(state_dict)
        model.eval()
        _places365_model = model
    return _places365_model

def classify_scene(image: Image.Image):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    model = get_places365_model()
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)
    scene_name = PLACES365_SCENE_CLASSES[top_catid[0]]
    return scene_name, float(top_prob[0]) 