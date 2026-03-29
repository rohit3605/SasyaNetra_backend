import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from PIL import Image

# ---------- CLASS NAMES ----------
class_names = [
    "Bacterial leaf blight",
    "Brown spot",
    "Healthy leaf",
    "Leaf Blast",
    "Leaf scald",
    "Leaf smut",
    "Narrow brown spot",
    "Sheath Blight"
]

NUM_CLASSES = len(class_names)

# ---------- LOAD MODEL ----------
model = mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, NUM_CLASSES)

model.load_state_dict(torch.load("backend/crop_model.pth", map_location="cpu"))
model.eval()

# ---------- TRANSFORM ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------- PREDICT ----------
def predict_image(image):
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    confidence, predicted_class = torch.max(probabilities, 0)

    label = class_names[predicted_class.item()]
    confidence = confidence.item()

    if confidence < 0.5:
        return "No disease detected", confidence

    return label, confidence