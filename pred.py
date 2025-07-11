import requests
from io import BytesIO
from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
import json
import torch.nn as nn
import os


# Define the PyTorch model class
class ScratchPredictor(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 2048),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x


# Load model
model_path = "new_data.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device,weights_only=False)
model.eval()
model.to(device)

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Flask app setup
app = Flask(__name__)

# Mistral AI setup
template = """
*STRICTLY GIVE BACK JSON NOTHING EXTRA ,NO HEADING TO THE TEXT, NO JSON WRITTEN IN FRONT OF THE JSON*
Provide the following details about the plant '{plant_name}' in STRICT JSON format.
{{
    "name": "...",
    "description": "...",
    "uses": "...",
    "natural_medicinal_benefits": "...",
    "pharmaceutical_uses": "...",
    "chemical_composition": "...",
    "plant_height": "...",
    "locations_in_india": [...],
    "climate": {{
        "temperature": "...",
        "rainfall": "..."
    }},
    "pharmaceutical_usage": {{
        "product1": "percentage of the plant used",
        "product2": "percentage of the plant used"
    }},
    "soil_conditions": {{
        "type": "...",
        "pH": "...",
        "best_conditions": "..."
    }},
    "varieties": [...],
    "fertilizer_requirement": "...",
    "irrigation": {{
        "summer": "...",
        "rainy": "...",
        "winter": "..."
    }},
    "harvesting": {{
        "method": "...",
        "frequency": "..."
    }},
    "coordinates": [...]
}}
"""
model_mistral = ChatMistralAI(model="mistral-small-latest", temperature=0.3, api_key="2yOxSxH78ymcjLIzqAErjYoScvGKJyMw")


def get_plant_info(plant_name):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model_mistral
    response = chain.invoke({"plant_name": plant_name})

    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response from AI"}


import requests
from io import BytesIO
from PIL import Image


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or "image_url" not in data:
        return jsonify({"error": "No image URL provided"}), 400

    image_url = data["image_url"]

    try:
        # Fetch image from the internet
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()  # Raise error if request fails

        image = Image.open(BytesIO(response.content)).convert("RGB")
        image = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            _, predicted_class = torch.max(output, 1)

        class_names = model.class_names
        predicted_label = class_names[predicted_class.item()]

        plant_info = get_plant_info(predicted_label)
        return jsonify(plant_info)

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to download image: {str(e)}"}), 400

    except Exception as e:
        return jsonify({"error": f"Image processing error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
