import os
import torch
from torchvision import datasets, transforms
from flask import Flask, jsonify
from model import Net

"""
Flask App that Serves MNIST Predictions
Author: Tai Wan Kim
Date: November, 2025
"""

MODEL_DIR = "artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "mnist.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Load model once ----------
model = Net().to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# ---------- Load MNIST test set ----------
transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

app = Flask(__name__)

@app.route("/predict/<int:item_id>", methods=["GET"])
def predict(item_id):
    if item_id < 0 or item_id >= len(test_dataset):
        return jsonify({
            "error": f"item_id must be in [0, {len(test_dataset) - 1}]"
        }), 400

    image, true_label = test_dataset[item_id]  # image: [1, 28, 28]

    x = image.unsqueeze(0).to(device)  # -> [1, 1, 28, 28]

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    pred_label = int(torch.argmax(probs).item())

    return jsonify({
        "id": item_id,
        "predicted": pred_label,
        "true_label": int(true_label),
        "probs": [float(p) for p in probs.tolist()],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
