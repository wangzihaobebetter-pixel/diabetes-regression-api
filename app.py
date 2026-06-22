from flask import Flask, request, jsonify
import torch
import torch.nn as nn

app = Flask(__name__)

class DiabetesRegressor(nn.Module):
    def __init__(self, input_features=10):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.ReLU(),
            nn.Dropout(p=0.10),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

model = DiabetesRegressor()
model.load_state_dict(
    torch.load(
        "diabetes_model.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()

@app.route("/")
def home():
    return {
        "message": "Diabetes Regression API is running"
    }

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    features = data["features"]

    x = torch.tensor(
        [features],
        dtype=torch.float32
    )

    with torch.no_grad():
        prediction = model(x).item()

    return jsonify({
        "prediction": prediction
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
