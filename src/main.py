# src/main.py
# Flask app serving predictions for the Wine dataset model

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__, static_folder="statics")

# Path to the saved model bundle
MODEL_PATH = "model/wine_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"{MODEL_PATH} not found. Run `python src/model_training.py` first "
        "or build the Docker image so Stage 1 can train the model."
    )

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]

# We used 4 features in model_training.py
FEATURE_NAMES = ["alcohol", "malic_acid", "color_intensity", "proline"]
FEATURE_LABELS = [
    "Alcohol",
    "Malic Acid",
    "Color Intensity",
    "Proline"
]

CLASS_LABELS = [
    "Wine Class 0",
    "Wine Class 1",
    "Wine Class 2"
]


@app.route("/", methods=["GET"])
def home():
    # Pass user-friendly labels to the template
    return render_template("predict.html", feature_labels=FEATURE_LABELS)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form

        # Extract values in the right order
        values = [
            float(data.get("alcohol")),
            float(data.get("malic_acid")),
            float(data.get("color_intensity")),
            float(data.get("proline")),
        ]

        arr = np.array(values).reshape(1, -1)
        arr_scaled = scaler.transform(arr)

        pred_idx = int(model.predict(arr_scaled)[0])
        predicted_class = CLASS_LABELS[pred_idx]

        return jsonify({"predicted_class": predicted_class})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # same port as before so Dockerfile/HOWTO still work
    app.run(debug=True, host="0.0.0.0", port=4000)
