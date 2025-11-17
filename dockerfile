# ===========================
# Stage 1: Build & Train Model
# ===========================
FROM python:3.9 AS model_training

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy full src so training script can import everything it needs
COPY src /app/src

# Run training to produce model/wine_model.pkl
RUN python src/model_training.py


# ===========================
# Stage 2: Serving Stage
# ===========================
FROM python:3.9 AS serving

WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy trained model from Stage 1
COPY --from=model_training /app/model /app/model

# Copy application code (Flask app)
COPY src /app/src

# Set Flask app entry
ENV FLASK_APP=src/main.py

# Expose the port your Flask app uses
EXPOSE 4000

# Run the Flask server
CMD ["python", "src/main.py"]
