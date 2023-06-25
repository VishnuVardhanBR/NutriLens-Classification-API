# NutriLens Flask API with TensorFlow Lite Model

This repository contains a Flask API that loads a TensorFlow Lite model and provides inference capabilities for image classification tasks.

## Prerequisites

- Python 3.6+
- TensorFlow 2.6.0
- Flask 2.0.2
- OpenCV-Python 4.5.3.56

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repository.git
   ```

2. Create a virtual environment (optional, but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Place your TensorFlow Lite model file (`model.tflite`) in the `model/` directory.

5. Create a `labels.txt` file in the `model/` directory and populate it with the labels corresponding to your model's output classes, one label per line.

## Usage

1. Start the Flask API:

   ```bash
   python app.py
   ```

   The API will be accessible at `http://localhost:8000/predict`.

2. Send a POST request to `http://localhost:8000/predict` with an image file included in the request. The API will return a JSON response containing the predicted label and confidence.

```
