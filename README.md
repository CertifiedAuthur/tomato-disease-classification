## Tomato Disease Classification API
#### Problem Statement
Tomato plants are susceptible to a range of diseases that can significantly affect crop yield and quality. Identifying these diseases early is crucial for effective intervention and management. This project addresses the problem by providing an automated solution for classifying tomato plant diseases using deep learning. The goal is to build an end-to-end system that allows users to upload images of tomato leaves and receive predictions about potential diseases, streamlining the diagnostic process for farmers and agricultural experts.

#### Project Overview
This project involves the development of a FastAPI-based web application that interacts with a TensorFlow Serving model for tomato disease classification. The system is designed to process images of tomato leaves, send them to a TensorFlow Serving instance for prediction, and return the classification results to the user. The end-to-end process includes model training, deployment, API development, testing, and final deployment to Google Cloud Platform (GCP).

#### Features

Image Upload: Users can upload images of tomato leaves through a REST API.

Disease Prediction: The image is processed and sent to TensorFlow Serving for classification.

Prediction Response: Users receive the predicted disease class and confidence score.

Model

The project uses a pre-trained deep learning model for tomato disease classification. The model identifies the following diseases:

Tomato Bacterial Spot,

Tomato Early Blight,

Tomato Late Blight,

Tomato Leaf Mold,

Tomato Septoria Leaf Spot,

Tomato Spider Mites (Two-spotted Spider Mite),

Tomato Target Spot,

Tomato Yellow Leaf Curl Virus,

Tomato Mosaic Virus,

Tomato Healthy.

#### Development and Testing Stages

Model Training and Export

Model Training: The model was trained using a dataset of labeled tomato leaf images. The training involved preprocessing the images, defining the model architecture, and optimizing hyperparameters.

Model Export: The trained model was saved in the .h5 format. It was then converted into the TensorFlow Serving format to be compatible with TensorFlow Serving.

import tensorflow as tf

model = tf.keras.models.load_model('path_to_trained_model/tomatoes.h5')

model.save('path_to_export_model/saved_model')

##### TensorFlow Serving Setup
Docker Container: TensorFlow Serving was set up using a Docker container. The model directory was mounted into the container, and TensorFlow Serving was configured to serve the model.

docker run -d -p 8501:8501 -v /path/to/your/models:/models tensorflow/serving --rest_api_port=8501 --model_config_file=/models/model.config

Model Configuration: A model.config file was created to specify the model path and versioning. This file is used by TensorFlow Serving to locate and serve the model.

#### FastAPI Development
API Development: A FastAPI application was developed to handle image uploads, send them to TensorFlow Serving, and return predictions.

from fastapi import FastAPI, File, UploadFile

import numpy as np

from io import BytesIO

from PIL import Image

import requests

app = FastAPI()

endpoint = "http://localhost:8501/v1/models/tomatoes_model:predict"

CLASS_NAMES = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', ...]  # Complete list of classes

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")

async def predict(file: UploadFile = File(...)):

    image = read_file_as_image(await file.read())
    
    img_batch = np.expand_dims(image, 0)
    
    json_data = {"instances": img_batch.tolist()}
    
    response = requests.post(endpoint, json=json_data)
    
    prediction = np.array(response.json()["predictions"][0])
    
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    
    confidence = np.max(prediction)
    
    return {'class': predicted_class, 'confidence': float(confidence)}

if __name__ == "__main__":

    uvicorn.run(app, host="localhost", port=8080)

#### Testing
Local Testing:

Postman: The API was tested using Postman to ensure that image uploads and predictions worked correctly.
FastAPI Testing: The FastAPI application was tested locally to verify that it correctly interacts with TensorFlow Serving.

#### React JS Frontend:

A React JS frontend was developed to provide a user interface for uploading images and displaying predictions. The frontend communicates with the FastAPI backend to send images and receive predictions.

#### GCP Deployment:

Deployment: The FastAPI application and TensorFlow Serving were deployed to Google Cloud Platform (GCP) for production use. This involved setting up virtual machines or managed services for hosting the application and serving the model.

Getting Started

Prerequisites

Python 3.7+

Docker (for TensorFlow Serving)

GCP Account (for deployment)

Installation

#### Clone the Repository:

git clone https://github.com/CertifiedAuthur/tomato-disease-classification-api.git

cd tomato-disease-classification-api

#### Create and Activate Virtual Environment:

python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`

#### Install Dependencies:

pip install -r requirements.txt

#### Run TensorFlow Serving:

docker run -d -p 8501:8501 -v /path/to/your/models:/models tensorflow/serving --rest_api_port=8501 --model_config_file=/models/model.config

#### Run the FastAPI Application:

python main_tf_serving.py

#### API Usage

Endpoint: /predict

Method: POST

Content-Type: multipart/form-data

Request: Upload an image file with the form-data key file.

Example Request

Using curl:

curl -X POST "http://localhost:8080/predict" -F "file=@path_to_your_image.jpg"

Example Response

json

{

  "class": "Tomato_Early_blight",
  
  "confidence": 0.87
  
}

#### Project Structure

main_tf_serving.py: FastAPI application for handling image uploads and predictions.

requirements.txt: Python dependencies.

model.config: Configuration for TensorFlow Serving.

Dockerfile (if applicable): Docker configuration.

README.md: Project documentation.

#### Troubleshooting

FileNotFoundError: Verify model paths in TensorFlow Serving and FastAPI configurations.

TensorFlow Serving Errors: Check Docker container logs for issues related to model serving.

API Errors: Review FastAPI logs and request responses for debugging.

#### Contributing

Contributions are welcome! Please submit pull requests or open issues for any bugs or feature requests.

#### License
This project is licensed under the MIT License - see the LICENSE file for details.

#### Acknowledgments

TensorFlow Serving for model deployment.

FastAPI for creating the API.

Docker for containerization.

Google Cloud Platform for deployment.
