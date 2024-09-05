#importing necessary packages
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import io
import requests
import wikipedia
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_recognition_model = models.resnet50(pretrained=True).to(device)
image_recognition_model.eval()

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
semantic_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize Wikipedia API with a user agent
wikipedia.set_lang('en')
user_agent = "YourAppName/1.0 (your.email@example.com)"
wikipedia.set_user_agent(user_agent)

## Fetch ImageNet labels
LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
labels_response = requests.get(LABELS_URL)
imagenet_labels = labels_response.json()

# Helper functions
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0).to(device)
    return image

def fetch_wikipedia_summary(title):
    try:
        summary = wikipedia.summary(title, sentences=2)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Disambiguation error: {e.options}"
    except wikipedia.exceptions.PageError:
        return "Page not found."

def answer_query(query, context):
    if context:
        result = qa_pipeline(question=query, context=context)
        return result['answer']
    return "Sorry, I couldn't find the answer to that."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found'}), 400

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    preprocessed_image = preprocess_image(image)
    
    with torch.no_grad():
        predictions = image_recognition_model(preprocessed_image)
        predicted_class = torch.argmax(predictions, dim=1).item()
    
    label = imagenet_labels[str(predicted_class)][1]
    wiki_summary = fetch_wikipedia_summary(label)
    
    return jsonify({
        'prediction': label,
        'wiki_summary': wiki_summary or f"Sorry, I couldn't find information about {label}."
    })

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query = data.get('query')
    context = data.get('context')
    
    if not query or not context:
        return jsonify({'error': 'Query and context are required.'}), 400
    
    answer = answer_query(query, context)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Specify the port here
