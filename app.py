from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import os
import re  # Add import for regular expressions

# Load models
label_encoder = joblib.load('label_encoder (1).joblib')
# Load tokenizer from local files
tokenizer = BertTokenizer.from_pretrained('./')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=8)
model.load_state_dict(torch.load('personality_model (1).pth', map_location=torch.device('cpu')))

model.eval()

# Create Flask App
app = Flask(__name__, static_folder="pic", static_url_path="/pic")

# Index page
@app.route('/')
def home():
    return render_template('index.html')

# Serve static files
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)

# Personality type to animal mapping
personality_to_animal = {
    'E': 'dolphin',
    'I': 'owl',
    'S': 'elephant',
    'N': 'eagle',
    'T': 'wolf',
    'F': 'dog',
    'J': 'bee',
    'P': 'fox'
}

def basic_preprocess(text):
    # Remove newline characters
    text = re.sub(r'\n', ' ', text)

    # tagging links
    text = re.sub(r'http\S+', ' ', text)

    # Remove repeating characters (3 or more occurrences)
    text = re.sub(r'([a-z])\1{2,}', r'\1', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Receive data from form and make prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['user_input']
        
        # Apply basic preprocessing to user input
        preprocessed_input = basic_preprocess(user_input)
        
        # Convert preprocessed data to tokens
        inputs = tokenizer(preprocessed_input, return_tensors='pt')
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            predicted_probability = probabilities[0][predicted_class].item()
        
        # Decode the predicted class
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        
        # Map personality type to animal
        predicted_animal = personality_to_animal.get(predicted_label, 'Unknown')
        
        return jsonify({
            'prediction': predicted_label,
            'probability': f"{predicted_probability * 100:.2f}%",
            'animal': predicted_animal
        })

if __name__ == '__main__':
    app.run(debug=True)
