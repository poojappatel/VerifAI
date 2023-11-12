from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer, BertModel, pipeline
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from model import BERT_Arch 
import requests
from torch.nn.functional import softmax

app = Flask(__name__)

def load_model(model_path):
    bert = BertModel.from_pretrained('bert-base-uncased')
    model = BERT_Arch(bert)  
    # Load the state dict with strict mode turned off
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict.pop("bert.embeddings.position_ids", None)  # Remove unexpected keys
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = load_model('fake_news_model.pt')

def preprocess(text):
    tokens = tokenizer.batch_encode_plus(
        [text],
        max_length=15,
        padding='max_length',
        truncation=True,
        return_tensors="pt"  # Ensure return type is PyTorch tensor
    )
    return tokens['input_ids'], tokens['attention_mask']

@app.route('/')
def index():
    # Render the HTML form
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        sentences = sent_tokenize(text)
        sentence_results = []

        for sentence in sentences:
            seq, mask = preprocess(sentence)
            with torch.no_grad():
                preds = model(seq, mask)
                preds = softmax(preds, dim=1)
                fake_prob = preds[0][1].item() * 100  # Probability of being fake
                real_prob = preds[0][0].item() * 100 # Probability of being real
                sentence_results.append((sentence, fake_prob, real_prob))

        return render_template('index.html', sentence_results=sentence_results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

