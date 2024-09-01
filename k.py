import io
import json
import re

import joblib
import tensorflow as tf
from nltk.corpus import stopwords

with open('Helper/dataset_tokenizer.json') as f:
    data = json.load(f)
    loaded_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

lstm_model = tf.keras.models.load_model('Models/LSTM_SentimentAnalysisModel.keras')

stopwords_list = set(stopwords.words('english'))
TAG_RE = re.compile(r'<[^>]+>')


def remove_tags(text):
    return TAG_RE.sub('', text)


def preprocess_text(text):
    text = text.lower()
    # Remove html tags
    sentence = remove_tags(text)
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Remove multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    # Remove Stopwords
    pattern = re.compile(r'\b(' + r'|'.join(stopwords_list) + r')\b\s*')
    sentence = pattern.sub('', sentence)
    return sentence



text = preprocess_text('hello world')
tokenize_text = loaded_tokenizer.texts_to_sequences(text)
padded = tf.keras.preprocessing.sequence.pad_sequences(tokenize_text, padding='post', maxlen=100)
print(padded)
answer = lstm_model.predict(padded)
print(answer[0][0])
