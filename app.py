import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import io
import json
import joblib
import re
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
import k

app = Flask(__name__)
with open('Helper/dataset_tokenizer.json') as f:
    data = json.load(f)
    loaded_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

lstm_model = tf.keras.models.load_model('Models/LSTM_SentimentAnalysisModel.keras')
cnn_model = tf.keras.models.load_model('Models/CNN_SentimentAnalysisModel.keras')
# Load the saved Decision Tree Classifier
dt_loaded = joblib.load('Models/decision_tree_classifier.pkl')
# Load the saved TfidfVectorizer
tf_loaded = joblib.load('Helper/tfidf_vectorizer.pkl')

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


def text_processing_dt(text):
    # convert to lowercase
    text = text.lower()
    # remove non letters
    text = re.sub(r"[^a-zA-Z]", " ", text)
    # tokenize
    words = word_tokenize(text)
    # remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    # apply Lemmatizer
    lm = nltk.WordNetLemmatizer()
    words = [lm.lemmatize(w) for w in words]
    # return list
    sentence = ' '.join(words)
    return sentence


def analyze_sentiment2(text):
    text = preprocess_text(text)
    tokenize_text = loaded_tokenizer.texts_to_sequences(text)
    padded = tf.keras.preprocessing.sequence.pad_sequences(tokenize_text, padding='post', maxlen=100)
    answer = lstm_model.predict(padded)
    print(answer[0][0])
    if answer[0][0] > 0.55:
        return 'Positive', answer[0][0] * 100
    else:
        return 'Negative', answer[0][0] * 2 * 100


def analyze_sentiment(text, m):
    answer = 0
    if m == 'lstm':
        print(text)
        text = preprocess_text(text)
        print(text)
        tokenize_text = loaded_tokenizer.texts_to_sequences(text)
        padded = tf.keras.preprocessing.sequence.pad_sequences(tokenize_text, padding='post', maxlen=100)
        print(padded)
        answer = lstm_model.predict(padded)
    elif m == 'cnn':
        text = preprocess_text(text)
        tokenize_text = loaded_tokenizer.texts_to_sequences(text)
        padded = tf.keras.preprocessing.sequence.pad_sequences(tokenize_text, padding='post', maxlen=100)
        print(m)
        answer = cnn_model.predict(padded)
    else:
        stra = text_processing_dt(text)
        stra = tf_loaded.transform([stra])
        answer = dt_loaded.predict(stra)
        print(answer[0])
        return answer[0], 100

    print(answer[0][0])
    if answer[0][0] > 0.55:
        return 'Positive', answer[0][0] * 100
    else:
        return 'Negative', (answer[0][0] - 0.05) * 2 * 100


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


def generate_graph(data):
    plt.figure(figsize=(8, 6))
    data['Sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'blue'])
    plt.title('Sentiment Analysis')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    return 'data:image/png;base64,{}'.format(graph_url)


def generate_graph2(Y):
    threshold = 0.55
    classes = np.where(Y > threshold, 1, 0)
    positive_count = np.sum(classes)
    negative_count = len(classes) - positive_count
    plt.figure(figsize=(6, 4))
    plt.bar(['Negative', 'Positive'], [negative_count, positive_count], color=['red', 'green'])
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Analysis')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    model = request.form['modal']
    sentiment, confidence = analyze_sentiment(text, model)
    return jsonify({'sentiment': sentiment, 'confi': confidence})


@app.route('/analyze', methods=['POST'])
def analyze():
    print("analyzimg 1")
    file = request.files['file']
    target_column = request.form['target_column']
    print("analyzimg 2")
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file format. Only CSV files are allowed.'})
    try:
        df = pd.read_csv(file)
        if target_column not in df.columns:
            return jsonify({'error': f'Target column "{target_column}" not found in the CSV file.'})
        model = request.form['modal']
        if model == 'Decision Tree Classifier':
            print("analyzimg 3")
            df[target_column] = df[target_column].apply(text_processing_dt)
            print("Analyzing 4")
            stra = tf_loaded.transform(df[target_column])
            print("analyzimg")
            preddt = dt_loaded.predict(stra)
            pred_df = pd.DataFrame({'Sentiment': preddt})
            print("analyzed")
            # Now you can use the groupby() method
            plt.figure(figsize=(10, 9))
            pred_df.groupby('Sentiment').size().plot(kind='bar', color=['green'])
            plt.xlabel('Sentiment')
            plt.ylabel('Count')
            plt.title('Sentiment Analysis')

            # Save the plot to a BytesIO object
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            graph_url = base64.b64encode(img.getvalue()).decode()

            plt.close()  # Close the plot to free up memory

            # Return the base64 encoded image string
            graph = 'data:image/png;base64,{}'.format(graph_url)
            return jsonify({'graph': graph})

        X = []
        sentences = list(df[target_column])
        for sen in sentences:
            X.append(preprocess_text(sen))
        X = loaded_tokenizer.texts_to_sequences(X)
        X = tf.keras.preprocessing.sequence.pad_sequences(X, padding='post', maxlen=100)
        model = request.form['modal']
        if model == 'lstm':
            Y = lstm_model.predict(X)
        else:
            Y = cnn_model.predict(X)
        graph = generate_graph2(Y)
        return jsonify({'graph': graph})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
