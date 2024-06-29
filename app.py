from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

# Load the trained model
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    prediction = model.predict([text])[0]
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    return jsonify({'text': text, 'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
