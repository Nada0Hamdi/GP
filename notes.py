from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


app = Flask(__name__)

model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name,force_download=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name,force_download=True)
sentiment_analyzer = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)


def analyze_sentiment(text):
    
    result = sentiment_analyzer(text)
    
    score = result[0]['score']
    label = result[0]['label']
    
    if label == 'positive':
        return 2
    elif label == 'negative':
        return 0
    else:
        return 1

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment_route():

    text = request.json['text']

    score = analyze_sentiment(text)

    return jsonify({'score': score})

if __name__ == '__main__':
    app.run(debug=True)
 

