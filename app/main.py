from flask import Flask, request, jsonify
from google.cloud import bigquery
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)
client = bigquery.Client()
analyzer = SentimentIntensityAnalyzer()

MODEL_NAME = 'willbigquery.willfinalproject.financialNewsML'

@app.route('/', methods=['GET'])
def hello_world():
    return 'hello world'

@app.route('/predict', methods=['POST'])
def predict_intraday():
    data = request.json
    title = data.get('title')

    if not title:
        return jsonify({'error': 'Please provide a title'}), 400

    # Vader Sentiment
    sentiment = analyzer.polarity_scores(title)
    sentiment_score = sentiment['compound']  

    query = f"""
    SELECT predicted_intraday, predicted_intraday_probs
    FROM ML.PREDICT(MODEL `{MODEL_NAME}`,
      (SELECT {sentiment_score} AS sentiment_score)
    )
    """

    query_job = client.query(query)
    result = query_job.result()

    prediction = None
    for row in result:
        prediction = {
            'sentiment_score': sentiment_score,
            'predicted_intraday': row.predicted_intraday,
            'probability': row.predicted_intraday_probs[1]
        }

    if prediction is None:
        return jsonify({'error': 'Prediction failed'}), 500

    return jsonify(prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

