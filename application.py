from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import json
from flask import (
    Flask,
    g,
    render_template,
    request,
    session,
    flash,
    redirect,
    url_for,
    abort,
    jsonify,
)

# create and initialize a new Flask app
application = Flask(__name__)

@application.route("/")
def index():
    return ('Hello!')

@application.route("/prediction", methods=['GET', 'POST'])
def load_model():
    # model loading
    loaded_model = None
    with open('basic_classifier.pkl', 'rb') as fid:
        loaded_model = pickle.load(fid)

    vectorizer = None
    with open('count_vectorizer.pkl', 'rb') as vd:
        vectorizer = pickle.load(vd)

    if request.method == 'POST':
        input = request.json.get('input')
        
        if input is None:
            return "Input is required", 400

        # Using model to predict
        prediction = loaded_model.predict(vectorizer.transform([input]))[0]
        
        # returning 'FAKE' if fake, 'REAL' if real
        return prediction

    return "Need POST Request", 405

if __name__ == '__main__':
    application.run(port=5000, debug=True)