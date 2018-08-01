import os, io

from flask import request, jsonify, Flask
from google.cloud import storage
from sklearn.externals import joblib

from core import TextClassifier

GCS_BUCKET = os.environ['GCS_BUCKET']
GCS_BLOB = os.environ['GCS_BLOB']

app = Flask(__name__)

@app.before_first_request
def _load_model():
    global model

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(GCS_BLOB)

    if blob.exists():
        f = io.BytesIO()
        blob.download_to_file(f)

        model = joblib.load(f)

    else:
        model = None

@app.route('/fit', methods=['GET'])
def fit_model():
    tmp_filename = 'model.tmp'

    model = TextClassifier()
    model.fit()

    joblib.dump(model, tmp_filename)

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)

    if not bucket.exists():
        bucket = client.create_bucket(GCS_BUCKET)

    blob = bucket.blob(GCS_BLOB)

    with open(tmp_filename, 'rb') as f:
        blob.upload_from_file(f)

    return 'Model successfully fitted and dumped to gs://{}'.format(os.path.join(GCS_BUCKET, GCS_BLOB))

@app.route('/predict', methods=['POST'])
def predict_from_model():
    if not model:
        _load_model()
        if not model:
            return 'Model not found at gs://{}'.format(os.path.join(GCS_BUCKET, GCS_BLOB))

    in_text = request.get_json()['text']

    return jsonify(model.predict(in_text))

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
