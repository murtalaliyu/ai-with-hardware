"""
PetScope — web UI for the Chapter 3 cats vs dogs CNN.
Loads best_model.keras and serves an upload + predict interface.
"""
import io
import os

from flask import Flask, jsonify, render_template, request
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

IMG_SIZE = (128, 128)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.keras')
# Below this confidence, surface an "uncertain" hint (binary models always pick a class)
CONFIDENCE_HINT_THRESHOLD = 0.65

# Explicit folders — required because the script filename has hyphens, which breaks
# Flask's default root-path detection for static/ and templates/
app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, 'static'),
    template_folder=os.path.join(BASE_DIR, 'templates'),
)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8 MB uploads
app.config['TEMPLATES_AUTO_RELOAD'] = True

model = None


def get_model():
    """Lazy-load the saved Keras model once."""
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No model at '{MODEL_PATH}'. Train via ch3-image-classification.py first."
            )
        model = load_model(MODEL_PATH)
    return model


def predict_image_bytes(file_bytes):
    """Run the same preprocessing as the CLI and return label + confidence."""
    img = load_img(io.BytesIO(file_bytes), target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = img_array.reshape(1, *IMG_SIZE, 3)

    dog_prob = float(get_model().predict(img_array, verbose=0)[0][0])
    if dog_prob > 0.5:
        label = 'Dog'
        confidence = dog_prob
    else:
        label = 'Cat'
        confidence = 1.0 - dog_prob

    return {
        'label': label,
        'confidence': round(confidence, 4),
        'dog_probability': round(dog_prob, 4),
        'uncertain': confidence < CONFIDENCE_HINT_THRESHOLD,
    }


@app.route('/')
def index():
    return render_template('ch3-image-classification-index.html')


@app.route('/api/status')
def status():
    ready = os.path.exists(MODEL_PATH)
    loaded = model is not None
    return jsonify({
        'model_file_present': ready,
        'model_loaded': loaded,
        'model_path': os.path.basename(MODEL_PATH),
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided. Use form field name "image".'}), 400

    file = request.files['image']
    if not file or not file.filename:
        return jsonify({'error': 'Empty filename.'}), 400

    allowed = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({'error': f'Unsupported file type "{ext}". Use JPG, PNG, or similar.'}), 400

    try:
        result = predict_image_bytes(file.read())
        result['filename'] = file.filename
        return jsonify(result)
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 503
    except Exception as exc:
        return jsonify({'error': f'Prediction failed: {exc}'}), 500


if __name__ == '__main__':
    # Warm the model at startup so the first upload is snappy
    try:
        get_model()
        print(f"Loaded model from {MODEL_PATH}")
    except FileNotFoundError as exc:
        print(f"Warning: {exc}")
        print("The UI will start, but predictions will fail until you train a model.")

    app.run(host='127.0.0.1', port=5000, debug=False)
