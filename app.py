# from flask import Flask, render_template, request
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import tensorflow as tf
# import numpy as np
# import os
# from PIL import Image
# app = Flask(__name__)

# # Load model
# # model = tf.keras.models.load_model("new_mobilenet_mask_model.h5",compile=False)   # your .h5 file
# # model = load_model("new_mobilenet_mask_model.h5", compile=False)

# model = load_model("new_mobilenet_mask_model_updated.h5", compile=False)

# # model = load_model("fixed_mask_model.h5", compile=False)

# # model.save("new_mobilenet_mask_model_fixed", save_format="tf")


# # orig_get_config = tf.keras.layers.Layer.get_config
# # def fixed_get_config(self):
# #     config = orig_get_config(self)
# #     config.pop('batch_input_shape', None)
# #     config.pop('batch_shape', None)
# #     return config
# # tf.keras.layers.Layer.get_config = fixed_get_config
# # ==================================================
# # Home page
# @app.route('/')
# def home():
#     return render_template('index.html')

# # Predict route
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return "No file uploaded"

#     file = request.files['file']
#     if file.filename == '':
#         return "No file selected"

#     filepath = os.path.join("static", file.filename)
#     file.save(filepath)

#     # Preprocess image
#     img = image.load_img(filepath, target_size=(224, 224))  # size used while training
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    
#     # img = Image.open(file.stream).convert('RGB')
#     # img = img.resize((224, 224))
#     # img_array = np.array(img) / 255.0
#     # img_array = np.expand_dims(img_array, axis=0)

#     prediction = model.predict(img_array)
#     class_idx = np.argmax(prediction, axis=1)[0]

#     label = "With Mask" if class_idx == 0 else "Without Mask"

#     return render_template('result.html', label=label, user_image=filepath)



# if __name__ == "__main__":
#     app.run(debug=True)




from flask import Flask, render_template, request,redirect,url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
app = Flask(__name__)

# ==============================
# 1. Model / paths config
# ==============================
IMG_SIZE = (224, 224)

CLASS_NAMES = ["with_mask", "without_mask"]

WEIGHTS_PATH = "mask_weights.weights.h5"

model = None

def build_model():
    """Recreate the same architecture used during training."""
    base_model = tf.keras.applications.MobileNetV2(
        # weights="imagenet",
        weights=None,
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )

    base_model.trainable = False

    model = tf.keras.Sequential(
        [
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )
    return model

def get_model():
    """Lazy-load model to avoid Render timeout & OOM"""
    global model
    if model is None:
        print("⏳ Loading model and weights...")
        model = build_model()
        model.load_weights(WEIGHTS_PATH)
        print("✅ Model loaded successfully")
    return model

# # Build model and load weights (no H5 full-model loading here)
# model = build_model()
# model.load_weights(WEIGHTS_PATH)
# print("✅ Model architecture built and weights loaded successfully.")


# ==============================
# 2. Routes
# ==============================
@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        # If someone opens /predict directly, send them to home
        return redirect(url_for('home'))
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    if file.filename == '':
        return "No file selected"

    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(224, 224))  # size used while training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    model = get_model() 

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]

    label = "With Mask" if class_idx == 0 else "Without Mask"

    return render_template('result.html', label=label, user_image=filepath)
if __name__ == "__main__":
    app.run(debug=True)
























# @app.route("/predict", methods=["POST"])
# def predict():
#     if "file" not in request.files:
#         return render_template("index.html", error="No file uploaded.")

#     file = request.files["file"]

#     if file.filename == "":
#         return render_template("index.html", error="No file selected.")

#     # Create upload folder
#     upload_folder = os.path.join("static", "uploads")
#     os.makedirs(upload_folder, exist_ok=True)

#     # Save file
#     filepath = os.path.join(upload_folder, file.filename)
#     file.save(filepath)

#     # ----------------------------
#     # Preprocess image
#     # ----------------------------
#     img = image.load_img(filepath, target_size=IMG_SIZE)
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0  # (1, 224, 224, 3)

#     # ----------------------------
#     # Prediction
#     # ----------------------------
#     preds = model.predict(img_array)
#     class_idx = int(np.argmax(preds, axis=1)[0])
#     confidence = float(preds[0][class_idx])

#     raw_label = CLASS_NAMES[class_idx]
#     label = "With Mask" if raw_label == "with_mask" else "Without Mask"

#     return render_template(
#         "result.html",
#         label=label,
#         confidence=f"{confidence * 100:.2f}",
#         user_image=filepath,
#     )