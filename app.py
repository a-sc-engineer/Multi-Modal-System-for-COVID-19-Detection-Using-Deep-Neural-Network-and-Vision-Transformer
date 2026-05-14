import os
import keras
import tensorflow as tf
from keras import layers
from flask import Flask, request, jsonify, render_template
from transformers import TFViTModel
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

# --- Custom Layer Definitions ---
# These MUST use @keras.saving.register_keras_serializable() so Keras 3
# can locate them when loading the .keras file saved on Kaggle.

@keras.saving.register_keras_serializable(package="Custom")
class GaussianNoise(keras.Layer):
    def __init__(self, stddev=0.1, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev

    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=self.stddev)
            return inputs + noise
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({'stddev': self.stddev})
        return config


@keras.saving.register_keras_serializable(package="Custom")
class ViTFeatureExtractor(keras.Layer):
    def __init__(self, vit_model=None, include_pooler=True, **kwargs):
        super().__init__(**kwargs)
        self.vit_model = vit_model
        self.include_pooler = include_pooler

    def call(self, inputs, training=None):
        pixel_values = tf.transpose(inputs, perm=[0, 3, 1, 2])
        outputs = self.vit_model(pixel_values=pixel_values, training=training)
        cls_token = outputs.last_hidden_state[:, 0, :]
        if self.include_pooler and hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return tf.concat([cls_token, outputs.pooler_output], axis=-1)
        return cls_token

    def get_config(self):
        config = super().get_config()
        config.update({
            "vit_model_name": "google/vit-base-patch16-224",
            "include_pooler": self.include_pooler,
        })
        return config

    @classmethod
    def from_config(cls, config):
        vit_model_name = config.pop("vit_model_name", "google/vit-base-patch16-224")
        vit_model = TFViTModel.from_pretrained(vit_model_name)
        return cls(vit_model=vit_model, **config)


@keras.saving.register_keras_serializable(package="Custom")
class CNNFeatureExtractor(keras.Layer):
    def __init__(self, name="cnn_extractor", **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv1 = layers.Conv2D(32, (3, 3), strides=2, padding='same', activation='swish', name="cnn_conv1")
        self.bn1   = layers.BatchNormalization(name="cnn_bn1")
        self.pool1 = layers.MaxPooling2D((2, 2), padding='same', name="cnn_pool1")
        self.conv2 = layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='swish', name="cnn_conv2")
        self.bn2   = layers.BatchNormalization(name="cnn_bn2")
        self.pool2 = layers.MaxPooling2D((2, 2), padding='same', name="cnn_pool2")
        self.conv3 = layers.Conv2D(128, (3, 3), strides=2, padding='same', activation='swish', name="cnn_conv3")
        self.bn3   = layers.BatchNormalization(name="cnn_bn3")
        self.gap   = layers.GlobalAveragePooling2D(name="cnn_gap")
        self.gmp   = layers.GlobalMaxPooling2D(name="cnn_gmp")
        self.mid_gap = layers.GlobalAveragePooling2D(name="cnn_mid_gap")

    def call(self, inputs, training=None):
        x1 = self.pool1(self.bn1(self.conv1(inputs), training=training))
        x2 = self.pool2(self.bn2(self.conv2(x1), training=training))
        x3 = self.bn3(self.conv3(x2), training=training)
        return tf.concat([self.gap(x3), self.gmp(x3), self.mid_gap(x2)], axis=-1)

    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable(package="Custom")
class CrossModalityAttention(keras.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units   = units
        self.query   = layers.Dense(units, dtype='float32', name="attn_query")
        self.key     = layers.Dense(units, dtype='float32', name="attn_key")
        self.value   = layers.Dense(units, dtype='float32', name="attn_value")
        self.softmax = layers.Softmax(axis=-1, dtype='float32', name="attn_softmax")

    def call(self, inputs):
        q = self.query(inputs[0])
        k = self.key(inputs[1])
        v = self.value(inputs[1])
        scores  = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.units, q.dtype))
        weights = self.softmax(scores)
        return tf.matmul(weights, v)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


@keras.saving.register_keras_serializable(package="Custom")
class FeatureFusionBlock(keras.Layer):
    def __init__(self, fusion_method='concat', name="feature_fusion_block", **kwargs):
        super().__init__(name=name, **kwargs)
        self.fusion_method = fusion_method
        if fusion_method == 'attention':
            self.attention_a_to_b = CrossModalityAttention(256, name=f"{name}_attn_a_to_b")
            self.attention_b_to_a = CrossModalityAttention(256, name=f"{name}_attn_b_to_a")
            self.dense = layers.Dense(512, activation='swish', name=f"{name}_dense_attn")
        elif fusion_method == 'bilinear':
            self.dense = layers.Dense(512, activation='swish', name=f"{name}_dense_bilinear")
        else:
            self.dense = layers.Dense(512, activation='swish', name=f"{name}_dense_concat")
        self.bn      = layers.BatchNormalization(name=f"{name}_bn")
        self.dropout = layers.Dropout(0.3, name=f"{name}_dropout")

    def call(self, inputs, training=None):
        features_a, features_b = inputs[0], inputs[1]
        if self.fusion_method == 'attention':
            fused = tf.concat([
                self.attention_a_to_b([features_a, features_b]),
                self.attention_b_to_a([features_b, features_a])
            ], axis=-1)
        elif self.fusion_method == 'bilinear':
            outer = tf.einsum('bi,bj->bij', features_a, features_b)
            fused = layers.Flatten()(outer)
        else:
            fused = tf.concat(inputs, axis=-1)
        x = self.dense(fused)
        x = self.bn(x, training=training)
        return self.dropout(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"fusion_method": self.fusion_method})
        return config


# --- Load Model ---
MODEL_PATH = 'final_hierarchical_multimodal_model.keras'
model = None

CUSTOM_OBJECTS = {
    'GaussianNoise':         GaussianNoise,
    'ViTFeatureExtractor':   ViTFeatureExtractor,
    'CNNFeatureExtractor':   CNNFeatureExtractor,
    'CrossModalityAttention':CrossModalityAttention,
    'FeatureFusionBlock':    FeatureFusionBlock,
}

try:
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        model = keras.models.load_model(MODEL_PATH, custom_objects=CUSTOM_OBJECTS)
        print(f"Model successfully loaded: {model.name}")
    else:
        print(f"WARNING: Model file not found at '{MODEL_PATH}'.")
except Exception as e:
    import traceback
    print(f"ERROR: Failed to load model: {e}")
    traceback.print_exc()

# --- API ---
IMAGE_SIZE = 224
CLASS_NAMES = ['COVID', 'NON_COVID']

def preprocess_image_from_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return tf.expand_dims(tf.convert_to_tensor(img_array), axis=0)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "running", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please ensure the .keras model file is present in the directory."}), 503

    if 'ct_image' not in request.files or 'xray_image' not in request.files:
        return jsonify({"error": "Both 'ct_image' and 'xray_image' files must be provided"}), 400

    ct_file   = request.files['ct_image']
    xray_file = request.files['xray_image']

    try:
        ct_tensor   = preprocess_image_from_bytes(ct_file.read())
        xray_tensor = preprocess_image_from_bytes(xray_file.read())

        predictions = model.predict([ct_tensor, xray_tensor])
        predicted_class_index = int(np.argmax(predictions[0]))
        predicted_class_name  = CLASS_NAMES[predicted_class_index]
        confidence            = float(predictions[0][predicted_class_index])

        return jsonify({
            "prediction":    predicted_class_name,
            "confidence":    confidence,
            "probabilities": {
                CLASS_NAMES[0]: float(predictions[0][0]),
                CLASS_NAMES[1]: float(predictions[0][1]),
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
