
import sys
import subprocess
import importlib
import traceback
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from transformers import TFViTModel # Ensure this is installed: pip install transformers tensorflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc # Added roc_curve and auc
import seaborn as sns

# --- Install packages if not present ---
def install_package(package):
    """Installs a package using pip, checking if it's already available."""
    try:
        base_package_name = package.split('[')[0].split('>')[0].split('<')[0].split('=')[0]
        importlib.import_module(base_package_name)
        print(f"{package} already installed.")
    except ImportError:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")
            print(f"Could not install required package: {package}")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred while checking/installing {package}: {e}")
            traceback.print_exc()
            sys.exit(1)

install_package("transformers")
install_package("tensorflow")
install_package("matplotlib")
install_package("Pillow") # Needed for image processing often implicitly by tf.image
install_package("scikit-learn")
install_package("seaborn")

print(f"TensorFlow version: {tf.__version__}")
try:
    transformers_module = importlib.import_module('transformers')
    print(f"Transformers version: {transformers_module.__version__}")
except ImportError:
    print("Transformers not fully installed or accessible.")
except Exception as e:
    print(f"Error checking Transformers version: {e}")

# --- GPU Check and Mixed Precision Setup ---
print("\n--- GPU and Mixed Precision Setup ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        mixed_precision.set_global_policy('mixed_float16')
        print(f"Mixed precision enabled with policy 'mixed_float16' on {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(f"RuntimeError in GPU setup (mixed precision might fail): {e}")
        print("Proceeding without mixed precision if it failed to set.")
    except Exception as e:
        print(f"Error during mixed precision setup: {e}")
else:
    print("No GPU found. Mixed precision not enabled. Training will be very slow.")

# --- Constants ---
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10 # Adjust as needed for real data
EPOCHS_PHASE2 = 20# Adjust as needed for real data

CLASS_NAMES = ['COVID', 'NON_COVID']
num_classes = len(CLASS_NAMES)
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# --- Kaggle Dataset Paths ---
print("\n--- Setting up Kaggle Dataset Paths ---")
KAGGLE_INPUT_PATH = "/kaggle/input/dataset-ct-xr"
DATASET_BASE_ROOT = os.path.join(KAGGLE_INPUT_PATH, 'dataset_ct_xr')

CT_ROOT = os.path.join(DATASET_BASE_ROOT, 'CT_scan_')
XRAY_ROOT = os.path.join(DATASET_BASE_ROOT, 'X_ray_')

print(f"Dataset base root: {DATASET_BASE_ROOT}")
print(f"CT scans directory: {CT_ROOT}")
print(f"X-ray directory: {XRAY_ROOT}")

# Verify paths exist
if not os.path.exists(DATASET_BASE_ROOT):
    print(f"Error: Dataset base directory not found at {DATASET_BASE_ROOT}")
    sys.exit(1)
if not os.path.exists(CT_ROOT):
    print(f"Error: CT root directory not found at {CT_ROOT}")
    sys.exit(1)
if not os.path.exists(XRAY_ROOT):
    print(f"Error: X-ray root directory not found at {XRAY_ROOT}")
    sys.exit(1)

print("All dataset paths verified successfully.")

def get_all_image_paths_and_labels_for_modality(root_dir, class_names, class_to_index):
    """
    Gathers all image paths and labels for a single modality (CT or X-ray).
    """
    image_paths = []
    labels = []
   
    print(f"Collecting images from: {root_dir}")
    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        if os.path.exists(class_dir):
            # Look for common image formats
            files = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                files.extend(glob.glob(os.path.join(class_dir, ext)))
           
            for f in files:
                image_paths.append(f)
                labels.append(class_to_index[class_name])
            print(f"  Found {len(files)} images for class {class_name}.")
        else:
            print(f"  Warning: Class directory not found: {class_dir}. Skipping.")
   
    if not image_paths:
        print(f"Error: No images found in {root_dir}. Check paths and class names.")
        sys.exit(1)
       
    return np.array(image_paths), np.array(labels)

print("\n--- Gathering All CT and X-ray Image Paths and Labels Independently ---")
ct_image_paths, ct_labels_int = get_all_image_paths_and_labels_for_modality(CT_ROOT, CLASS_NAMES, CLASS_TO_INDEX)
xray_image_paths, xray_labels_int = get_all_image_paths_and_labels_for_modality(XRAY_ROOT, CLASS_NAMES, CLASS_TO_INDEX)

print(f"Total CT images found: {len(ct_image_paths)}")
print(f"Total X-ray images found: {len(xray_image_paths)}")

# Split each modality's dataset independently
# This ensures a good distribution for each modality in train/val/test sets
# and then we'll combine them for "random pairs"
train_ct_paths, test_ct_paths, train_ct_labels_int, test_ct_labels_int = train_test_split(
    ct_image_paths, ct_labels_int, test_size=0.10, random_state=42, shuffle=True, stratify=ct_labels_int
)
train_ct_paths, val_ct_paths, train_ct_labels_int, val_ct_labels_int = train_test_split(
    train_ct_paths, train_ct_labels_int, test_size=(0.10 / 0.90), random_state=42, shuffle=True, stratify=train_ct_labels_int
)

# X-ray Split
train_xray_paths, test_xray_paths, train_xray_labels_int, test_xray_labels_int = train_test_split(
    xray_image_paths, xray_labels_int, test_size=0.10, random_state=42, shuffle=True, stratify=xray_labels_int
)
train_xray_paths, val_xray_paths, train_xray_labels_int, val_xray_labels_int = train_test_split(
    train_xray_paths, train_xray_labels_int, test_size=(0.10 / 0.90), random_state=42, shuffle=True, stratify=train_xray_labels_int
)

print(f"Train CT samples: {len(train_ct_paths)}")
print(f"Validation CT samples: {len(val_ct_paths)}")
print(f"Test CT samples: {len(test_ct_paths)}")
print(f"Train X-ray samples: {len(train_xray_paths)}")
print(f"Validation X-ray samples: {len(val_xray_paths)}")
print(f"Test X-ray samples: {len(test_xray_paths)}")

# --- Data Pipeline Creation ---
def preprocess_image(image_path):
    """Preprocess image for model input."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = image / 255.0  # Normalize to [0, 1]
    return image



# Custom Gaussian Noise Layer
class GaussianNoise(layers.Layer):
    def __init__(self, stddev=0.1, **kwargs):
        super(GaussianNoise, self).__init__(**kwargs)
        self.stddev = stddev
   
    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=self.stddev)
            return inputs + noise
        return inputs
   
    def get_config(self):
        config = super(GaussianNoise, self).get_config()
        config.update({'stddev': self.stddev})
        return config

# Enhanced data augmentation with Gaussian noise
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.4),
    layers.RandomZoom(0.3),
    layers.RandomContrast(0.2),
    GaussianNoise(stddev=0.02)  # Add Gaussian noise with standard deviation of 0.02
], name="data_augmentation")


def create_proper_paired_dataset(ct_paths, xray_paths, ct_labels, xray_labels, batch_size, shuffle=True):
    """
    Create properly paired dataset ensuring CT and X-ray are from same patient/class
    """
    # Group by labels to ensure proper pairing
    covid_ct_paths = ct_paths[ct_labels == 0]  # COVID class
    covid_xray_paths = xray_paths[xray_labels == 0]
   
    non_covid_ct_paths = ct_paths[ct_labels == 1]  # NON_COVID class  
    non_covid_xray_paths = xray_paths[xray_labels == 1]
   
    # Create balanced pairs
    min_covid = min(len(covid_ct_paths), len(covid_xray_paths))
    min_non_covid = min(len(non_covid_ct_paths), len(non_covid_xray_paths))
   
    # Take equal samples from each class
    final_ct_paths = np.concatenate([
        covid_ct_paths[:min_covid],
        non_covid_ct_paths[:min_non_covid]
    ])
    final_xray_paths = np.concatenate([
        covid_xray_paths[:min_covid],
        non_covid_xray_paths[:min_non_covid]
    ])
    final_labels = np.concatenate([
        np.zeros(min_covid),  # COVID labels
        np.ones(min_non_covid)  # NON_COVID labels
    ])
   
    # Shuffle while maintaining pairs
    if shuffle:
        indices = np.random.permutation(len(final_labels))
        final_ct_paths = final_ct_paths[indices]
        final_xray_paths = final_xray_paths[indices]
        final_labels = final_labels[indices]
   
    # Create TensorFlow datasets
    ct_dataset = tf.data.Dataset.from_tensor_slices(final_ct_paths).map(preprocess_image)
    xray_dataset = tf.data.Dataset.from_tensor_slices(final_xray_paths).map(preprocess_image)
    label_dataset = tf.data.Dataset.from_tensor_slices(
        tf.keras.utils.to_categorical(final_labels, 2)
    )
   
    paired_dataset = tf.data.Dataset.zip(((ct_dataset, xray_dataset), label_dataset))
    return paired_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)



class ViTFeatureExtractor(layers.Layer):
    """ViT feature extractor from a pre-trained TFViTModel."""
    def __init__(self, vit_model, include_pooler=True, **kwargs):
        super().__init__(**kwargs)
        self.vit_model = vit_model
        self.include_pooler = include_pooler
       
    def call(self, inputs, training=None):
        # Hugging Face ViT models expect channels-first (B, C, H, W)
        pixel_values = tf.transpose(inputs, perm=[0, 3, 1, 2])
        outputs = self.vit_model(pixel_values=pixel_values, training=training)
       
        cls_token = outputs.last_hidden_state[:, 0, :]
        if self.include_pooler and hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
            return tf.concat([cls_token, pooled_output], axis=-1)
        return cls_token

    def get_config(self):
        config = super().get_config()
        config.update({
            "vit_model_name": "microsoft/cvt-13", # Store the model name for reconstruction
            "include_pooler": self.include_pooler,
        })
        return config

    @classmethod
    def from_config(cls, config):
        vit_model_name = config.pop("vit_model_name", "microsoft/cvt-13")
        vit_model = TFViTModel.from_pretrained(vit_model_name)
        return cls(vit_model=vit_model, **config)

class CNNFeatureExtractor(layers.Layer):
    """Custom CNN feature extractor."""
    def __init__(self, name="cnn_extractor", **kwargs):
        super().__init__(name=name, **kwargs)
       
        self.conv1 = layers.Conv2D(32, (3,3), strides=2, padding='same', activation='swish', name="cnn_conv1")
        self.bn1 = layers.BatchNormalization(name="cnn_bn1")
        self.pool1 = layers.MaxPooling2D((2,2), padding='same', name="cnn_pool1")
       
        self.conv2 = layers.Conv2D(64, (3,3), strides=2, padding='same', activation='swish', name="cnn_conv2")
        self.bn2 = layers.BatchNormalization(name="cnn_bn2")
        self.pool2 = layers.MaxPooling2D((2,2), padding='same', name="cnn_pool2")
       
        self.conv3 = layers.Conv2D(128, (3,3), strides=2, padding='same', activation='swish', name="cnn_conv3")
        self.bn3 = layers.BatchNormalization(name="cnn_bn3")
       
        self.gap = layers.GlobalAveragePooling2D(name="cnn_gap")
        self.gmp = layers.GlobalMaxPooling2D(name="cnn_gmp")
       
    def call(self, inputs, training=None):
        x1 = self.conv1(inputs)
        x1 = self.bn1(x1, training=training)
        x1 = self.pool1(x1)
       
        x2 = self.conv2(x1)
        x2 = self.bn2(x2, training=training)
        x2 = self.pool2(x2)
       
        x3 = self.conv3(x2)
        x3 = self.bn3(x3, training=training)
       
        gap = self.gap(x3)
        gmp = self.gmp(x3)
        mid_gap = layers.GlobalAveragePooling2D()(x2) # Apply GAP to x2 directly
       
        return tf.concat([gap, gmp, mid_gap], axis=-1)

    def get_config(self):
        config = super().get_config()
        return config

class CrossModalityAttention(layers.Layer):
    """Cross-modality attention layer for feature fusion."""
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.query = layers.Dense(units, dtype='float32', name="attn_query")
        self.key = layers.Dense(units, dtype='float32', name="attn_key")  
        self.value = layers.Dense(units, dtype='float32', name="attn_value")
        self.softmax = layers.Softmax(axis=-1, dtype='float32', name="attn_softmax")
       
    def call(self, inputs):
        q = self.query(inputs[0])
        k = self.key(inputs[1])
        v = self.value(inputs[1])
       
        attention_scores = tf.matmul(q, k, transpose_b=True)
        attention_scores = attention_scores / tf.math.sqrt(tf.cast(self.units, attention_scores.dtype))
        attention_weights = self.softmax(attention_scores)
       
        output = tf.matmul(attention_weights, v)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

class FeatureFusionBlock(layers.Layer):
    """Advanced feature fusion block."""
    def __init__(self, fusion_method='concat', name="feature_fusion_block", **kwargs):
        super().__init__(name=name, **kwargs)
        self.fusion_method = fusion_method
       
        if fusion_method == 'attention':
            self.attention_a_to_b = CrossModalityAttention(256, name=f"{name}_attn_a_to_b")
            self.attention_b_to_a = CrossModalityAttention(256, name=f"{name}_attn_b_to_a")
            self.dense = layers.Dense(512, activation='swish', name=f"{name}_dense_attn")
        elif fusion_method == 'bilinear':
            self.dense = layers.Dense(512, activation='swish', name=f"{name}_dense_bilinear")
        else: # Default is 'concat'
            self.dense = layers.Dense(512, activation='swish', name=f"{name}_dense_concat")
           
        self.bn = layers.BatchNormalization(name=f"{name}_bn")
        self.dropout = layers.Dropout(0.3, name=f"{name}_dropout")
       
    def call(self, inputs, training=None):
        features_a, features_b = inputs[0], inputs[1]

        if self.fusion_method == 'attention':
            attn_a_to_b = self.attention_a_to_b([features_a, features_b])
            attn_b_to_a = self.attention_b_to_a([features_b, features_a])
            fused = tf.concat([attn_a_to_b, attn_b_to_a], axis=-1)
        elif self.fusion_method == 'bilinear':
            outer = tf.einsum('bi,bj->bij', features_a, features_b) # b = batch, i,j = feature dims
            fused = layers.Flatten()(outer)
        else: # 'concat'
            fused = tf.concat(inputs, axis=-1)
           
        x = self.dense(fused)
        x = self.bn(x, training=training)
        return self.dropout(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "fusion_method": self.fusion_method,
        })
        return config

# --- New Hierarchical Model Architecture ---
def build_hierarchical_fusion_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=2):
    """
    Builds a hierarchical multimodal fusion model combining CNN and ViT branches.
    Each branch performs feature-level fusion, and their outputs are then concatenated.
    """
    input_ct = layers.Input(shape=input_shape, name="ct_input")
    input_xray = layers.Input(shape=input_shape, name="xray_input")
   
    # --- CNN Branch ---
    cnn_feature_extractor_ct = CNNFeatureExtractor(name="cnn_extractor_ct")
    cnn_feature_extractor_xray = CNNFeatureExtractor(name="cnn_extractor_xray")
   
    ct_cnn_features = cnn_feature_extractor_ct(input_ct)
    xray_cnn_features = cnn_feature_extractor_xray(input_xray)
   
    cnn_fusion_block = FeatureFusionBlock(fusion_method='attention', name="cnn_feature_fusion_block")
    fused_cnn_features = cnn_fusion_block([ct_cnn_features, xray_cnn_features])
   
    # --- ViT Branch ---
    vit_base_model_ct = TFViTModel.from_pretrained("google/vit-base-patch16-224", name="vit_base_model_ct")
    vit_base_model_xray = TFViTModel.from_pretrained("google/vit-base-patch16-224", name="vit_base_model_xray")
   
    vit_feature_extractor_ct = ViTFeatureExtractor(vit_base_model_ct, name="vit_extractor_ct")
    vit_feature_extractor_xray = ViTFeatureExtractor(vit_base_model_xray, name="vit_extractor_xray")
   
    ct_vit_features = vit_feature_extractor_ct(input_ct)
    xray_vit_features = vit_feature_extractor_xray(input_xray)
   
    vit_fusion_block = FeatureFusionBlock(fusion_method='attention', name="vit_feature_fusion_block")
    fused_vit_features = vit_fusion_block([ct_vit_features, xray_vit_features])

    # --- Final Hierarchical Fusion (Concatenation of Fused Features) ---
    final_fused_features = layers.Concatenate(axis=-1, name="final_hierarchical_fusion_concat")(
        [fused_cnn_features, fused_vit_features]
    )
   
    # --- Classification Head ---
    x = layers.Dense(512, activation='swish', name="dense_final_1")(final_fused_features)
    x = layers.BatchNormalization(name="bn_final_1")(x)
    x = layers.Dropout(0.7, name="dropout_final_1")(x)
   
    x = layers.Dense(256, activation='swish', name="dense_final_2")(x)
    x = layers.BatchNormalization(name="bn_final_2")(x)
    x = layers.Dropout(0.4, name="dropout_final_2")(x)
   
    output = layers.Dense(num_classes, activation='softmax', dtype='float32', name="output")(x)
   
    return models.Model(inputs=[input_ct, input_xray], outputs=output, name="Hierarchical_Multimodal_Model")




# --- Model Training ---
print("\n--- Building Hierarchical Multimodal Model ---")
fusion_model = build_hierarchical_fusion_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=num_classes)
fusion_model.summary()

# --- Phase 1: Train only fusion and classification layers (Backbones frozen) ---
print("\n--- Phase 1 Training (Frozen Backbones & Initial Fusion Blocks) ---")

# Freeze all backbone and internal feature extractor layers
for layer in fusion_model.layers:
    if "cnn_extractor" in layer.name: # Freezes CNNFeatureExtractor instances
        layer.trainable = False
    elif "vit_extractor" in layer.name: # Freezes ViTFeatureExtractor instances
        layer.trainable = False
        # Also ensure the underlying TFViTModel is frozen
        if hasattr(layer, 'vit_model') and isinstance(layer.vit_model, TFViTModel):
            layer.vit_model.trainable = False

fusion_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4), # A moderate learning rate for initial layers
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

if EPOCHS_PHASE1 > 0:
    print(f"Starting Phase 1 training for {EPOCHS_PHASE1} epochs...")
    history_phase1 = fusion_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS_PHASE1,
        callbacks=[
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.ModelCheckpoint(
                'phase1_best_hierarchical.keras',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            )
        ]
    )
else:
    print("Skipping Phase 1 training as EPOCHS_PHASE1 is 0.")

# --- Phase 2: Fine-tune the entire model (Unfrozen Backbones) ---
print("\n--- Phase 2 Training (Unfrozen All Layers for Fine-tuning) ---")

# Unfreeze all layers for fine-tuning
for layer in fusion_model.layers:
    layer.trainable = True
    if hasattr(layer, 'vit_model') and isinstance(layer.vit_model, TFViTModel):
        layer.vit_model.trainable = True # Also ensures the TFViTModel is trainable

fusion_model.compile(
    optimizer=tf.keras.optimizers.Adam(5e-6), # Lower learning rate for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

if EPOCHS_PHASE2 > 0:
    print(f"Starting Phase 2 training for {EPOCHS_PHASE2} epochs (total epochs: {EPOCHS_PHASE1 + EPOCHS_PHASE2})...")
    history_phase2 = fusion_model.fit(
        train_dataset,
        validation_data=val_dataset,
        initial_epoch=EPOCHS_PHASE1,
        epochs=EPOCHS_PHASE1 + EPOCHS_PHASE2,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                'phase2_best_hierarchical.keras',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=7,
                restore_best_weights=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            )
        ]
    )
else:
    print("Skipping Phase 2 training as EPOCHS_PHASE2 is 0.")

# Save final model
print("\n--- Saving Final Model ---")
fusion_model.save('final_hierarchical_multimodal_model.keras')
print("Model saved as 'final_hierarchical_multimodal_model.keras'.")

# --- Final Evaluation on the Test Set ---
print("\n--- Final Evaluation on Test Set ---")
# Define custom objects needed for loading the model
CUSTOM_OBJECTS = {
    'GaussianNoise': GaussianNoise,
    'ViTFeatureExtractor': ViTFeatureExtractor,
    'CNNFeatureExtractor': CNNFeatureExtractor,
    'CrossModalityAttention': CrossModalityAttention,
    'FeatureFusionBlock': FeatureFusionBlock
}

try:
    if EPOCHS_PHASE2 > 0 and os.path.exists('phase2_best_hierarchical.keras'):
        final_model = tf.keras.models.load_model('phase2_best_hierarchical.keras', custom_objects=CUSTOM_OBJECTS)
        print("Loaded best model from phase 2 for final evaluation.")
    elif EPOCHS_PHASE1 > 0 and os.path.exists('phase1_best_hierarchical.keras'):
        final_model = tf.keras.models.load_model('phase1_best_hierarchical.keras', custom_objects=CUSTOM_OBJECTS)
        print("Loaded best model from phase 1 for final evaluation (Phase 2 skipped or failed).")
    else:
        final_model = fusion_model
        print("Using the current state of the model for final evaluation (no best model checkpoint found).")
except Exception as e:
    print(f"Error loading best model, using the last trained model state: {e}")
    traceback.print_exc() # Print full traceback for debugging load errors
    final_model = fusion_model

print("\nEvaluating final model on **Test Dataset**...")
test_loss, test_accuracy = final_model.evaluate(test_dataset, verbose=1)
print(f"Final **Test Loss**: {test_loss:.4f}")
print(f"Final **Test Accuracy**: {test_accuracy:.4f}")

# --- Generate ROC Curve and Confusion Matrix ---
print("\n--- Generating ROC Curve and Confusion Matrix ---")

# Collect true labels and predicted probabilities
y_true = []
y_pred_probs = []

for inputs, labels in test_dataset:
    y_true.extend(np.argmax(labels.numpy(), axis=1)) # Convert one-hot to class index
    y_pred_probs.extend(final_model.predict(inputs))

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)

# Ensure y_pred_probs is float32 for metric calculations if mixed_precision is active
y_pred_probs = tf.cast(y_pred_probs, tf.float32).numpy()

# 1. ROC Curve
# Determine positive class index (e.g., 'COVID' usually is the positive case)
# Based on CLASS_NAMES = ['COVID', 'NON_COVID'], COVID is index 0.
positive_class_idx = CLASS_TO_INDEX['COVID']
y_pred_positive_class_probs = y_pred_probs[:, positive_class_idx]

fpr, tpr, thresholds = roc_curve(y_true, y_pred_positive_class_probs, pos_label=positive_class_idx)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(9, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# 2. Confusion Matrix
y_pred_classes = np.argmax(y_pred_probs, axis=1) # Get predicted class indices

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            cbar=False) # cbar=False if you don't need the color bar
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print("\n--- Training and Evaluation Complete ---")