"""
Picture System for the Food Chatbot
This handles all computer vision tasks:
- Image classification using a local TensorFlow/Keras model.
- Image analysis (object detection, descriptions) using the Azure Computer Vision API.
"""
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import requests
from io import BytesIO

# Try to import TensorFlow/Keras, but handle gracefully if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Image classification features will be limited.")

class PictureSystem:
    def __init__(self, image_path="images", azure_credentials=None):
        # I set up the vision system, including Azure credentials and model paths
        self.IMAGE_PATH = image_path
        self.AZURE_CV_KEY, self.AZURE_CV_ENDPOINT = azure_credentials or (None, None)
        self.cv_client = None
        self.image_classifier = None
        self.model_save_path = 'classical_nn_model.h5'
        self.initialize_image_classifier()
        if self.AZURE_CV_KEY and self.AZURE_CV_ENDPOINT:
            self.initialize_azure_computer_vision()

    def initialize_image_classifier(self):
        """I load a pre-trained Keras model or the default MobileNetV2."""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Image classification will use basic image operations.")
            self.image_classifier = None
            return
            
        try:
            if os.path.exists(self.model_save_path):
                self.image_classifier = keras.models.load_model(self.model_save_path)
                print(f"Loaded trained model from {self.model_save_path}")
                # Verify the model can be used
                try:
                    # Test with a dummy input to check if model works
                    input_shape = self.image_classifier.input_shape
                    if input_shape and len(input_shape) >= 3:
                        test_input = np.random.random((1, input_shape[1], input_shape[2], input_shape[3]))
                        _ = self.image_classifier.predict(test_input, verbose=0)
                        print(f"Model verification successful. Input shape: {input_shape}")
                    else:
                        raise ValueError("Invalid input shape")
                except Exception as e:
                    print(f"Saved model verification failed: {e}")
                    print("Falling back to MobileNetV2...")
                    self.image_classifier = MobileNetV2(weights='imagenet')
                    print("Loaded pre-trained MobileNetV2 model (saved model was incompatible).")
            else:
                self.image_classifier = MobileNetV2(weights='imagenet')
                print("Loaded pre-trained MobileNetV2 model (no local model found).")
        except Exception as e:
            print(f"Error loading image classifier model: {e}")
            print("Falling back to MobileNetV2...")
            try:
                self.image_classifier = MobileNetV2(weights='imagenet')
                print("Loaded pre-trained MobileNetV2 model as fallback.")
            except Exception as e2:
                print(f"Failed to load MobileNetV2 as well: {e2}")
                self.image_classifier = None

    def initialize_azure_computer_vision(self):
        """I initialize the Azure Computer Vision client."""
        try:
            from azure.cognitiveservices.vision.computervision import ComputerVisionClient
            from msrest.authentication import CognitiveServicesCredentials
            credentials = CognitiveServicesCredentials(self.AZURE_CV_KEY)
            self.cv_client = ComputerVisionClient(self.AZURE_CV_ENDPOINT, credentials)
            print("Azure Computer Vision client initialized.")
        except ImportError:
            print("Azure SDK not installed. Run 'pip install azure-cognitiveservices-vision-computervision'.")
            self.cv_client = None
        except Exception as e:
            print(f"Error initializing Azure CV client: {e}")
            self.cv_client = None

    def get_random_image(self, dish_name):
        """I get a random image for a given dish from the images folder."""
        dish_folder = os.path.join(self.IMAGE_PATH, dish_name.replace(' ', '_'))
        if not os.path.exists(dish_folder):
            return None, f"I'm sorry, I don't have any images for {dish_name}."
        images = glob.glob(os.path.join(dish_folder, '*.jpg'))
        if not images:
            return None, f"No JPG images found for {dish_name} in the folder."
        random_image_path = np.random.choice(images)
        return random_image_path, None

    def display_image(self, image_path):
        """I display an image using matplotlib."""
        try:
            img = mpimg.imread(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show(block=False)
            plt.pause(1)
            plt.close()
        except Exception as e:
            print(f"Error displaying image: {e}")

    def classify_food_image(self, image_path):
        """I classify a food image using the loaded model."""
        if not TENSORFLOW_AVAILABLE:
            # Fallback when TensorFlow is not available
            try:
                self.display_image(image_path)
                # Get basic image information
                img = Image.open(image_path)
                width, height = img.size
                file_size = os.path.getsize(image_path) / 1024  # KB
                return f"TensorFlow not available for AI classification.\nImage info: {width}x{height} pixels, {file_size:.1f} KB\nThis appears to be a food image based on the filename."
            except Exception as e:
                return f"Error processing image: {e}"
                
        if self.image_classifier is None:
            return "Image classifier is not available."
        try:
            # Add debugging information
            print(f"Model type: {type(self.image_classifier)}")
            print(f"Model input shape: {self.image_classifier.input_shape}")
            
            # Check if this is a custom trained model or MobileNetV2 based on input shape
            input_shape = self.image_classifier.input_shape
            if input_shape and len(input_shape) >= 3:
                # Check if input shape is (224, 224, 3) which indicates MobileNetV2
                is_mobilenet = input_shape[1] == 224 and input_shape[2] == 224
                print(f"Input shape analysis: {input_shape[1]}x{input_shape[2]}x{input_shape[3]}")
                print(f"Detected as MobileNet: {is_mobilenet}")
            else:
                # Fallback: check layer names for MobileNetV2
                is_mobilenet = hasattr(self.image_classifier, 'layers') and any('mobilenet' in layer.name.lower() for layer in self.image_classifier.layers)
                print(f"Fallback detection - MobileNet: {is_mobilenet}")
            
            if is_mobilenet:
                # Use MobileNetV2 preprocessing and decoding
                print("Using MobileNetV2 preprocessing (224x224)")
                img = image.load_img(image_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array_expanded = np.expand_dims(img_array, axis=0)
                processed_img = preprocess_input(img_array_expanded)
                predictions = self.image_classifier.predict(processed_img)
                decoded_predictions = decode_predictions(predictions, top=3)[0]
                self.display_image(image_path)
                result_str = "I think this is a picture of:\n"
                for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
                    result_str += f"{i+1}: {label.replace('_', ' ')} ({score:.2%})\n"
                return result_str
            else:
                # Use custom model preprocessing
                print("Using custom model preprocessing (180x180)")
                img = image.load_img(image_path, target_size=(180, 180))
                img_array = image.img_to_array(img)
                img_array_expanded = np.expand_dims(img_array, axis=0)
                # Normalize to [0,1] range for custom model
                processed_img = img_array_expanded / 255.0
                print(f"Processed image shape: {processed_img.shape}")
                predictions = self.image_classifier.predict(processed_img)
                self.display_image(image_path)
                
                # Get class names from the training data
                data_dir = self.IMAGE_PATH
                if os.path.exists(data_dir):
                    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
                    result_str = "I think this is a picture of:\n"
                    # Get top 3 predictions
                    top_indices = np.argsort(predictions[0])[-3:][::-1]
                    for i, idx in enumerate(top_indices):
                        if idx < len(class_names):
                            class_name = class_names[idx].replace('_', ' ').title()
                            confidence = predictions[0][idx]
                            result_str += f"{i+1}: {class_name} ({confidence:.2%})\n"
                    return result_str
                else:
                    return "Custom model predictions available but class names not found."
                    
        except Exception as e:
            return f"An error occurred during image classification: {e}"

    def analyze_image_with_azure_cv(self, image_path):
        """I analyze an image using Azure Computer Vision for object detection and description."""
        if not self.cv_client:
            return "Azure Computer Vision service is not available."
        try:
            with open(image_path, "rb") as image_stream:
                description_results = self.cv_client.describe_image_in_stream(image_stream)
            desc = "No description generated."
            if description_results.captions:
                desc = f"I see: {description_results.captions[0].text} (Confidence: {description_results.captions[0].confidence:.2%})"
            with open(image_path, "rb") as image_stream:
                object_detection_results = self.cv_client.detect_objects_in_stream(image_stream)
            self.display_image_with_azure_analysis(image_path, object_detection_results.objects, desc)
            return f"Azure Vision Analysis:\n{desc}"
        except Exception as e:
            return f"Error using Azure CV: {e}"

    def display_image_with_azure_analysis(self, image_path, detected_objects, description):
        """I display an image with bounding boxes for detected objects."""
        try:
            image_obj = Image.open(image_path)
            draw = ImageDraw.Draw(image_obj)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()
            for obj in detected_objects:
                rect = obj.rectangle
                left, top, right, bottom = rect.x, rect.y, rect.x + rect.w, rect.y + rect.h
                draw.rectangle([left, top, right, bottom], outline="red", width=3)
                label = f"{obj.object_property} ({obj.confidence:.2%})"
                text_size = draw.textbbox((0,0), label, font=font)
                text_width = text_size[2] - text_size[0]
                text_height = text_size[3] - text_size[1]
                if top - text_height >= 0:
                    text_location = (left, top - text_height)
                else:
                    text_location = (left, top + 1)
                draw.rectangle([text_location[0], text_location[1], text_location[0] + text_width, text_location[1] + text_height], fill="red")
                draw.text(text_location, label, fill="white", font=font)
            plt.figure(figsize=(12, 8))
            plt.imshow(image_obj)
            plt.title(description, wrap=True)
            plt.axis('off')
            plt.show(block=False)
            plt.pause(2)
            plt.close()
        except Exception as e:
            print(f"Error displaying image with bounding boxes: {e}")

    def train_classical_nn(self, epochs=3):
        """I train a simple CNN on the images dataset."""
        if not TENSORFLOW_AVAILABLE:
            return "TensorFlow is not available. Cannot train neural network models."
            
        print("Starting CNN training...")
        data_dir = self.IMAGE_PATH
        if not os.path.exists(data_dir):
            return "Image directory not found for training."
        image_size = (180, 180)
        batch_size = 32
        train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="both",
            seed=1337,
            image_size=image_size,
            batch_size=batch_size,
        )
        class_names = train_ds.class_names
        num_classes = len(class_names)
        # I use basic data augmentation
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
        ])
        # I build the CNN model
        model = keras.Sequential([
            keras.Input(shape=image_size + (3,)),
            data_augmentation,
            layers.Rescaling(1./255),
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax'),
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        print("Training the model...")
        model.fit(train_ds, epochs=epochs, validation_data=val_ds)
        print("Training complete.")
        self.image_classifier = model
        self.save_model()
        return f"Successfully trained and updated the image classification model. It can now classify {num_classes} categories."

    def save_model(self):
        """I save the currently active image classifier model."""
        if not TENSORFLOW_AVAILABLE:
            return "TensorFlow is not available. Cannot save models."
            
        if self.image_classifier and isinstance(self.image_classifier, keras.Model):
            try:
                self.image_classifier.save(self.model_save_path)
                return f"Model successfully saved to {self.model_save_path}"
            except Exception as e:
                return f"Error saving model: {e}"
        return "No trained model available to save." 