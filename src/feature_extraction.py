import os
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image as kimage
import pickle
import numpy as np

# Load in InceptionV3 Model
# Documentation for InceptionV3: https://keras.io/api/applications/inceptionv3/

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Remove classification layers
pooled_output = base_model.layers[-1].output
encoded_features = GlobalAveragePooling2D()(pooled_output)

# Create updated model w/o classification layers
feature_extraction = Model(inputs=base_model.input, outputs=encoded_features)

# Freeze layers in base model
for layer in base_model.layers:
    layer.trainable = False # ensures these layers aren't retrained 
    
    
# Preprocess images
# Path to your dataset directory
dataset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'Flicker8k', 'Images')


# Create a list of image file paths
image_paths = []
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(".jpg"):
            image_paths.append(os.path.join(root, file))


print(len(image_paths))

# Preprocess and extract features for each image
features_dict = {}
for img_path in image_paths:
    img = kimage.load_img(img_path, target_size=(224, 224))
    img_array = kimage.img_to_array(img)
    img_array = img_array/255.0
    img_array = img_array.reshape(1, 224, 224, 3)
    extracted_features = feature_extraction.predict(img_array)
    img_id = img_path.split('/')[-1].split('.')[0]
    features_dict[img_id] = extracted_features
    
    
# Save extracted features via pkl file
with open('extracted_feats.pkl', 'wb') as f:
    pickle.dump(features_dict, f)

# Save Model
feature_extraction.save('features.h5')
