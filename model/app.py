from flask import Flask, jsonify, request
import numpy as np
from PIL import Image
from keras.preprocessing import image
import lime
from lime import lime_image
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.io import imread
from skimage.segmentation import mark_boundaries
from keras.models import load_model
import cv2
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.utils import load_img,img_to_array,array_to_img
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.applications import ResNet50
from keras.applications.mobilenet import preprocess_input
from keras.models import Model
import efficientnet.tfkeras as efn
import keras.backend as K
import base64
from flask_cors import CORS
import json

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/predictGradcam', methods=['POST'])
# @cross_origin()
def predictGradcam():

    # model = load_model("chest_xray.hdf5")
    filename = request.get_data()
    json_object = json.loads(filename.decode())

    encoded_data = json_object['data']
    decoded_data=base64.b64decode((encoded_data))
    #write the decoded data back to original format in  file
    img_file = open('image.jpg', 'wb')
    img_file.write(decoded_data)
    img_file.close()
    
    img_path = "image.jpg"
    
    
    # model = load_model("grad_cam_final.h5")
    model = load_model("Final2.h5")
    model2 = load_model("grad_updated.h5")


    trained_conv_layer = model.get_layer('conv5_block3_out')
    last_conv_layer_name='conv5_block3_out'
    classifier_layer_names = [
        "bn",
        "relu",
        "averagepooling2d_head",
        "flatten_head",
        "dense_head",
        "dropout_head",
        "predictions_head"
    ]
    # Load and preprocess an image for GradCAM
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    cv2.imwrite("../app/src/Images/image.jpg", img)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    img_size = (224, 224)

    # Convert the input image to a NumPy array
    img_array = np.array(img)

    # Convert the input image to RGB format
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    # Predict the class probabilities
    preds = model.predict(x)
    class_index = np.argmax(preds[0])

    # Obtain the last convolutional layer
    last_conv_layer = model.get_layer('conv5_block3_out')

    # Compute the gradient of the class output value with respect to the feature map of the last convolutional layer
    grad_model = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
    with tf.GradientTape() as tape:
        preds, conv_outputs = grad_model(x)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)[0]

    # Compute the channel-wise mean of the gradients
    weights = tf.reduce_mean(grads, axis=(0, 1))

    # Multiply each channel in the feature map array by "how important this channel is" with regard to the class
    cam = np.ones(conv_outputs.shape[1:3], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[0, :, :, i]
    # Normalize the heatmap between 0 and 255
    heatmap = cam.numpy()
    heatmap = cv2.resize(heatmap, (224,224))
    heatmap = np.maximum(heatmap, 0)
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)
    # Apply a color map to the heatmap
    jet = cm.get_cmap("jet")
    heatmap = jet(heatmap)
    heatmap = np.uint8(heatmap * 255)

    # Convert the heatmap to RGB format
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGBA2RGB)

    # Resize the heatmap to be the same size as the original image
    heatmap = cv2.resize(heatmap, (224,224))

    # Combine the heatmap with the original image
    superimposed_img = cv2.addWeighted(np.array(img), 0.5, heatmap, 0.5, 0)

    # Show the result
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.savefig('../app/src/Images/xray_cam.jpg', bbox_inches='tight', pad_inches=0.0)

    # for index, resized_img in enumerate(img_arr):
    #     explainer = lime_image.LimeImageExplainer()
    #     explanation = explainer.explain_instance(resized_img.astype('double'), model2.predict, top_labels=5, hide_color=0, num_samples=1000)
    #     temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    #     plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    #     plt.axis('off')
    #     plt.savefig('../app/src/Images/result-lime.png', bbox_inches='tight')

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(x[0].astype('double'), model2.predict, top_labels=2, hide_color=0, num_samples=1000, distance_metric='cosine')
    temp, mask = explanation.get_image_and_mask(label=1, positive_only=False,  num_features=15, hide_rest=False, min_weight=0.00000004)
    tempp = np.interp(temp, (temp.min(), temp.max()), (0, +1))
    plt.imshow(mark_boundaries(tempp, mask))
    plt.axis('off')
    plt.savefig('../app/src/Images/result-lime.png', bbox_inches='tight', pad_inches=0.0)

    pred = model2.predict(x)

    if preds[0,0] >= 0.5:
      response =  'Our network is {:.2%} sure this is NORMAL'.format(preds[0][0])
      return jsonify({"success": response})
    else: 
      response = 'Our network is {:.2%} sure this is PNEUMONIA'.format(1-preds[0][0])
      return jsonify({"success": response})

    # return ({"Success": "Successfully executed"})