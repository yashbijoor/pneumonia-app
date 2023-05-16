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

    img_file2 = open('../app/src/Images/image.jpg', 'wb')
    img_file2.write(decoded_data)
    img_file2.close()
    
    img_path = "image.jpg"
    
    def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer
        last_conv_layer = model.get_layer(last_conv_layer_name)
        last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

        # Second, we create a model that maps the activations of the last conv
        # layer to the final class predictions
        classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        for layer_name in classifier_layer_names:
            x = model.get_layer(layer_name)(x)
        classifier_model = keras.Model(classifier_input, x)

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            # Compute activations of the last conv layer and make the tape watch it
            last_conv_layer_output = last_conv_layer_model(img_array)
            tape.watch(last_conv_layer_output)
            # Compute class predictions
            preds = classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]

        # This is the gradient of the top predicted class with regard to
        # the output feature map of the last conv layer
        grads = tape.gradient(top_class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]

        # The channel-wise mean of the resulting feature map
        # is our heatmap of class activation
        heatmap = np.mean(last_conv_layer_output, axis=-1)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        return heatmap, top_pred_index.numpy()
    
    def superimposed_img(image, heatmap):
        # We rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # We use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # We use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # We create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((224, 224))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * 0.4 + image
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
        return superimposed_img
    
    def categorical_smooth_loss(y_true, y_pred, label_smoothing=0.1):
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)
        return loss
    
    model = load_model("grad_cam_model.h5", custom_objects={'categorical_smooth_loss': categorical_smooth_loss})

    last_conv_layer_name = "conv5_block32_concat"
    classifier_layer_names = [
        "bn",
        "relu",
        "averagepooling2d_head",
        "flatten_head",
        "dense_head",
        "dropout_head",
        "predictions_head"
    ]

    # test image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224), interpolation=cv2.INTER_NEAREST)
    img = np.expand_dims(img,axis=0)
    x_img = preprocess_input(img)

    heatmap, top_index = make_gradcam_heatmap(img, model, last_conv_layer_name, classifier_layer_names)

    s_img = superimposed_img(img[0], heatmap)

    s_img.save("../app/src/Images/xray_cam.jpg")

    model2 = load_model("grad_updated.h5")

    img = cv2.imread(img_path)
    resized_img = cv2.resize(img, (224, 224))
    np_img = np.array(resized_img)
    np_img = np_img/255
    img_arr = np.array([np_img])

    # for index, resized_img in enumerate(img_arr):
    #     explainer = lime_image.LimeImageExplainer()
    #     explanation = explainer.explain_instance(resized_img.astype('double'), model2.predict, top_labels=5, hide_color=0, num_samples=1000)
    #     temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    #     plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    #     plt.axis('off')
    #     plt.savefig('../app/src/Images/result-lime.png', bbox_inches='tight')

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(x_img[0].astype('double'), model2.predict, top_labels=2, hide_color=0, num_samples=1000, distance_metric='cosine')
    temp, mask = explanation.get_image_and_mask(label=1, positive_only=False,  num_features=15, hide_rest=False, min_weight=0.00000004)
    tempp = np.interp(temp, (temp.min(), temp.max()), (0, +1))
    plt.imshow(mark_boundaries(tempp, mask))
    plt.axis('off')
    plt.savefig('../app/src/Images/result-lime.png', bbox_inches='tight', pad_inches=0.0)

    pred = model2.predict(x_img)

    if pred[0,0] >= 0.5:
      response =  'Our network is {:.2%} sure this is NORMAL'.format(pred[0][0])
      return jsonify({"success": response})
    else: 
      response = 'Our network is {:.2%} sure this is PNEUMONIA'.format(1-pred[0][0])
      return jsonify({"success": response})

    # return ({"Success": "Successfully executed"})


@app.route('/predictLime', methods=['POST'])
def predictLime():

    model = load_model("grad_updated.h5")

    img = cv2.imread("image.jpg")
    img = cv2.resize(img, (224, 224))
    np_img = np.array(img)

    np_img = np_img/255
    image_array = np.array([np_img])

    for index, img in enumerate(image_array):
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(img.astype('double'), model.predict, top_labels=5, hide_color=0, num_samples=1000)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.axis('off')
        plt.savefig('../app/src/Images/result-lime.png', bbox_inches='tight')

    prediction = model.predict(image_array)

    class_x=np.argmax(prediction,axis=1)

    if class_x[0] == 1:
        predicted_class = "Pneumonia"
    else:
        predicted_class = "Normal"
    
    print(predicted_class)
    
    return ({"Output": predicted_class})
