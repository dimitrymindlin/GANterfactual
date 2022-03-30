import tensorflow as tf
import cv2
import numpy as np
from albumentations import Compose, CLAHE
from tensorflow.keras.models import Model
from skimage.color import gray2rgb

AUGMENTATIONS_TEST = Compose([
    CLAHE(always_apply=True)
])

def resize_with_pad(im):
  desired_size = 512
  old_size = im.shape[:2] # old_size is in (height, width) format

  ratio = float(desired_size)/max(old_size)
  new_size = tuple([int(x*ratio) for x in old_size])

  # new_size should be in (width, height) format

  im = cv2.resize(im, (new_size[1], new_size[0]))

  delta_w = desired_size - new_size[1]
  delta_h = desired_size - new_size[0]
  top, bottom = delta_h//2, delta_h-(delta_h//2)
  left, right = delta_w//2, delta_w-(delta_w//2)

  color = [0, 0, 0]
  return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
      value=color)

def preprocess(img):
    """
    Loading and preprocessing of the image
    """
    img = AUGMENTATIONS_TEST(image=img)["image"]
    if len(img.shape) < 3:
        img = tf.expand_dims(img, axis=-1)
    if img.shape[-1] != 3:
        img = tf.image.grayscale_to_rgb(img)
    img = tf.image.resize_with_pad(img, 512, 512)
    image_batch = tf.expand_dims(img, axis=0)
    return tf.keras.applications.inception_v3.preprocess_input(image_batch)


def get_activations_at(input_image, model):
    """
    Get the activations of the model in layer i. The last conv layer is usually used, but it applies to any.
    """
    def find_target_layer(model):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for idx, layer in enumerate(reversed(model.layers)):
            # check to see if the layer has a 4D output
            try:
                if len(layer.output_shape) == 4:
                    return idx
            except AttributeError:
                print("Output ...")
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
    # index the layer
    out_layer = model.layers[find_target_layer(model)]

    # change the output of the model
    submodel = Model(inputs=model.inputs, outputs=out_layer.output)

    # return the activations
    return submodel.predict(input_image)


def postprocess_activations(activations):
    """
    Transform activations into a workable image format.
    """

    # using the approach in https://arxiv.org/abs/1612.03928
    output = np.abs(activations)
    output = np.sum(output, axis=-1).squeeze()

    # resize and convert to image
    output = cv2.resize(output, (512, 512))
    output /= output.max()
    output *= 255
    return 255 - output.astype('uint8')


def apply_heatmap2(weights, img):
    """
    Combine the initial image with the image of the activations to generate the attention heatmap.
    """
    combined = gray2rgb(weights * img)
    return combined / 127.5 - 1. # 3 channel image [-1, 1]
    """plt.imshow(weights)
    plt.show()
    plt.imshow(combined, cmap='gray')
    plt.show()
    plt.imshow(img, cmap='gray')
    plt.show()
    plt.savefig('activation_map.jpg')"""


def generate_img_attention(model, img):
    """
    Flow to get the attention map.
    """

    input_image = preprocess(img)
    resized_img = resize_with_pad(img)
    activations = get_activations_at(input_image, model)
    weights = postprocess_activations(activations)
    apply_heatmap2(weights, resized_img)


"""root = '/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/valid/XR_WRIST/'

classifier_folder = f"../checkpoints/2022-03-24--12.42/model"
classifier = tf.keras.models.load_model(classifier_folder, compile=False)
img = imread(root + "patient11186/study2_positive/image1.png")
generate_img_attention(classifier, img)
"""