import tensorflow as tf
import cv2
from albumentations import Compose, CLAHE
from tensorflow.keras.models import Model

AUGMENTATIONS_TEST = Compose([
    CLAHE(always_apply=True)
])

def get_activations_at(model, img):
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
    return submodel(img)


def postprocess_activations(activations):
    """
    Transform activations into a workable image format.
    """

    # using the approach in https://arxiv.org/abs/1612.03928
    output = tf.math.abs(activations)
    output = tf.squeeze(tf.math.reduce_sum(output, axis=-1))

    # resize and convert to image
    output = tf.image.resize_with_pad(output, 512, 512)
    output = tf.math.divide(output, tf.math.reduce_max(output))
    output = tf.math.multiply(output, 255)
    return tf.math.subtract(tf.cast(255, tf.uint8), tf.cast(output, tf.uint8))


def apply_heatmap2(weights, img):
    """
    Combine the initial image with the image of the activations to generate the attention heatmap.
    """
    weights = tf.cast(weights, tf.float32)
    normalized_weights = tf.keras.applications.inception_v3.preprocess_input(weights) # put [0,255] to [-1,1]
    combined = tf.math.multiply(normalized_weights, img)
    return combined # should have [-1,1] and (None, 512, 512, 3)
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
    activations = get_activations_at(model, img)
    weights = postprocess_activations(activations)
    return apply_heatmap2(weights, img)

