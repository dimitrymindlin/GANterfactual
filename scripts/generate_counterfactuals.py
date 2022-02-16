from GANterfactual.cyclegan import CycleGAN
from configs.gan_training_config import gan_config

gan = CycleGAN(gan_config)
gan.load_existing(cyclegan_folder="../checkpoints/GAN",
                  classifier_weight=gan_config['train']['classifier_weight'])

gan.predict()


"""from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils
import cv2

# load the two input images
imageA = imgs[0]
imageB = imgs[1]

# convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
"""