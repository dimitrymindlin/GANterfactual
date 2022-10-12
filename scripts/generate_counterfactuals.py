from GANterfactual.cyclegan import CycleGAN
from configs.gan_training_config import gan_config
from skimage.metrics import structural_similarity as compare_ssim
import imutils
import cv2
import matplotlib.pyplot as plt

gan = CycleGAN(gan_config)
gan.load_existing(cyclegan_folder="../checkpoints/GAN")

image_list = gan.predict()

for imgs in image_list:
    original_image = imgs[0]
    counterfactual_image = imgs[1]

    # convert the images to grayscale
    grayA = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(counterfactual_image, cv2.COLOR_BGR2GRAY)

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(counterfactual_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # show the output images
    print("showing the pics")
    fig = plt.figure(figsize=(4, 4))
    images = [original_image, counterfactual_image, diff, thresh]
    columns = 2
    rows = 2
    for idx, i in enumerate(range(1, columns * rows + 1)):
        fig.add_subplot(rows, columns, i)
        plt.imshow(images[idx])
    plt.show()


