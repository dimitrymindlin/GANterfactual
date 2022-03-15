from GANterfactual.cyclegan import CycleGAN
from configs.gan_training_config import gan_config
from utils.image_utils import show_diff_in_pngs

gan = CycleGAN(gan_config)
gan.load_existing(cyclegan_folder="../checkpoints/GANterfactual_2022-03-07--23.04")
gan.evaluate()