from GANterfactual.cyclegan import CycleGAN
from configs.gan_training_config import gan_config
from utils.image_utils import show_diff_in_pngs
import tensorflow as tf


gan = CycleGAN(gan_config)
gan.load_existing(cyclegan_folder="../checkpoints/GANterfactual_2022-03-26--06.18")
gan.evaluate()
oracle = tf.keras.models.load_model(f"../checkpoints/2022-03-24--12.18/model", compile=False)
gan.evaluate_oracle_score(oracle)
