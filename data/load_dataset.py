import numpy as np

from GANterfactual.dataloader import DataLoader
from configs.gan_training_config import gan_config
from data.mura_dataset import MuraDataset
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

gan_config["dataset"]["download"] = False

"""(train, validation, test), info = tfds.load(
            'MuraGANImages',
            split=['train[:80%]', 'train[80%:]', 'test'],
            shuffle_files=True,
            as_supervised=True,
            download=mura_config["dataset"]["download"],
            with_info=True,
        )

fig = tfds.visualization.show_examples(test, info)"""

#dataset = MuraDataset(gan_config)

data_loader = DataLoader(config=gan_config)

for neg, pos in data_loader.load_batch():
    raw_pic = np.asarray(neg[0]).astype('float32') / 1
    print('Min: %.3f, Max: %.3f' % (raw_pic.min(), raw_pic.max()))
    plt.imshow(raw_pic, cmap="Greys", interpolation='nearest')
    plt.show()
    new_pic = 0.5 * raw_pic + 0.5
    print('Min: %.3f, Max: %.3f' % (new_pic.min(), new_pic.max()))
    plt.imshow(new_pic, cmap="Greys")
    plt.show()
    """new_pic = raw_pic / 255.0
    print('Min: %.3f, Max: %.3f' % (new_pic.min(), new_pic.max()))
    plt.imshow(new_pic, cmap="Greys")
    plt.show()"""
    quit()

#dataset.benchmark()

#get_num_of_samples(dataset)
