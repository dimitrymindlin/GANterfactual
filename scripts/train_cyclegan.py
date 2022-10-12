from __future__ import print_function, division
import os
from GANterfactual.cyclegan import CycleGAN

if __name__ == '__main__':
    # local_path = "/Users/dimitrymindlin/tensorflow_datasets/rsna_data"
    local_path = "../tensorflow_datasets/rsna_data"
    gan = CycleGAN()
    gan.construct(classifier_path=f"checkpoints/inception/model", classifier_weight=1)
    gan.train(dataset_name=local_path, epochs=20, batch_size=1, print_interval=10,
              sample_interval=100)
    gan.save(os.path.join('models', 'GANterfactual_inception'))
