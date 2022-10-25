from GANterfactual.classifier import get_adapted_alexNet
import tensorflow.keras as keras

TFDS_PATH = "/Users/dimitrymindlin/tensorflow_datasets/rsna_data"
checkpoint_path = "../checkpoints/alexnet_rsna/model/2022-10-13--13.03/model"
model = keras.models.load_model(f"../checkpoints/alexnet_rsna/2022-10-13--13.03/model", compile=True)
image_size = 512
batch_size = 32

model = get_adapted_alexNet(image_size)
model.load_weights(checkpoint_path)

train_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=(lambda x: x / 127.5 - 1.))

test_data = train_gen.flow_from_directory(
    directory=f"{TFDS_PATH}/test",
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    color_mode='rgb',
    classes={'normal': 0,
             'abnormal': 1}
)

print("Loaded weights")
print("Evaluation")
result = model.evaluate(test_data, batch_size=batch_size, )
result = dict(zip(model.metrics_names, result))
result_matrix = [[k, str(w)] for k, w in result.items()]
for metric, value in result.items():
    print(metric, ": ", value)
