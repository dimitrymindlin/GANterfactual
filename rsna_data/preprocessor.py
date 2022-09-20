import tqdm
from PIL import Image
import os
import argparse
from sklearn.model_selection import train_test_split
import pydicom as dcm
import pandas as pd
import random

"""ap = argparse.ArgumentParser()
ap.add_argument('-i', '--in', required=True, help='input folder')
#ap.add_argument('-o', '--out', required=True, help='output folder')
ap.add_argument('-t', '--test', required=True, help='proportion of images used for test')
ap.add_argument('-v', '--validation', required=True, help='proportion of images used for validation')
ap.add_argument('-d', '--dimension', required=True, help='new dimension for files')
args = vars(ap.parse_args())"""

def preprocess(in_path, out_path, test_size, val_size, dim):
    ### Create df with patient_id, image_info and labels, without the label 'Not Normal'
    train_labels_df = pd.read_csv(in_path + '/stage_2_train_labels.csv')
    class_info_df = pd.read_csv(in_path + '/stage_2_detailed_class_info.csv')
    class_info_df = class_info_df[class_info_df["class"].str.contains("Not Normal") == False]
    train_class_df = class_info_df.merge(train_labels_df.drop_duplicates(subset=['patientId']), how='left')
    train_class_df = train_class_df.drop_duplicates(
        subset=['patientId'])  # results in 14863 images as in GANterfactual paper!

    ### Make train rsna_data tuple list with (path: str, label: int)
    train_data_list = []
    for n, row in train_class_df.iterrows():
        # --- Initialize patient entry into parsed
        pid = row['patientId']
        if row['Target'] == 0:
            train_data_list.append(((f'{in_path}/stage_2_train_images/{pid}.dcm'), 0))
        else:
            train_data_list.append(((f'{in_path}/stage_2_train_images/{pid}.dcm'), 1))

    ### train test split
    random.shuffle(train_data_list)
    train, test_val = train_test_split(train_data_list, test_size=test_size + val_size)
    test, val = train_test_split(test_val, test_size=val_size / (test_size + val_size))

    print('n train samples', len(train))
    print('n valid samples', len(val))
    print('n test samples', len(test))  # results in partitions as in GANterfactual paper!

    ### Make dirs for new rsna_data
    dirs = ["normal", "abnormal"]
    for label in dirs:
        train_path = os.path.join(out_path, 'train', label)
        os.makedirs(train_path, exist_ok=True)
        test_path = os.path.join(out_path, 'test', label)
        os.makedirs(test_path, exist_ok=True)
        val_path = os.path.join(out_path, 'validation', label)
        os.makedirs(val_path, exist_ok=True)

        ### Resize and save rsna_data to new dirs
        resize_and_save(train_path, train, dim, label)
        resize_and_save(test_path, test, dim, label)
        resize_and_save(val_path, val, dim, label)


def resize_and_save(out_path, images, dim, label):
    for image_in_path in tqdm.tqdm(images, desc=label):
        # Check that image has correct label
        if (label == "normal" and image_in_path[1]) == (label == "abnormal" and image_in_path[1] == 1):
                image_out_path = os.path.join(out_path, image_in_path[0].split("/")[-1])[:-4] + ".png"
                im = dcm.dcmread(image_in_path[0]).pixel_array
                im = Image.fromarray(im)
                im_resized = im.resize((dim, dim), Image.ANTIALIAS)
                im_resized.save(image_out_path, 'png', quality=100)


# preprocess(args['in'], os.path.join('.'), float(args['test']), float(args['validation']), int(args['dimension']))
preprocess("/Users/dimitrymindlin/tensorflow_datasets/downloads/rsna-pneumonia-detection-challenge", ".", 0.2, 0.1, 512)
