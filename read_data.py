import os
import numpy as np
import scipy.ndimage
import glob
import pickle

dataset_path = "tiny-imagenet-200/train/"

nb_classes= 10

def read_training_data(filename):
    if os.path.isfile(filename):
        return
    X_data = []
    label_data = []
    image_directories = glob.glob('tiny-imagenet-200/train/n*')
    image_directories = image_directories[:nb_classes]
    # print(image_directories)
    for d in image_directories:
        image_filenames = os.listdir(d + '/images')
        for fname in image_filenames:
            # print("reading file {}".format(fname))
            X = scipy.ndimage.imread(d + '/images/'+fname, mode='RGB')
            label = fname.split('_')[0]
            X_data.append(X)
            label_data.append(label)

    print("len X_data = {}".format(len(X_data)))
    X_data = np.stack(X_data, axis=0)
    labels_unique = np.unique(label_data)
    y_unique = range(len(labels_unique))
    label_y_map = dict(zip(labels_unique, y_unique))
    y_data = [label_y_map[label] for label in label_data]
    y_data = np.asarray(y_data)
    assert X_data.shape[1] == 64
    assert X_data.shape[2] == 64
    assert X_data.shape[3] == 3
    with open(filename, 'wb') as f:
        pickle.dump({'features': X_data, 'labels': y_data}, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    print("reading data...")
    read_training_data('train.p')
