from os.path import abspath, join
import tensorflow as tf
import numpy as np
import csv
import progressbar
import os
import Class_Image as Ci
    
def safe_folder_creation(folder_path):
    """
    Safely create folder and return the new path value if a change occurred.

    :param folder_path: proposed path for the folder that must be created
    :type folder_path: str
    :return: path of the created folder
    :rtype: str
    """
    # Boolean initialization
    folder = True

    while folder:
        # Check path validity
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            folder = False
        else:
            folder_path = input(Back.RED + 'Folder already exists : {}\n Please enter a new path !'.format(folder_path)
                                + Style.RESET_ALL)
    return folder_path


def get_label_comparison_from_string(string):
    """
    Create the label probabilities from the string comparison value

    The value being either 'left', 'right' or 'No preference'
    :param string: Comparison value
    :type string: str
    :return: label probabilities
    :rtype: list
    """
    if string == "left":
        return [1, 0]
    elif string == "right":
        return [0, 1]


def get_label_score_from_string(string):
    """
    Create the label probabilities from the string comparison value

    The value being either 'left', 'right' or 'No preference'
    :param string: Comparison value
    :type string: str
    :return: label score
    :rtype: int
    """
    if string == "left":
        return 1
    elif string == "right":
        return 0

#Taken from https://github.com/laggiss/OttawaGreenspacesComparisons
def preprocessing_duels(csv_path, img_size, image_folder, save_folder, test):
    """
    Create the inputs of the comparison network from the comparisons csv and save them as npy

    A csv line has the following format:
        image_key_1, image_key_2, winner, ip_address
        the image keys are 22 character-long string
        winner is one of the 3 following string : left, right, equivalent
    :param csv_path: path of the comparisons csv
    :type csv_path: str
    :param img_size: input image size
    :type img_size: int
    :param image_folder: path of the image folder
    :type image_folder: str
    :param save_folder: path of the folder where npy are saved
    :type save_folder: str
    :param test: value to determine the size of the data dedicated to the test set.
                It can either be a proportion inside [0,1] or the number of comparisons kept aside.
    :type test: float
    :return:
    :rtype:
    """

    # List initialization
    left_images = []
    right_images = []
    labels = []
    labels_score = []
    print(csv_path)
    # Get data from csv
    with open(csv_path, 'r') as csvfileReader:
        reader = csv.reader(csvfileReader, delimiter=',')
        print("Creating inputs from csv ...")
        pbar = progressbar.ProgressBar()
        #for r in reader:
        #    print(r)

        for line in reader:#pbar(reader):
            # Do not include No preference comparisons
            if line != [] and line[2] != 'No preference':
                # Create Image instances
                #print(line)
                left_image_path = get_filename_from_key(line[0], image_folder)
                right_image_path = get_filename_from_key(line[1], image_folder)
                #print(left_image_path)
                left_img = Ci.Image(left_image_path)
                right_img = Ci.Image(right_image_path)

                # Add images to list
                left_images.append(left_img.preprocess_image(img_size))
                right_images.append(right_img.preprocess_image(img_size))

                # Add labels to list
                labels.append(get_label_comparison_from_string(line[2]))
                labels_score.append(get_label_score_from_string(line[2]))

    # Compute number of comparisons kept for test set
    if len(labels) > test > 1:
        nb_test = int(test)
    elif 0 <= test <= 1:
        nb_test = int(test * len(labels))
    else:
        raise ValueError

    print("Done\nSaving test set ...")
    # Create test dataset
    test_left = np.array(left_images[:nb_test])
    test_right = np.array(right_images[:nb_test])
    test_labels = np.array(labels[:nb_test])
    test_labels_score = np.array(labels_score[:nb_test])

    # Save testing dataset as npy
    test_folder = os.path.join(save_folder, "test")
    test_folder = safe_folder_creation(test_folder)
    np.save(os.path.join(test_folder, "test_left_{}".format(img_size)), np.array(test_left))
    np.save(os.path.join(test_folder, "test_right_{}".format(img_size)), np.array(test_right))
    np.save(os.path.join(test_folder, "test_labels_{}".format(img_size)), np.array(test_labels))
    np.save(os.path.join(test_folder, "test_labels_score_{}".format(img_size)), np.array(test_labels_score))
    
    del test_left, test_right, test_labels,test_labels_score
    
    print("Done\nSaving train set ...")
    # Create training dataset
    train_left = np.array(left_images[nb_test:])
    train_right = np.array(right_images[nb_test:])
    train_labels = np.array(labels[nb_test:])
    train_labels_score = np.array(labels_score[nb_test:])

    # Save training dataset as npy
    train_folder = os.path.join(save_folder, "train")
    train_folder = safe_folder_creation(train_folder)
    np.save(os.path.join(train_folder, "train_left_{}".format(img_size)), np.array(train_left))
    del train_left
    np.save(os.path.join(train_folder, "train_right_{}".format(img_size)), np.array(train_right))
    del train_right
    np.save(os.path.join(train_folder, "train_labels_{}".format(img_size)), np.array(train_labels))
    np.save(os.path.join(train_folder, "train_labels_score_{}".format(img_size)), np.array(train_labels_score))
    print("Done")
    
# ----------------------------------------------------------------------------------------------------------------------
def get_filename_from_key(key, image_folder):
    """
    Get the complete path of an image from its key

    :param key: image key
    :type key: str
    :param image_folder: folder containing images
    :type image_folder: str
    :return: image path
    :rtype: str
    """
    images = os.listdir(image_folder)
    for image in images:
        if key in image:
            return os.path.join(image_folder, image)
        
def npy_to_tfrecords(DATA_FOLDER, val_split=0.2):
    tfrecord = tf.io.TFRecordWriter(join(DATA_FOLDER, "data_train.tfrecord"))
    
    data_left = np.load(join(DATA_FOLDER, "train", "train_left_224.npy"), mmap_mode=None)
    data_right = np.load(join(DATA_FOLDER, "train", "train_right_224.npy"),  mmap_mode=None)
    data_label = np.load(join(DATA_FOLDER, "train", "train_labels_224.npy"), mmap_mode=None)
    labels_score = np.load(join(DATA_FOLDER, "train", "train_labels_score_224.npy"),  mmap_mode=None)
    for j in range(data_left.shape[0]): #iterate through all rows
        if(j == int(data_left.shape[0] * (1 - val_split)) ):
            tfrecord = tf.io.TFRecordWriter(join(DATA_FOLDER, "data_val.tfrecord"))
        features = {
        'data_label' : tf.train.Feature(int64_list=tf.train.Int64List(value=data_label[j])),
        'labels_score': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array([labels_score[j]]))),
        'data_left': tf.train.Feature(int64_list=tf.train.Int64List(value=data_left[j].astype(int).flatten())),
        'data_right': tf.train.Feature(int64_list=tf.train.Int64List(value=data_right[j].astype(int).flatten())),
            
        }
        
        example = tf.train.Example(features=tf.train.Features(feature=features))
        tfrecord.write(example.SerializeToString())

def build_dataset(csv_path, img_size, image_folder, save_folder, test=0.2, val=0.2):
    print("Saving as .npy")
    preprocessing_duels(csv_path, img_size, image_folder, save_folder, test)
    print("Converting to .tfrecords")
    npy_to_tfrecords(save_folder, val)
    print("Done")
    
