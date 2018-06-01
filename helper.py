import numpy as np
from glob import glob
import os.path
from imageio import imread
import random


def map_labels(orig_vec):
    """
    Maps original 13 classes to new 3 classes
    Roads (7) and road markers (6) are assigned class label 1
    Vehicles (10) are assigned class label 2
    All other labels are assigned 0
    :param orig_vec: label matrix with 1st dim: (x*y) pixels of label image, 2nd dim (13) one-hot encoded labels (from 0 to 12)
    :return: label matrix with 1st dim identical to orig_vec, 2nd dim (3) one-hot encoded labels (from 0 to 2)
    """
    a = np.ones((13,1))
    b = np.zeros((13, 2))
    label_vec = np.concatenate((a, b), axis = 1)
    label_vec[6][1] = 1
    label_vec[6][0] = 0
    label_vec[7][1] = 1
    label_vec[7][0] = 0
    label_vec[10][2] = 1
    label_vec[10][0] = 0
    return np.matmul(orig_vec, label_vec)

def gen_batch_function(image_folder, label_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size, num_classes):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(image_folder, '*.png'))
        label_paths = glob(os.path.join(label_folder, '*.png'))
        im_height, im_width = image_shape
        mask = mask_hood() # masks out hood pixels as class zero

        # Shuffle both image and label files in the same order
        indices = random.sample(range(len(image_paths)), len(image_paths))
        image_paths = [image_paths[i] for i in indices]
        label_paths = [label_paths[i] for i in indices]

        for batch_i in range(0, len(image_paths), batch_size):
            #time_start = time.time()
            images = []
            labels = []
            vertical_start = 169
            vertical_end = vertical_start + im_height
            
            for image_file, label_file in zip(
                image_paths[batch_i:batch_i+batch_size], label_paths[batch_i:batch_i+batch_size]):

                # Loads images and resizes them. For labels, only R channel used
                image = imread(image_file)
                image = image[vertical_start:vertical_end, :im_width, :]
                image = (image / 255.0) - 0.5
                label = imread(label_file)
                # Labels are stored in the R channel of RGB (channel 0)
                label = label[:,:,0]
                # Transforms hood pixels. Before: vehicle (class 2), after: class 0. Mask is 0 for all hood pixels, 1 otherwise
                label = np.multiply(label, mask)
                # Crops image below hood and above horizon where all pixels are of class 0 and therefore not of interest. Speeds up training.
                label = label[vertical_start:vertical_end, :im_width]

                label_shape = label.shape
                label = (np.array(label)).reshape(-1)
                #label = [label_map[elem] for elem in label]
                label = np.eye(13)[label] # Convert label vector to one-hot
                label = map_labels(label) # Convert labels to 0,1 or 2

                #label = np.eye(num_classes)[label]
                #label = (np.array(label)).reshape((label_shape[0], label_shape[1], num_classes)).tolist()
                label = label.reshape((label_shape[0], label_shape[1], num_classes))

                images.append(image)
                labels.append(label)

            #print("Time for loading and converting images/labels: {0}".format(time.time() - time_start))

            yield np.array(images), labels
    return get_batches_fn

def mask_hood():
    hood_img_path = './Train/CameraSeg/0.png'
    hood_img = imread(hood_img_path)
    hood_img = hood_img[:,:,0]
    
    # Mask all pixels below y=500 which are vehicles (covers hood of car)
    mask = hood_img[480:,:] == 10
    mask = mask * np.ones((120,800))
    mask = 1 - mask
    mask = np.concatenate((np.ones((480,800)),mask), axis = 0)
    mask = mask.astype(int)
    return mask
