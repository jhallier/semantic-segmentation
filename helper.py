import numpy as np
from glob import glob
import os.path
from imageio import imread
import random
import cv2


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
    :param image_folder: Path to folder that contains all the training images
    :param label_folder: Path to folder with all corresponding labels
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

            images = []
            labels = []
            vertical_start = 169
            vertical_end = vertical_start + im_height
            
            for image_file, label_file in zip(
                image_paths[batch_i:batch_i+batch_size], label_paths[batch_i:batch_i+batch_size]):

                # Loads images and resizes them. For labels, only R channel used
                image = imread(image_file)
                image = image[vertical_start:vertical_end, :im_width, :]
                # Modifies H and L color channel of HLS to augment images and reduce overfitting
                image = color_transform(image)
                # Normalize images for training
                image = (image / 255.0) - 0.5

                label = imread(label_file)
                # Labels are stored in the R channel of RGB (channel 0)
                label = label[:,:,0]
                # Transforms hood pixels. Before: vehicle (class 2), after: class 0. Mask is 0 for all hood pixels, 1 otherwise
                label = np.multiply(label, mask)

                # Performs a l/r flip of the image/label with a 1/3 probability
                randFlip = np.random.randint(low=0, high=3)
                if randFlip == 0:
                    image, label = flip_image_label(image, label)

                # Crops image below hood and above horizon where all pixels are of class 0 and therefore not of interest. Speeds up training.
                label = label[vertical_start:vertical_end, :im_width]

                label_shape = label.shape
                label = (np.array(label)).reshape(-1)

                label = np.eye(13)[label] # Convert label vector to one-hot
                label = map_labels(label) # Convert labels to 0,1 or 2

                label = label.reshape((label_shape[0], label_shape[1], num_classes))

                images.append(image)
                labels.append(label)

            yield np.array(images), labels
    return get_batches_fn

def mask_hood():
    """
    Takes all pixels that are associated with the hood of the car and returns a mask with all hood pixels set to 0, all others set to 1.
    This mask can be multiplied with any image and all hood pixels will consequently be assigned 0
    """
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

def restore_image(image_vec, im_size = (352,800,3)):
    """
    Restores image to original size with each label (0, 1, 2) restored to a RGB value defined in the color dictionary
    :param image_vec: vector of length image width*height
    :param im_size: New image size after resizing (width,height,color channels)
    :return: RGB image of size im_size
    """
    color = {
    0: [0, 0, 0],
    1: [0, 0, 255],
    2: [0, 255, 0],
    }
    image_vec = [color[elem] for elem in image_vec]
    image_vec = np.array(image_vec)
    image = image_vec.reshape(im_size)
    return image

def color_transform(image):
    '''
    Performs a random change of the H and L channel of the HLS color space
    args: image: image array to be modified
    output: modified image
    '''
    HLS_before = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    h, l, s = cv2.split(HLS_before)

    randH = np.random.randint(low=0, high=20, dtype=np.uint8)
    signH = np.random.randint(low=0, high=2)
    if signH == 1:
        h = h + randH
    else:
        mask = (h >= randH) # Prevents for h values to be below 0
        h = h - mask * randH
    randL = np.random.randint(low=0, high=30, dtype=np.uint8)
    signL = np.random.randint(low=0, high=2)
    if signL == 1:
        l = l + randL
    else:
        mask = l >= randL
        l = l - mask * randL

    # Thresholding H to be of a maximum of 179 (maximum value in OpenCV)
    retval, h = cv2.threshold(h, thresh = 179, maxval = 179, type=cv2.THRESH_TRUNC)
    HLS_after = cv2.merge((h,l,s))

    output = cv2.cvtColor(HLS_after, cv2.COLOR_HLS2RGB)

    return output


def flip_image_label(image, label):

    image = cv2.flip(image, flipCode=1)
    label = cv2.flip(label, flipCode=1)

    return image, label

# From user MHKang on the Lyftchallenge slack channel
def lyft_score(y_true, y_pred):
   true = K.reshape(y_true,(-1,im_w*im_h,3))
   pred = K.reshape(y_pred,(-1,im_w*im_h,3))

   inte = true * pred
   true_sum = K.sum(true,axis=1)
   pred_sum = K.sum(pred,axis=1)
   inte_sum = K.sum(inte,axis=1)

   precision = inte_sum / (pred_sum + 1)
   recall = inte_sum / (true_sum + 1)

   beta2_r = 0.5**2
   beta2_v = 2.0**2
   beta2np = np.asarray([1.0, beta2_r, beta2_v])
   beta2 = K.constant(value=beta2np)

   fscore_num = (1.0 + beta2) * precision * recall
   fscore_den = beta2 * precision + recall + 1e-6

   fscore = fscore_num / fscore_den

   avg_weightsnp = np.asarray([0.0, 0.5, 0.5]) # None, Road, Vehicles
   avg_weights = K.constant(value=avg_weightsnp)
   favg = K.sum(avg_weights * fscore,axis=1)
   return favg

def lyft_score_loss(y_true, y_pred):
   return 1. - lyft_score(y_true, y_pred)

