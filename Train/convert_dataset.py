import tensorflow as tf
from os.path import isfile
from imageio import imread
import os
import sys
import getopt
import numpy as np
import cv2


# Dictionary mapping the original labels to those of interest for the challenge
# Roads (7) and road markers (6) are assigned class label 1
# Vehicles (10) are assigned class label 2
# All other labels are assigned 0
label_map = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 1,
        7: 1,
        8: 0,
        9: 0,
        10: 2,
        11: 0,
        12: 0
    }

def load_image_resize(filename, resize=False, new_size=(800, 600)):

    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if resize:
        width, height = new_size
        image = image[:height,:width,:]
    return image

def load_map_label(filename, map=False, resize=False, new_size=(800, 600)):
    label = cv2.imread(filename)
    label = label[:,:,2] # labels are stored in R channel, OpenCV reads in BGR order -> channel 2 is the R channel
    if map:
        label_shape = label.shape
        label = label.reshape(-1) # Unroll label vector
        label = [label_map[elem] for elem in label]
        label = np.array(label)
        label = label.reshape(label_shape)
    if resize:
        width, height = new_size
        label = label[:height,:width]
    return label

def create_tf_record(image_filename, label_filename, im_size):
  '''
  param xmins: List of normalized left x coordinates in bounding box (1 per box)
  param xmaxs: List of normalized right x coordinates in bounding box (1 per box)
  param ymins: List of normalized top y coordinates in bounding box (1 per box)
  param ymaxs: List of normalized bottom y coordinates in bounding box (1 per box)
  param classes_text: List of string class name of bounding box (1 per box)
  param classes: List of integer class id of bounding box (1 per box)
  '''
  image_data = load_image_resize(image_filename, resize = True, new_size = im_size)
  label_data = load_map_label(label_filename, map = True, resize=True, new_size = im_size)

  im_width, im_height = im_size

  encoded_image_data = tf.compat.as_bytes(image_data.tostring()) # Encoded image bytes
  filename = tf.compat.as_bytes(image_filename)
  image_format = 'png'.encode('utf8') # b'jpeg' or b'png'

  label_data = label_data.reshape(-1)

  '''
  tf_single_dataset = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(im_height),
      'image/width': dataset_util.int64_feature(im_width),
      'image/filename': dataset_util.bytes_feature(os.path.basename(filename)),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/class/label': dataset_util.int64_list_feature(mapped_labels)
  }))
  '''
  tf_single_dataset = tf.train.Example(features=tf.train.Features(feature={
      'train/image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image_data])),
      'train/label': tf.train.Feature(int64_list=tf.train.Int64List(value=label_data))
  }))

  return tf_single_dataset

def convert_dataset(images_path, labels_path, output_path):

    # The following image properties must be set according to the dataset you are importing
    height = 576 # Image height
    width = 800 # Image width

    im_size = (width, height)

    writer = tf.python_io.TFRecordWriter(output_path)

    index = 0
    _, __, train_images = os.walk(images_path).__next__()
    len_train_images = len(train_images)
    image_file = images_path + str(index) + '.png'
    label_file = labels_path + str(index) + '.png'

    # Loads train and label image files and stores them in lists X, Y
    while (isfile(image_file) and isfile(label_file)):
        if ((index % 10) == 0):
            print("Converting image {0} of {1}".format(index, len_train_images))
  
        tf_single_dataset = create_tf_record(image_file, label_file, im_size)
        writer.write(tf_single_dataset.SerializeToString())

        index = index + 1
        image_file = images_path + str(index) + '.png'
        label_file = labels_path + str(index) + '.png'
    
    print("Dataset successfully converted!")
    print("Writing tfrecord file... (this may take some time)")
    writer.close()


def main(argv):
    images = './CameraRGB/'
    label = './CameraSeg/'
    outfile = './lyftdata.tfrecord'
    try:
        opts, args = getopt.getopt(argv, "", ["images_file=", "label_file=", "output_file="])
    except getopt.GetoptError:
        print("Syntax error. Correct syntax: python convert_dataset.py --images_path=[path to images] --label_path=[path to label images] --output_file=[path and filename to store tfrecord file]")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "images_file":
            images = arg
        if opt == "--label_file":
            label = arg
        if opt == "--output_file":
            outfile = arg
    convert_dataset(images, label, outfile)

if __name__ == "__main__":
    main(sys.argv[1:])
