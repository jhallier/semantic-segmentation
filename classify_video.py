import sys, skvideo.io, json, base64
import numpy as np
#from PIL import Image
from io import BytesIO, StringIO
import tensorflow as tf
import cv2

file = sys.argv[-1]

# Settings
video_method = 'skvideo'
process_video = 'multiple_frames'
frozen_graph = './inference/frozen_fcn_vgg16_10e-2048.pb'
vertical_start = 169
im_height = 352
im_height_orig = 600
lower_height = im_height_orig - im_height - vertical_start
im_width = 800
batch_size = 8

# Define encoder function
# Thanks to @phmagic on the slack channel #lyftchallenge
def encode(array):
    retval, buffer = cv2.imencode('.png', array)
    return base64.b64encode(buffer).decode("utf-8")

if video_method == 'skvideo':
    video = skvideo.io.vread(file)
    nr_frames = video.shape[0]
elif video_method == 'opencv':
    video = cv2.VideoCapture(file)

answer_key = {}

# Frame numbering for the json result, starting at 1
frame = 1

pixels = im_height * im_width
vertical_end = vertical_start + im_height
frame_end = 0

# Open inference graph file
with tf.gfile.GFile(frozen_graph, "rb") as file:
    graph_def = tf.GraphDef()    
    graph_def.ParseFromString(file.read())

G = tf.Graph()

with tf.Session(graph=G) as sess:

    logits, = tf.import_graph_def(graph_def, return_elements=['logits:0'])
    input_image = G.get_tensor_by_name("import/nn_input:0")
    keep_prob = G.get_tensor_by_name("import/keep_prob:0")
 

    if video_method == 'skvideo':

        if process_video == 'multiple_frames':
        
            for batch in range(0, nr_frames, batch_size):
                if (frame + batch_size) <= nr_frames:
                    frame_end = batch + batch_size
                else:
                    frame_end = nr_frames
                batch = video[batch:frame_end]
                batch_length = batch.shape[0]
                batch_resized = batch[:, vertical_start:vertical_end, :, :]
                batch_resized = (batch_resized / 255.0) - 0.5

                pred = sess.run(logits, feed_dict={input_image:batch_resized, keep_prob:1.0})
                #softmax = sess.run(tf.nn.softmax(pred))
                
                for index in range(batch_resized.shape[0]):
                    single_frame = pred[index*pixels:(index+1)*pixels, :]
                    index = np.argmax(single_frame, axis = 1)
                    binary_car_result = np.where(index == 2, 1, 0).astype('uint8')
                    binary_road_result = np.where(index == 1, 1, 0).astype('uint8')
                    binary_car_result = binary_car_result.reshape(im_height, im_width)
                    binary_road_result = binary_road_result.reshape(im_height, im_width)
                    binary_car_result = np.concatenate((np.zeros((vertical_start, im_width)), binary_car_result), axis = 0)
                    binary_car_result = np.concatenate((binary_car_result, np.zeros((lower_height, im_width))), axis = 0)
                    binary_road_result = np.concatenate((np.zeros((vertical_start, im_width)), binary_road_result), axis = 0)
                    binary_road_result = np.concatenate((binary_road_result, np.zeros((lower_height, im_width))), axis = 0)

                    answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
                    
                    # Increment frame
                    frame+=1

        if process_video == 'single_frames':
            for rgb_frame in video:
                    
                rgb_frame = rgb_frame[vertical_start:vertical_end, :, :]
                rgb_4d = np.expand_dims(rgb_frame, axis=0)
                rgb_4d = (rgb_4d / 255.0) - 0.5

                pred = sess.run(logits, feed_dict={input_image:rgb_4d, keep_prob:1.0})
                softmax = sess.run(tf.nn.softmax(pred))
                index = np.argmax(softmax, axis = 1)
                segment = index.reshape(im_height, im_width)
                segment = np.concatenate((np.zeros((vertical_start, im_width)), segment), axis = 0)
                segment = np.concatenate((segment, np.zeros((im_height_orig - segment.shape[0], im_width))), axis = 0)

                binary_car_result = np.where(segment == 2, 1, 0).astype('uint8')
                binary_road_result = np.where(segment == 1, 1, 0).astype('uint8')

                answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
                
                # Increment frame
                frame+=1
    
    if video_method == 'opencv':
        while(video.isOpened()):
            batch = []
            for index in range(batch_size):
                ret, frame = video.read()
                if ret == True:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tolist()
                    batch.append(rgb_frame)
            batch = np.array(batch)
            if index == 1:
                batch = np.expand_dims(batch, axis=0)

            batch_length = batch.shape[0]
            batch_resized = batch[:, vertical_start:vertical_end, :, :]
            print(batch_resized)
            batch_resized = (batch_resized / 255.0) - 0.5
            
            pred = sess.run(logits, feed_dict={input_image:batch_resized, keep_prob:1.0})
            softmax = sess.run(tf.nn.softmax(pred))

            for index in range(batch_resized.shape[0]):
                single_frame = softmax[index*pixels:(index+1)*pixels, :]
                index = np.argmax(single_frame, axis = 1)
                binary_car_result = np.where(index == 2, 1, 0).astype('uint8')
                binary_road_result = np.where(index == 1, 1, 0).astype('uint8')
                binary_car_result = binary_car_result.reshape(im_height, im_width)
                binary_road_result = binary_road_result.reshape(im_height, im_width)
                binary_car_result = np.concatenate((np.zeros((vertical_start, im_width)), binary_car_result), axis = 0)
                binary_car_result = np.concatenate((binary_car_result, np.zeros((lower_height, im_width))), axis = 0)
                binary_road_result = np.concatenate((np.zeros((vertical_start, im_width)), binary_road_result), axis = 0)
                binary_road_result = np.concatenate((binary_road_result, np.zeros((lower_height, im_width))), axis = 0)

                answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
                frame+=1

# Print output in proper json format
print (json.dumps(answer_key))