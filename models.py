import tensorflow as tf

def vgg16(keep_prob, num_classes, image_shape):

    nn_input = tf.placeholder(tf.float32, shape = (None, image_shape[0], image_shape[1], 3), name = 'nn_input')
    
    # First layers. Input 800x352x3, output 400x176x64
    conv1_1 = tf.layers.conv2d(nn_input, filters = 64, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv1_1')
        
    conv1_2 = tf.layers.conv2d(conv1_1, filters = 64, kernel_size = 3, 
    strides = 1, padding='SAME', activation=tf.nn.relu,             kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv1_2')

    pool1 = tf.nn.max_pool(conv1_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME', name = 'pool1')

    # Second layers. Input 400x176,64, output 200x88x128
    conv2_1 = tf.layers.conv2d(pool1, filters = 128, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu,                             kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv2_1')

    conv2_2 = tf.layers.conv2d(conv2_1, filters = 128, kernel_size = 3, 
        strides = 1, padding='SAME', activation=tf.nn.relu,    kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv2_2')

    pool2 = tf.nn.max_pool(conv2_2, ksize = [1, 2, 2, 1], 
        strides = [1, 2, 2, 1], padding='SAME', name = 'pool2')

    # Third layers. Input 200x88x128, output 100x44x256
    conv3_1 = tf.layers.conv2d(pool2, filters = 256, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv3_1')

    conv3_2 = tf.layers.conv2d(conv3_1, filters = 256, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv3_2')

    pool3 = tf.nn.max_pool(conv3_2, ksize = [1, 2, 2, 1], 
        strides = [1, 2, 2, 1], padding='SAME', name = 'pool3')

    # Fourth layers. Input 100x44x256, output 50x22x512
    conv4_1 = tf.layers.conv2d(pool3, filters = 512, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv4_1')

    conv4_2 = tf.layers.conv2d(conv4_1, filters = 512, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv4_2')

    conv4_3 = tf.layers.conv2d(conv4_2, filters = 512, kernel_size = 3,strides = 1, padding='SAME', activation=tf.nn.relu,                         kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv4_3')

    pool4 = tf.nn.max_pool(conv4_3, ksize = [1, 2, 2, 1], 
        strides = [1, 2, 2, 1], padding='SAME', name = 'pool4') 

    # Fifth layers. Input 50x22x512, output 25x11x512
    conv5_1 = tf.layers.conv2d(pool4, filters = 512, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu,kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv5_1')

    conv5_2 = tf.layers.conv2d(conv5_1, filters = 512, kernel_size = 3, 
        strides = 1, padding='SAME', activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv5_2')

    conv5_3 = tf.layers.conv2d(conv5_2, filters = 512, kernel_size = 3, 
        strides = 1, padding='SAME', activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv5_3')

    pool5 = tf.nn.max_pool(conv5_3, ksize = [1, 2, 2, 1], 
        strides = [1, 2, 2, 1], padding='SAME', name = 'pool5')

    # Fully convolutional layers
    fc1 = tf.layers.conv2d(pool5, filters= 2048, kernel_size = 1, strides = 1, padding='SAME', activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name='fc1')

    drop1 = tf.nn.dropout(fc1, keep_prob, name='drop1')

    fc2 = tf.layers.conv2d(drop1, filters= 2048, kernel_size = 1, strides = 1, padding='SAME', activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name='fc2')

    drop2 = tf.nn.dropout(fc2, keep_prob, name='drop2')        

    # 1x1 convolution to reduce the last layer from 4096 to 3 classes (scores)
    fc3 = tf.layers.conv2d(drop2, filters = num_classes, kernel_size = 1,strides = 1, padding='SAME', kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name='fc3')


    # First decoder layer. Upsampling from 25x11xnc to 50x22xnc
    decoder1 = tf.layers.conv2d_transpose(fc3, filters=num_classes,kernel_size = 4, strides = 2, padding = 'SAME', 
    kernel_initializer = tf.random_normal_initializer(stddev=0.01), name = 'decoder1')
    # 1x1 convolution of pool4 to 50x22xnc
    #pool4_scaled = tf.multiply(pool4, 0.01)
    pool4_1x1 = tf.layers.conv2d(pool4, filters = num_classes, kernel_size = 1, strides = 1, padding='SAME', kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d())
    # Add both layers (Skip layer connection)
    decoder1_out = tf.add(decoder1, pool4_1x1, name = 'decoder1_out')

    # Second decoder layer. Upsampling from 50x22xnc to 100x44xnc
    decoder2 = tf.layers.conv2d_transpose(decoder1_out, filters=num_classes,kernel_size = 4, strides = 2, padding = 'SAME', kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'decoder2')
    # 1x1 convolutions of pool3 to 100x72xnc
    #pool3_scaled = tf.multiply(pool3, 0.01)
    pool3_1x1 = tf.layers.conv2d(pool3, filters = num_classes, kernel_size = 1,strides = 1, padding='SAME', kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d())
    # Add both layers (Skip connection)
    decoder2_out = tf.add(decoder2, pool3_1x1, name = 'decoder2_out')

    # Final decoder layer. Upsampling from 100x44xnc to 800x352xnc
    nn_output = tf.layers.conv2d_transpose(decoder2_out, filters=num_classes, kernel_size = 16, strides = 8, padding = 'SAME', kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'out_node')

    return nn_input, nn_output

def resnet_block(keep_prob, num_classes, image_shape):
    pass
