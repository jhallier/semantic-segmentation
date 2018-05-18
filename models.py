import tensorflow as tf

def vgg16(nn_input, keep_prob, num_classes, image_shape):
    
    # First layers. Input 800x576x3, output 400x288x64
    conv1_1 = tf.layers.conv2d(nn_input, filters = 64, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_regularizer = tf.contrib.layers.l2_regularizer(0.001), kernel_initializer = tf.random_normal_initializer(stddev=0.01), name = 'conv1_1')
        
    conv1_2 = tf.layers.conv2d(conv1_1, filters = 64, kernel_size = 3, 
    strides = 1, padding='SAME', activation=tf.nn.relu,             kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),                  kernel_initializer= tf.random_normal_initializer(stddev=0.01), name = 'conv1_2')

    pool1 = tf.nn.max_pool(conv1_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME', name = 'pool1')

    # Second layers. Input 400x288,64, output 200x144x128
    conv2_1 = tf.layers.conv2d(pool1, filters = 128, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu,                             kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),kernel_initializer= tf.random_normal_initializer(stddev=0.01), name = 'conv2_1')

    conv2_2 = tf.layers.conv2d(conv2_1, filters = 128, kernel_size = 3, 
        strides = 1, padding='SAME', activation=tf.nn.relu,    kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer= tf.random_normal_initializer(stddev=0.01), name = 'conv2_2')

    pool2 = tf.nn.max_pool(conv2_2, ksize = [1, 2, 2, 1], 
        strides = [1, 2, 2, 1], padding='SAME', name = 'pool2')

    # Third layers. Input 200x144x128, output 100x72x256
    conv3_1 = tf.layers.conv2d(pool2, filters = 256, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),                  kernel_initializer= tf.random_normal_initializer(stddev=0.01), name = 'conv3_1')

    conv3_2 = tf.layers.conv2d(conv3_1, filters = 256, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),kernel_initializer=tf.random_normal_initializer(stddev=0.01), name = 'conv3_2')

    pool3 = tf.nn.max_pool(conv3_2, ksize = [1, 2, 2, 1], 
        strides = [1, 2, 2, 1], padding='SAME', name = 'pool3')

    # Fourth layers. Input 100x72x256, output 50x36x512
    conv4_1 = tf.layers.conv2d(pool3, filters = 512, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),kernel_initializer=tf.random_normal_initializer(stddev=0.01), name = 'conv4_1')

    conv4_2 = tf.layers.conv2d(conv4_1, filters = 512, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer= tf.random_normal_initializer(stddev=0.01), name = 'conv4_2')

    conv4_3 = tf.layers.conv2d(conv4_2, filters = 512, kernel_size = 3,strides = 1, padding='SAME', activation=tf.nn.relu,                         kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),              kernel_initializer= tf.random_normal_initializer(stddev=0.01), name = 'conv4_3')

    pool4 = tf.nn.max_pool(conv4_3, ksize = [1, 2, 2, 1], 
        strides = [1, 2, 2, 1], padding='SAME', name = 'pool4') 

    # Fifth layers. Input 50x36x512, output 25x18x512
    conv5_1 = tf.layers.conv2d(pool4, filters = 512, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),kernel_initializer= tf.random_normal_initializer(stddev=0.01), name = 'conv5_1')

    conv5_2 = tf.layers.conv2d(conv5_1, filters = 512, kernel_size = 3, 
        strides = 1, padding='SAME', activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),kernel_initializer= tf.random_normal_initializer(stddev=0.01), name = 'conv5_2')

    conv5_3 = tf.layers.conv2d(conv5_2, filters = 512, kernel_size = 3, 
        strides = 1, padding='SAME', activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),kernel_initializer=tf.random_normal_initializer(stddev=0.01), name = 'conv5_3')

    pool5 = tf.nn.max_pool(conv5_3, ksize = [1, 2, 2, 1], 
        strides = [1, 2, 2, 1], padding='SAME', name = 'pool5')

    # Fully connected layers
    fc1 = tf.layers.conv2d(pool5, filters= 4096, kernel_size = 1, strides = 1, padding='SAME', activation = tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.random_normal_initializer(stddev=0.01), name='fc1')

    drop1 = tf.nn.dropout(fc1, keep_prob, name='drop1')

    fc2 = tf.layers.conv2d(drop1, filters= 4096, kernel_size = 1, strides = 1, padding='SAME', activation = tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.random_normal_initializer(stddev=0.01), name='fc2')

    drop2 = tf.nn.dropout(fc2, keep_prob, name='drop2')        

    # 1x1 convolution to reduce the last layer from 4096 to 3 classes (scores)
    fc3 = tf.layers.conv2d(drop2, filters = num_classes, kernel_size = 1,strides = 1, padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.random_normal_initializer(stddev=0.01), name='fc3')

    # Decoder (FCN32): Upsampling from 25x18xnc to 800x576xnc
    nn_output = tf.layers.conv2d_transpose(fc3, filters=num_classes, kernel_size = 64, strides = 32, padding = 'SAME', 
    kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer = tf.random_normal_initializer(stddev=0.01), name = 'nn_output')

    '''
    # First decoder layer. Upsampling from 25x18xnc to 50x36xnc
    decoder1 = tf.layers.conv2d_transpose(fc3, filters=num_classes,kernel_size = 4, strides = 2, padding = 'SAME', 
    kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer = tf.random_normal_initializer(stddev=0.01), name = 'decoder1')

    # 1x1 convolution of pool4 to 50x36xnc
    pool4_1x1 = tf.layers.conv2d(pool4, filters = num_classes, kernel_size = 1, strides = 1, padding='SAME', kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer = tf.random_normal_initializer(stddev=0.01))
    # Add both layers (Skip layer connection)
    decoder1_out = tf.add(decoder1, pool4_1x1, name = 'decoder1_out')

    # Second decoder layer. Upsampling from 50x36xnc to 100x72xnc
    decoder2 = tf.layers.conv2d_transpose(decoder1_out, filters=num_classes,kernel_size = 4, strides = 2, padding = 'SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.random_normal_initializer(stddev=0.01), name = 'decoder2')
    # 1x1 convolutions of pool3 to 100x72xnc
    pool3_1x1 = tf.layers.conv2d(pool3, filters = num_classes, kernel_size = 1,strides = 1, padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    # Add both layers (Skip connection)
    decoder2_out = tf.add(decoder2, pool3_1x1, name = 'decoder2_out')

    # Third decoder layer. Upsampling from 100x72xnc to 200x144xnc
    decoder3 = tf.layers.conv2d_transpose(decoder2_out, filters=num_classes,kernel_size = 4, strides = 2, padding = 'SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.random_normal_initializer(stddev=0.01), name = 'decoder3')
    # 1x1 convolutions of pool2 to 200x144xnc
    pool2_1x1 = tf.layers.conv2d(pool2, filters = num_classes, kernel_size = 1,strides = 1, padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    # Add both layers (Skip connection)
    decoder3_out = tf.add(decoder3, pool2_1x1, name = 'decoder3_out')

    # Fourth decoder layer. Upsampling from 200x144xnc to 400x288xnc
    decoder4 = tf.layers.conv2d_transpose(decoder3_out, filters=num_classes,kernel_size = 4, strides = 2, padding = 'SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.random_normal_initializer(stddev=0.01), name = 'decoder4')
    # 1x1 convolutions of pool1 to 400x288x2
    pool1_1x1 = tf.layers.conv2d(pool1, filters = num_classes, kernel_size = 1, strides = 1, padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    # Add both layers (Skip connection)
    decoder4_out = tf.add(decoder4, pool1_1x1, name = 'decoder4_out')

    # Final decoder layer. Upsampling from 400x288x2 to 800x576x2
    nn_output = tf.layers.conv2d_transpose(decoder4_out, filters=num_classes, kernel_size = 4, strides = 2, padding = 'SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.random_normal_initializer(stddev=0.01), name = 'out_node')
    '''

    return nn_output