import tensorflow as tf

def vgg16(keep_prob, num_classes, image_shape):

    nn_input = tf.placeholder(tf.float32, shape = (None, image_shape[0], image_shape[1], 3), name = 'nn_input')

    C = 4 # Number of upconvolutional filters, C * number_of_classes
    
    # First layers. Input 800x352x3, output 400x176x64
    conv1_1 = tf.layers.conv2d(nn_input, filters = 64, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv1_1')

    #bn1_1 = tf.layers.batch_normalization(conv1_1)
        
    conv1_2 = tf.layers.conv2d(conv1_1, filters = 64, kernel_size = 3, 
    strides = 1, padding='SAME', activation=tf.nn.relu,             kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv1_2')

    #bn1_2 = tf.layers.batch_normalization(conv1_2)

    pool1 = tf.nn.max_pool(conv1_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME', name = 'pool1')

    # Second layers. Input 400x176,64, output 200x88x128
    conv2_1 = tf.layers.conv2d(pool1, filters = 128, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu,                             kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv2_1')

    #bn2_1 = tf.layers.batch_normalization(conv2_1)

    conv2_2 = tf.layers.conv2d(conv2_1, filters = 128, kernel_size = 3, 
        strides = 1, padding='SAME', activation=tf.nn.relu,    kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv2_2')

    #bn2_2 = tf.layers.batch_normalization(conv2_2)

    pool2 = tf.nn.max_pool(conv2_2, ksize = [1, 2, 2, 1], 
        strides = [1, 2, 2, 1], padding='SAME', name = 'pool2')

    # Third layers. Input 200x88x128, output 100x44x256
    conv3_1 = tf.layers.conv2d(pool2, filters = 256, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv3_1')

    #bn3_1 = tf.layers.batch_normalization(conv3_1)

    conv3_2 = tf.layers.conv2d(conv3_1, filters = 256, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv3_2')

    #bn3_2 = tf.layers.batch_normalization(conv3_2)

    pool3 = tf.nn.max_pool(conv3_2, ksize = [1, 2, 2, 1], 
        strides = [1, 2, 2, 1], padding='SAME', name = 'pool3')

    # Fourth layers. Input 100x44x256, output 50x22x512
    conv4_1 = tf.layers.conv2d(pool3, filters = 512, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv4_1')

    #bn4_1 = tf.layers.batch_normalization(conv4_1)

    conv4_2 = tf.layers.conv2d(conv4_1, filters = 512, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv4_2')

    #bn4_2 = tf.layers.batch_normalization(conv4_2)

    conv4_3 = tf.layers.conv2d(conv4_2, filters = 512, kernel_size = 3,strides = 1, padding='SAME', activation=tf.nn.relu,                         kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv4_3')

    #bn4_3 = tf.layers.batch_normalization(conv4_3)

    pool4 = tf.nn.max_pool(conv4_3, ksize = [1, 2, 2, 1], 
        strides = [1, 2, 2, 1], padding='SAME', name = 'pool4') 

    # Fifth layers. Input 50x22x512, output 25x11x512
    conv5_1 = tf.layers.conv2d(pool4, filters = 512, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu,kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv5_1')

    #bn5_1 = tf.layers.batch_normalization(conv5_1)

    conv5_2 = tf.layers.conv2d(conv5_1, filters = 512, kernel_size = 3, 
        strides = 1, padding='SAME', activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv5_2')

    #bn5_2 = tf.layers.batch_normalization(conv5_2)

    conv5_3 = tf.layers.conv2d(conv5_2, filters = 512, kernel_size = 3, 
        strides = 1, padding='SAME', activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv5_3')

    #bn5_3 = tf.layers.batch_normalization(conv5_3)

    pool5 = tf.nn.max_pool(conv5_3, ksize = [1, 2, 2, 1], 
        strides = [1, 2, 2, 1], padding='SAME', name = 'pool5')

    # Fully convolutional layers
    fc1 = tf.layers.conv2d(pool5, filters= 1024, kernel_size = 3, strides = 1, padding='SAME', activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name='fc1')

    drop1 = tf.nn.dropout(fc1, keep_prob, name='drop1')

    fc2 = tf.layers.conv2d(drop1, filters= 1024, kernel_size = 1, strides = 1, padding='SAME', activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name='fc2')

    drop2 = tf.nn.dropout(fc2, keep_prob, name='drop2')        

    # 1x1 convolution to reduce the last layer from 4096 to 3 classes (scores)
    fc3 = tf.layers.conv2d(drop2, filters = num_classes, kernel_size = 1,strides = 1, padding='SAME', kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name='fc3')

    # First decoder layer. Upsampling from 25x11xnc to 50x22xnc
    decoder1 = tf.layers.conv2d_transpose(fc3, filters=C*num_classes,kernel_size = 4, strides = 2, padding = 'SAME', 
    kernel_initializer = tf.random_normal_initializer(stddev=0.01), name = 'decoder1')
    #bn_dec_1 = tf.layers.batch_normalization(decoder1)
    # 1x1 convolution of pool4 to 50x22xnc
    #pool4_scaled = tf.multiply(pool4, 0.01)
    pool4_1x1 = tf.layers.conv2d(pool4, filters = C*num_classes, kernel_size = 1, strides = 1, padding='SAME', kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d())
    # Add both layers (Skip layer connection)
    decoder1_out = tf.add(decoder1, pool4_1x1, name = 'decoder1_out')

    # Second decoder layer. Upsampling from 50x22xnc to 100x44xnc
    decoder2 = tf.layers.conv2d_transpose(decoder1_out, filters= C*num_classes,kernel_size = 4, strides = 2, padding = 'SAME', kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'decoder2')
    #bn_dec_2 = tf.layers.batch_normalization(decoder2)
    # 1x1 convolutions of pool3 to 100x72xnc
    #pool3_scaled = tf.multiply(pool3, 0.01)
    pool3_1x1 = tf.layers.conv2d(pool3, filters = C*num_classes, kernel_size = 1,strides = 1, padding='SAME', kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d())
    # Add both layers (Skip connection)
    decoder2_out = tf.add(decoder2, pool3_1x1, name = 'decoder2_out')

    # Third decoder layer. Upsampling from 100x44xnc to 200x88xnc
    decoder3 = tf.layers.conv2d_transpose(decoder2_out, filters= C*num_classes,kernel_size = 4, strides = 2, padding = 'SAME', kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'decoder3')
    #bn_dec_3 = tf.layers.batch_normalization(decoder3)
    # 1x1 convolutions of pool3 to 100x72xnc
    #pool3_scaled = tf.multiply(pool3, 0.01)
    pool2_1x1 = tf.layers.conv2d(pool2, filters = C*num_classes, kernel_size = 1,strides = 1, padding='SAME', kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d())
    # Add both layers (Skip connection)
    decoder3_out = tf.add(decoder3, pool2_1x1, name = 'decoder3_out')

    # Fourth decoder layer. Upsampling from 200x88xnc to 400x176xnc
    decoder4 = tf.layers.conv2d_transpose(decoder3_out, filters= C*num_classes,kernel_size = 4, strides = 2, padding = 'SAME', kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'decoder4')
    #bn_dec_4 = tf.layers.batch_normalization(decoder4)
    # 1x1 convolutions of pool3 to 100x72xnc
    #pool3_scaled = tf.multiply(pool3, 0.01)
    pool1_1x1 = tf.layers.conv2d(pool1, filters = C*num_classes, kernel_size = 1,strides = 1, padding='SAME', kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d())
    # Add both layers (Skip connection)
    decoder4_out = tf.add(decoder4, pool1_1x1, name = 'decoder4_out')

    # Final decoder layer. Upsampling from 400x176xnc to 800x352xnc
    nn_output = tf.layers.conv2d_transpose(decoder4_out, filters=num_classes, kernel_size = 4, strides = 2, padding = 'SAME', kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'out_node')

    return nn_input, nn_output

def resnet_block(input, filter1, filter2, layer):

    conv1 = tf.layers.conv2d(input, filters = filter1, kernel_size = 1, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv_'+layer+'_1')

    conv2 = tf.layers.conv2d(conv1, filters = filter1, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv_'+layer+'_2')

    conv3 = tf.layers.conv2d(conv2, filters = filter2, kernel_size = 1, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv_'+layer+'_3')

    output = tf.add(input, conv3)

    return output

def resnet50(keep_prob, num_classes, image_shape):

    C = 1 # C * number_of_classes for upconvolution

    nn_input = tf.placeholder(tf.float32, shape = (None, image_shape[0], image_shape[1], 3), name = 'nn_input')

    # Input: 800x352x3, output: 400x176x64
    conv1 = tf.layers.conv2d(nn_input, filters = 64, kernel_size = 7, strides = 2, padding='SAME', activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv1')

    # Input: 400x176x64, output: 200x88x64
    pool1 = tf.nn.max_pool(conv1, ksize = [1, 3, 
    3, 1], strides = [1, 2, 2, 1], padding='SAME', name = 'pool1')

    conv2_1 = resnet_block(pool1, 64, 256, '2_1')
    conv2_2 = resnet_block(conv2_1, 64, 256, '2_2')
    conv2_3 = resnet_block(conv2_2, 64, 256, '2_3')

    #Input: 200x88x256, output: 100x44x256
    pool2 = tf.nn.max_pool(conv1, ksize = [1, 3, 
    3, 1], strides = [1, 2, 2, 1], padding='SAME', name = 'pool2')

    conv3_1 = resnet_block(pool2, 128, 512, '3_1')
    conv3_2 = resnet_block(conv3_1, 128, 512, '3_2')
    conv3_3 = resnet_block(conv3_2, 128, 512, '3_3')
    conv3_4 = resnet_block(conv3_3, 128, 512, '3_4')

    #Input: 100x44x512, output: 50x22x512
    pool3 = tf.nn.max_pool(conv1, ksize = [1, 3, 
    3, 1], strides = [1, 2, 2, 1], padding='SAME', name = 'pool3')

    conv4_1 = resnet_block(pool3, 256, 1024, '4_1')
    conv4_2 = resnet_block(conv4_1, 256, 1024, '4_2')
    conv4_3 = resnet_block(conv4_2, 256, 1024, '4_3')
    conv4_4 = resnet_block(conv4_3, 256, 1024, '4_4')
    conv4_5 = resnet_block(conv4_4, 256, 1024, '4_5')
    conv4_6 = resnet_block(conv4_5, 256, 1024, '4_6')

    #Input: 50x22x1024, output: 25x11x1024
    pool4 = tf.nn.max_pool(conv1, ksize = [1, 3, 
    3, 1], strides = [1, 2, 2, 1], padding='SAME', name = 'pool4')

    conv5_1 = resnet_block(pool4, 512, 2048, '5_1')
    conv5_2 = resnet_block(conv5_1, 512, 2048, '5_2')
    conv5_3 = resnet_block(conv5_2, 512, 2048, '5_3')

    drop1 = tf.nn.dropout(conv5_3, keep_prob, name='drop1')

    fc1 = tf.layers.conv2d(drop1, filters= 1024, kernel_size = 1, strides = 1, padding='SAME', activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name='fc1')

    drop2 = tf.nn.dropout(fc1, keep_prob, name='drop2')        

    # 1x1 convolution to reduce the last layer from 1024 to 3 classes (scores)
    fc2 = tf.layers.conv2d(drop2, filters = num_classes, kernel_size = 1,strides = 1, padding='SAME', kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name='fc2')

    # First decoder layer. Upsampling from 25x11xnc to 50x22xnc
    decoder1 = tf.layers.conv2d_transpose(fc2, filters=C*num_classes,kernel_size = 4, strides = 2, padding = 'SAME', 
    kernel_initializer = tf.random_normal_initializer(stddev=0.01), name = 'decoder1')
    # 1x1 convolution of pool4 to 50x22xnc
    #pool4_scaled = tf.multiply(pool4, 0.01)
    pool3_1x1 = tf.layers.conv2d(pool3, filters = C*num_classes, kernel_size = 1, strides = 1, padding='SAME', kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d())
    # Add both layers (Skip layer connection)
    decoder1_out = tf.add(decoder1, pool3_1x1, name = 'decoder1_out')

    # Second decoder layer. Upsampling from 50x22xnc to 100x44xnc
    decoder2 = tf.layers.conv2d_transpose(decoder1_out, filters= C*num_classes,kernel_size = 4, strides = 2, padding = 'SAME', kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'decoder2')
    # 1x1 convolutions of pool2 to 100x72xnc
    #pool3_scaled = tf.multiply(pool3, 0.01)
    pool2_1x1 = tf.layers.conv2d(pool2, filters = C*num_classes, kernel_size = 1,strides = 1, padding='SAME', kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d())
    # Add both layers (Skip connection)
    decoder2_out = tf.add(decoder2, pool2_1x1, name = 'decoder2_out')

    # Third decoder layer. Upsampling from 100x44xnc to 200x88xnc
    decoder3 = tf.layers.conv2d_transpose(decoder2_out, filters= C*num_classes,kernel_size = 4, strides = 2, padding = 'SAME', kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'decoder3')
    # 1x1 convolutions of pool3 to 100x72xnc
    #pool3_scaled = tf.multiply(pool3, 0.01)
    pool1_1x1 = tf.layers.conv2d(pool1, filters = C*num_classes, kernel_size = 1,strides = 1, padding='SAME', kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d())
    # Add both layers (Skip connection)
    decoder3_out = tf.add(decoder3, pool1_1x1, name = 'decoder3_out')

    # Fourth decoder layer. Upsampling from 200x88xnc to 400x176xnc
    decoder4 = tf.layers.conv2d_transpose(decoder3_out, filters= C*num_classes,kernel_size = 4, strides = 2, padding = 'SAME', kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'decoder4')
    # 1x1 convolutions of pool3 to 100x72xnc
    #pool3_scaled = tf.multiply(pool3, 0.01)
    conv1_1x1 = tf.layers.conv2d(conv1, filters = C*num_classes, kernel_size = 1,strides = 1, padding='SAME', kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d())
    # Add both layers (Skip connection)
    decoder4_out = tf.add(decoder4, conv1_1x1, name = 'decoder4_out')

    # Final decoder layer. Upsampling from 400x176xnc to 800x352xnc
    nn_output = tf.layers.conv2d_transpose(decoder4_out, filters=num_classes, kernel_size = 4, strides = 2, padding = 'SAME', kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = 'out_node')

    return nn_input, nn_output


# Function from the Udacity Self-Driving car Engineer Semantic Segmentation Project
def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    vgg_graph = tf.get_default_graph()

    keep_prob = vgg_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    nn_input = vgg_graph.get_tensor_by_name(vgg_input_tensor_name)
    l3 = vgg_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    l4 = vgg_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    l7 = vgg_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return nn_input, keep_prob, l3, l4, l7

# Function from the Udacity Self-Driving car Engineer Semantic Segmentation Project
def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # 1x1 convolutions to reduce the last layer (25x18x4096) to (25x18xnc)
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1), kernel_initializer=tf.random_normal_initializer(stddev=0.01))

    # First layer: Upsampling to (50x36xnc)
    decoder1 = tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, strides=(2,2), padding = 'same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.random_normal_initializer(stddev=0.01))

    # Scaling of pooling layer 4 as implemented in the original FCN paper
    #layer4_scaled = tf.multiply(vgg_layer4_out, 0.01, name='layer4_scaled')

    # 1x1 convolution of l4_out (50x36x4096) to (50x36xnc)
    l4_conv1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1,1),padding='same', kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    # Add both layers (Skip layer)
    decoder1_out = tf.add(decoder1, l4_conv1x1)

    # Second layer: Upsampling to (100x72xnc)
    decoder2 = tf.layers.conv2d_transpose(decoder1_out, num_classes, 4, strides=(2,2), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    # Scaling of pooling layer 3 as implemented in the original FCN paper
    #layer3_scaled = tf.multiply(vgg_layer3_out, 0.0001, name='layer3_scaled')
    # 1x1 convolution of l3_out (100x72x4096) to (100x72xnc)
    l3_conv1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1,1),padding='same', kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    # Add both layers (Skip layer)
    decoder2_out = tf.add(decoder2, l3_conv1x1)

    # Third layer: Upsampling to (800x576xnc)
    decoder3 = tf.layers.conv2d_transpose(decoder2_out, num_classes, 16, strides=(8,8), padding='same',  kernel_initializer=tf.random_normal_initializer(stddev=0.01))

    return decoder3