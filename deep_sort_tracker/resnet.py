import tensorflow as tf

import tensorflow.contrib.slim as slim

import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

from visualization_utilities import read_img


def preprocess(input_img, bbox_2d, crop_size):
    '''
        Crop and resize the region of interest including 
        the object in the input image.
        Inputs: input_img is the input image of the current scene.
                bboxes_2d is the detected 2d bbox in the format
                of [x,y,w,h]
                crop_size gives the new size of the cropped region.
        Output: obj_roi is the 4D numpy array of the shape
                [1, image_height, image_width, depth] 
    '''
    # the top-left coordinate
    x = max(0,bbox_2d[0])
    y = max(0,bbox_2d[1])
    
    # the top-left cannot be negative
    x1 = tf.constant(x, tf.float32)
    y1 = tf.constant(y, tf.float32)
    
    # the  width and height of the bbox
    dim_x = bbox_2d[2]
    dim_y = bbox_2d[3] 
    
    # width and height of the input image
    w = float(input_img.shape[1])
    h = float(input_img.shape[0])
    
    # the bottom-right cannot exceed the image dimensions
    x2 = min(x+dim_x,w)
    y2 = min(y+dim_y,h)

    # Calculate bottom-right corners
    x2 = tf.constant(x2, tf.float32)
    y2 = tf.constant(y2, tf.float32)
    
    # build the graph to preprocess the input image 
    with tf.variable_scope('preprocess'):
        
        # dummy nodes that provide entry points for input image to the graph
        in_img_pl = tf.placeholder(tf.float32,[None, None, 3],"in")
        
        # add batch dimension
        img_input_batches = tf.expand_dims(in_img_pl, axis=0)
        
        # bbox corners in the format of [y1, x1, y2, x2] where (y1, x1) and
        # (y2, x2) are the coordinates of any diagonal pair of box corners
        box_corners = tf.stack([y1, x1, y2, x2], axis=0, name='bbox_corners')

        # the normalization constants for bbox corners
        norm_const = tf.constant([h, w, h, w], name = 'img_shape')
    
        # bbox corners in the normalized coordinates
        box_corners_norm = tf.div(box_corners,norm_const)
        
        box_corners_norm_batches = tf.expand_dims(box_corners_norm, axis=0)
    
        roi = tf.image.crop_and_resize(img_input_batches, box_corners_norm_batches, [0], crop_size)        
        
    # run the  graph 
    with tf.Session() as sess:
        obj_roi = sess.run(roi,feed_dict = {in_img_pl : input_img})
      
    return obj_roi
    

class resnet(object):
    
    def __init__(self, resnet_ckpt_path):
        '''
        Resnet constructor.
        Inputs: resnet_ckpt_path is the path to the directory where the 
                 checkpoint files of the pre-trained resnet is located.
        '''
        
        # define the resnet graph
        self.resnet_graph = tf.Graph()
        
        img_h = 64    #image height in pixels
        img_w = 128   #image height in pixels
        img_c = 3     #the number of channels
        
        # the shape eof the input image placeholder
        self.ph_shape = [None, img_h, img_w, img_c]
        
        self.ckpt_path = resnet_ckpt_path
    
        
    def resnet_arg_scope(self, weight_decay=1e-8):
        '''
            Specify default arguments which will be passed to the layers in resnet.
            Input:  weigth_decay is the l2 regularization coefficient for conv and fc 
                    weights.
            Output: default arguments to the layers of the network. 
        '''

        conv_fc_reg_fn = slim.l2_regularizer(weight_decay)
        
        def batch_norm_fn(x):
            return slim.batch_norm(x, scope = tf.get_variable_scope().name + "/bn")
        
        # disable learning the batch normalization and dropout ops
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=False):
            
            # default arguments passed to the conv2d and fc layers
            with slim.arg_scope([slim.conv2d, slim.fully_connected], 
                                activation_fn = tf.nn.elu, 
                                normalizer_fn = batch_norm_fn,
                                weights_regularizer = conv_fc_reg_fn):
                
                with slim.arg_scope([slim.conv2d], 
                                    padding='SAME', 
                                    stride = 1, 
                                    kernel_size = [3, 3]) as arg_scope:
                    
                    return arg_scope
                
    def define_inner_block(self, in_tensor, scope, increase_dim=False):
        '''
            Define the two conv layers. Follow two simple design rules: 
            (i) for the same output feature map size, the layers have 
            the same number of filters; and (ii) if the feature map size
            is halved, the number of filters is doubled.
            Inputs: in_tensor is the input tensor object, i,e., the 
                    output from the last operation of the network.
                    increase_dim is the bool indicating if the number 
                    of kernels will be increased and the feature map 
                    size will be halved.
            Output: the stacked layer consisting of 2 conv layers.
        '''

        # default stride
        s = 1
        
        # the number of filters used in the previous residual layer
        n = in_tensor.get_shape().as_list()[-1]
        
        # design rule
        if (increase_dim):
            # double the number of filters
            n *= 2
            # halve the feature map size
            s = 2
    
        net = slim.conv2d(in_tensor, n, stride = s, scope = scope + "/1")
    
        net = slim.dropout(net, keep_prob=0.6)
    
        net = slim.conv2d(net, n, stride = 1, activation_fn=None,
                          normalizer_fn=None, scope = scope + "/2")
        
        return net
    
    def residual_layer(self, in_tensor, scope, increase_dim = False, is_first = False):
        '''
            Build the stacked layers that learn the residual mapping.
            Then, add the shortcut and the residual mapping to obtain
            the desired underlying mapping.
            Inputs:  in_tensor is the input tensor object, i,e., the 
                     output tensor from the last operation of the network.
                     increase_dim is the bool indicating if the number of
                     kernels will be increased and the feature map size
                     will be halved.
                     is_first is the bool indicating if this is the first
                     residual layer in the network.                 
            Outputs: the desired underlying mapping,i.e., y = F(x) + x
                     where x is the shortcut and F(x) is the residual
                     mapping.
         '''
        
        # check if this is the first residual layer
        if not(is_first):
            net = slim.batch_norm(in_tensor, scope=scope + "/bn")
        else:
            net = in_tensor
        
        shortcut = net
        residual_mapping = self.define_inner_block(net,scope,increase_dim) 

        # the numbers of feature maps from the shortcut and residual learning
        shortcut_dim = shortcut.get_shape().as_list()[-1]
        residual_dim = residual_mapping.get_shape().as_list()[-1]
        
        # if the residual dimension increases,
        if (shortcut_dim != residual_dim):
            # check if the assumption about dims are true 
            assert (residual_dim == 2 * shortcut_dim),  \
                    '{0:d} != {1:d}'.format(residual_dim, 2 * shortcut_dim) 
            
            # match the depth dim using 1×1 convolution and 
            # the width and height dims using the stride of 2 
            projection = slim.conv2d(shortcut, 
                                     num_outputs = residual_dim,
                                     kernel_size = 1,
                                     stride = 2, 
                                     activation_fn = None,
                                     normalizer_fn = None,
                                     biases_initializer = None,     #skip biases   
                                     scope = scope +"/projection")
            
            net = projection + residual_mapping
        
        else:
            net = residual_mapping + shortcut 
            
        return net
                                        
    
    def build(self):
        '''
            Build the graph defining the resnet. The resnet consists
            of two conv layers followed by six residual layers. The
            output tensor from the residual layers are then processed 
            by a fully connected layer. Finally, a batch normalization
            and l2 normalization ops are applied to obtain feature the
            vector projected onto the unit hypersphere.
            Output: appearance feature map/descriptor which is a numpy 
                    array with the dimension of 128.
        '''
        
        # build the resnet operations
        with self.resnet_graph.as_default():
            
            with tf.variable_scope('input'):
                # dummy nodes that provide entry points for input image to the graph
                img_input = tf.placeholder(tf.float32,self.ph_shape,"roi")

            with slim.arg_scope( self.resnet_arg_scope() ):
                    
                # conv and max pooling layers
                net = slim.conv2d(img_input, num_outputs = 32, scope='conv1_1')
                
                net = slim.conv2d(net, num_outputs = 32, scope = 'conv1_2' )
                
                net = slim.max_pool2d(net, [3, 3], scope="pool1")
                
                # residual layers that learn the residual mappings
                net  = self.residual_layer(net, 'conv2_1',
                                           increase_dim=False, 
                                           is_first=True)
            
                net = self.residual_layer(net, 'conv2_3', 
                                          increase_dim=False)
                
                net = self.residual_layer(net, 'conv3_1',  
                                          increase_dim=True)
                
                net = self.residual_layer(net, 'conv3_3', 
                                          increase_dim=False)
                
                net = self.residual_layer(net, 'conv4_1', 
                                          increase_dim=True)
                
                net = self.residual_layer(net, 'conv4_3', 
                                          increase_dim=False)

                # the number of feature maps obtained from the last residual layer
                feat_num = net.get_shape().as_list()[-1]
                
                # flattens the last feature maps into 1D for each batch
                net = slim.flatten(net,  scope = 'feat_1d')
            
                # add dropout op to the input to the fc layer
                net = slim.dropout(net, keep_prob=0.6, scope = 'dropout')
                
                features = slim.fully_connected(net, feat_num, scope='fc1')
            
                # remove the feature distribution change before normalization
                features = slim.batch_norm(features, scope='ball')
                
                #  add euclidean norm op
                euc_norm = tf.norm(features, axis = [-2,-1], keep_dims=True) 
                           
                # add a small constant to prevent division by zero           
                euc_norm += tf.constant(1e-8,tf.float32)
                                       
                # project the features to unit hypersphere                            
                features = tf.div(features, euc_norm, 'norm_feat') 
     
    def restore_resnet(self, sess):
        '''
            Restore the resnet graph from its checkpoint file.
            Input: sess is the Session which sets tensors, and 
                   executes the operations defined in the resnet 
                   graph.
        '''   
            
        # the list of parameters in the resnet to be restored 
        var_to_restore = slim.get_variables_to_restore()
        
        '''
        for t in var_to_restore:
            print(t.name[:-2])
        
        var_to_restore = {t.name[:-2]:t for t in var_to_restore}
        '''
        
        # add operation to restore 
        saver = tf.train.Saver(var_to_restore)
        
        # restore the parameter values of the trained network
        saver.restore(sess, self.ckpt_path)
            
        print('The resnet restored!')

              
    def get_features(self, input_img, bboxes_2d):
        '''
            launch the resnet graph to extract object appearance descriptors.
            Input:  in_img is the input image of the current scene.
                    bboxes_2d are the list of the detected 2d bboxes in the 
                    format of [x,y,w,h].
            Output: the list of the appearance descriptors of the detected objects. 
        '''
        
        # the list of appearance descriptors
        app_des = []
        
        obj_rois = []
        
        # access the input image placeholder variable to feed the network
        img_input_ph = self.resnet_graph.get_tensor_by_name('input/roi:0')
        
        # access the op that you want to run. 
        feature_maps_out = self.resnet_graph.get_tensor_by_name('norm_feat:0')
        
        # execute the operations in the resnet graph
        with tf.Session(graph=self.resnet_graph) as sess:
            
            self.restore_resnet(sess)
            
            for i, bbox in enumerate(bboxes_2d):
                
                # cropped and resized region of interest including the detection
                roi = preprocess(input_img, bbox, self.ph_shape[1:3])
                
                obj_rois.append(roi[0])
          
                # obtain appearance descriptor for each bounding box
                feat_map = sess.run(feature_maps_out,feed_dict = {img_input_ph:roi})
                
                #print('Appearance descriptor obtained for bbox {0:d}'.format(i))
                
                app_des.append(feat_map)
                
        return app_des, obj_rois

if __name__ ==  '__main__':
    
    img_path = os.path.expanduser('~') + \
               '/resnet detections/img1'
               
    img_no = 148
    
    img_name = '{0:0>6}'.format(str(img_no)) + '.jpg'
    
    # Load an color image without changing
    rgb_img = read_img(img_path,img_name)
    
    fig  = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(rgb_img)
    ax.set_title('input image')
    ax.axis('off')  
    
    gt_path = os.path.expanduser('~') + \
               '/resnet detections/gt'
    
    # read the bbox detections at frame k
    with open(gt_path + '/' + 'gt.txt','r') as f:
       
        # the list of 2d bbox coordinates in the current image
        bboxes_2d = []
        
        # read each detections obtained at time k 
        for line in f:
            
            # remove trailing whitespace characters and split the line
            obj_info = line.rsplit()[0].split(',')
            
            frame_num = int(obj_info[0])
        
            # there is no more detections at time k
            if (frame_num > img_no):
                continue
            # do not consider previous detections
            elif (frame_num < img_no):
                continue
            
            obj_score = float(obj_info[6])
            if (obj_score < 0.2):
                continue
            
            visibility = float(obj_info[-1])
            if (visibility == 0):
                continue
            
            # 2d bbox coordinates: left,top, width and height 
            bbox_2d = [float(x) for x in obj_info[2:6]]
            
            bboxes_2d.append(bbox_2d)
            
    
            # width and height of the bboxes
            w = bbox_2d[2] 
            h = bbox_2d[3] 
            
            # draw the detection bbox
            ax.add_patch(patches.Rectangle(bbox_2d[0:2], w, h,  edgecolor='w', fill=False) )

    # path to the resnet chekcpoint file
    resnet_path = os.path.realpath(__file__) 

    resnet_ckpt_path = os.path.dirname(resnet_path) +  '/{0}/{1}'.format('resenet ckpt',
                                                        'mars-small128.ckpt-68577')

    resnet_feat_extractor = resnet(resnet_ckpt_path)

    resnet_feat_extractor.build()
    
    app_des,obj_rois = resnet_feat_extractor.get_features(rgb_img,bboxes_2d)
    
    
    fig = plt.figure()
    for i,roi in enumerate(obj_rois,1):
        ax = fig.add_subplot(len(bboxes_2d),1,i)
        img_roi = roi/255
        ax.imshow(img_roi)
        ax.axis('off')
    plt.suptitle('input to the resnet')
    
    # cosine distance function
    cos_dist = lambda x,y: 1 - np.dot(x,y.T)
    dists = []
    
    fig = plt.figure()
    for i,app_i in enumerate(app_des,1):
        ax = fig.add_subplot(len(app_des),1,i)
        ax.imshow(np.reshape(app_i,(8,16)))
        ax.axis('off')
        
        # compare if they are similar 
        dists.append([cos_dist(app_des,app_i)])
    plt.suptitle('Appearance descriptors')
    
    plt.show()
    