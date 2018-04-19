import os

from track import track

from track import MAX_COST

from matching_cascade import associate

from resnet import resnet

from visualization_utilities import draw_annotated_bbox as draw

from visualization_utilities import read_img


def get_paths():
    '''
       Return the paths to detection file, images, resnet
       checkpoint, and tracking results.
       Output: paths to detections, images, checkpoint, and
               tracking results, respectively.
    '''  
    
    # path  the input path where all input files are located.
    in_path = os.path.expanduser('~')  + \
              '/{0}'.format('resnet detections')
    
    det_path = in_path + '/{0}'.format('gt')              
    
    imgs_path = in_path + '/{0}'.format('img1')
    
    # path to the resnet chekcpoint file
    resnet_path  =  os.path.realpath(__file__) 

    ckpt_path = os.path.dirname(resnet_path) +  '/{0}/{1}'.format('resenet ckpt',
                                                 'mars-small128.ckpt-68577')
    
    track_path = in_path + '/{0}'.format('tracking_results')
    
    return det_path, imgs_path, ckpt_path, track_path

        
def mot_tracking():
    
    # find files where bbox coordinates written for each frame
    det_path, imgs_path, ckpt_path, track_path = get_paths()
    
    # find all images and sort them in time
    _, _, images = os.walk(imgs_path).__next__()
    images = sorted(images)
    
    # build the resnet feature extractor
    resnet_feat_extractor = resnet(ckpt_path)

    resnet_feat_extractor.build()
    
    # the list of tentative or confirmed tracks
    T = []

    for k,img in enumerate(images,1):
        
        img_k = read_img(imgs_path, img)
    
        # read the bbox detections at frame k
        with open(det_path + '/' + 'gt.txt','r') as f:
            
            # the list of 2d bbox coordinates in the current image
            bboxes_2d = []
            
            # read each detections obtained at time k 
            for line in f:
                
                # remove trailing whitespace characters and split the line
                obj_info = line.rsplit()[0].split(',')
                
                frame_num = int(obj_info[0])

                # there is no more detections at time k
                if (frame_num > k):
                    continue
                
                # do not consider previous detections
                if (frame_num < k):
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
                
                
            # the number of detections at time k
            M = len(bboxes_2d)       
                
            # extract the appearance descriptors using the resnet
            app_des,_ = resnet_feat_extractor.get_features(img_k,bboxes_2d)
              
        # initialize the first tracks    
        if (k == 1):
            for z,app in zip(bboxes_2d,app_des):
                T.append(track(z,app))
                
        else:
            '''State Check, Pruning and Prediction'''
            for t in T:
                
                t.check_state()
                
                # find and remove terminated tracks
                if (t.is_terminated()):
                    T.remove(t)
                # call kf to predict new states of existing targets
                else:
                    t.predict()
                    
            '''Validation and Appearance-based Cost Computation'''
            costs = []
            # validate and compute the combined cost
            for t in T:
                # compute the squared Mahalabobis distances
                m_dist = t.Mahalanobis_dist(bboxes_2d)
                # compute the cosine distance
                c_dist = t.cosine_distance(app_des)
                # combine the two distances for validated measurements
                t_cost = t.validate_cost(m_dist,c_dist,M)
                
                costs.append(t_cost)
        
            # form a list of pairs in the format of (bbox_2d_i,appear_des_i)        
            Z = list(zip(bboxes_2d,app_des))
        
            '''Main Association, Update, and IOU-based Cost Computation'''
            # find the possible associations and unassociated measurements
            assoc1,U1 = associate(T, Z, costs, MAX_COST)
            
            # the list of detected tracks ids
            detec_ids1 = [t_id for  t_id,_ in assoc1]
            
            # the list of tentative but unassociated tracks at the age of 1
            Tt = []
            # the intersection of union cost for trac
            iou_costs = []
            # the list of unassoicated bbox measurements
            u_Z = [z for z,_ in U1]
            
            for t in T:
                if (t.trackId in detec_ids1):
                    # find the index of the track in the list
                    t_idx = detec_ids1.index(t.trackId)
                    # find the index of the associated measurement
                    _, meas_idx = assoc1[t_idx]
                    # update the track
                    t.update(bboxes_2d[meas_idx],
                             app_des[meas_idx])
                    
                # find tentative but unassociated tracks at the age
                # of 1 for intersection of union matching matching
                else:
                    if (t.is_tentative() and t.num_md == 1):
                        Tt.append(t)
                
                        # compute the iou cost
                        c_iou = t.iou_cost(u_Z)
                        
                        iou_costs.append(c_iou)
            
            '''IoU based Association for tentative tracks, and Update'''      
            # associations for tracks in Tt and unassociated measurements
            assoc2,U2 = associate(Tt, U1, iou_costs, MAX_COST)
            
            # the list of detected tracks ids
            detec_ids2 = [t_id for  t_id,_ in assoc2]
    
            # update tracks with measurements according to assoc2
            for t in Tt:
                if (t.trackId in detec_ids2):
                    # find the index of the track in the list
                    t_idx = detec_ids2.index(t.trackId)
                    # find the index of the associated measurement
                    _, meas_idx = assoc2[t_idx]
                    # update the track
                    t.update(bboxes_2d[meas_idx],
                             app_des[meas_idx])
            
            ''''Initialize Newborn Tracks'''
            for z,app in U2:
                T.append(track(z,app))
                
                
        '''Draw Tracks on the Image'''
        draw(img_k,track_path,img,T)
            
if __name__ == '__main__':
    mot_tracking()

    