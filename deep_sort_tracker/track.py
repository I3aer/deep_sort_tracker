import numpy as np
from filterpy.kalman import kalman_filter as kf
from enum import Enum
from scipy.stats import chi2
from scipy import linalg

# max cost used for impossible associations
MAX_COST = 10e10

# reject assignments where iou is less than this value
MIN_IOU = 0.3

class track_state(Enum):
    """
    track state for management.
    """

    TENTATIVE = 1   # newly initialized tracks
    CONFIRMED = 2   # tracks received measurements 
    TERMINATED = 3  # missed or terminated tracks


class track(object):
    '''
        Track class is used to track a specific target in time.
        Target state is given by [x,y,w,h,v_x,v_y,v_w,v_h] 
        where (x,y) is the top-left position of a bbox, w, 
        is box width, h is the box height, and the successive 
        elements are their velocities in image coordinates. 
        
        Measurements are given by a linear Gaussian model: z=Hx+w
        where z is the bbox coordinates, i.e., [x,y,w,h] on image 
        plane.
    '''
    # class/static variable used to uniquely label each track
    id_counter = 0
    
    def __init__(self, x0, app0, dt=1):
        '''
            Initialize a track for an unassociated measurement.
            Inputs: x0 is the 4 dimensional measurement vector, i.e, 
                    the bbox coordinates: [x,y,w,h]
                    app0 is the 128 dimensional unit array of the
                    appearance descriptor.
                    trackId: a unique id assigned to each target.
                    dt is the time step between the kalman updates. 
        '''
        # dimension of the state vector
        self.dimx = 8
        # dimension of the measurement vector
        self.dimz = 4
        
        # time step constant
        self.dt = dt
        
        # create target state and its covariance matrix
        self.x = np.zeros(shape=self.dimx,dtype=np.float32)
        self.P = np.zeros(shape=(self.dimx,self.dimx),dtype=np.float32)
        
        # set observed states to the bbox coordinates 
        self.x[0:4] = x0[:] 
        
        # set the initial system error covariance 
        self.P = self.P0_Q_R()
        
        # CV motion model transition matrix
        self.F = np.eye(self.dimx, self.dimx)
        
        for i in range(self.dimz):
            self.F[i][self.dimz + i] = self.dt
            
        # system process noise covariance 
        self.Q = np.zeros(shape=(self.dimx,self.dimx),dtype=np.float32)
            
        # state to measurement transition matrix    
        self.H = np.eye(self.dimz, self.dimx)
        
        # measurement noise covariance  
        self.R = np.zeros(shape=(self.dimz,self.dimz),dtype=np.float32)
        
        # Allocate 100X128 dimensional matrix for appearance descriptors
        self.app_des = np.zeros(shape=(100,128),dtype = np.float32)
        # save the first appearance descriptor
        self.app_des[0][:] = app0
    
        # set the track state to tentative
        self.state = track_state.TENTATIVE
        
        # set a unique track Id 
        self.trackId = track.id_counter
    
        # increase the track id 
        track.id_counter += 1
        
        # number of successive missed detections
        self.num_md = 0
        
        # age of the track
        self.age = 0
        
        # track confirmation age
        self.conf_age = 2
        
        # max number of missed detections
        self.max_md = 30
        
        # gating threshold for the squared Mahalanobis distance
        up_tail_prob = 0.95
        self.gt_th = chi2.ppf(up_tail_prob, df=self.dimz)
        
        # threshold for cosine distance 
        self.cos_th = 1e-1
        
        # weight used to combine the cosine and the squared mahalanobis distances
        self.gamma = 0
      
    def P0_Q_R(self, z = None, is_R = False, is_P0 = True, up_tail_prob=0.95):
        '''
            The state covariance and measurement covariance
            matrix are set under the following assumptions:
            The NEES for the bbox center (x,y) are chi square
            rv's with degree of 1 for each axis. Assuming the 
            true center is within the bbox, these NEES are less
            than a predetermined chi-square threshold. 
            
            Assume that the dimensions of the true bbox are 
            somewhere within 0.5 and 1.5 times dimensions of 
            the detected bbox. Thus, the NEES for w and h are 
            modeled as chi-square rv's and they must be less 
            than a predetermined threshold.
            
            At time t, take a crop of frame t − 1 centered
            at (x, y) with a width and height of k1w and k1h, 
            respectively. Search the target in this region
            under the smooth motion constraints. Set k = 2.
            
            Velocities are unobservable and therefore they must
            be set to higher values when initializing system
            covariance matrix. 
            
            Inputs: is_R is a bool set to True for measurement
                    covariance matrix R.
                    is_P0 is a bool set to True for  initial 
                    system covariance matrix.
                    tail_prob is the tail probability to validate
                    detected bbox coordinates.
            Return: elements of the computed covariance matrix.
        '''  
        
        # scale used to set variances of velocities 
        if (is_P0): 
            alpha = 1    # for P0
        else:
            alpha = 0.05  # for P and Q
        
        
        # compute the threshold for the given tail probability
        chi2_th = chi2.ppf(up_tail_prob, df=1)
        
        if not(is_R):
            covM = np.zeros(shape=(self.dimx,self.dimx),dtype=np.float32)
            # width and height of the track's bbox
            w = self.x[2]
            h = self.x[3]
        else:
            covM = np.zeros(shape=(self.dimz,self.dimz),dtype=np.float32)
            # width and height of the measurement bbox
            w = z[2]
            h = z[3]
        
        
        for i in range(self.dimz):
            
            if (i % 2 == 0):
                max_diff = w/2 
            else:
                max_diff = h/2 
            
            # the infimum of the variance to satisfy the tail probability:
            #                (x - mu)^2/var <= chi2_th
            covM[i][i] =  alpha*np.square(max_diff)/chi2_th
        
            # set var of velocities for P0 or Q
            if not(is_R):
                # for var_vx and var_vy
                if (i<2):
                    # max vel in the search area is max_diff*sqrt(2)/dt
                    covM[i+self.dimz][i+self.dimz] = 2*covM[i][i] / (self.dt**2)
                # for var_vw and var_h
                else:
                    covM[i+self.dimz][i+self.dimz] = covM[i][i] / (self.dt**2)
               
        
        # return elements of the matrix       
        return covM[:]
            
        
    def predict(self):
        '''
            Propagate the target state in time according to 
            the constant velocity motion mode using the Kalman 
            fiter.In addition, increase the counter used to 
            measure successive missed detections.
        '''
        
        self.Q[:] = self.P0_Q_R(is_P0 = False)
        
        # predict the state according to CV motion model
        self.x, self.P = kf.predict(self.x, self.P, self.F, self.Q)
        
        # increase the  missed detection number
        self.num_md += 1
        
        # increase the age of the track
        self.age += 1
    
        
    def update(self,z,app_des):
        '''
            Update the target state using the associated
            measurement. Reset the missed detection num-
            ber to zero.
            Inputs: z is the 4 dimensional measurement vector 
                    of the bbox coordinates, i.e, [x,y,w,h]
                    where (x,y) is the top-left corner of the
                    bbox, w is the box width, and h is the box
                    height. 
                    app_des is the 128-D appearance descriptor 
                    array of the associated measurement.
        '''   
       
        # measurement noise covariance
        self.R[:] = self.P0_Q_R(z,is_R = True,is_P0=False)
       
        # Kalman uodate
        self.x, self.P = kf.update(self.x, self.P, 
                                   z, self.R, self.H)
        
        
        # shift the array of descriptors to right by one element
        self.app_des[1:] = self.app_des[0:-1]
        
        # save the new appearance descriptor in the first element
        self.app_des[0][:] = app_des
        
        # reset the missed detection number
        self.num_md = 0
        
    
    def check_state(self):
        """
            Check the status of the track to confirm or delete it.
        """
        
        # confirm or delete the tentative track
        if (self.state == track_state.TENTATIVE):
            # delete the track if it is missed before the confirmation age
            if (self.age < self.conf_age and self.num_md > 0):
                self.state = track_state.TERMINATED
            # confirm the track if it is successfully detected until the confirmation age
            elif (self.age >= self.conf_age and self.num_md == 0):
                self.state = track_state.CONFIRMED

        # delete the confirmed track if it is missed more than the missed detection threshold
        elif (self.state == track_state.CONFIRMED):        
            if (self.num_md > self.max_md ):
                self.state = track_state.TERMINATED
                
                
    def is_tentative(self):
        '''
            Returns True if the track is tentative.
        '''
        return self.state == track_state.TENTATIVE

    def is_confirmed(self):
        '''
            Returns True if the track is confirmed.
        '''
        return self.state == track_state.CONFIRMED

    def is_terminated(self):
        '''
            Returns True if the track is terminated.
        '''
        return self.state == track_state.TERMINATED
        
    def Mahalanobis_dist(self,Z):
        '''
            Compute the squared Mahalanobis distance to validate 
            measurements.
            Input: Z is the list of measurements.
            Output: the list of the squared Mahalanobis distances,
                    and the number of the measurements, i.e. M.
        '''
        
        # predicted measurement
        z_mean = np.dot(self.H,self.x)
        
        # the number of measurements
        M = len(Z)
        
        # the squared Mahalanobis distances
        dist = [0]*M
        
        # the first term of the innovation covariance: H*P*H.T 
        S = np.dot(( np.dot(self.H,self.P)),self.H.T) 
        
        for i,z in enumerate(Z):
            
            self.R[:] = self.P0_Q_R(z,is_R = True,is_P0=False)
  
            # measurement innovation
            inno = z_mean - z
            
            # compute the inverse of the innovation covariance
            inv_S = linalg.inv(S + self.R)
            
            # Mahalonobis distance
            dist[i] = np.dot( np.dot(inno.T,inv_S), inno )
            
        return dist
    
    def cosine_distance(self,m_app_des):
        '''
            Compute the cosine distance between the unit arrays as
            follows:
                    dist(i) = min{1-x_j*y_i|x_j in X} 
            Input: m_app_des is the list of 128 dimensional normalized
                   appearance descriptors of the detected bboxes.
            Output: the list of the min cosine distances between the 
                    the last 100 associated descriptors of the target
                    and each of the measurement appearance descriptors 
        '''
        
        # the list of cosine distances
        dist = []
    
        # cosine distance function
        cos_dist = lambda x,y: 1 - np.dot(x,y.T)
        
        # available appearance features
        if (self.age < self.app_des.shape[0] - 1):
            n = self.age + 1
        else:
            n = self.app_des.shape[0]
        
        for m_app_des_i in m_app_des:
            min_dist = min(cos_dist(self.app_des[0:n],m_app_des_i))
            
            dist.append(min_dist)

        return dist
    
    
    def validate_cost(self,mh_dist,cos_dist,M):
        '''
            Validate measurements using the Mahalanobis and
            cosine distances. Then, compute the combined 
            distance/cost as the weighted sum of these 
            two metrics for validated measurements.
            Inputs: mh_dist is the list  of Mahalanobis 
                    distances.
                    cos_dist is the list of cosine 
                    distances.
                    M is the number of measurements.
            Return: 1D array of the combined cost values.
        '''
        
        
        # set the cost of array to inf initially because of no validation
        cost = np.asarray([MAX_COST]*M, dtype= np.float64)
        
        # validate and then compute the combined cost
        for m in range(M):
            
            if (mh_dist[m] <= self.gt_th and  cos_dist[m] <= self.cos_th):
                cost[m] = self.gamma*mh_dist[m] +  \
                          (1 - self.gamma)*cos_dist[m]
        
        
        return cost
    
    def iou_cost(self,Z):
        '''
            Compute the intersection of union cost for the track 
            and measurements Z using iou's between the track's 
            bbox and those of the measurements. The iou cost is 
            defined as
                    iou_cost = 1 - iou.
            Input: the list of measurements.
            Output: the list of iou costs.
        '''
    
        # intersection of union cost
        iou_cost = [MAX_COST]*len(Z)
        
        # bbox coordinates of the target
        bbox_t = self.x[0:4]
        
        # compute the bbox area of the target
        t_area = np.prod(bbox_t[2:])
        
        for j,z in enumerate(Z):
            # determine the (x, y)-coordinates of the intersection rectangle
            xmin_max = max(bbox_t[0] - bbox_t[2]/2, z[0] - z[2]/2)
            ymin_max = max(bbox_t[1] - bbox_t[3]/2, z[1] - z[3]/2)
            
            xmax_min = min(bbox_t[0] + bbox_t[2]/2, z[0] + z[2]/2)
            ymax_min = min(bbox_t[1] + bbox_t[3]/2, z[1] + z[3]/2)
         
            # compute the area of intersection rectangle
            inter_area = max(0,(xmax_min - xmin_max))*max(0,(ymax_min - ymin_max))
            
            # compute the bbox area of the measurement
            z_area =  np.prod(z[2:])
         
            # compute the intersection over union 
            iou = inter_area / float(t_area + z_area - inter_area)
            
            # validate the measurement
            if (iou > MIN_IOU):
                iou_cost[j] = 1 - iou
    
        return iou_cost