import numpy as np


from sklearn.utils.linear_assignment_ import linear_assignment

def associate(T, Z, C, max_cost):
    '''
      Associate tracks to measurements by giving priority to
      the more frequently detected targets. The solver for 
      this binary integer programming is the Hungarian algorithm.
      Inputs: T is the list of track objects.
              Z is the list of M number of measurements.
              C is the list of M dimensional array of costs 
              computed for each track.
              max_cost is used to label impossible associations.
      Output: A is the list of pairs (i,j) where i denotes
              the track_id, and j denotes the measurement index.
              U is the list of unassociated measurements. 
    '''
    # initialize the list of valid association pairs
    A = []
    
    # the global indices of the unassociated measurements
    u_idx = [i for i in range(len(Z))]
    
    # the set of missed detection numbers 
    md_nums  = set([t.num_md for t in T])
    
    # sort the number of missed detections in ascending order
    md_nums = sorted(md_nums)
    
    # select tracks and solve the linear assignment problem
    for n in md_nums:
        # the list of costs for the selected tracks
        Cn = []
        # the list of the selected tracks
        Tn = []
        
        for i,t in enumerate(T):
            # find the tracks which were missed detected n times
            if (t.num_md == n):
                Tn.append(t) 
                # the costs of the remainig measurements
                c = [C[i][j] for j in u_idx]
                Cn.append(c)
                     
        # convert the list to the matrix of tracks vs their costs for measurements 
        cost_m = np.asarray(Cn,dtype=np.float64)
        
        # solve the binary integer programming problem
        assoc = linear_assignment(cost_m)
        
        # the list of the last association pairs
        An = []
        for a in assoc:
            i = a[0]  # target index
            j = a[1]  # measurement index
        
            # check if this association probable
            if (cost_m[i][j] < max_cost):
                # append the pair (track id,global measurement index) to the list M 
                An.append( (Tn[i].trackId, u_idx[j]) )
        
        # update the list of  valid association pairs by An
        A = A + An 
        
        # Remove the indices of the associated measurements u_idx
        u_idx = list(set(u_idx) - set([j for  _,j in An]))
        
    
    # the list of unassociated measurements
    U = [Z[j] for j in u_idx]
        
    return  A, U    
        

def iou_matching(self, Tt, Z, C, max_cost):
    '''
        Using intersection of union cost associate the 
        tentative and unmatched tracks at age of 1 with 
        one of unassociated measurements.
        Inputs:  Tt is the list of tentative and unmatched
                 tracks.
                 Z is the list of unassociated measurements.
                 C is the list of iou costs computed for each 
                 track and unassociated measurements.
                 max_cost is used to label impossible associations.
        Outputs: A is the list of pairs (i,j) where i denotes
                 the track_id, and j denotes the measurement index.
                 U is the list of the unassociated measurements which 
                 can be used to initialize new tentative tracks.
    '''
    A, U = associate(Tt, Z, C, max_cost)
    
    return A, U
        
        
        