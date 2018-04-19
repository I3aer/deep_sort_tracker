from track import track

from track import MAX_COST

from matching_cascade import associate

import numpy as np

if __name__ == '__main__':
    
    # dummy measurements and appearance vectors
    z0 = np.asarray([125,100,10,20],dtype=np.float32)
    app0 = np.random.rand(128)
    app0 = app0/np.linalg.norm(app0)
    
    z1 = np.asarray([120,100,10,15], dtype =np.float32)
    app1 = np.random.rand(128)
    app1 = app1/np.linalg.norm(app1)
    
    # generate two dummy tracks
    t1 = track(z0,app0)
    T = [t1]
    t2 = track(z1, app1)
    T.append(t2)
    
    # predict their next states
    for t in T:
        t.predict()
    
    # new dummy  measurements
    z0 =  np.asarray([130,95,11,17],dtype=np.float32)
    Z = [z0]
    app0 = app0 + 1e-2*np.random.randn(128)
    app0 = app0/np.linalg.norm(app0)
    A = [app0]
    
    #false alarm/clutter
    z2 = np.asarray([25,95,8,16],dtype=np.float32)
    Z.append(z2)
    app2 = np.random.rand(128)
    app2 = app2/np.linalg.norm(app2)
    A.append(app2)
    
    z1 = np.asarray([122,107,13,18],dtype=np.float32)
    Z.append(z1)
    app1 = app1 + 1e-2*np.random.randn(128)
    app1 = app1/np.linalg.norm(app1)
    A.append(app1)
    
    M = len(Z)
    costs = []
    for t in T:
        m_dist = t.Mahalanobis_dist(Z)
        c_dist = t.cosine_distance(A)
        t_cost = t.validate_cost(m_dist,c_dist,M)
        costs.append(t_cost)

    assoc,U = associate(T, Z, costs, MAX_COST)
    
    for a in assoc:
        print('association pair (track Id,measurement index): {0}'.format(a))
        
        
    # update
    for i,t in enumerate(T):
        j  =  assoc[i][1]
        t.update(Z[j], A[j])