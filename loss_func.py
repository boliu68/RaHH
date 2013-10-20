import numpy as np
import scipy.spatial as sp
import math


def loss_func(hash_1, hash_2, R_pq, W, trade_off):
    #concerned about the fact that the homogeneous similarly matrix is identical matrix
    #the homogeneous part of loss function is not concerned
    
    hash = [hash_1, hash_2]
    mp = np.shape(hash_1)[1]
    mq = np.shape(hash_2)[1]
    J = []
    
    for p in range(2):
        
        q = 1 - p
        
        rp = np.shape(hash[p])[0]
        J.append(0)
        
        print np.shape(R_pq)
        
        for k in range(rp):
            for i in range(mp):
                for j in range(mq):
                   #print -R_pq[i,j]*hash[q][k,j]
                   #print np.dot(np.transpose(W[:,k]),hash[p][:,i])
                   #print 'k', k, 'i', i, 'j', j
                   
                   #hash[q][k,j] * np.dot(np.transpose(W[:,k]),hash[p][:,i])
                   l = math.log(1+math.exp(-R_pq[i,j]*hash[q][k,j] * np.dot(np.transpose(W[:,k]),hash[p][:,i])))
                   J[p] = J[p] + l

            J[p] = J[p] + trade_off * math.pow((sp.distance.norm(W[:,k], 2)),2)
            
    loss = sum(J)
    
    return loss


def train(img_fea, tag_fea, H_img, H_tag, S, W, R_pq, R_p, R_q):
    
    #def loss_func(hash_1, hash_2, R_pq, W, trade_off):
    trade_off = 0.1
    a = loss_func(H_img, H_tag, R_pq, W, trade_off)
    print a

def update_S(fea, hash):
    
    #print np.shape(hash)
    #print np.shape(fea)
    
    S_0 = np.dot(hash, np.transpose(fea))
    S_1 = np.dot(hash, np.ones([np.shape(fea)[1],1]))
    S_2 = np.subtract(np.dot(hash, np.transpose(hash)),np.multiply(np.shape(fea)[1],np.eye(np.shape(hash)[0])))

            

                
                
                
        
        
        

    
    