import numpy as np
import scipy.spatial as sp
import scipy.linalg as lag
import math


def loss_func(hash_1, hash_2, R_pq, W, S, trade_off, beta, gamma1, gamma2, gamma3):
    #concerned about the fact that the homogeneous similarly matrix is identical matrix
    #the homogeneous part of loss function is not concerned
    
    hash = [hash_1, hash_2]
    mp = np.shape(hash_1)[1]
    mq = np.shape(hash_2)[1]
    J = []
    
    theta1 = 0
    theta2 = 0 
    theta3 = 0
    
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
        
        #print np.eye(np.shape(hash[p])[0],np.shape(hash[p])[1])
        #print np.eye(np.shape(hash[p]))
        
        theta1 = theta1 + math.pow(lag.norm((hash[p] * hash[p] - np.eye(np.shape(hash[p])[0],np.shape(hash[p])[1])), 'fro'),2)
        theta2 = theta2 + math.pow(lag.norm(S[p][1],'fro'),2)
        theta3 = theta3 + math.pow(lag.norm(S[p][2],'fro'),2)
        
    loss = sum(J) + theta1 + theta2 + theta3
    
    return loss


def train(img_fea, tag_fea, H_img, H_tag, S, W, R_pq, R_p, R_q):
    
    #def loss_func(hash_1, hash_2, R_pq, W, trade_off):
    trade_off = 0.1
    old_loss = 0
    beta = 1
    gamma1 = 0.1
    gamma2 = 0.1
    gamma3 = 0.1
    lambda_w = 1e-3
    lambda_h = 1e-3
    new_loss = loss_func(H_img, H_tag, R_pq, W, S, trade_off, beta, gamma1, gamma2, gamma3)
    
    converge_threshold = 1e2
    #print a
    fea = [img_fea, tag_fea]
    H = [H_img, H_tag]
    m= [np.shape(img_fea)[1], np.shape(tag_fea)[1]]
    
    W = np.transpose(W)
    R_pq = np.transpose(R_pq)
    print R_pq
    
    print '---------Training---------------'
    
    while (old_loss - new_loss < converge_threshold):
        
        old_loss = new_loss
        print old_loss
        #update the hash code
        #and update the statistics S
        for p  in range(2):
            print 'One Dmain update'
            q = 1 - p
            W = np.transpose(W)
            R_pq = np.transpose(R_pq)
            
            print 'Updating H'
            [H, S] = update_h(fea, H, W, S, R_pq, p, lambda_h)
            #S[p] = update_S( fea[p], hash[p])
            print 'updating W'
            #update the mapping function w
            W = update_w(H, R_pq, W, p, lambda_w)
            
    
        new_loss = loss_func(H[0], H[1], R_pq, W, S, trade_off, beta, gamma1, gamma2, gamma3)
    
    H_img = np.sign(H[0])
    H_tag = np.sign(H[1])
    
    return [H_img, H_tag, S, R_pq]
        
def update_h(fea, H, W, S, R_pq, p, lambda_h):
    #Does not consider the homogeneous similarity
    #Thus the part to upate the homogeneous is ignored.
    
    q = 1 - p
    
    rp = np.shape(H[p])[0]
    mp = np.shape(H[p])[1]
    
    rq = np.shape(H[q])[0]
    mq = np.shape(H[q])[1]
    
    
    #The derivative
    Gradient = np.zeros([rp, mp])
    gd_1 = 4*((H[p] * H[p] - np.eye(np.shape(H[p])[0],np.shape(H[p])[1]))*H[p])
    gd_2 = np.multiply(2, S[p][1])
    
    #print np.shape(S[p][3])
    
    gd_3 = np.multiply(4, np.dot(S[p][2], H[p]))
    
    
    Gradient = Gradient + gd_1 + gd_2 + gd_3
    
    for k in range(rp):
        for i in range(mp):
            
            gd = 0
            
            for j in range(mq):
                
                for g in range(rq):

                    
                    gd = gd + (-R_pq[i,j] * H[q][g,j] * W[k,g]) / (1 + math.exp( R_pq[i,j] * H[q][g,j] * np.dot(np.transpose(W[:,g]), H[p][:, i])))

                gd = gd + (-R_pq[i,j] * np.dot(W[k, :] , H[q][:, j])) / (1 + math.exp(R_pq[i,j] * H[p][k, i] * np.dot(W[k, :], H[q][:, j])))
        
        Gradient[k, i] = Gradient[k,i] + gd
        
        H[p][k,i] = H[p][k,i] - lambda_h * Gradient[k,i]
        
        S[p] = update_S(fea[p], H[p])
    
    return [H, S]


def update_w(H, R_pq, W, p, lambda_w):

    q = 1 - p
    
    mp = np.shape(H[p])[1]
    mq = np.shape(H[q])[1]
    
    rp = np.shape(H[p])[0]
    rq = np.shape(H[q])[0]

    for k in range(rq):
        
        gd = 0
        for i in range(mp):
            for j in range(mq):
                
                gd = gd + (-R_pq[i,j] * H[q][k,j] * H[p][:, i]) / (1 + math.exp(R_pq[i,j] * H[q][k,j] * np.dot(W[:,k], H[p][:,i])))
                
        gd_vec = gd - lambda_w * W[:,k]
        
        print gd_vec
        
        W[:,k] = W[:,k] - gd_vec
        
    return W
        
def update_S(fea, hash):
    
    #print np.shape(hash)
    #print np.shape(fea)
    
    S_0 = np.dot(hash, np.transpose(fea))
    S_1 = np.dot(hash, np.ones([np.shape(fea)[1],1]))
    S_2 = np.subtract(np.dot(hash, np.transpose(hash)),np.multiply(np.shape(fea)[1],np.eye(np.shape(hash)[0])))

    return [S_0, S_1, S_2]


            

                
                
                
        
        
        

    
    