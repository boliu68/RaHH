from numpy import *
from scipy.spatial import *
import scipy.linalg as lag
import scipy.spatial as sp
import math


def loss_func(img_fea, tag_fea, hash_1, hash_2, R_pq, Rp, Rq, W, S, alpha, beta, gamma1, gamma2, gamma3, lambda_reg, lambda_alpha):
    #concerned about the fact that the homogeneous similarly matrix is identical matrix

    fea = [img_fea, tag_fea]
    hash = [hash_1, hash_2]
    R = [Rp, Rq]
#    mp = shape(hash_1)[1]
#    mq = shape(hash_2)[1]
#    rp = hash_1.shape[0]
#    rq = hash_2.shape[0]
    J = []

    theta1 = 0
    theta2 = 0
    theta3 = 0

    R_pqT = R_pq.transpose()

    #avoid modify W
    W_temp = W

    print '--------Loss begin---------'

    for p in range(2):

        J.append(0)
        R_pqT = R_pqT.transpose()
        q = 1 - p
        rp = hash[p].shape[0]
        rq = hash[q].shape[0]
	mp = hash[p].shape[1]
	mq = hash[q].shape[1]

        #Homogeneous

        Ap = dot(fea[p].transpose(), fea[p]) + alpha * R[p]
	print 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
	print Ap
	print Ap.shape

	return

        H_distance = sp.distance_matrix(hash[p].transpose(), hash[p].transpose()) ** 2
        J_homo = (Ap * H_distance).sum()# / (mp * mp)
        J[p] += J_homo * lambda_alpha

        print 'J homo:', J_homo * lambda_alpha

        #Heterogeneous
        #caused identical matrix is utlized to represent the homogeneous similarity
        W_temp = W_temp.transpose()
        hash_p_maped = dot(W_temp, hash[p])
	
	loss_sum = 0 
	regu_sum = 0

        for k in range(rq):
	    #Heterogeneous Loss
            hashq_k = tile(hash[q][k, :], (R_pqT.shape[0], 1))
            hashp_mapped_k = tile((hash_p_maped[k, :]), (R_pqT.shape[1], 1)).transpose()
            J_tmp = (R_pqT * hashq_k) * hashp_mapped_k
            J[p] += beta * sum(log(1 + exp(-J_tmp)))# / (mp * mq * rq * rp) 
	    #Regularization loss
	    J[p] += beta * lambda_reg * math.pow((distance.norm(W_temp[k, :], 2)), 2) #/ (rp * rq)
	    
	    #just for print
	    loss_sum += beta * sum(log(1 + exp(-J_tmp))) #/ (mp * mq)
	    regu_sum += beta * lambda_reg * math.pow((distance.norm(W_temp[k, :], 2)), 2) 

	
	print 'Heterogeneous Loss', loss_sum
	print 'Regularization Loss', regu_sum

        #regularization part of loss function
        theta1 += math.pow(lag.norm((hash[p] * hash[p] - ones((shape(hash[p])[0], shape(hash[p])[1]))), 'fro'), 2) #/ (mp)#* rp)
        theta2 += math.pow(lag.norm(S[p][1], 'fro'), 2)# / (mp**2)# * rp)
        theta3 += math.pow(lag.norm(S[p][2], 'fro'), 2)#/ (mp ** 2)# * (rp ** 2))
	
    loss = (sum(J) + gamma1 * theta1 + gamma2 * theta2 + gamma3 * theta3)# / (mp * mq * (rp + rq)) #* (alpha + beta + gamma1 + gamma2 + gamma3))#
   
    print 'Loss composition:'
    print sum(J)
    print gamma1 * theta1
    print gamma2 * theta2
    print gamma3 * theta3
    print '----------------'

    return loss
