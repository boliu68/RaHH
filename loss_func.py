from numpy import *
from scipy.spatial import *
import scipy.linalg as lag
import scipy.spatial as sp
import math


def loss_func(img_fea, tag_fea, hash_1, hash_2, R_pq, Rp, Rq, W, S, alpha, beta, gamma1, gamma2, gamma3):
    #concerned about the fact that the homogeneous similarly matrix is identical matrix

    fea = [img_fea, tag_fea]
    hash = [hash_1, hash_2]
    R = [Rp, Rq]
#    mp = shape(hash_1)[1]
#    mq = shape(hash_2)[1]
#    rp = hash_1.shape[0]
#    rq = hash_2.shape[0]
    J = []
    J_l = []

    theta1 = 0
    theta2 = 0
    theta3 = 0

    R_pqT = R_pq.transpose()

    W_temp = W

    for p in range(2):
        J.append(0)
        J_l.append(0)
        R_pqT = R_pqT.transpose()
        q = 1 - p
        rp = hash[p].shape[0]
        rq = hash[q].shape[0]
	mp = hash[p].shape[1]
	mq = hash[1].shape[1]

        #Homogeneous
        Ap = dot(fea[p].transpose(), fea[p]) #+ alpha * R[p]
        H_distance = sp.distance_matrix(hash[p].transpose(), hash[p].transpose()) ** 2
	print 'Disance:', H_distance
        J_homo = (Ap * H_distance).sum() / (mp ** 2 * rp)
        J[p] += alpha * J_homo

        print 'J homo:', alpha * J_homo

        #Heterogeneous
        #caused identical matrix is utlized to represent the homogeneous similarity
        W_temp = W_temp.transpose()
        hash_p_maped = dot(W_temp, hash[p])

        for k in range(rq):

            hashq_k = tile(hash[q][k, :], (R_pqT.shape[0], 1))
            hashp_mapped_k = tile((hash_p_maped[k, :]), (R_pqT.shape[1], 1)).transpose()

            J_tmp = (R_pqT * hashq_k) * hashp_mapped_k

            J[p] += sum(log(1 + exp(-J_tmp))) / (mp * mq * rq) 
	    print 'J composition:'
	    print 'Loss:', sum(log(1 + exp(-J_tmp))) / (mp * mq * rq)
            J[p] += beta * math.pow((distance.norm(W_temp[k, :], 2)), 2) / rq
	    print 'Regularization:', beta * math.pow((distance.norm(W_temp[k, :], 2)), 2) / rq
        print 'Hetero:', J[p] - alpha * J_homo

        #regularization part of loss function
        theta1 += math.pow(lag.norm((hash[p] * hash[p] - eye(shape(hash[p])[0], shape(hash[p])[1])), 'fro'), 2) / (mp * rp)
        theta2 += math.pow(lag.norm(S[p][1], 'fro'), 2) / (mp**2 * rp)
        theta3 += math.pow(lag.norm(S[p][2], 'fro'), 2) / ((mp ** 2) * (rp ** 2))

    loss = (sum(J) + gamma1 * theta1 + gamma2 * theta2 + gamma3 * theta3)# / (mp * mq * (rp + rq)) #* (alpha + beta + gamma1 + gamma2 + gamma3))#
   
    print 'Loss composition:'
    print sum(J)
    print gamma1 * theta1
    print gamma2 * theta2
    print gamma3 * theta3
    print '----------------'

    return loss
