from numpy import *
from scipy.spatial import *
import scipy.linalg as lag
import math


def loss_func(hash_1, hash_2, R_pq, W, S, beta, gamma1, gamma2, gamma3):
    #concerned about the fact that the homogeneous similarly matrix is identical matrix
    #the homogeneous part of loss function is not conerned

    hash = [hash_1, hash_2]
    mp = shape(hash_1)[1]
    mq = shape(hash_2)[1]
    J = []
    J_l = []

    theta1 = 0
    theta2 = 0
    theta3 = 0

    R_pqT = R_pq.transpose()

    for p in range(2):

        R_pqT = R_pqT.transpose()

        #print p, 'domain begin'

        q = 1 - p

        rp = hash[p].shape[0]
        rq = hash[q].shape[0]

        J.append(0)
        J_l.append(0)

        #Homogeneous loss function is considered as zero caused identical matrix is utlized to represent the homogeneous similarity
        hash_p_maped = dot(W, hash[q])

        for k in range(rq):
            #print 'k:', k

            hashq_k = tile(hash[q][k, :], (R_pqT.shape[0], 1))
            hashp_mapped_k = tile(hash_p_maped[k, :], (R_pqT.shape[0], 1))

            J_tmp = (R_pqT * hashq_k) * hashp_mapped_k

            J[p] += sum(log(1 + exp(-J_tmp)))

            J[p] += beta * math.pow((distance.norm(W[:, k], 2)), 2)
        #print 'finish one domain'

        #regularization part of loss function
        theta1 += math.pow(lag.norm((hash[p] * hash[p] - eye(shape(hash[p])[0], shape(hash[p])[1])), 'fro'), 2)
        theta2 += math.pow(lag.norm(S[p][1], 'fro'), 2)
        theta3 += math.pow(lag.norm(S[p][2], 'fro'), 2)

    loss = (sum(J) + gamma1 * theta1 + gamma2 * theta2 + gamma3 * theta3) / (mp * mq)

    return loss


def train(img_fea, tag_fea, H_img, H_tag, S, W, R_pq, R_p, R_q, OutofSample):
    print 'Train func begin'

    beta = 100 #heterogeneous
    gamma1 = 10 #regularization 1
    gamma2 = 3e-3 #regularization 2
    gamma3 = 3 #regualariation 3
    lambda_w = 0.1
    lambda_h = 0.1
    lambda_reg = 1e-1

    print 'begin to calculate loss func'
    new_loss = loss_func(H_img, H_tag, R_pq, W, S, beta, gamma1, gamma2, gamma3)
    old_loss = new_loss + 20  # just for start

    converge_threshold = 1

    fea = [img_fea, tag_fea]
    H = [H_img, H_tag]

    W = W.transpose()
    R_pq = R_pq.transpose()

    print '---------Training---------------'

    iteration = 0

    while old_loss - new_loss > converge_threshold:

        iteration += 1

        print '-------------------------------'
        print iteration, 'times iteration'

        old_loss = new_loss
        print old_loss

        #update the hash code
        #and update the statistics S
        for p in range(2):
            q = 1 - p
            W = W.transpose()
            R_pq = R_pq.transpose()

            #print 'Updating H'
            [H, S] = update_h(fea, H, W, S, R_pq, p, lambda_h, OutofSample)

            #print 'updating W'
            #update the mapping function w
            if not OutofSample:
                W = update_w(H, R_pq, W, p, lambda_w, lambda_reg)

        new_loss = loss_func(H[0], H[1], R_pq.transpose(), W.transpose(), S, beta, gamma1, gamma2, gamma3)

    print 'old:', old_loss, 'new', new_loss

    H_img = sign(H[0])
    H_tag = sign(H[1])

    return [H_img, H_tag, W.transpose(), S]


def update_h(fea, H, W, S, R_pq, p, lambda_h, OutofSample):
    #Does not consider the homogeneous similarity
    #Thus the part to upate the homogeneous is ignored.

    #used for two domain situation, only, currently
    q = 1 - p

    rp = shape(H[p])[0]
    mp = shape(H[p])[1]

    rq = shape(H[q])[0]
    mq = shape(H[q])[1]


    #The derivative
    Gradient = zeros([rp, mp])

    gd_1 = 4 * ((H[p] * H[p] - eye(rp, mp)) * H[p])
    gd_2 = multiply(2, S[p][1]) #rp \times 1
    gd_2 = tile(gd_2, (1, mp))
    gd_3 = multiply(4, dot(S[p][2], H[p]))

    Gradient = Gradient + gd_1 + gd_2 + gd_3

    Hq_map = dot(W.transpose(), H[p]) #hash code of q acquired from mapping Hp
    Hp_map = dot(W, H[q]) #hash code of p acquired from mapping Hq

    for k in range(rp):
        #print 'Update H:', k

        for i in range(mp):
            gd = 0

            R_pqij = tile(R_pq[i, :], (rq, 1))
            H_qgj = H[q]
            W_pqkg = tile(W[k, :].transpose(), (mq, 1)).transpose()
            Wg_Hip = tile(Hq_map[:, i], (mq, 1)).transpose()

            gd += sum((-R_pqij * H_qgj * W_pqkg) / (1 + exp(R_pqij * H_qgj * Wg_Hip)))

            gd += sum((-R_pq[i, :] * Hp_map[k, :]) / (1 + exp(R_pq[i, :] * Hp_map[k, :] * H[p][k, i])))

            #print i, 'instance by matrix:', gd
            #print 'Current Method', gd1, gd
            #print 'Current Method',

            Gradient[k, i] += gd

        H[p][k, :] = H[p][k, :] - lambda_h * Gradient[k, :] / (mp * mq)

        if not OutofSample:
            S[p] = update_S(fea[p], H[p])

    #print 'graident finished'

    return [H, S]


def update_w(H, R_pq, W, p, lambda_reg, lambda_w):
    q = 1 - p

    mp = H[p].shape[1]
    mq = H[q].shape[1]

    rp = H[p].shape[0]
    rq = H[q].shape[0]

    Hq_maped = dot(W.transpose(), H[p])

    for k in range(rq):

        gd = zeros([1, rp])

        #mp \times 1
        Rpq_Hkj = R_pq * tile(H[q][k, :], (mp, 1))

        gd_i = (-Rpq_Hkj) / (1 + exp(Rpq_Hkj * tile(Hq_maped[k, :], (mq, 1)).transpose()))
        gd_i = sum(gd_i, 1)
        gd = dot(H[p], gd_i.transpose())
        gd += 2 * lambda_reg * W[:, k]

        W[:, k] = W[:, k] - lambda_w * gd / (mp * mq)

    return W


def update_S(fea, hash):

    S_0 = dot(hash, transpose(fea))
    S_1 = dot(hash, ones([shape(fea)[1], 1]))
    S_2 = subtract(dot(hash, transpose(hash)), multiply(shape(fea)[1], eye(shape(hash)[0])))

    return [S_0, S_1, S_2]
