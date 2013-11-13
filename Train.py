__author__ = 'Bo Liu'
from numpy import *
from loss_func import *
from update import *

def train(img_fea, tag_fea, H_img, H_tag, S, W, R_pq, R_p, R_q, OutofSample, up_mp, up_mq):
    #print 'Train func begin'

    alpha = 1000
    beta = 10 #heterogeneous
    gamma1 = 1 #regularization 1
    gamma2 = 3e-3 #regularization 2
    gamma3 = .3 #regualariation 3
    lambda_w = 0.3
    lambda_h = 0.3
    lambda_reg = 1e-1
    converge_threshold = 1e-4 #/ sqrt(img_fea.shape[1] * tag_fea.shape[1])
    #print 'converge_threshol:', converge_threshold

    #print 'begin to calculate loss func'
    new_loss = loss_func(img_fea, tag_fea, H_img, H_tag, R_pq, R_p, R_q, W, S, alpha, beta, gamma1, gamma2, gamma3)
    old_loss = new_loss + 20  # just for start

    fea = [img_fea, tag_fea]
    H = [H_img, H_tag]

    W = W.transpose()
    R_pq = R_pq.transpose()

    #print '---------Training---------------'

    iteration = 0

    while (old_loss - new_loss > converge_threshold) and (iteration < 200):

        iteration += 1

        #print '-------------------------------'
        #print iteration, 'times iteration'

        old_loss = new_loss
        #print old_loss

        #update the hash code
        #and update the statistics S
        for p in range(2):
            q = 1 - p
            W = W.transpose()
            R_pq = R_pq.transpose()

            #print 'Updating H'
            [H, S] = update_h(fea, H, W, S, R_pq, R_p.transpose(), R_q.transpose(), p, alpha, beta, gamma1, gamma2, gamma3, lambda_h, OutofSample, up_mp, up_mq)

            #print 'updating W'
            #update the mapping function w
            if not OutofSample:
                W = update_w(H, R_pq, W, p, lambda_w, lambda_reg)

        new_loss = loss_func(img_fea, tag_fea, H[0], H[1], R_pq.transpose(), R_p.transpose(), R_q.transpose(), W.transpose(), S, alpha, beta, gamma1, gamma2, gamma3)

    #print 'old:', old_loss, 'new', new_loss

    H_img = sign(H[0])
    H_tag = sign(H[1])

    return [H_img, H_tag, W.transpose(), S]