from numpy import *
from cvh import *
from loss_func import *

#Author: Bo Liu, bliuab@cse.ust.hk
#Date: 2013.11.8
#References:
#Ou, M., Cui, P., Wang, F., Wang, J., Zhu, W., & Yang, S. (2013). Comparing apples to oranges: a scalable solution with heterogeneous hashing.
#Paper presented at the Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining.
#Given the trained statistics, transforming matrix, Hash code. And new data.
#Efficiently train the hash code of new out of samples.

def OutSample_Train(x, img_fea, rpq, W, H_img_Tr, S, bit):

    #x: the new n data sample
    #img_fea: the features of images
    #rp: homogeneous similarity. Intuitively, it is assume to be [0...1..0] here without consideration of homogeneous similarity
    #rpq: the heterogeneous similarity. import component for training here.
    #W: trained transformation matrix to mapping the hash code between different domain.
    #H_img_Tr: trained Hash code for existed image
    #S: trained statistics to accelerate training.
    #Rpq: exist heterogeneous similarity to build a new heterogeneous similarity
    #bit: hash code should be uniform length the previous trained data.

    #Initialize hash based on CVH

    H_img, H_qa, A_img, A_qa = cvh(rpq, img_fea, x, bit)

    #Parameters
    beta = 100 #heterogeneous
    gamma1 = 10 #regularization 1
    gamma2 = 3e-3 #regularization 2
    gamma3 = 3 #regualariation 3
    lambda_h = 1e-3 / x.shape[1]
    p = 1#qa
    q = 1 - p #img
    H = [H_img_Tr, H_qa]

    #H[p] = hstack((H[p], H_qa))
    new_loss = loss_func(H[q], H[p], rpq, W, S, beta, gamma1, gamma2, gamma3)
    old_loss = new_loss + 200  # just for start
    converge_threshold = 1e2


    while old_loss - new_loss > converge_threshold:

        H = update_h([img_fea, x], H, W.transpose(), S, rpq, p, lambda_h, False)
        new_loss = loss_func(H[q], H[p], rpq, W, S, beta, gamma1, gamma2, gamma3)

        print 'old_loss:', old_loss, ' new loss:', new_loss

    return H[p]


