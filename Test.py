#Test function for RaHH algorithm as well as CVH algorithm
#Given the image features, qa features, image hash, qa hash
#And aslo Rpq to mapping the hash code form image domain 
#to QA domain. The global precision, global recall and also MAP
#aacording to references are calcuated. In order to test the performance
#of CVH just set Rpq = eye() to disable it 
#Author: Bo Liu,
#bliuab@cse.ust.hk, 2013.11.2
from numpy import *
from HamDist import *
import pylab as pl

def test(img_hash, qa_hash, groundtruth):

    dist = zeros((img_hash.shape[1],qa_hash.shape[1]))
    for i in range(img_hash.shape[1]):
        for j in range(qa_hash.shape[1]):
            #print i,j,HamDist(img_hash[:,i],qa_hash[:,j])
            dist[i, j] = HamDist(img_hash[:, i], qa_hash[:, j])


    step = 40
    dist_threshold = linspace(dist.min() + 0.1, 1, step)
    GP = arange(step)
    GR = arange(step)


    i = 0

    for thre in dist_threshold:
    #set_printoptions(threshold='nan')

        TP_FP = sum(dist <= thre)
        TP = sum((dist <= thre) * (groundtruth == 1))
        P = sum(groundtruth == 1)


        GP[i] = 100.000000 * TP / TP_FP
        GR[i] = 100.000000 * TP / P

        print 'TP_FP:', TP_FP, 'TP:', TP, 'P:', P
        print 'GP:', GP[i], 'GR:', GR[i]

        i += 1
        #neg_mean = Tst_sim[dist < thre]
        #pos_mean = Tst_sim[dist >= thre]
        #print 'neg:', mean(neg_mean)
        #print 'pos:', mean(pos_mean)
        #
        #pl.close()
        #pl.plot(neg_mean.ravel(), ones((len(neg_mean.ravel()), 1)), 'go')
        #pl.plot(pos_mean.ravel(), ones((len(pos_mean.ravel()), 1)) + 1, 'ro')
        #pl.savefig(str(i) + '.jpg')
        #
        #i += 1
