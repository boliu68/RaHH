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

def test(img_hash, qa_hash ,bit, groundtruth):

    dist_threshold = [.2,.4,.6,.8]
    GP = arange(4)
    GR = arange(4)
        
    dist = zeros((img_hash.shape[1],qa_hash.shape[1]))
    for i in range(img_hash.shape[1]):
        for j in range(qa_hash.shape[1]):
            #print i,j,HamDist(img_hash[:,i],qa_hash[:,j])
            dist[i, j] = HamDist(img_hash[:, i], qa_hash[:, j])
        
    for d in range(len(dist_threshold)):
    #set_printoptions(threshold='nan')
    #print dist
        print 'bit:', bit, 'threshold:', dist_threshold[d]
        thre = dist_threshold[d]

        TP_FP = sum(dist < thre) 
        TP = sum((dist<thre) * (groundtruth == 1))
        P = sum(groundtruth == 1)

        GP[d] = 100.000 * TP / TP_FP
        GR[d] = 100.000 * TP/P
        
        print 'TP_FP:', TP_FP, 'TP:', TP, 'P:', P
        print 'GP:', GP[d], 'GR:', GR[d]
