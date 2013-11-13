import load_data
import numpy as np
from cvh import cvh
from loss_func import *
from Test import *
from OutSample import *
from init import *
from Train import *

#Relation-aware Heterogeneous Hashing(RaHH)
#Puesdo-Code
#Author: Bo Liu
#Date: Oct. 05 2013
#References:
#Ou, M., Cui, P., Wang, F., Wang, J., Zhu, W., & Yang, S. (2013). Comparing apples to oranges: a scalable solution with heterogeneous hashing.
#Paper presented at the Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining.
#Rahh()
#Input: X^p/data, R_p,intra_domain relation, R_pq inter-domain relation. r: the number of bit for each domain
#Output: H^p: hash function, W: map function to map the hash code to another Hamming space.
def subsampling(fea1, fea2, sim, lin, num):

    if lin == 0:

        fea1_ind = random.random_integers(0, fea1.shape[1] -1 , num)
        fea2_ind = random.random_integers(0, fea2.shape[1] - 1, num)
    else:

        if num == 0:
            fea1_ind = random.random_integers(0, fea1.shape[1] - 1, fea1.shape[1] / lin)
            fea2_ind = random.random_integers(0, fea2.shape[1] - 1, fea2.shape[1] / lin)

    fea1 = fea1[:, fea1_ind]
    fea2 = fea2[:, fea2_ind]
    sim = (sim[fea1_ind, :])[:, fea2_ind]

    return fea1, fea2, sim

def RaHH():
    #R is the similarity to keep the consistent with origin paper

    #Tr_sim_path = 'Data/Train/similarity.txt'
    #Tr_img_path = 'Data/Train/images_features.txt'
    #Tr_tag_path = 'Data/Train/tags_features.txt'
    #Tst_img_path = 'Data/Test/images_features.txt'
    #Tst_qa_path = 'Data/Test/QA_features.txt'
    #gd_path = 'Data/Test/groundtruth.txt'
    Tr_img_path = 'Data/NUS_Wide_Processed/Training/images_features.txt'
    Tr_tag_path = 'Data/NUS_Wide_Processed/Training/tags_features.txt'
    Tr_sim_path = 'Data/NUS_Wide_Processed/Training/similarity.txt'
    Tst_img_path = 'Data/NUS_Wide_Processed/Query/images_features.txt'
    Tst_qa_path = 'Data/NUS_Wide_Processed/Test/tags_features.txt'
    gd_path = 'Data/NUS_Wide_Processed/Query/groundtruth.txt'

    [Tr_sim, Tr_img, Tr_tag, Tst_img, Tst_qa, gd] = load_data.analysis(Tr_sim_path, Tr_img_path, Tr_tag_path, Tst_img_path, Tst_qa_path, gd_path)
    #image_fea d_p * m_p
    #tag_fea d_q* m_q
    #similarty : m_p * m_q
    #QA_fea = d_p * m_p
    #GD = #img * #QA
    Tr_img, Tr_tag, Tr_sim = subsampling(Tr_img, Tr_tag, Tr_sim, 0, 300)

    print 'Loading Data finish'
    print 'Train sim:', Tr_sim.shape, 'Train Img:', Tr_img.shape, 'Tr_tag:', Tr_tag.shape
    print 'Tst Img:', Tst_img.shape, 'Tst_qa:', Tst_qa.shape, 'GD:', gd.shape

    print '----------------CVH finish----------------------'

    [H_img, H_tag, W, S, R_p, R_q, A_img, A_tag] = initialize(Tr_img, Tr_tag, Tr_sim)

    print 'begin RaHH train'
    [H_img, H_tag, W, S] = train(Tr_img, Tr_tag, H_img, H_tag, S, W, Tr_sim, R_p, R_q, False, 0, 0)

    print '---------------begin Test----------------------'

    Tst_img, Tst_qa, gd = subsampling(Tst_img, Tst_qa, gd, 20, 0)

    OutSample_Test(Tr_img, Tr_tag, Tr_sim, Tst_img, Tst_qa, W, S, H_img, H_tag, gd)

    #[H_img_Tst, H_qa_Tst, W_Tst, S_Tst, Rp_Tst, Rq_Tst, A_img_Tst, A_qa_Tst] = initialize(Tst_img, Tst_qa, Tst_sim)
    #[H_img_Tst, H_qa_Tst, W_Tst, S_Tst] = train(Tst_img, Tst_qa, H_img_Tst, H_qa_Tst, S, W, Tst_sim, Rp_Tst, Rq_Tst, True)
    #print '---------------Result---------------------------'
    #H_img_Tst = np.sign(dot(W.transpose(), H_img_Tst))
    #test(H_img_Tst, H_qa_Tst, gd)

if __name__ == '__main__':
    
    RaHH()
