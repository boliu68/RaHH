import load_data
import numpy as np
from cvh import cvh
from loss_func import *
from Test import *
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

def get_attr(image_fea, tag_fea, similarity):
    #The attribute of the input data
    #K #domain
    #N #data instances
    #dim #data feature dimension
    K = 2
    N = [int(np.size(image_fea,1)),int(np.size(tag_fea,1))]
    dim = [int(np.size(image_fea,0)),int(np.size(tag_fea,0))]

    return [K, N, dim]


def initialize(image_fea, tag_fea, similarity):
    #Initialize the hash code for image and tag using CVH
    #Initialize the inter domain mapping matrix W as I 
    #Initialize the matrix S
    
    [K, N, dim] = get_attr(image_fea, tag_fea, similarity)

    #the hash code length
    hash_bit = [32, 32]
    
    #Hash code rp*mp
    [H_img, H_tag, A_img, A_tag] = cvh(similarity, image_fea, tag_fea,hash_bit) 
    #Heterogeneous mapping by image_hash'*W = tag_hash 
    W = np.eye(hash_bit[0], hash_bit[1])
    
    S_img = update_S(image_fea, H_img)
    S_tag = update_S(tag_fea, H_tag)
    S = [S_img, S_tag]
    
    R_p = np.eye(np.shape(image_fea)[1])
    R_q = np.eye(np.shape(tag_fea)[1]) 
    return [H_img, H_tag, W, S, R_p, R_q, A_img, A_tag]

def RaHH():
    #R is the similarity to keep the consistent with origin paper

    #Tr_sim_path = 'Data/Train/similarity.txt'
    #Tr_img_path = 'Data/Train/images_features.txt'
    #Tr_tag_path = 'Data/Train/tags_features.txt'
    #Tst_sim_path = 'Data/Test/similarity.txt'
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
    img_ind = random.random_integers(0, Tr_img.shape[1] - 1, Tr_img.shape[1] / 50)
    tag_ind = random.random_integers(0, Tr_tag.shape[1] - 1, Tr_tag.shape[1] / 50)


    #sub sampling
    Tr_img = Tr_img[:, img_ind]
    Tr_tag = Tr_tag[:, tag_ind]
    Tr_sim = (Tr_sim[img_ind, :])[:, tag_ind]

    print 'Loading Data finish'
    print 'Train sim:', Tr_sim.shape, 'Train Img:', Tr_img.shape, 'Tr_tag:', Tr_tag.shape
    print 'Tst Img:', Tst_img.shape, 'Tst_qa:', Tst_qa.shape, 'GD:', gd.shape

    #[K, N, dim] = get_attr(Tr_img, Tr_tag, Tr_sim)
    print 'CVH finish'
    
    #bits = [8,16,32,64]
    [H_img, H_tag, W, S, R_p, R_q, A_img, A_tag] = initialize(Tr_img, Tr_tag, Tr_sim)

    print 'begin RaHH train'
    [H_img, H_tag, W, S] = train(Tr_img, Tr_tag, H_img, H_tag, S, W, Tr_sim, R_p, R_q, False)

    print 'begin Test'
    [H_img_Tst, H_qa_Tst, W_Tst, S_Tst, Rp_Tst, Rq_Tst, A_img_Tst, A_qa_Tst] = initialize(Tst_img, Tst_qa, gd)
    [H_img_Tst, H_qa_Tst, W_Tst, S_Tst] = train(Tst_img, Tst_qa, H_img_Tst, H_qa_Tst, S_Tst, W_Tst, gd, Rp_Tst, Rq_Tst, False)

    test(H_img_Tst, H_qa_Tst, 16, gd, Tst_sim)

if __name__ == '__main__':
    
    RaHH()
