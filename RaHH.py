import load_data
import numpy as np
from cvh import cvh
from loss_func import *
#Relation-aware Heterogeneous Hashing(RaHH)
#Puesdo-Code
#Author: Bo Liu
#Date: Oct. 05 2013
#Reference: 
#Ou, M., Cui, P., Wang, F., Wang, J., Zhu, W., & Yang, S. (2013). Comparing apples to oranges: a scalable solution with heterogeneous hashing. Paper presented at the Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining.
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
    hash_bit = [16,16]
    
    #Hash code rp*mp
    [H_img, H_tag] = cvh(similarity, image_fea, tag_fea,hash_bit) 
    #Heterogeneous mapping by image_hash'*W = tag_hash 
    W = np.eye(hash_bit[0],hash_bit[1])
    
    S_img = update_S(image_fea, H_img)
    S_tag = update_S(tag_fea, H_tag)
    S = [S_img, S_tag]
    
    R_p = np.eye(np.shape(image_fea)[1])
    R_q = np.eye(np.shape(tag_fea)[1])
    
    return [H_img, H_tag, W, S, R_p, R_q]

def RaHH():
    #R is the similarity to keep the consistent with origin paper
    [R_pq, image_fea, tag_fea] = load_data.analysis()
    #image_fea d_p * m_p
    #tag_fea d_q* m_q
    #similarty : m_p * m_q
    
    [K, N, dim] = get_attr(image_fea, tag_fea, R_pq)
    [H_img, H_tag, W, S, R_p, R_q] = initialize(image_fea, tag_fea, R_pq)
    
    train(image_fea, tag_fea, H_img, H_tag, S, W, R_pq, R_p, R_q)
    
    

if __name__ == '__main__':
    
    RaHH()
    