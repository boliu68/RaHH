import load_data
import numpy as np
from cvh import cvh
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

def train(img_fea, tag_fea, H_img, H_tag, S, W, R_pq, R_p, R_q):
    
    print 1

def update_S(fea, hash):
    
    S_0 = np.dot(hash, np.transpost(fea))
    S_1 = np.dot(hash, np.ones([np.shape(fea,1),1]))
    S_2 = np.subtract(np.dot(hash, np.transpose(hash)),np.multiply(np.shape(fea,1),np.eye(np.shape(hash,0))))

    return [S_0, S_1, S_2]

def initialize(image_fea, tag_fea, similarity):
    #Initialize the hash code for image and tag using CVH
    #Initialize the inter domain mapping matrix W as I 
    #Initialize the matrix S
    
    [K, N, dim] = get_attr(image_fea, tag_fea, similarity)
    
    #the hash code length
    hash_bit = [16,16]
    
    #Hash code rp*mp
    [H_img, H_tag] = cvh(similarity, image_fea, tag_fea) 
    #Heterogeneous mapping by image_hash'*W = tag_hash 
    W = np.eye(hash[0],hash[1])
    
    S_img = updast_S(image_fea, H_img)
    S_tag = update_S(image_tag, H_tag)
    S = {'img': S_img, 'tag': S_tag}
    
    return [H_img, H_tag, W, S]

def RaHH():
    #R is the similarity to keep the consistent with origin paper
    [R_pq, image_fea, tag_fea] = load_data.analysis()
    #image_fea d_p * m_p
    #tag_fea d_q* m_q
    #similarty : m_p * m_q
    
    [K, N, dim] = get_attr(image_fea, tag_fea, R_pq)
    [H_img, H_tag, W, S] = initialize(image_fea, tag_fea, R_pq)
    
    
    

if __name__ == '__main__':
    
    RaHH()
    