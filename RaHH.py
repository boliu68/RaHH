import load_data
import numpy as np
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
    
    
    
    [K, N, dim] = get_attr(image_fea, tag_fea, similarity)
    hash = [16,16]#the hash code length
    
    H_0 = np.zeros([hash[0], N[0]])
    H_1 = np.zeros([hash[1], N[1]])
    
    W = np.eye(hash[0],hash[1])
    
    S = []
    
    return [H_0, H_1, W]

def RaHH():
    #R is the similarity to keep the consistent with origin paper
    [R, image_fea, tag_fea] = load_data.analysis()
    #image_fea d_p * m_p
    #tag_fea d_q* m_q
    #similarty : m_p * m_q
    
    [K, N, dim] = get_attr(image_fea, tag_fea, R)
    [H_0, H_1, W] = initialize(image_fea, tag_fea, R)
    
    
    

if __name__ == '__main__':
    
    RaHH()
    