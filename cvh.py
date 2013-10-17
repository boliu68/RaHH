import numpy as np
import load_data

#Reference : Kumar, S., & Udupa, R. (2011). 
#Learning hash functions for cross-view similarity search. 
#Paper presented at the Proceedings of the Twenty-Second international joint conference 
#on Artificial Intelligence-Volume Volume Two.
#Input: X: n dim-dimensional instances.
#       W: n*n matrix to represent the similarity between each instance
#Output: The hash function for each view/domain/task. 
#For initialize RaHH. Can also be used as baseline.

def domain2view(fea_1,fea_2,similarity):
    #Transform the setting of data
    #the image and tag whose similarity is greater than 
    #threshold is chosen as the different view for each (concept)
    #After transforming,the I is used to represent
    
    threshold = np.median(similarity)
    indicator = similarity > threshold #For choosing the pair that will be used

    dim= [np.size(fea_1,1), np.size(fea_2,1)]
    
    num_x = np.sum(indicator)
    
    
    X_1 = np.zeros([num_x,dim[0]])
    X_2 = np.zeros([num_x,dim[1]])
    
    count = 0
    
    for i in range(np.size(similarity, 0)):
        for j in range(np.size(similarity,1)):
            
            if similarity[i, j] > threshold:
                
                X_1[count] = fea_1[i]
                X_2[count] = fea_2[j]
                count = count + 1
    
    return [X_1, X_2]



def hash_function(X_1, X_2):
    #Based two view X_1, and X_2
    #return the hash function each view A_1 and A_2
    X_1t = np.transpose(X_1)
    X_2t = np.transpose(X_2)
    
    B = np.dot(np.linalg.pinv((np.dot(X_1t,X_1))),np.dot(X_1t,X_2))
    C = np.dot(np.linalg.pinv(np.dot(X_2t,X_2)),np.dot(X_2t,X_1))
    
    B = np.dot(B,C)
    
    [eig_val, A_1] = np.linalg.eig(B)
    
    print eig_val
    
    A_2 = np.dot(np.dot(C, A_1),np.linalg.pinv((np.sqrt(eig_val))))
    
    return [A_1, A_2]

def cvh():

    [image_tags_cross_similarity, image_features, tag_features] = load_data.analysis()
    
    [X_1, X_2] = domain2view(image_features, tag_features, image_tags_cross_similarity)
    
    [A_1, A_2] = hash_function(X_1, X_2)
    
    hash_dim = 100
    
    hash_1 = np.multiply(image_features, A_1)
    hash_2 = np.multiply(tag_features, A_2)
    
    return [hash_1, hash_2]
    
if __name__ == '__main__':
    
    [hash_1, hash_2 ] = cvh()
    
    print hash_1
    print hash_2

    
