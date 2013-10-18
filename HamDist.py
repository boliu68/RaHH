#To calculate the Hamming Distance given two string
import numpy

def HamDis(v1, v2):
    
    distance = np.sum(v1 ^ v2)
    
    return distance