#Load format data from text
#Author: Bo Liu
#Date: 2013.10.15

import numpy as np
import os

def analysis():
    
	path = os.getcwd()
	    
	similarity_path = os.path.join(path,'Data/image_tags_cross_similarity.txt')
	image_features = os.path.join(path,'Data/images_features_sparse.txt')
	tag_features = os.path.join(path,'Data/tags_features_sparse.txt')
	
	image_tags_cross_similarity_file = open(similarity_path).readlines()
	image_features_file = open(image_features).readlines()
	tag_features_file = open(tag_features).readlines()

	image_tags_cross_similarity = np.zeros([170,170]) #the similarity between image and tag in 170*170
	image_features = np.zeros([len(image_features_file),500])
	tag_features  = np.zeros([len(tag_features_file),1000])


	#similarity matrix #image * #tag
	for line in image_tags_cross_similarity_file:
		
		similarity = line.split('\t')
		image_tags_cross_similarity[int(similarity[0])-1][int(similarity[1])-1] = float(similarity[2])
	
	#Image feature n*500
	for i in range(len(image_features_file)):
		
		line = image_features_file[i].split(' ')
		
		for feature in line:
			if (':') in feature:
				fea_idx = feature.split(':')[0]
				fea_val = feature.split(':')[1]
		
				image_features[i][int(fea_idx)-1] = int(fea_val)
	
	#Tag feature n*1000
	for i in range(len(tag_features_file)):
		
		line = tag_features_file[i].split(' ')
		
		for feature in line:
			
			if (':') in feature:
				
				fea_idx = feature.split(':')[0]
				fea_val = feature.split(':')[1]
				tag_features[i][int(fea_idx)-1] = float(fea_val)


	return [image_tags_cross_similarity, image_features, tag_features]
				

if __name__ == '__main__':

    [image_tags_cross_similarity, image_features, tag_features] = analysis()

    print image_features

