#Load format data from text
#Author: Bo Liu
#Date: 2013.10.15
import numpy as np
import os


def analysis(Tr_sim, Tr_fea1, Tr_fea2, Tst_sim, Tst_fea1, Tst_fea2, GD_Path):
    num_img_fea = 500
    num_tag_fea = 1000

    path = os.getcwd()

    #Construct the path
    Tr_sim_path = os.path.join(path, Tr_sim)
    Tr_img_path = os.path.join(path, Tr_fea1)
    Tr_tag_path = os.path.join(path, Tr_fea2)

    Tst_sim_path = os.path.join(path, Tst_sim)
    Tst_img_path = os.path.join(path, Tst_fea1)
    Tst_qa_path = os.path.join(path, Tst_fea2)

    gd_path = os.path.join(path, GD_Path)

    #Read
    Tr_sim_file = open(Tr_sim_path).readlines()
    Tr_img_file = open(Tr_img_path).readlines()
    Tr_tag_file = open(Tr_tag_path).readlines()

    Tst_sim_file = open(Tst_sim_path).readlines()
    Tst_img_file = open(Tst_img_path).readlines()
    Tst_qa_file = open(Tst_qa_path).readlines()
    gd_file = open(gd_path).readlines()

    #Init matrix
    Tr_sim = np.zeros((len(Tr_img_file), len(Tr_sim_file))) #the similarity between image and tag in 170*170
    Tr_img = np.zeros((len(Tr_img_file), num_img_fea))
    Tr_tag = np.zeros((len(Tr_tag_file), num_tag_fea))

    Tst_sim = np.zeros((len(Tst_img_file), len(Tst_qa_file)))
    Tst_img = np.zeros((len(Tst_img_file), num_img_fea))
    Tst_qa = np.zeros((len(Tst_qa_file), num_tag_fea))
    gd = np.zeros((len(gd_file), len(Tst_qa_file)))

    #QA feature matrix #qa * # bag of word
    for i in range(len(Tst_qa_file)):
        line = Tst_qa_file[i].split('\t')
        for j in range(len(line)):
            Tst_qa[i, j] = float(line[j])

        #Ground Truth
    for i in range(len(gd_file)):
        line = gd_file[i].split('\t')
        for j in range(len(line)):
            gd[i, j] = int(line[j])

    #Tst similarit
    for i in range(len(Tst_sim_file)):
        line = Tst_sim_file[i].split('\t')
        for j in range(len(line)):
            Tst_sim[i, j] = float(line[j])

    #Tst Images
    for i in range(len(Tst_img_file)):
        line = Tst_img_file[i].split('\t')
        for j in range(len(line)):
            Tst_img[i, j] = float(line[j])

    #Train similarity matrix #image * #tag
    for i in range(len(Tr_sim_file)):
    #similarity = line.split('\t')
    #Tr_sim[int(similarity[0])-1][int(similarity[1])-1] = float(similarity[2])
        line = Tr_sim_file[i].split('\t')
        for j in range(len(line)):
            Tr_sim[i, j] = float(line[j])

        #Image feature n*500
    for i in range(len(Tr_img_file)):
        line = Tr_img_file[i].split('\t')
        for j in range(len(line)):
             #if (':') in feature:
                #fea_idx = feature.split(':')[0]
                #fea_val = feature.split(':')[1]
                #Tr_img[i][int(fea_idx) - 1] = int(fea_val)
            Tr_img[i, j] = float(line[j])

        #Tag feature n*1000
    for i in range(len(Tr_tag_file)):
        line = Tr_tag_file[i].split('\t')
        for j in range(len(line)):
            #if (':') in feature:
                #fea_idx = feature.split(':')[0]
                #fea_val = feature.split(':')[1]
            Tr_tag[i, j] = float(line[j])

    #Return the data set matrix that is dimension \times instance
    #Ground Truth is images \times QA
    return [Tr_sim, Tr_img.transpose(), Tr_tag.transpose(), Tst_sim, Tst_img.transpose(), Tst_qa.transpose(), gd]


if __name__ == '__main__':
    [image_tags_cross_similarity, image_features, tag_features, QA_fea, GD] = analysis()

    print QA_fea.shape
    print GD.shape
