function load_data()

image_tags_cross_similarity = load('image_tags_cross_similarity.txt');
image_tags_cross_similarity_mat = zeros(max(image_tags_cross_similarity(:,1)),max(image_tags_cross_similarity(:,2)));
similarity = image_tags_cross_similarity(:,3);

for i = 1:max(image_tags_cross_similarity(:,1))
    for j = 1:max(image_tags_cross_similarity(:,2))
       
        index = image_tags_cross_similarity(:,1)==i & image_tags_cross_similarity(:,2) == j;
        
        if max(index) ~=0
            image_tags_cross_similarity_mat(i,j) = similarity(index);
        end
    end
end

image_tags_cross_similarity_mat(i,j);

end