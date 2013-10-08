function A  = cvh(X,W)

%Reference : Kumar, S., & Udupa, R. (2011). Learning hash functions for cross-view similarity search. Paper presented at the Proceedings of the Twenty-Second international joint conference on Artificial Intelligence-Volume Volume Two.
%Input: X: n dim-dimensional instances.
%       W: n*n matrix to represent the similarity between each instance
%Output: The hash function for each view/domain/task. 
%For initialize RaHH. Can also be used as baseline.

K = size(X,1);%The number of tasks.
N = size(X,2);;%The number of instances
dim = size(X,3)%The number of dim.

D_ii = zeros(n,1);
for n = 1:N   
    D_ii(n) = sum(W(i,:));
end
D = diag(D_ii);%Initialize D

L = D-W;
L_ = 2*L + (K-1)*D;


for k = 1:K
   
    eval()('A',num2str(k),'=zeros[',num2str(N(k)),',',num2str(dim(k)));
    Left = X(k)'*L_*X(k);
    Right = X(k)'*X(k)
    
    [A_k,lambda] = eig(Left,Right);
    
    [lambda,order] = sort(diag(lambda),'descend');
    A_k = A_k(:,order);
    
    eval()('A',num2str(k),'=A_k');
end


