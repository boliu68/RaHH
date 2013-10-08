function A  = cvh(W,varargin)

%Reference : Kumar, S., & Udupa, R. (2011). Learning hash functions for cross-view similarity search. Paper presented at the Proceedings of the Twenty-Second international joint conference on Artificial Intelligence-Volume Volume Two.
%Input: X: n dim-dimensional instances.
%       W: n*n matrix to represent the similarity between each instance
%Output: The hash function for each view/domain/task. 
%For initialize RaHH. Can also be used as baseline.

K = nargin-1;%The number of tasks.
N = zeros(K,1);
dim = zeros(K,1);

for k = 1:K
    
    N(k) = size(varargin{k},1);
    dim(k) = size(varargin{k},2);
    
end

D_ii = zeros(N,1);
for n = 1:N   
    D_ii(n) = sum(W(n,:));
end
D = diag(D_ii);%Initialize D

L = D-W;
L_ = 2*L + (K-1)*D;


for k = 1:K
   
    eval(['A_' num2str(k) '=zeros(' num2str(N(k)) ',' num2str(dim(k)) ');']);
    Left = varargin{k}'*L_*varargin{k};
    Right = varargin{k}'*varargin{k};
    
    [A_k,lambda] = eig(Left,Right);
    
    [lambda,order] = sort(diag(lambda),'descend');
    A_k = A_k(:,order);
   
    A{k} = A_k
end


