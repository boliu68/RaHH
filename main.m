[H,W] = function RaHH(X,R_p,R_pq,r)
%Relation-aware Heterogeneous Hashing(RaHH)
%Puesdo-Code
%Author: Bo Liu
%Date: Oct. 05 2013
%Reference: 
%Ou, M., Cui, P., Wang, F., Wang, J., Zhu, W., & Yang, S. (2013). Comparing apples to oranges: a scalable solution with heterogeneous hashing. Paper presented at the Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining.
%Rahh()
%Input: X^p/data, R_p,intra_domain relation, R_pq inter-domain relation. r: the number of bit for each domain
%Output: H^p: hash function, W: map function to map the hash code to another Hamming space.

dim = size(X,1); %The dimension of input data
m = size(X,2); %The number of instances
P = size(r,1)%The number of domains
bit = [1,2] %The number of bit for each domain. Equal to r_p in origin paper


for p = 1:P
	%initialize H_p
	
end

%initialize S

% J = loss_function(H,W, X, R_p, R_pq)
converge_threshold = 10

while (J - J_old < converge_threshold)

    for p = 1:P
	for i = 1:m(p)
	    %Gradient to h_i_p
	    %update h_i_p
	    %update S
	end

	for q = 1:P

		for k = 1:bit(p)
			
			%calculate gradient to w_pq_k
			%update w_pq_k

		end
	end

    end

	
end

