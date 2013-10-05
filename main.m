%Relation-aware Heterogeneous Hashing(RaHH)
%Puesdo-Code
%Author: Bo Liu
%Date: Oct. 05 2013
%Reference: 
%Ou, M., Cui, P., Wang, F., Wang, J., Zhu, W., & Yang, S. (2013). Comparing apples to oranges: a scalable solution with heterogeneous hashing. Paper presented at the Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining.
%Rahh()
%Input: X^p/data, R_p,intra_domain relation, R_pq inter-domain relation.
%Output: H^p: hash function, W: map function to map the hash code to another Hamming space.
%initialize H by cross-view-hashing and w as identity matrix.
%initialize S.
%while the value of objective function don't converge do.
%for each domain p do
%	for each entity i in domain p do:
%		calculate the gradient with respect to h_i^p
%		update h_i^p
%		update S
%	end for
%	for each domain q do:
%		for each bit k of domain q do
%			calculate gradients with respects to w_k^pq
%			update w_k^pq
%		end for
%	end for
%end for
%end while.
[H,W] = function RaHH(X,R_p,R_pq)

P = size(X,1); %The number of domain
m = size(X,2); %The number of instances
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

