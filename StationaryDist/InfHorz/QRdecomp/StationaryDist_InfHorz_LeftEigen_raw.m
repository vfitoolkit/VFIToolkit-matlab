function StationaryDistKron=StationaryDist_InfHorz_LeftEigen_raw(PolicyIndexesKron,N_d,N_a,N_z,pi_z,simoptions)
%Will treat the agents as being on a continuum of mass 1.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel

% eigs() only works well for full cpu matrices
if N_d==0 %length(n_d)==1 && n_d(1)==0
    optaprime=reshape(PolicyIndexesKron,[1,N_a*N_z]);
else
    optaprime=reshape(PolicyIndexesKron(2,:,:),[1,N_a*N_z]);
end
Ptranspose=zeros(N_a,N_a*N_z);
Ptranspose(optaprime+N_a*(0:1:N_a*N_z-1))=1;
Ptranspose=(kron(pi_z',ones(N_a,N_a))).*(kron(ones(N_z,1),Ptranspose));


%% Calculate the stationary distribution as left eigenvector of the transition matrix
% Eigenvector approach doesn't work very well. Partly because the matrices
% are so sparse, partly because some of the grid ends up with zero mass and
% eigenvectors seem to have trouble with this (this is my guess of what is
% going wrong)

% fprintf('Starting Eigenvector calculation \n')

% [V,~] = eig(Ptranspose);
% % V are the right eigenvectors.
% % But the (transposes of) the right eigenvectors of Ptranspose are in fact the left
% % eigenvectors of P, and what we want are the left eigenvectors of P.
% % The left eigenvector, normalised to one, will give the stationary distribution.
% StationaryDistKron=V(:, 1)/sum(V(:, 1));
% % Note that I didn't need to take the transpose of this answer as I want a column vector 
% % anyway (the underlying math from which I took this used row vectors)

% We are only interested in the largest eigenvector.
% In principle we can do this using eigs(), but this doesn't seem any
% faster than just using eig() for the kind of Ptraspose matrices in actual
% economic models (eigs() was faster when I just created random matrices,
% but when I then implemented it here it was slower)
% Following commented out line is what I had
% [V,~] = eigs(Ptranspose,1); % We are only interested in the largest eigenvector
% Following lines are alternative I found in MNS2016. It includes a bunch of checks of input and output
assert(all(abs(sum(Ptranspose)-1)<1e-10));
opts.disp=0;
[x,eval] = eigs(Ptranspose,[],1,1+1e-10,opts);
assert(abs(eval-1)<1e-10);
V = x/sum(x);
% assert(min(V)>-1e-12); % Replaced with if statement
if min(V)>-1e-11
    StationaryDistKron=1;
    return
    % Failed, just return 1 and the code that calls this knows to try another method
end
V = max(V,0);

StationaryDistKron=V/sum(V);

% Note that we could check the first eigenvalue, D(1, 1), which should be 1
% (otherwise it is indicating that the stationary distribution can be
% reduced, it would be D in [V,D] = eig(Ptranspose',1);).
% The second eigenvalue would tell us how quickly the markov process
% converges to the stationary distribution, specifically 1/SecondEigenvalue gives the order of rate of convergence. 

%% 
if simoptions.parallel==2 || simoptions.parallel==4 % Return answer on gpu
    StationaryDistKron=gpuArray(StationaryDistKron);
end

end


%% Personal notes on trying to speed things up.
% eigs() is faster than eig() as we are only interested in the first eigenvector (which corresponds to the largest eigenvalue)
% Matlab cannot do eigs for gpu. eig() on gpuArray is slower than eig() on standard array.
% eig() and eigs() are no faster due to the extreme sparseness of Ptranspose than they would be for a matrix with no zero elements
% making Pstranspose a sparse matrix (i.e., sparse(Ptranspose)) just makes eigs() run slower

