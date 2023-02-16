function StationaryDistKron=StationaryDist_Case1_Iteration_raw(StationaryDistKron,PolicyIndexesKron,N_d,N_a,N_z,pi_z,simoptions)
%Will treat the agents as being on a continuum of mass 1.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel

%kstep tells the code to use the k-step transition matrix P^k, instead of
%P, when calculating the steady state distn
%kstep=100;
%THIS does not seem to be a good idea as it uses way to much memory and
%appears to in fact slow the code down. NOTE: this is no longer used
%anywhere in code, I leave it here as a reminder that I tried this and it
%did not work well. This is particularly true now that I use sparse
%matrices.

%% If the Ptranspose matrix could be stored as a full matrix then it would have used the eigenvector method (StationaryDist_Case1_LeftEigen_raw)
% So the following only does sparse matrix (or sparse gpu matrix)

%% This commented out section was the full matrix code that is no longer used.
% if simoptions.parallel<2
% %     % The following commented out version was producing machine precision
% %     % level errors in Ptranspose that lead to machine precision level
% %     % errors in the individual points of the StationaryDist, but that
% %     % summed up accross the StationaryDist this errors were numerically
% %     % important and causing problems.
% %     %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
% %     P=zeros(N_a,N_z,N_a,N_z); %P(a,z,aprime,zprime)=proby of going to (a',z') given in (a,z)
% %     for a_c=1:N_a
% %         for z_c=1:N_z
% %             if N_d==0 %length(n_d)==1 && n_d(1)==0
% %                 optaprime=PolicyIndexesKron(a_c,z_c);
% %             else
% %                 optaprime=PolicyIndexesKron(2,a_c,z_c);
% %             end
% %             for zprime_c=1:N_z
% %                 P(a_c,z_c,optaprime,zprime_c)=pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
% %             end
% %         end
% %     end
% %     P=reshape(P,[N_a*N_z,N_a*N_z]);
% %     Ptranspose=P';
% 
%     if N_d==0 %length(n_d)==1 && n_d(1)==0
%         optaprime=reshape(PolicyIndexesKron,[1,N_a*N_z]);
%     else
%         optaprime=reshape(PolicyIndexesKron(2,:,:),[1,N_a*N_z]);
%     end
%     Ptranspose=zeros(N_a,N_a*N_z);
%     Ptranspose(optaprime+N_a*(0:1:N_a*N_z-1))=1;
%     Ptranspose=(kron(pi_z',ones(N_a,N_a))).*(kron(ones(N_z,1),Ptranspose));
%     
% elseif simoptions.parallel==2 % Using the GPU
%     % First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
% 
%     if N_d==0 %length(n_d)==1 && n_d(1)==0
%         optaprime=reshape(PolicyIndexesKron,[1,N_a*N_z]);
%     else
%         optaprime=reshape(PolicyIndexesKron(2,:,:),[1,N_a*N_z]);
%     end
%     Ptranspose=zeros(N_a,N_a*N_z,'gpuArray');
%     Ptranspose(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=1;
%     Ptranspose=(kron(pi_z',ones(N_a,N_a,'gpuArray'))).*(kron(ones(N_z,1,'gpuArray'),Ptranspose));
    
%% Following is the sparse matrix code
% I create the sparse matrix Ptranspose on the cpu
% If simoptions.parallel=2 it is then transferred to the gpu
% Would probably be faster if I rewrote this code to just create Ptranspose
% on the gpu when appropriate.

%     % The following commented out version was producing machine precision
%     % level errors in Ptranspose that lead to machine precision level
%     % errors in the individual points of the StationaryDist, but that
%     % summed up accross the StationaryDist this errors were numerically
%     % important and causing problems.
%     tic;
%     %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
%     Ptranspose=sparse(N_a*N_z,N_a*N_z); %P(a,z,aprime,zprime)=proby of going to (a',z') given in (a,z) [ So Ptranspose(aprime,zprime,a,z)=proby of going to (a',z') given in (a,z)]
%     if N_d==0 %length(n_d)==1 && n_d(1)==0
%         PolicyIndexesKron=reshape(PolicyIndexesKron(:,:),[N_a*N_z,1]);
%     else
%         PolicyIndexesKron=reshape(PolicyIndexesKron(2,:,:),[N_a*N_z,1]);
%     end
%     
%     parfor az_c=1:N_a*N_z
%         Ptranspose_az=sparse(N_a*N_z,1);
%         optaprime=PolicyIndexesKron(az_c);
%         az_sub=ind2sub_homemade([N_a,N_z],az_c);
%         pi_z_temp=pi_z(az_sub(2),:);
%         sum_pi_z_temp=sum(pi_z_temp); % COULD PRECOMPUTE THIS OUTSIDE THE PARFOR-LOOP (other than rounding error this will be 1, but just to be sure; I added this line as the rounding error turned out to matter for larger models)
%         for zprime_c=1:N_z % THIS COULD BE DONE AS MATRIX OPERATION, would be faster than this for-loop.
%             optaprimezprime=sub2ind_homemade([N_a,N_z],[optaprime,zprime_c]);
%             Ptranspose_az(optaprimezprime)=pi_z_temp(zprime_c)/sum_pi_z_temp;
%         end
%         Ptranspose(:,az_c)=Ptranspose_az;
%     end
%     toc


if N_d==0 %length(n_d)==1 && n_d(1)==0
    optaprime=reshape(PolicyIndexesKron,[1,N_a*N_z]);
else
    optaprime=reshape(PolicyIndexesKron(2,:,:),[1,N_a*N_z]);
end
PtransposeA=sparse(N_a,N_a*N_z);
PtransposeA(optaprime+N_a*(0:1:N_a*N_z-1))=1;

pi_z=sparse(pi_z);
try % Following formula only works if pi_z is already sparse, otherwise kron(pi_z',ones(N_a,N_a)) is not sparse.
    Ptranspose=kron(pi_z',ones(N_a,N_a)).*kron(ones(N_z,1),PtransposeA);
catch % Otherwise do something slower but which is sparse regardless of whether pi_z is sparse
    pi_z=gather(pi_z); % The indexing used can only be donoe on cpu
    Ptranspose=kron(ones(N_z,1),PtransposeA);
    for ii=1:N_z
        Ptranspose(:,(1:1:N_a)+N_a*(ii-1))=Ptranspose(:,(1:1:N_a)+N_a*(ii-1)).*kron(pi_z(ii,:)',ones(N_a,N_a));
    end
end

if simoptions.parallel==2
    Ptranspose=gpuArray(Ptranspose);
    pi_z=gpuArray(pi_z);
end

%% The rest is essentially the same regardless of which simoption.parallel is being used
%SteadyStateDistKron=ones(N_a*N_z,1)/(N_a*N_z); % This line was handy when checking/debugging. Have left it here.
if simoptions.parallel==2
    StationaryDistKronOld=gpuArray(sparse(N_a*N_z,1)); % sparse() creates a matrix of zeros
else
    StationaryDistKronOld=sparse(N_a*N_z,1); % sparse() creates a matrix of zeros
end

StationaryDistKron=sparse(StationaryDistKron);

currdist=full(sum(abs(StationaryDistKron-StationaryDistKronOld)));
counter=0;
while currdist>simoptions.tolerance && counter<simoptions.maxit  % Matlab objects to using currdist here if I don't 'full' it
    
    for jj=1:100
        StationaryDistKron=Ptranspose*StationaryDistKron; %No point checking distance every single iteration. Do 100, then check.
    end
    StationaryDistKronOld=StationaryDistKron;
    StationaryDistKron=Ptranspose*StationaryDistKron; % Base the tolerance on 10 iterations. (For some reason just using one iteration worked perfect on gpu, but was not accurate enough on cpu)
    currdist=full(sum(abs(StationaryDistKron-StationaryDistKronOld)));
    
    counter=counter+1;
    if simoptions.verbose==1
        if rem(counter,50)==0
            fprintf('StationaryDist_Case1: after %i iterations the current distance ratio is %8.6f (currdist/tolerance, convergence when reaches 1) \n', counter, currdist/simoptions.tolerance)            
            maxdist=full(max(gather(abs(StationaryDistKron-StationaryDistKronOld))));
            fprintf('StationaryDist_Case1: after %i iterations the max distance %8.12f \n', counter, maxdist)
        end
    end
end

%% Turn the resulting agent distribution into a full matrix
StationaryDistKron=full(StationaryDistKron);

if ~(counter<simoptions.maxit)
    disp('WARNING: SteadyState_Case1 stopped due to reaching simoptions.maxit, this might be causing a problem')
end 

end
