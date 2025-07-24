function StationaryDistKron=StationaryDist_InfHorz_Iteration_raw(StationaryDistKron,PolicyIndexesKron,N_d,N_a,N_z,pi_z,simoptions)
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
end

%% The rest is essentially the same regardless of which simoption.parallel is being used
% SteadyStateDistKron=ones(N_a*N_z,1)/(N_a*N_z); % This line was handy when checking/debugging. Have left it here.
StationaryDistKron=sparse(StationaryDistKron);

currdist=Inf;
counter=0;
while currdist>simoptions.tolerance && counter<simoptions.maxit  % Matlab objects to using currdist here if I don't 'full' it
    
    StationaryDistKron=Ptranspose*StationaryDistKron; % Base the tolerance on 10 iterations. (For some reason just using one iteration worked perfect on gpu, but was not accurate enough on cpu)

    % Only check covergence every couple of iterations
    if rem(counter,simoptions.multiiter)==0
        StationaryDistKronOld=StationaryDistKron;
    elseif rem(counter,simoptions.multiiter)==10
        currdist=max(gather(abs(StationaryDistKron-StationaryDistKronOld))); % Note: changed over to max distance, to match the Tan improvement version
    end

    counter=counter+1;

    if simoptions.verbose==1
        if rem(counter,50)==0
            fprintf('StationaryDist_Case1: after %i iterations the current distance ratio is %8.6f (currdist/tolerance, convergence when reaches 1) \n', counter, currdist/simoptions.tolerance)            
        end
    end
end

%% Turn the resulting agent distribution into a full matrix
StationaryDistKron=full(StationaryDistKron);

if ~(counter<simoptions.maxit)
    warning('SteadyState_Case1 stopped due to reaching simoptions.maxit, this might be causing a problem')
end 


end
