function StationaryDistKron=StationaryDist_Case1_IterationTan_raw(StationaryDistKron,PolicyIndexesKron,N_d,N_a,N_z,pi_z,simoptions)
% Will treat the agents as being on a continuum of mass 1.
% Uses the improvement of: Tan (2020) - A fast and low computational memory algorithm for non-stochastic simulations in heterogeneous agent models

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel
%  simoptions.multiiter

% tic;

% First, get Gamma
if N_d==0
    optaprime=reshape(PolicyIndexesKron,[1,N_a*N_z]);
else
    optaprime=reshape(PolicyIndexesKron(2,:,:),[1,N_a*N_z]);
end

Gammatranspose=sparse(N_a*N_z,N_a*N_z);
firststep=optaprime+kron(N_a*(0:1:N_z-1),ones(1,N_a));
Gammatranspose(firststep+N_a*N_z*(0:1:N_a*N_z-1))=1;

pi_z=sparse(pi_z);

if simoptions.parallel==2
    Gammatranspose=gpuArray(Gammatranspose);
    pi_z=gpuArray(pi_z);
end

% rdk1=toc
% 
% tic;

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
    
    % Do multiple iterations before checking the tolerance (as this is faster)
    for jj=1:simoptions.multiiter
        % Two steps of the Tan improvement
        StationaryDistKron=reshape(Gammatranspose*StationaryDistKron,[N_a,N_z]); %No point checking distance every single iteration. Do 100, then check.
        StationaryDistKron=reshape(StationaryDistKron*pi_z,[N_a*N_z,1]);
    end

    StationaryDistKronOld=StationaryDistKron;

    % Base the tolerance on a single iteration.
    StationaryDistKron=Gammatranspose*StationaryDistKron; %No point checking distance every single iteration. Do 100, then check.
    StationaryDistKron=reshape(StationaryDistKron,[N_a,N_z])*pi_z;
    StationaryDistKron=StationaryDistKron(:); % effectively reshape(StationaryDistKron,[N_a*N_z,1]), but faster

    % currdist=sum(abs(StationaryDistKron-StationaryDistKronOld));
    currdist=max(abs(StationaryDistKron-StationaryDistKronOld));
    
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

% rdk2=toc

end
