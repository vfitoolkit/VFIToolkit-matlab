function StationaryDistKron=StationaryDist_Case1_Iteration_EntryExit2_raw(StationaryDistKron,PolicyIndexesKron,N_d,N_a,N_z,pi_z, ExitProb, EntryDist, simoptions)
%Will treat the agents as being on a continuum of mass 1.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel

% Note that EntryDist is of size N_a*N_z-by-1.

% First, get Gamma
if N_d==0
    optaprime=gather(reshape(PolicyIndexesKron,[1,N_a*N_z]));
else
    optaprime=gather(reshape(PolicyIndexesKron(2,:,:),[1,N_a*N_z]));
end
ExitProb=gather(ExitProb);
pi_z=sparse(gather(pi_z));

Gammatranspose=sparse(optaprime+kron(N_a*(0:1:N_z-1),ones(1,N_a)),1:1:N_a*N_z,(1-ExitProb)*ones(N_a*N_z,1),N_a*N_z,N_a*N_z);


%% The rest is essentially the same regardless of which simoption.parallel is being used
% StationaryDistKron=sparse(N_a*N_z,1);
StationaryDistKron=sparse(gather(StationaryDistKron));

currdist=Inf;
counter=0;
while currdist>simoptions.tolerance && (100*counter)<simoptions.maxit
    
    for jj=1:100
        % Tan improvement
        StationaryDistKron=reshape(Gammatranspose*StationaryDistKron,[N_a,N_z]); %No point checking distance every single iteration. Do 100, then check.
        StationaryDistKron=reshape(StationaryDistKron*pi_z,[N_a*N_z,1]);
        StationaryDistKron=StationaryDistKron+ExitProb*EntryDist;
    end
    StationaryDistKronOld=StationaryDistKron;

    % Tan improvement
    StationaryDistKron=reshape(Gammatranspose*StationaryDistKron,[N_a,N_z]); %No point checking distance every single iteration. Do 100, then check.
    StationaryDistKron=reshape(StationaryDistKron*pi_z,[N_a*N_z,1]);
    StationaryDistKron=StationaryDistKron+ExitProb*EntryDist;

    currdist=sum(abs(StationaryDistKron-StationaryDistKronOld));
    
    counter=counter+1;
    if simoptions.verbose==1
        if rem(counter,50)==0
            fprintf('StationaryDist_Case1: after %i iterations the current distance is %8.4f (tolerance=%8.4f) \n', counter, currdist, simoptions.tolerance)            
        end
    end
end

if simoptions.parallel==2
    StationaryDistKron=gpuArray(full(StationaryDistKron));
else
    StationaryDistKron=full(StationaryDistKron);
end

if ~(counter<simoptions.maxit)
    disp('WARNING: SteadyState_Case1 stopped due to reaching simoptions.maxit, this might be causing a problem')
end 




end
