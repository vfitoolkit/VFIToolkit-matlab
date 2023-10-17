function StationaryDistKron=StationaryDist_FHorz_Case1_Iteration_noz_e_raw(jequaloneDistKron,AgeWeightParamNames,PolicyIndexesKron,N_d,N_a,N_e,N_j,pi_e_J,Parameters,simoptions)
%Will treat the agents as being on a continuum of mass 1.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel
%  simoptions.parovere


% Ran a bunch of runtime tests. Tan improvement is always faster.
% Seems loop over e vs parallel over e is essentially break-even.

if simoptions.loopovere==0

    if N_d==0
        PolicyIndexesKron=gather(reshape(PolicyIndexesKron,[1,N_a*N_e,N_j]));
    else
        PolicyIndexesKron=gather(reshape(PolicyIndexesKron(2,:,:,:),[1,N_a*N_e,N_j]));
    end

    StationaryDistKron=zeros(N_a*N_e,N_j);
    StationaryDistKron(:,1)=gather(jequaloneDistKron);

    StationaryDist_jj=sparse(gather(jequaloneDistKron));

    
    for jj=1:(N_j-1)
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end

        optaprime=PolicyIndexesKron(1,:,jj);

        Gammatranspose=sparse(optaprime,1:1:N_a*N_e,ones(N_a*N_e,1),N_a,N_a*N_e);

        pi_e=sparse(gather(pi_e_J(:,jj))); % Note: this cannot be moved outside the for-loop as Matlab only allows sparse for 2-D arrays (so cannot, e.g., do sparse(pi_z_J)).

        % Tan improvement not really relevant for without z shock
        StationaryDist_jj=Gammatranspose*StationaryDist_jj;

        StationaryDist_jj=kron(pi_e,StationaryDist_jj);

        StationaryDistKron(:,jj+1)=full(StationaryDist_jj);
    end
    if simoptions.parallel==2 % Move result to gpu
        StationaryDistKron=gpuArray(StationaryDistKron);
        % Note: sparse gpu matrices do exist in matlab, but cannot index nor reshape() them. So cannot do Tan improvement with them.
    end

elseif simoption.loopovere==1
    StationaryDistKron=zeros(N_a,N_e,N_j);
    StationaryDist_jj=gather(reshape(jequaloneDistKron,[N_a,N_e]));
    StationaryDistKron(:,:,1)=StationaryDist_jj;
    
    if N_d==0
        PolicyIndexesKron=gather(reshape(PolicyIndexesKron,[1,N_a,N_e,N_j]));
    else
        PolicyIndexesKron=gather(reshape(PolicyIndexesKron(2,:,:,:,:),[1,N_a,N_e,N_j]));
    end

    for jj=1:(N_j-1)
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end

        for e_c=1:N_e % you can probably parfor this?
            optaprime=PolicyIndexesKron(1,:,e_c,jj);
            StationaryDist_jjee=sparse(StationaryDist_jj(:,e_c));

            Gammatranspose=sparse(optaprime,1:1:N_a,ones(N_a,1),N_a,N_a); % from a to a' (but transpose)

            % Tan improvement not really relevant for without z shock
            StationaryDist_jjee=Gammatranspose*StationaryDist_jjee; %No point checking distance every single iteration. Do 100, then check.

            StationaryDist_jj(:,e_c)=full(StationaryDist_jjee);
        end

        StationaryDist_jj=sum(StationaryDist_jj,2);
        StationaryDist_jj=StationaryDist_jj.*pi_e_J(:,jj)';

        StationaryDistKron(:,:,jj+1)=full(StationaryDist_jj);
    end
    if simoptions.parallel==2 % Move result to gpu
        StationaryDistKron=gpuArray(StationaryDistKron);
        % Note: sparse gpu matrices do exist in matlab, but cannot index nor reshape() them. So cannot do Tan improvement with them.
    end
elseif simoption.loopovere==2 % loop over e, but using a parfor loop
    StationaryDistKron=zeros(N_a,N_e,N_j);
    StationaryDist_jj=gather(reshape(jequaloneDistKron,[N_a,N_e]));
    StationaryDistKron(:,:,1)=StationaryDist_jj;

    if N_d==0
        PolicyIndexesKron=reshape(PolicyIndexesKron,[1,N_a,N_e,N_j]);
    else
        PolicyIndexesKron=reshape(PolicyIndexesKron(2,:,:,:),[1,N_a,N_e,N_j]);
    end

    for jj=1:(N_j-1)
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end

        parfor e_c=1:N_e % you can probably parfor this?
            optaprime=PolicyIndexesKron(1,:,e_c,jj);
            StationaryDist_jjee=sparse(StationaryDist_jj(:,e_c));

            Gammatranspose=sparse(optaprime,1:1:N_a,ones(N_a,1),N_a,N_a); % from a to a' (but transpose)

            % Tan improvement not really relevant for without z shock
            StationaryDist_jjee=Gammatranspose*StationaryDist_jjee;

            StationaryDist_jj(:,e_c)=full(StationaryDist_jjee);
        end

        StationaryDist_jj=sum(StationaryDist_jj,2);
        StationaryDist_jj=StationaryDist_jj.*pi_e_J(:,jj)';

        StationaryDistKron(:,:,jj+1)=full(StationaryDist_jj);
    end
    if simoptions.parallel==2 % Move result to gpu
        StationaryDistKron=gpuArray(StationaryDistKron);
        % Note: sparse gpu matrices do exist in matlab, but cannot index nor reshape() them. So cannot do Tan improvement with them.
    end
end


% Reweight the different ages based on 'AgeWeightParamNames'. (it is assumed there is only one Age Weight Parameter (name))
try
    AgeWeights=Parameters.(AgeWeightParamNames{1});
catch
    error('Unable to find the AgeWeightParamNames in the parameter structure')
end
% I assume AgeWeights is a row vector
if size(AgeWeights,2)==1 % If it seems to be a column vector, then transpose it
    AgeWeights=AgeWeights';
end

if simoptions.loopovere>0
    StationaryDistKron=StationaryDistKron.*shiftdim(AgeWeights,-1); %.*repmat(shiftdim(AgeWeights,-1),N_a*N_z,N_e,1);
else
    StationaryDistKron=StationaryDistKron.*AgeWeights;
end


end
