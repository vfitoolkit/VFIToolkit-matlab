function StationaryDist=StationaryDist_FHorz_CPU(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Parameters,simoptions)

if prod(n_d)==0
    l_d=0;
else
    l_d=length(n_d);
end
% l_a=1; % hardcoded for CPU that only one endogenous state

N_a=prod(n_a);
N_z=prod(n_z);
if isfield(simoptions,'n_e')
    N_e=prod(simoptions.n_e);
else
    N_e=0;
end

jequaloneDist=gather(jequaloneDist);
Policy=gather(Policy);
pi_z=gather(pi_z);


%% Deal with the no z and no e first (as it needs different shapes to the rest)
if N_z==0 && N_e==0
    jequaloneDist=reshape(jequaloneDist,[N_a,1]);
    Policy=reshape(Policy,[size(Policy,1),N_a,N_j]);

    % Policy_aprime
    Policy_aprime=Policy(l_d+1,:,:);
    Policy_aprime=shiftdim(Policy_aprime,1);

    StationaryDistKron=zeros(N_a,N_j);
    StationaryDistKron(:,1)=jequaloneDist;

    StationaryDistKron_jj=sparse(jequaloneDist);

    IIind=(1:1:N_a)';
    JJind=ones(N_a,1);

    for jj=1:(N_j-1)

        Gammatranspose=sparse(Policy_aprime(:,jj),IIind,JJind,N_a,N_a);

        StationaryDistKron_jj=Gammatranspose*StationaryDistKron_jj;
        StationaryDistKron(:,jj+1)=full(StationaryDistKron_jj);
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

    StationaryDistKron=StationaryDistKron.*AgeWeights;

    StationaryDist=reshape(StationaryDistKron,[n_a,N_j]);

else % N_z>0 or N_e>0
    if N_z==0
        n_ze=simoptions.n_e;
        N_ze=N_e;
    elseif N_e==0
        n_ze=n_z;
        N_ze=N_z;
    else % neither is zero
        n_ze=[n_z,simoptions.n_e];
        N_ze=N_z*N_e;
    end

    jequaloneDist=reshape(jequaloneDist,[N_a*N_ze,1]);
    Policy=reshape(Policy,[size(Policy,1),N_a,N_ze,N_j]);

    % Policy_aprime
    Policy_aprime=Policy(l_d+1,:,:,:);
    Policy_aprime=shiftdim(Policy_aprime,1);

    Policy_aprimez=Policy_aprime+N_a*(0:1:N_ze-1); % Note: add z' index following the z dimension [Tan improvement, z stays where it is]

    StationaryDistKron=zeros(N_a*N_ze,N_j);
    StationaryDistKron(:,1)=jequaloneDist;

    StationaryDist_jj=sparse(jequaloneDist);

    IIind=1:1:N_a*N_ze;
    JJind=ones(N_a,N_ze);

    if N_e==0
        pi_z=sparse(pi_z);
    else
        pi_z=sparse(repmat(pi_z,N_e));
    end

    for jj=1:(N_j-1)

        Gammatranspose=sparse(Policy_aprimez(:,:,jj),IIind,JJind,N_a*N_ze,N_a*N_ze);

        % First step of Tan improvement
        StationaryDist_jj=reshape(Gammatranspose*StationaryDist_jj,[N_a,N_ze]); %No point checking distance every single iteration. Do 100, then check.

        % Second step of Tan improvement
        StationaryDist_jj=reshape(StationaryDist_jj*pi_z,[N_a*N_ze,1]);

        StationaryDistKron(:,jj+1)=full(StationaryDist_jj);
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

    StationaryDistKron=StationaryDistKron.*AgeWeights;

    StationaryDist=reshape(StationaryDistKron,[n_a,n_ze,N_j]);

end


end