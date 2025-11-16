function StationaryDist=StationaryDist_FHorz_CPU(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Parameters,simoptions)

if prod(n_d)==0
    l_d=0;
else
    l_d=length(n_d);
end
% l_a=1; % hardcoded for CPU that only one endogenous state

N_a=prod(n_a);
N_z=prod(n_z);

jequaloneDist=gather(jequaloneDist);
Policy=gather(Policy);
pi_z=gather(pi_z);


%% Deal with the no z first (as it needs different shapes to the rest)
if N_z==0
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

else % N_z>0
    jequaloneDist=reshape(jequaloneDist,[N_a*N_z,1]);
    Policy=reshape(Policy,[size(Policy,1),N_a,N_z,N_j]);

    % Policy_aprime
    Policy_aprime=Policy(l_d+1,:,:,:);
    Policy_aprime=shiftdim(Policy_aprime,1);

    Policy_aprimez=Policy_aprime+N_a*(0:1:N_z-1); % Note: add z' index following the z dimension [Tan improvement, z stays where it is]

    StationaryDistKron=zeros(N_a*N_z,N_j);
    StationaryDistKron(:,1)=jequaloneDist;

    StationaryDist_jj=sparse(jequaloneDist);

    IIind=1:1:N_a*N_z;
    JJind=ones(N_a,N_z);

    pi_z=sparse(pi_z);

    for jj=1:(N_j-1)

        Gammatranspose=sparse(Policy_aprimez(:,:,jj),IIind,JJind,N_a*N_z,N_a*N_z);

        % First step of Tan improvement
        StationaryDist_jj=reshape(Gammatranspose*StationaryDist_jj,[N_a,N_z]); %No point checking distance every single iteration. Do 100, then check.

        % Second step of Tan improvement
        StationaryDist_jj=reshape(StationaryDist_jj*pi_z,[N_a*N_z,1]);

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

    StationaryDist=reshape(StationaryDistKron,[n_a,n_z,N_j]);

end


end