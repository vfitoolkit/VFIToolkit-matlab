function P=CreatePTransitionMatrix(Policy,l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,pi_semiz,pi_z,pi_e,Parameters,simoptions)

if simoptions.experienceasset==1
    %% Experience asset
    error('Not yet implemented')
elseif simoptions.inheritanceasset==1
    %% Inheritance asset
    % Not yet implmeented for semiz or e
    Policy=reshape(Policy, [size(Policy,1),N_a,N_z]);
    % Not yet implemented with grid interpolation layer
    
    %% Setup related to inheritance asset
    n_d2=n_d(end);
    % Split endogenous assets into the standard ones and the inheritance asset
    if isscalar(n_a)
        n_a1=0;
    else
        n_a1=n_a(1:end-1);
    end
    n_a2=n_a(end); % n_a2 is the inheritance asset

    if ~isfield(simoptions,'aprimeFn')
        error('To use an inheritance asset you must define simoptions.aprimeFn')
    end
    if isfield(simoptions,'a_grid')
        % a_grid=simoptions.a_grid;
        % a1_grid=simoptions.a_grid(1:sum(n_a1));
        a2_grid=simoptions.a_grid(sum(n_a1)+1:end);
    else
        error('To use an inheritance asset you must define simoptions.a_grid')
    end
    if isfield(simoptions,'d_grid')
        d_grid=simoptions.d_grid;
    else
        error('To use an inheritance asset you must define simoptions.d_grid')
    end
    if isfield(simoptions,'z_grid')
        z_gridvals=CreateGridvals(n_z,simoptions.z_grid,1);
    else
        error('To use an inheritance asset you must define simoptions.z_grid')
    end
    
    % aprimeFnParamNames in same fashion
    l_d2=length(n_d2);
    l_z=length(n_z);
    temp=getAnonymousFnInputNames(simoptions.aprimeFn);
    if length(temp)>(l_d2+2*l_z)
        aprimeFnParamNames={temp{l_d2+2*l_z+1:end}}; % the first inputs will always be (d2,a2)
    else
        aprimeFnParamNames={};
    end

    N_zprime=N_z; % just to make code easier to read

    % Policy is currently about d and a1prime. Convert it to being about aprime as that is what we need for simulation.
    Policy_a2prime=zeros(N_a,N_z,N_zprime,2,'gpuArray'); % the lower grid point
    PolicyProbs=zeros(N_a,N_z,N_zprime,2,'gpuArray'); % The fourth dimension is lower/upper grid point
    whichisdforinheritasset=length(n_d);  % is just saying which is the decision variable that influences the experience asset (it is the 'last' decision variable)
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames);
    [a2primeIndexes, a2primeProbs]=CreateaprimePolicyInheritanceAsset(Policy,simoptions.aprimeFn, whichisdforinheritasset, n_d, n_a1,n_a2, n_z, n_z, gpuArray(d_grid), a2_grid, gpuArray(z_gridvals), gpuArray(z_gridvals), aprimeFnParamsVec);
    % Note: aprimeIndexes and aprimeProbs are both [N_a,N_z,N_zprime]
    % Note: aprimeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the aprimeProbs are the probability of this lower point (prob of upper point is just 1 minus this).
    Policy_a2prime(:,:,:,1)=a2primeIndexes; % lower grid point
    Policy_a2prime(:,:,:,2)=a2primeIndexes+1; % upper grid point
    PolicyProbs(:,:,:,1)=a2primeProbs; % probability of lower grid point
    PolicyProbs(:,:,:,2)=1-a2primeProbs; % probability of upper grid point

    if l_a==1 % just inheritanceasset
        Policy_aprime=Policy_a2prime;
    elseif l_a==2 % one other asset, then inheritance asset
        Policy_aprime(:,:,:,1)=reshape(Policy(l_d+1,:,:),[N_a,N_z,1])+n_a(1)*(Policy_a2prime(:,:,:,1)-1);
        Policy_aprime(:,:,:,2)=reshape(Policy(l_d+1,:,:),[N_a,N_z,1])+n_a(1)*Policy_a2prime(:,:,:,1); % Note: upper grid point minus 1 is anyway just lower grid point
    elseif l_a==3 % two other assets, then inheritance asset
        Policy_aprime(:,:,:,1)=reshape(Policy(l_d+1,:,:),[N_a,N_z,1])+n_a(1)*reshape(Policy(l_d+2,:,:),[N_a,N_z,1])+n_a(1)*n_a(2)*(Policy_a2prime(:,:,:,1)-1);
        Policy_aprime(:,:,:,2)=reshape(Policy(l_d+1,:,:),[N_a,N_z,1])+n_a(1)*reshape(Policy(l_d+2,:,:),[N_a,N_z,1])+n_a(1)*n_a(2)*Policy_a2prime(:,:,:,1); % Note: upper grid point minus 1 is anyway just lower grid point
    elseif l_a>3
        error('Not yet implemented inheritanceasset with length(n_a)>3')
    end

    N_probs=2;
    
    % Policy_aprime and PolicyProbs are currently [N_a,N_z,N_zprime,N_probs]
    Policy_aprimezprime=Policy_aprime+N_a*shiftdim(gpuArray(0:1:N_z-1),-1);  % Note: add z' index following the z' dimension
    Policy_aprimezprime=gather(reshape(Policy_aprimezprime,[N_a*N_z,N_zprime*N_probs])); % sparse() requires inputs to be 2-D
    PolicyProbs=gather(reshape(PolicyProbs,[N_a*N_z,N_zprime*N_probs])); % sparse() requires inputs to be 2-D

    II2=repmat((1:1:N_a*N_z)',1,N_zprime*N_probs); %  Index for this period (a,z), note the N_zprime*N_probs-copies

    pi_z=sparse(gather(repelem(repmat(pi_z,1,N_probs),N_a,1)));

    % Transition matrix
    P=sparse(II2,Policy_aprimezprime,PolicyProbs.*pi_z,N_a*N_z,N_a*N_z); % Note: sparse() will accumulate at repeated indices

else
    %% Standard Endogenous States
    if N_semiz==0 && N_z>0 && N_e==0
        Policy=reshape(Policy, [size(Policy,1),N_a,N_z]);

        if simoptions.gridinterplayer==0
            Policy_aprime=zeros(N_a,N_z,1,'gpuArray');
        elseif simoptions.gridinterplayer==1
            Policy_aprime=zeros(N_a,N_z,2,'gpuArray'); % Policy_aprime has an additional final dimension of length 2 which is the two points (and contains only the aprime indexes, no d indexes as would usually be the case).
        end

        if l_a==1
            Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:),1);
        elseif l_a==2
            Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1),1);
        elseif l_a==3
            Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:)-1),1);
        elseif l_a==4
            Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy(l_d+4,:,:)-1),1);
        else
            error('EvalFnOnAgentDist_CorrTransProbs_InfHorz cannot handle length(n_a)>4, contact me if you need this')
        end

        % Constructing Gammatranspose is done differently if using grid interpolation layers
        if simoptions.gridinterplayer==0
            Policy_aprimezprime=reshape(Policy_aprime,[N_a*N_z,1])+N_a*gpuArray(0:1:N_z-1);  % Note: add z' index in a new dimension, so all possible zprime
            Policy_aprimezprime=gather(reshape(Policy_aprimezprime,[N_a*N_z,N_z])); % sparse() requires inputs to be 2-D
            % Precompute
            II2=repmat((1:1:N_a*N_z)',1,N_z); %  Index for this period (a,z)

            % P: full transition matrix on (a,z)
            P=sparse(II2,Policy_aprimezprime,repelem(pi_z,N_a,1),N_a*N_z,N_a*N_z); % Note: sparse() will accumulate at repeated indices

        elseif simoptions.gridinterplayer==1
            Policy_aprime(:,:,2)=Policy_aprime(:,:,1)+1; % upper index, add one to index for a1

            % First, get Gamma
            PolicyProbs=zeros(N_a,N_z,2,'gpuArray');% PolicyProbs are the corresponding probabilities of each of these two.

            PolicyProbs(:,:,2)=shiftdim((Policy(end,:,:)-1)/(simoptions.ngridinterp+1),1); % probability of upper grid point
            PolicyProbs(:,:,1)=1-PolicyProbs(:,:,2); % probability of lower grid point

            % Policy_aprime and PolicyProbs are currently [N_a,N_z,2]
            Policy_aprimezprime=reshape(Policy_aprime,[N_a*N_z*2,1])+N_a*gpuArray(0:1:N_z-1);  % Note: add z' index in a new dimension, so all possible zprime
            Policy_aprimezprime=gather(reshape(Policy_aprimezprime,[N_a*N_z,2*N_z])); % sparse() requires inputs to be 2-D
            PolicyProbs=gather(reshape(PolicyProbs,[N_a*N_z,2])); % sparse() requires inputs to be 2-D

            % Precompute
            II2=repmat((1:1:N_a*N_z)',1,2*N_z); %  Index for this period (a,z), note the N_probs-copies

            % P: full transition matrix on (a,z)
            P=sparse(II2,Policy_aprimezprime,repmat(PolicyProbs,1,N_z).*repelem(pi_z,N_a,2),N_a*N_z,N_a*N_z); % Note: sparse() will accumulate at repeated indices
        end
    else
        error('Not yet implemented')
    end
end









end