function StationaryDistKron=StationaryDist_FHorz_Case1_RiskyAsset_Iteration_e_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,n_d,n_a,n_z,n_e,n_u,N_j,d_grid,a_grid,u_grid,pi_z_J,pi_e_J,pi_u,aprimeFn,Parameters,aprimeFnParamNames,simoptions)
% Case3: aprime(d,u)
% Will treat the agents as being on a continuum of mass 1.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);
N_u=prod(n_u);

% Parallel<2, just loop
% 2: gpu
% 3: sparse matrix on cpu
% 4: sparse matrix on gpu
% 5: loop over e on cpu
% 6: loop over e on gpu

if simoptions.parallel<2
    
    StationaryDistKron=zeros(N_a*N_z*N_e,N_j);
    StationaryDistKron(:,1)=jequaloneDistKron;
        
    for jj=1:(N_j-1)
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end

        pi_z=pi_z_J(:,:,jj);
        pi_e=pi_e_J(:,jj);
        
        aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
        [aprimeIndex,aprimeProbs]=CreateaprimeFnMatrix_Case3(aprimeFn, n_d, n_a, n_u, d_grid, a_grid, u_grid, aprimeFnParamsVec,0); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)

        %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
        P=zeros(N_a,N_z,N_e,N_a,N_z); %P(a,z,e,aprime,zprime)=proby of going to (a',z') given in (a,z,e)
        for a_c=1:N_a
            for z_c=1:N_z
                for e_c=1:N_e
                    optd=PolicyKron(a_c,z_c,e_c,jj);
                    for zprime_c=1:N_z
                        for u_c=1:N_u
                            optaprime=aprimeIndex(optd+N_d*(u_c-1)); % Case3; a'(d,u)
                            % Add to lower grid point
                            P(a_c,z_c,e_c,optaprime,zprime_c)=P(a_c,z_c,e_c,optaprime,zprime_c)+aprimeProbs(u_c)*pi_u(u_c)*pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
                            % Add to upper grid point
                            P(a_c,z_c,e_c,optaprime,zprime_c)=P(a_c,z_c,e_c,optaprime,zprime_c)+(1-aprimeProbs(u_c))*pi_u(u_c)*pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
                        end
                    end
                end
            end
        end
        P=reshape(P,[N_a*N_z*N_e,N_a*N_z]);
        P=P';
        
        StationaryDistKron(:,jj+1)=kron(pi_e,(P*StationaryDistKron(:,jj)));
    end
    
elseif simoptions.parallel==2 % Using the GPU

    StationaryDistKron=zeros(N_a*N_z*N_e,N_j,'gpuArray');
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    % First, generate the transition matrix P=g of Q (the convolution of the
    % optimal policy function and the transition fn for exogenous shocks)
    for jj=1:(N_j-1)
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end

        pi_z=pi_z_J(:,:,jj);
        pi_e=pi_e_J(:,jj);
        
        aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
        [aprimeIndex,aprimeProbs]=CreateaprimeFnMatrix_Case3(aprimeFn, n_d, n_a, n_u, d_grid, a_grid, u_grid, aprimeFnParamsVec,0); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
        
        Ptran=zeros(N_a,N_a*N_z*N_e,'gpuArray'); % Start with P (a,z,e) to a' (Note, create P')
        
        optd=reshape(PolicyKron(:,:,:,jj),[1,N_a*N_z*N_e]);
        for u_c=1:N_u
            optaprime_jj=aprimeIndex(optd,u_c)'; % Case3; a'(d,u)
            % Lower grid point
            Ptran(optaprime_jj+N_a*(gpuArray(0:1:N_a*N_z*N_e-1)))=Ptran(optaprime_jj+N_a*(gpuArray(0:1:N_a*N_z*N_e-1)))+aprimeProbs(u_c)*pi_u(u_c)*ones(1,N_a*N_z*N_e); % Fill in the a' transitions based on Policy
            % Upper grid point
            Ptran(optaprime_jj+1+N_a*(gpuArray(0:1:N_a*N_z*N_e-1)))=Ptran(optaprime_jj+1+N_a*(gpuArray(0:1:N_a*N_z*N_e-1)))+(1-aprimeProbs(u_c))*pi_u(u_c)*ones(1,N_a*N_z*N_e); % Fill in the a' transitions based on Policy
        end
        
        % Now use pi_z to switch to P (a,z,e) to (a',z')
        Ptran=(kron(ones(1,N_e),kron(pi_z',ones(N_a,N_a,'gpuArray')))).*(kron(ones(N_z,1,'gpuArray'),Ptran));
        
        StationaryDistKron(:,jj+1)=kron(pi_e, Ptran*StationaryDistKron(:,jj) );
    end

elseif simoptions.parallel==3 % Sparse matrix instead of a standard matrix for P, on cpu
    StationaryDistKron=sparse(N_a*N_z*N_e,N_j);
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    % First, generate the transition matrix P=g of Q (the convolution of the
    % optimal policy function and the transition fn for exogenous shocks)
    for jj=1:(N_j-1)
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end
        
        pi_z=sparse(gather(pi_z_J(:,:,jj)));
        pi_e=sparse(gather(pi_e_J(:,jj)));
        
        aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
        [aprimeIndex,aprimeProbs]=CreateaprimeFnMatrix_Case3(aprimeFn, n_d, n_a, n_u, d_grid, a_grid, u_grid, aprimeFnParamsVec,0); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
        
        %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
        PtransposeA=sparse(N_a,N_a*N_z*N_e);  % Start with P (a,z,e) to a' (Note, create P')
        optd=reshape(PolicyKron(:,:,:,jj),[1,N_a*N_z*N_e]);
        for u_c=1:N_u
            optaprime_jj=aprimeIndex(optd,u_c)'; % Case3; a'(d,u)
            % Lower grid point
            PtransposeA(optaprime_jj+N_a*(gpuArray(0:1:N_a*N_z*N_e-1)))=PtransposeA(optaprime_jj+N_a*(gpuArray(0:1:N_a*N_z*N_e-1)))+aprimeProbs(u_c)*pi_u(u_c)*ones(1,N_a*N_z*N_e); % Fill in the a' transitions based on Policy
            % Upper grid point
            PtransposeA(optaprime_jj+1+N_a*(gpuArray(0:1:N_a*N_z*N_e-1)))=PtransposeA(optaprime_jj+1+N_a*(gpuArray(0:1:N_a*N_z*N_e-1)))+(1-aprimeProbs(u_c))*pi_u(u_c)*ones(1,N_a*N_z*N_e); % Fill in the a' transitions based on Policy
        end
        
        % Note: Create Ptranspose as (a,z,e)-to-(a',z') as best to use to
        % multiply lag of agent dist, and then just iid distribute over e
        % later in a 'seperate' step (is same line of code, but splits the
        % steps)
        try % Following formula only works if pi_z is already sparse, otherwise kron(pi_z',ones(N_a,N_a)) is not sparse.
            Ptranspose=kron(ones(1,N_e),kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),PtransposeA);
        catch % Otherwise do something slower but which is sparse regardless of whether pi_z is sparse
            pi_z=gather(pi_z); % The indexing used can only be done on cpu
            Ptranspose=sparse(N_a*N_z,N_a*N_z*N_e);
            aaa=(kron(ones(1,N_z),(1:1:N_a))+N_a*kron(((1:1:N_z)-1),ones(1,N_a)));
            for ii=1:N_e
                Ptranspose(:,aaa+(ii-1)*N_a*N_z)=kron(ones(N_z,1),PtransposeA(:,aaa+(ii-1)*N_a*N_z)).*kron(pi_z',ones(N_a,N_a));
            end
        end
                
        StationaryDistKron(:,jj+1)=kron(pi_e, Ptranspose*StationaryDistKron(:,jj));
    end
    
    StationaryDistKron=full(StationaryDistKron); % Why do I do this? Why not just leave it sparse?
    % Move the solution to the gpu
    StationaryDistKron=gpuArray(StationaryDistKron);
    
elseif simoptions.parallel==4 % Sparse matrix instead of a standard matrix for P, on gpu
    % NOTE: Because the use of indexing it is not really possible to do sparse gpu matrices. So following is not very useful.
    
    StationaryDistKron=sparse(N_a*N_z*N_e,N_j);
    StationaryDistKron(:,1)=jequaloneDistKron;
        
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end

        pi_z=pi_z_J(:,:,jj);
        pi_e=pi_e_J(:,jj);
        
        aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
        [aprimeIndex,aprimeProbs]=CreateaprimeFnMatrix_Case3(aprimeFn, n_d, n_a, n_u, d_grid, a_grid, u_grid, aprimeFnParamsVec,0); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
        
        %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
        PtransposeA=sparse(N_a,N_a*N_z*N_e);
        optd=reshape(PolicyKron(:,:,:,jj),[1,N_a*N_z*N_e]);
        for u_c=1:N_u
            optaprime_jj=aprimeIndex(optd,u_c)'; % Case3; a'(d,u)
            % Lower grid point
            PtransposeA(optaprime_jj+N_a*(gpuArray(0:1:N_a*N_z*N_e-1)))=PtransposeA(optaprime_jj+N_a*(gpuArray(0:1:N_a*N_z*N_e-1)))+aprimeProbs(u_c)*pi_u(u_c)*ones(1,N_a*N_z*N_e); % Fill in the a' transitions based on Policy
            % Upper grid point
            PtransposeA(optaprime_jj+1+N_a*(gpuArray(0:1:N_a*N_z*N_e-1)))=PtransposeA(optaprime_jj+1+N_a*(gpuArray(0:1:N_a*N_z*N_e-1)))+(1-aprimeProbs(u_c))*pi_u(u_c)*ones(1,N_a*N_z*N_e); % Fill in the a' transitions based on Policy
        end

        pi_z=sparse(pi_z);
        try % Following formula only works if pi_z is already sparse, otherwise kron(pi_z',ones(N_a,N_a)) is not sparse.
            Ptranspose=kron(ones(1,N_e),kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),PtransposeA);
        catch % Otherwise do something slower but which is sparse regardless of whether pi_z is sparse
            pi_z=gather(pi_z); % The indexing used can only be donoe on cpu
            Ptranspose=sparse(N_a*N_z,N_a*N_z*N_e);
            aaa=(kron(ones(1,N_z),(1:1:N_a))+N_a*kron(((1:1:N_z)-1),ones(1,N_a)));
            for ii=1:N_e
                Ptranspose(:,aaa+(ii-1)*N_a*N_z)=kron(ones(N_z,1),PtransposeA(:,aaa+(ii-1)*N_a*N_z)).*kron(pi_z',ones(N_a,N_a));
            end
        end
        
        Ptranspose=gpuArray(Ptranspose);
        pi_z=gpuArray(pi_z);
        pi_e=sparse(gpuArray(pi_e));
        
        try
            StationaryDistKron(:,jj+1)=kron(pi_e, Ptranspose*StationaryDistKron(:,jj));
        catch
            error('The transition matrix is big, please use simoptions.parallel=3 (instead of 4) \n')
        end
    end
    
    StationaryDistKron=full(StationaryDistKron);
    StationaryDistKron=gpuArray(StationaryDistKron);
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


end
