function StationaryDistKron=StationaryDist_FHorz_Case2_3_Iteration_e_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,n_d,n_a,n_z,n_e,N_j,d_grid, a_grid, z_grid,pi_z,pi_e,Phi_aprime,Case2_Type,Parameters,PhiaprimeParamNames,simoptions)
% Case2_Type=3: aprime(d,z')
% Will treat the agents as being on a continuum of mass 1.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

eval('fieldexists_ExogShockFn=1;simoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;simoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')
eval('fieldexists_pi_z_J=1;simoptions.pi_z_J;','fieldexists_pi_z_J=0;')

eval('fieldexists_EiidShockFn=1;simoptions.EiidShockFn;','fieldexists_EiidShockFn=0;')
eval('fieldexists_EiidShockFnParamNames=1;simoptions.EiidShockFnParamNames;','fieldexists_EiidShockFnParamNames=0;')
eval('fieldexists_pi_e_J=1;simoptions.pi_e_J;','fieldexists_pi_e_J=0;')


if simoptions.parallel<2
    
    StationaryDistKron=zeros(N_a*N_z*N_e,N_j);
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    if simoptions.phiaprimedependsonage==0
        PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
        if simoptions.lowmemory==0
            Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid, PhiaprimeParamsVec);
        end
    end    
    
    for jj=1:(N_j-1)
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end
        
        if fieldexists_pi_z_J==1
            z_grid=simoptions.z_grid_J(:,jj);
            pi_z=simoptions.pi_z_J(:,:,jj);
        elseif fieldexists_ExogShockFn==1
            if fieldexists_ExogShockFnParamNames==1
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [z_grid,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
            else
                [z_grid,pi_z]=simoptions.ExogShockFn(jj);
            end
        end
        
        if fieldexists_pi_e_J==1
            pi_e=simoptions.pi_e_J(:,jj);
        elseif fieldexists_EiidShockFn==1
            if fieldexists_EiidShockFnParamNames==1
                EiidShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.EiidShockFnParamNames,jj);
                EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                for ii=1:length(EiidShockFnParamsVec)
                    EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                end
                [~,pi_e]=simoptions.EiidShockFn(EiidShockFnParamsCell{:});
            else
                [~,pi_e]=simoptions.EiidShockFn(jj);
            end
        end
        
        if simoptions.phiaprimedependsonage==1
            PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
            Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
        end

        %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
        P=zeros(N_a,N_z,N_e,N_a,N_z); %P(a,z,e,aprime,zprime)=proby of going to (a',z') given in (a,z,e)
        for a_c=1:N_a
            for z_c=1:N_z
                for e_c=1:N_e
                    optd=PolicyKron(a_c,z_c,e_c,jj);
                    for zprime_c=1:N_z
                        optaprime=Phi_aprimeMatrix(optd,zprime_c); % Case2_Type==3; a'(d,z')
                        P(a_c,z_c,e_c,optaprime,zprime_c)=pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
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

    if simoptions.phiaprimedependsonage==0
        PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
        if simoptions.lowmemory==0
            Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
        end
    end
    
    % First, generate the transition matrix P=g of Q (the convolution of the
    % optimal policy function and the transition fn for exogenous shocks)
    for jj=1:(N_j-1)
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end

        if fieldexists_pi_z_J==1
            z_grid=simoptions.z_grid_J(:,jj);
            pi_z=simoptions.pi_z_J(:,:,jj);
        elseif fieldexists_ExogShockFn==1
            if fieldexists_ExogShockFnParamNames==1
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [z_grid,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
                z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
            else
                [z_grid,pi_z]=simoptions.ExogShockFn(jj);
                z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
            end
        end
        
        if fieldexists_pi_e_J==1
            pi_e=simoptions.pi_e_J(:,jj);
        elseif fieldexists_EiidShockFn==1
            if fieldexists_EiidShockFnParamNames==1
                EiidShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.EiidShockFnParamNames,jj);
                EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                for ii=1:length(EiidShockFnParamsVec)
                    EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                end
                [~,pi_e]=simoptions.EiidShockFn(EiidShockFnParamsCell{:});
            else
                [~,pi_e]=simoptions.EiidShockFn(jj);
            end
        end

        if simoptions.phiaprimedependsonage==1
            PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
            Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
        end
        
        optd=reshape(PolicyKron(:,:,:,jj),[1,N_a*N_z*N_e]);
        for zprime_c=1:N_z
            optaprime_jj=Phi_aprimeMatrix(optd,zprime_c); % Case2_Type==3; a'(d,z')
        end
        Ptran=zeros(N_a,N_a*N_z*N_e,'gpuArray'); % Start with P (a,z,e) to a' (Note, create P')
        Ptran(optaprime_jj+N_a*(gpuArray(0:1:N_a*N_z*N_e-1)))=1; % Fill in the a' transitions based on Policy
        
        % Now use pi_z to switch to P (a,z,e) to (a',z')
        Ptran=(kron(ones(1,N_e),kron(pi_z',ones(N_a,N_a,'gpuArray')))).*(kron(ones(N_z,1,'gpuArray'),Ptran));
        
        StationaryDistKron(:,jj+1)=kron(pi_e, Ptran*StationaryDistKron(:,jj) );
    end

elseif simoptions.parallel==3 % Sparse matrix instead of a standard matrix for P, on cpu
    
    
    StationaryDistKron=sparse(N_a*N_z*N_e,N_j);
    StationaryDistKron(:,1)=jequaloneDistKron;

    if simoptions.phiaprimedependsonage==0
        PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
        if simoptions.lowmemory==0
            Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
        end
    end
    
    % First, generate the transition matrix P=g of Q (the convolution of the
    % optimal policy function and the transition fn for exogenous shocks)
    for jj=1:(N_j-1)
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end

        if fieldexists_pi_z_J==1
            z_grid=simoptions.z_grid_J(:,jj);
            pi_z=simoptions.pi_z_J(:,:,jj);
        elseif fieldexists_ExogShockFn==1
            if fieldexists_ExogShockFnParamNames==1
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [z_grid,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
                z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
            else
                [z_grid,pi_z]=simoptions.ExogShockFn(jj);
                z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
            end
        end
        
        if fieldexists_pi_e_J==1
            pi_e=simoptions.pi_e_J(:,jj);
        elseif fieldexists_EiidShockFn==1
            if fieldexists_EiidShockFnParamNames==1
                EiidShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.EiidShockFnParamNames,jj);
                EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                for ii=1:length(EiidShockFnParamsVec)
                    EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                end
                [~,pi_e]=simoptions.EiidShockFn(EiidShockFnParamsCell{:});
            else
                [~,pi_e]=simoptions.EiidShockFn(jj);
            end
        end

        if simoptions.phiaprimedependsonage==1
            PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
            Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
        end
        
        %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
        optd=reshape(PolicyKron(:,:,:,jj),[1,N_a*N_z*N_e]);
        PtransposeA=sparse(N_a,N_a*N_z*N_e);  % Start with P (a,z,e) to a' (Note, create P')
        for zprime_c=1:N_z
            optaprime_jj=Phi_aprimeMatrix(optd,zprime_c); % Case2_Type==3; a'(d,z')
            PtransposeA(optaprime_jj+N_a*(0:1:N_a*N_z*N_e-1))=PtransposeA(optaprime_jj+N_a*(0:1:N_a*N_z*N_e-1))+1/N_z; % Fill in the a' transitions based on Policy
        end
        
        % Note: Create Ptranspose as (a,z,e)-to-(a',z') as best to use to
        % multiply lag of agent dist, and then just iid distribute over e
        % later in a 'seperate' step (is same line of code, but splits the
        % steps)
        pi_z=sparse(pi_z);
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
    
elseif simoptions.parallel==4 % Sparse matrix instead of a standard matrix for P, on cpu
    
    StationaryDistKron=sparse(N_a*N_z*N_e,N_j);
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    if simoptions.phiaprimedependsonage==0
        PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
        if simoptions.lowmemory==0
            Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
        end
    end
    
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end

        if fieldexists_pi_z_J==1
            pi_z=simoptions.pi_z_J(:,:,jj);
        elseif fieldexists_ExogShockFn==1
            if fieldexists_ExogShockFnParamNames==1
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [~,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
            else
                [~,pi_z]=simoptions.ExogShockFn(jj);
            end
        end
        if fieldexists_pi_e_J==1
            pi_e=simoptions.pi_e_J(:,jj);
        elseif fieldexists_EiidShockFn==1
            if fieldexists_EiidShockFnParamNames==1
                EiidShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.EiidShockFnParamNames,jj);
                EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                for ii=1:length(EiidShockFnParamsVec)
                    EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                end
                [~,pi_e]=simoptions.EiidShockFn(EiidShockFnParamsCell{:});
            else
                [~,pi_e]=simoptions.EiidShockFn(jj);
            end
        end
        
        if simoptions.phiaprimedependsonage==1
            PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
            Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
        end
        
        %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
        optd=reshape(PolicyKron(:,:,:,jj),[1,N_a*N_z*N_e]);
        PtransposeA=sparse(N_a,N_a*N_z*N_e);
        for zprime_c=1:N_z
            optaprime_jj=Phi_aprimeMatrix(optd,zprime_c); % Case2_Type==3; a'(d,z')
            PtransposeA(optaprime_jj+N_a*(0:1:N_a*N_z*N_e-1))=PtransposeA(optaprime_jj+N_a*(0:1:N_a*N_z*N_e-1))+1/N_z;
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
        
        try
            StationaryDistKron(:,jj+1)=kron(pi_e, Ptranspose*StationaryDistKron(:,jj));
        catch
            error('The transition matrix is big, please use simoptions.parallel=3 (instead of 4) \n')
        end
    end
    
    StationaryDistKron=full(StationaryDistKron);
    StationaryDistKron=gpuArray(StationaryDistKron);
    
elseif simoptions.parallel==5 % Same as 2, except loops over e (uses gpu arrays)
    
    StationaryDistKron=zeros(N_a*N_z,N_e,N_j,'gpuArray');
    StationaryDistKron(:,:,1)=reshape(jequaloneDistKron,[N_a*N_z,N_e]);
    
    PolicyKron=gpuArray(PolicyKron);
    
    if simoptions.phiaprimedependsonage==0
        PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
        if simoptions.lowmemory==0
            Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
        end
    end
    
    % First, generate the transition matrix P=g of Q (the convolution of the 
    % optimal policy function and the transition fn for exogenous shocks)
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end

        if fieldexists_pi_z_J==1
            pi_z=simoptions.pi_z_J(:,:,jj);
        elseif fieldexists_ExogShockFn==1
            if fieldexists_ExogShockFnParamNames==1
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [~,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
                pi_z=gpuArray(pi_z);
            else
                [~,pi_z]=simoptions.ExogShockFn(jj);
                pi_z=gpuArray(pi_z);
            end
        end
        if fieldexists_pi_e_J==1
            pi_e=simoptions.pi_e_J(:,jj);
        elseif fieldexists_EiidShockFn==1
            if fieldexists_EiidShockFnParamNames==1
                EiidShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.EiidShockFnParamNames,jj);
                EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                for ii=1:length(EiidShockFnParamsVec)
                    EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                end
                [~,pi_e]=simoptions.EiidShockFn(EiidShockFnParamsCell{:});
            else
                [~,pi_e]=simoptions.EiidShockFn(jj);
            end
        end
        
        if simoptions.phiaprimedependsonage==1
            PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
            Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
        end
        
        bigpi_zt=kron(gather(pi_z'),ones(N_a,N_a)); % Note already includes taking transpose of pi_z

        % Take advantage of fact that for each (a,z,e) we can map into (a',z'), looping over e
        StatDisttemp=zeros(N_a*N_z,1,'gpuArray');
        for e_c=1:N_e
            %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
            optd=reshape(PolicyKron(:,:,e_c,jj),[N_a*N_z,1]); % d(a,z)
            optaprime_jj=reshape(Phi_aprimeMatrix(optd+N_d*(0:1:N_z-1)),[N_a*N_z*N_z,1]); % a'(a,z,z')
            % We have index for aprime, with each element corresponding to a specific (a,z,a')
            % We want to convert this into a index for (a,z,a',z')

%             A=kron(ones(N_z*N_z,1),(1:1:N_a)'); % corresponding a
%             B=kron(kron(ones(N_z,1),N_a*((1:1:N_z)-1)'),ones(N_a,1)); % corresponding z
%             C=N_a*N_z*(optaprime_jj-1); % aprime in terms of (a,z,z')
%             D=kron(N_a*N_a*N_z*((1:1:N_z)-1)',ones(N_a*N_z,1)); % corresponding zprime
%             fullindex=A+B+C+D;
            
            % The above commented out "fullindex=A+B+C+D;" is an easier to
            % follow version of the next line (putting it in one line is marginally faster)
            fullindex=kron(ones(N_z*N_z,1,'gpuArray'),gpuArray(1:1:N_a)')+kron(kron(ones(N_z,1,'gpuArray'),N_a*(gpuArray(1:1:N_z)-1)'),ones(N_a,1))+N_a*N_z*(optaprime_jj-1)+kron(N_a*N_a*N_z*(gpuArray(1:1:N_z)-1)',ones(N_a*N_z,1,'gpuArray'));
            
            P=zeros(N_a*N_z,N_a*N_z,'gpuArray');
            P(fullindex)=1;

            StatDisttemp=StatDisttemp+(bigpi_zt.*gather(P'))*StationaryDistKron(:,e_c,jj);
        end
        % And now just distribute over the e'
        StationaryDistKron(:,:,jj+1)=StatDisttemp.*pi_e';
        
    end
    
elseif simoptions.parallel==6 % Same as 4, except loops over e (uses full cpu matrices)
    
    StationaryDistKron=zeros(N_a*N_z,N_e,N_j,'gpuArray');
    StationaryDistKron(:,:,1)=reshape(jequaloneDistKron,[N_a*N_z,N_e]);
    
    if simoptions.phiaprimedependsonage==0
        PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
        if simoptions.lowmemory==0
            Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
        end
    end
    
    % First, generate the transition matrix P=g of Q (the convolution of the 
    % optimal policy function and the transition fn for exogenous shocks)
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end

        if fieldexists_pi_z_J==1
            pi_z=simoptions.pi_z_J(:,:,jj);
        elseif fieldexists_ExogShockFn==1
            if fieldexists_ExogShockFnParamNames==1
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [~,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
                pi_z=gpuArray(pi_z);
            else
                [~,pi_z]=simoptions.ExogShockFn(jj);
                pi_z=gpuArray(pi_z);
            end
        end
        if fieldexists_pi_e_J==1
            pi_e=simoptions.pi_e_J(:,jj);
        elseif fieldexists_EiidShockFn==1
            if fieldexists_EiidShockFnParamNames==1
                EiidShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.EiidShockFnParamNames,jj);
                EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                for ii=1:length(EiidShockFnParamsVec)
                    EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                end
                [~,pi_e]=simoptions.EiidShockFn(EiidShockFnParamsCell{:});
            else
                [~,pi_e]=simoptions.EiidShockFn(jj);
            end
        end
        
        if simoptions.phiaprimedependsonage==1
            PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
            Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
        end
        
        bigpi_zt=kron(gather(pi_z'),ones(N_a,N_a)); % Note already includes taking transpose of pi_z

        % Take advantage of fact that for each (a,z,e) we can map into (a',z'), looping over e
        StatDisttemp=zeros(N_a*N_z,1);
        for e_c=1:N_e % Can I parfor over this??
            %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
            optd=reshape(PolicyKron(:,:,e_c,jj),[N_a*N_z,1]); % d(a,z)
            optaprime_jj=reshape(Phi_aprimeMatrix(optd+N_d*(0:1:N_z-1)),[N_a*N_z*N_z,1]); % a'(a,z,z')
            % We have index for aprime, with each element corresponding to a specific (a,z,a')
            % We want to convert this into a index for (a,z,a',z')

%             A=kron(ones(N_z*N_z,1),(1:1:N_a)'); % corresponding a
%             B=kron(kron(ones(N_z,1),N_a*((1:1:N_z)-1)'),ones(N_a,1)); % corresponding z
%             C=N_a*N_z*(optaprime_jj-1); % aprime in terms of (a,z,z')
%             D=kron(N_a*N_a*N_z*((1:1:N_z)-1)',ones(N_a*N_z,1)); % corresponding zprime
%             fullindex=A+B+C+D;
            
            % The above commented out "fullindex=A+B+C+D;" is an easier to
            % follow version of the next line (putting it in one line is marginally faster)
            fullindex=kron(ones(N_z*N_z,1),(1:1:N_a)')+kron(kron(ones(N_z,1),N_a*((1:1:N_z)-1)'),ones(N_a,1))+N_a*N_z*(optaprime_jj-1)+kron(N_a*N_a*N_z*((1:1:N_z)-1)',ones(N_a*N_z,1));

            P=zeros(N_a*N_z,N_a*N_z);
            P(fullindex)=1;
            
            StatDisttemp=StatDisttemp+(bigpi_zt.*(P'))*StationaryDistKron(:,e_c,jj);
        end
        % And now just distribute over the e'
        StationaryDistKron(:,:,jj+1)=StatDisttemp.*pi_e';
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

StationaryDistKron=StationaryDistKron.*shiftdim(AgeWeights,-1);

end
