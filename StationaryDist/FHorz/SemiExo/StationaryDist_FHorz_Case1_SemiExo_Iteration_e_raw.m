function StationaryDistKron=StationaryDist_FHorz_Case1_SemiExo_Iteration_e_raw(jequaloneDistKron,AgeWeightParamNames,PolicyIndexesKron,n_d1,n_d2,N_a,N_z,N_semiz,N_e,N_j,pi_z,pi_semiz_J,pi_e,Parameters,simoptions)
%Will treat the agents as being on a continuum of mass 1.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel

eval('fieldexists_ExogShockFn=1;simoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;simoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')
eval('fieldexists_pi_z_J=1;simoptions.pi_z_J;','fieldexists_pi_z_J=0;')

eval('fieldexists_EiidShockFn=1;simoptions.EiidShockFn;','fieldexists_EiidShockFn=0;')
eval('fieldexists_EiidShockFnParamNames=1;simoptions.EiidShockFnParamNames;','fieldexists_EiidShockFnParamNames=0;')
eval('fieldexists_pi_e_J=1;simoptions.pi_e_J;','fieldexists_pi_e_J=0;')

N_bothz=N_z*N_semiz;

if simoptions.parallel<2
    
    StationaryDistKron=zeros(N_a*N_bothz*N_e,N_j);
    StationaryDistKron(:,1)=jequaloneDistKron;
    
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

        pi_semiz_jj=pi_semiz_J(:,:,:,jj);
           
        % z transitions based on semiz
        optdprime=reshape(PolicyIndexesKron(1,:,:,:,jj),[N_a*N_bothz*N_e,1]); % Note has to be column vector to use in next line
        dsub=ind2sub_vec_homemade([n_d1,n_d2],optdprime);
        d2_c=dsub(:,end); % This is the decision variable that is determining the transition probabilities for the semi-exogenous state
        % Get the right part of pi_semiz_J % d2 depends on (a,z,semiz,e), and
        % pi_semiz is going to be about (semiz,semiz'), so I need to put it all
        % together as (a,z,semiz,e,semiz').
        semizindexcorrespondingtod2_c=kron(kron(ones(N_e,1),(1:1:N_semiz)'),ones(N_a*N_z,1));
        fullindex=semizindexcorrespondingtod2_c+N_semiz*((1:1:N_semiz)-1)+(N_semiz*N_semiz)*(d2_c-1);
        semiztransitions=pi_semiz_jj(fullindex); % (a,z,semiz,semiz')

        optaprime=reshape(PolicyIndexesKron(2,:,:,:,jj),[1,N_a*N_bothz*N_e]);
        Ptranspose=zeros(N_a,N_a*N_bothz*N_e);  % Start with P (a,z,semiz,e) to a' (Note, create P')
        Ptranspose(optaprime+N_a*(gpuArray(0:1:N_a*N_bothz*N_e-1)))=1;
        Ptranspose=kron(kron(ones(N_semiz,N_semiz*N_e),pi_z'),ones(N_a,N_a)).*kron(semiztransitions',ones(N_a*N_z*N_e,1)).*kron(ones(N_bothz,1),Ptranspose);  % Now use pi_z and pi_semiz to switch to P (a,z,semiz,e) to (a',z',semiz')
        
        StationaryDistKron(:,jj+1)=kron(pi_e,(Ptranspose*StationaryDistKron(:,jj)));
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

        pi_semiz_jj=pi_semiz_J(:,:,:,jj);

        % z transitions based on semiz
        optdprime=reshape(PolicyIndexesKron(1,:,:,:,jj),[N_a*N_bothz*N_e,1]); % Note has to be column vector to use in next line
        dsub=ind2sub_vec_homemade([n_d1,n_d2],optdprime);
        d2_c=dsub(:,end); % This is the decision variable that is determining the transition probabilities for the semi-exogenous state
        % Get the right part of pi_semiz_J % d2 depends on (a,z,semiz,e), and
        % pi_semiz is going to be about (semiz,semiz'), so I need to put it all
        % together as (a,z,semiz,e,semiz').
        semizindexcorrespondingtod2_c=kron(kron(ones(N_e,1,'gpuArray'),(1:1:N_semiz)'),ones(N_a*N_z,1,'gpuArray'));
        fullindex=semizindexcorrespondingtod2_c+N_semiz*((1:1:N_semiz)-1)+(N_semiz*N_semiz)*(d2_c-1);
        semiztransitions=pi_semiz_jj(fullindex); % (a,z,semiz,semiz')

        optaprime=reshape(PolicyIndexesKron(2,:,:,:,jj),[1,N_a*N_bothz*N_e]);
        Ptranspose=zeros(N_a,N_a*N_bothz*N_e,'gpuArray');  % Start with P (a,z,semiz,e) to a' (Note, create P')
        Ptranspose(optaprime+N_a*(gpuArray(0:1:N_a*N_bothz*N_e-1)))=1;
        Ptranspose=kron(kron(ones(N_semiz,N_semiz*N_e,'gpuArray'),pi_z'),ones(N_a,N_a,'gpuArray')).*kron(semiztransitions',ones(N_a*N_z*N_e,1,'gpuArray')).*kron(ones(N_bothz,1,'gpuArray'),Ptranspose);  % Now use pi_z and pi_semiz to switch to P (a,z,semiz,e) to (a',z',semiz')

        StationaryDistKron(:,jj+1)=kron(pi_e, Ptranspose*StationaryDistKron(:,jj) );
    end
    
elseif simoptions.parallel==3 % Sparse matrix instead of a standard matrix for P, on cpu
    
    StationaryDistKron=sparse(N_a*N_z*N_e,N_j);
    StationaryDistKron(:,1)=jequaloneDistKron;
    
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

        pi_semiz_jj=pi_semiz_J(:,:,:,jj);


        % z transitions based on semiz
        optdprime=reshape(PolicyIndexesKron(1,:,:,:,jj),[N_a*N_bothz*N_e,1]); % Note has to be column vector to use in next line
        dsub=ind2sub_vec_homemade([n_d1,n_d2],optdprime);
        d2_c=dsub(:,end); % This is the decision variable that is determining the transition probabilities for the semi-exogenous state
        % Get the right part of pi_semiz_J % d2 depends on (a,z,semiz,e), and
        % pi_semiz is going to be about (semiz,semiz'), so I need to put it all
        % together as (a,z,semiz,e,semiz').
        semizindexcorrespondingtod2_c=kron(kron(ones(N_e,1),(1:1:N_semiz)'),ones(N_a*N_z,1));
        fullindex=semizindexcorrespondingtod2_c+N_semiz*((1:1:N_semiz)-1)+(N_semiz*N_semiz)*(d2_c-1);
        semiztransitions=pi_semiz_jj(fullindex); % (a,z,semiz,semiz')

        optaprime=reshape(PolicyIndexesKron(2,:,:,:,jj),[1,N_a*N_bothz*N_e]);
        Ptranspose=zeros(N_a,N_a*N_bothz*N_e);  % Start with P (a,z,semiz,e) to a' (Note, create P')
        Ptranspose(optaprime+N_a*(gpuArray(0:1:N_a*N_bothz*N_e-1)))=1;
        Ptranspose=kron(kron(ones(N_semiz,N_semiz*N_e),sparse(pi_z)'),ones(N_a,N_a)).*kron(sparse(semiztransitions'),ones(N_a*N_z*N_e,1)).*kron(ones(N_bothz,1),Ptranspose);  % Now use pi_z and pi_semiz to switch to P (a,z,semiz,e) to (a',z',semiz')

        StationaryDistKron(:,jj+1)=kron(pi_e, Ptranspose*StationaryDistKron(:,jj) );
    end
    
    StationaryDistKron=full(StationaryDistKron); % Why do I do this? Why not just leave it sparse?
    % Move the solution to the gpu
    StationaryDistKron=gpuArray(StationaryDistKron);
    
elseif simoptions.parallel==4 % Sparse matrix instead of a standard matrix for P, on cpu
    
    StationaryDistKron=sparse(N_a*N_bothz*N_e,N_j);
    StationaryDistKron(:,1)=jequaloneDistKron;
    
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

        pi_semiz_jj=pi_semiz_J(:,:,:,jj);
        
        % z transitions based on semiz
        optdprime=reshape(PolicyIndexesKron(1,:,:,:,jj),[N_a*N_bothz*N_e,1]); % Note has to be column vector to use in next line
        dsub=ind2sub_vec_homemade([n_d1,n_d2],optdprime);
        d2_c=dsub(:,end); % This is the decision variable that is determining the transition probabilities for the semi-exogenous state
        % Get the right part of pi_semiz_J % d2 depends on (a,z,semiz,e), and
        % pi_semiz is going to be about (semiz,semiz'), so I need to put it all
        % together as (a,z,semiz,e,semiz').
        semizindexcorrespondingtod2_c=kron(kron(ones(N_e,1),(1:1:N_semiz)'),ones(N_a*N_z,1));
        fullindex=semizindexcorrespondingtod2_c+N_semiz*((1:1:N_semiz)-1)+(N_semiz*N_semiz)*(d2_c-1);
        semiztransitions=pi_semiz_jj(fullindex); % (a,z,semiz,semiz')

        optaprime=reshape(PolicyIndexesKron(2,:,:,:,jj),[1,N_a*N_bothz*N_e]);
        Ptranspose=zeros(N_a,N_a*N_bothz*N_e);  % Start with P (a,z,semiz,e) to a' (Note, create P')
        Ptranspose(optaprime+N_a*(gpuArray(0:1:N_a*N_bothz*N_e-1)))=1;
        Ptranspose=kron(kron(ones(N_semiz,N_semiz*N_e),sparse(pi_z)'),ones(N_a,N_a)).*kron(sparse(semiztransitions'),ones(N_a*N_z*N_e,1)).*kron(ones(N_bothz,1),Ptranspose);  % Now use pi_z and pi_semiz to switch to P (a,z,semiz,e) to (a',z',semiz')

        StationaryDistKron(:,jj+1)=kron(pi_e, Ptranspose*StationaryDistKron(:,jj) );
    end
    StationaryDistKron=full(StationaryDistKron);
    
    if simoptions.parallel==4 % Move solution to gpu
        StationaryDistKron=gpuArray(StationaryDistKron);
    end
    
elseif simoptions.parallel==5 % Same as 2, except loops over e
    
    StationaryDistKron=zeros(N_a*N_bothz,N_e,N_j,'gpuArray');
    StationaryDistKron(:,:,1)=reshape(jequaloneDistKron,[N_a*N_bothz,N_e]);
    
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

        pi_semiz_jj=pi_semiz_J(:,:,:,jj);
        
        % Take advantage of fact that for each (a,z,e) we can map into
        % (a',z'), looping over e
        StatDisttemp=zeros(N_a*N_z,1,'gpuArray');
        for e_c=1:N_e
            % z transitions based on semiz
            optdprime=reshape(PolicyIndexesKron(1,:,:,e_c,jj),[N_a*N_bothz,1]); % Note has to be column vector to use in next line
            dsub=ind2sub_vec_homemade([n_d1,n_d2],optdprime);
            d2_c=dsub(:,end); % This is the decision variable that is determining the transition probabilities for the semi-exogenous state
            % Get the right part of pi_semiz_J % d2 depends on (a,z,semiz,e), and
            % pi_semiz is going to be about (semiz,semiz'), so I need to put it all
            % together as (a,z,semiz,e,semiz').
            semizindexcorrespondingtod2_c=kron((1:1:N_semiz)',ones(N_a*N_z,1,'gpuArray'));
            fullindex=semizindexcorrespondingtod2_c+N_semiz*((1:1:N_semiz)-1)+(N_semiz*N_semiz)*(d2_c-1);
            semiztransitions=pi_semiz_jj(fullindex); % (a,z,semiz,semiz')

            optaprime=reshape(PolicyIndexesKron(2,:,:,e_c,jj),[1,N_a*N_bothz]);
            Ptranspose=zeros(N_a,N_a*N_bothz,'gpuArray');  % Start with P (a,z,semiz,e) to a' (Note, create P')
            Ptranspose(optaprime+N_a*(gpuArray(0:1:N_a*N_bothz-1)))=1;
            Ptranspose=kron(kron(ones(N_semiz,N_semiz,'gpuArray'),pi_z'),ones(N_a,N_a,'gpuArray')).*kron(semiztransitions',ones(N_a*N_z,1,'gpuArray')).*kron(ones(N_bothz,1,'gpuArray'),Ptranspose);  % Now use pi_z and pi_semiz to switch to P (a,z,semiz,e) to (a',z',semiz')
            
            StatDisttemp=StatDisttemp+Ptranspose*StationaryDistKron(:,e_c,jj);
        end
        % And now just distribute over the e'
        StationaryDistKron(:,:,jj+1)=StatDisttemp.*pi_e';

    end
    
elseif simoptions.parallel==6 % Same as 4, except loops over e
    
    StationaryDistKron=zeros(N_a*N_z,N_e,N_j,'gpuArray');
    StationaryDistKron(:,:,1)=reshape(jequaloneDistKron,[N_a*N_z,N_e]);
    
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

        pi_semiz_jj=pi_semiz_J(:,:,:,jj);
        
        % Take advantage of fact that for each (a,z,e) we can map into (a',z'), looping over e
        StatDisttemp=sparse(N_a*N_bothz,1);
        for e_c=1:N_e
            % z transitions based on semiz
            optdprime=reshape(PolicyIndexesKron(1,:,:,e_c,jj),[N_a*N_bothz,1]); % Note has to be column vector to use in next line
            dsub=ind2sub_vec_homemade([n_d1,n_d2],optdprime);
            d2_c=dsub(:,end); % This is the decision variable that is determining the transition probabilities for the semi-exogenous state
            % Get the right part of pi_semiz_J % d2 depends on (a,z,semiz,e), and
            % pi_semiz is going to be about (semiz,semiz'), so I need to put it all
            % together as (a,z,semiz,e,semiz').
            semizindexcorrespondingtod2_c=kron((1:1:N_semiz)',ones(N_a*N_z,1));
            fullindex=semizindexcorrespondingtod2_c+N_semiz*((1:1:N_semiz)-1)+(N_semiz*N_semiz)*(d2_c-1);
            semiztransitions=pi_semiz_jj(fullindex); % (a,z,semiz,semiz')

            optaprime=reshape(PolicyIndexesKron(2,:,:,e_c,jj),[1,N_a*N_bothz]);
            Ptranspose=zeros(N_a,N_a*N_bothz);  % Start with P (a,z,semiz,e) to a' (Note, create P')
            Ptranspose(optaprime+N_a*(gpuArray(0:1:N_a*N_bothz-1)))=1;
            Ptranspose=kron(kron(ones(N_semiz,N_semiz),pi_z'),ones(N_a,N_a)).*kron(semiztransitions',ones(N_a*N_z,1)).*kron(ones(N_bothz,1),Ptranspose);  % Now use pi_z and pi_semiz to switch to P (a,z,semiz,e) to (a',z',semiz')
            
            StatDisttemp=StatDisttemp+Ptranspose*StationaryDistKron(:,e_c,jj);
        end
        % And now just distribute over the e'
        StationaryDistKron(:,:,jj+1)=full(gpuArray(StatDisttemp.*pi_e'));

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

if simoptions.parallel==5 || simoptions.parallel==6 
    StationaryDistKron=StationaryDistKron.*shiftdim(AgeWeights,-1); %.*repmat(shiftdim(AgeWeights,-1),N_a*N_z,N_e,1);
else
    StationaryDistKron=StationaryDistKron.*AgeWeights;
end


end
