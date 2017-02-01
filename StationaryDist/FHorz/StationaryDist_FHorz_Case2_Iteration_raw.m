function StationaryDistKron=StationaryDist_FHorz_Case2_Iteration_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid,pi_z,Phi_aprime,Case2_Type,Parameters,PhiaprimeParamNames,simoptions)
%Will treat the agents as being on a continuum of mass 1.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

eval('fieldexists_ExogShockFn=1;simoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;simoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')


if simoptions.parallel~=2
    
    StationaryDistKron=zeros(N_a*N_z,N_j);
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    if simoptions.phiaprimedependsonage==0
        PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
        if simoptions.lowmemory==0
            Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
        end
    end    
    
    for jj=1:(N_j-1)
        if fieldexists_ExogShockFn==1
            if fieldexists_ExogShockFnParamNames==1
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                [~,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsVec);
            else
                [~,pi_z]=simoptions.ExogShockFn(jj);
            end
        end
        
        if simoptions.phiaprimedependsonage==1
            PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
            Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
        end
        
        %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
        P=zeros(N_a,N_z,N_a,N_z); %P(a,z,aprime,zprime)=proby of going to (a',z') given in (a,z)
        for a_c=1:N_a
            for z_c=1:N_z
                optd=PolicyKron(a_c,z_c,jj);
                if Case2_Type==1 % a'(d,a,z,z')
                    for zprime_c=1:N_z
                        optaprime=Phi_aprimeMatrix(optd,a_c,z_c,zprime_c);
                    end
                elseif Case2_Type==11 % a'(d,a,z')
                    for zprime_c=1:N_z
                        optaprime=Phi_aprimeMatrix(optd,a_c,zprime_c);
                    end
                elseif Case2_Type==12 % a'(d,a,z)
                    optaprime=Phi_aprimeMatrix(optd,a_c,z_c);
                elseif Case2_Type==2 % a'(d,z,z')
                    for zprime_c=1:N_z
                        optaprime=Phi_aprimeMatrix(optd,z_c,zprime_c);
                    end
                end
                for zprime_c=1:N_z
                    P(a_c,z_c,optaprime,zprime_c)=pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
                end
            end
        end
        P=reshape(P,[N_a*N_z,N_a*N_z]);
        P=P';
        
        StationaryDistKron(:,jj+1)=P*StationaryDistKron(:,jj);
    end
    
elseif simoptions.parallel==2 % Using the GPU
    
    StationaryDistKron=zeros(N_a*N_z,N_j,'gpuArray');
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
        if fieldexists_ExogShockFn==1
            if fieldexists_ExogShockFnParamNames==1
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                [~,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsVec);
                pi_z=gpuArray(pi_z);
            else
                [~,pi_z]=simoptions.ExogShockFn(jj);
                pi_z=gpuArray(pi_z);
            end
        end
        
        if simoptions.phiaprimedependsonage==1
            PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
            Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
        end
        
        %optdprime=reshape(PolicyKron,[1,N_a*N_z]);
        if Case2_Type==1 % phi(d,a,z,z')
            disp('ERROR: StationaryDist_FHorz_Case2_Iteration_raw() not yet implemented for Case2_Type==1')
            % Create P matrix
        elseif Case2_Type==11 % phi(d,a,z')
            disp('ERROR: StationaryDist_FHorz_Case2_Iteration_raw() not yet implemented for Case2_Type==11')
            % Create P matrix
        elseif Case2_Type==12 % phi(d,a,z)
            % Create P matrix
            % optaprime is here replaced by Phi_of_Policy, which is a different shape
            Phi_of_Policy=zeros(N_a,N_z,'gpuArray'); %a'(a,z)
            for z_c=1:N_z
                temp=[PolicyKron(:,z_c,jj),(1:N_a)',z_c*ones(N_a,1)];
                temp2=sub2ind([N_d,N_a,N_z],temp(:,1),temp(:,2),temp(:,3));
                Phi_of_Policy(:,z_c)=Phi_aprimeMatrix(temp2); % WORK IN PROGRESS HERE
            end
            Ptemp=zeros(N_a,N_a*N_z*N_z,'gpuArray');
            Ptemp(reshape(permute(Phi_of_Policy,[2,1,3]),[1,N_a*N_z*N_z])+N_a*(gpuArray(0:1:N_a*N_z*N_z-1)))=1;
            %        Ptemp(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=1;
            Ptran=kron(pi_z',ones(N_a,N_a,'gpuArray')).*reshape(Ptemp,[N_a*N_z,N_a*N_z]);            
        elseif Case2_Type==2  % phi(d,z',z)
            % optaprime is here replaced by Phi_of_Policy, which is a different shape
            Phi_of_Policy=zeros(N_a,N_z,N_z,'gpuArray'); %a'(d,z',z)
            for z_c=1:N_z
                Phi_of_Policy(:,:,z_c)=Phi_aprimeMatrix(PolicyKron(:,z_c,jj),:,z_c);
            end
            Ptemp=zeros(N_a,N_a*N_z*N_z,'gpuArray');
            Ptemp(reshape(permute(Phi_of_Policy,[2,1,3]),[1,N_a*N_z*N_z])+N_a*(gpuArray(0:1:N_a*N_z*N_z-1)))=1;
            %        Ptemp(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=1;
            Ptran=kron(pi_z',ones(N_a,N_a,'gpuArray')).*reshape(Ptemp,[N_a*N_z,N_a*N_z]);
        end
        
        StationaryDistKron(:,jj+1)=Ptran*StationaryDistKron(:,jj);
    end
end

% Reweight the different ages based on 'AgeWeightParamNames'. (it is
% assumed there is only one Age Weight Parameter (name))
FullParamNames=fieldnames(Parameters);
nFields=length(FullParamNames);
found=0;
for iField=1:nFields
    if strcmp(AgeWeightParamNames{1},FullParamNames{iField})
        AgeWeights=Parameters.(FullParamNames{iField});
        found=1;
    end
end
if found==0 % Have added this check so that user can see if they are missing a parameter
    fprintf(['FAILED TO FIND PARAMETER ',AgeWeightParamNames{1}])
end
% I assume AgeWeights is a row vector
StationaryDistKron=StationaryDistKron.*(ones(N_a*N_z,1)*AgeWeights);

end
