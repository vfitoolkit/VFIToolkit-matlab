function StationaryDist=StationaryDist_FHorz_SemiExo(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_semiz,n_z,N_j,pi_semiz_J,pi_z_J,Parameters,simoptions)

%%
l_d=length(n_d);
l_d1=l_d-simoptions.l_dsemiz;
l_d2=simoptions.l_dsemiz; % decision variables that determine the semi-exo state

%%
l_a=length(n_a);

N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
if isfield(simoptions,'n_e')
    N_e=prod(simoptions.n_e);
else
    N_e=0;
end

%%
if N_z==0
    if N_e==0
        n_bothze=simoptions.n_semiz;
        N_bothze=N_semiz;
    else
        n_bothze=[simoptions.n_semiz,simoptions.n_e];
        N_bothze=N_semiz*N_e;
    end
else
    if N_e==0
        n_bothze=[simoptions.n_semiz,n_z];
        N_bothze=N_semiz*N_z;
    else
        n_bothze=[simoptions.n_semiz,n_z,simoptions.n_e];
        N_bothze=N_semiz*N_z*N_e;
    end
end

jequaloneDist=gpuArray(jequaloneDist); % make sure it is on gpu
jequaloneDist=reshape(jequaloneDist,[N_a*N_bothze,1]);
Policy=reshape(Policy,[size(Policy,1),N_a,N_bothze,N_j]);

%% Policy_aprime
% Policy_aprime=zeros(N_a,N_bothze,N_j,'gpuArray'); % the lower grid point
if l_a==1 % one endo state
    Policy_aprime=Policy(l_d+1,:,:,:);
elseif l_a==2 % two endo states
    Policy_aprime=Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1);
elseif l_a==3 % three endo states
    Policy_aprime=Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:,:)-1);
elseif l_a==4 % four endo states
    Policy_aprime=Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy(l_d+4,:,:,:)-1);
else
    error('Not yet implemented standard endogenous states with length(n_a)>4')
end
Policy_aprime=shiftdim(Policy_aprime,1); % [N_a,N_bothze,N_j]


%% Policy_dsemiexo

% d2 is the variable relevant for the semi-exogenous asset. 
if l_d2==1
    Policy_dsemiexo=Policy(l_d1+1,:,:,:);
elseif l_d2==2
    Policy_dsemiexo=Policy(l_d1+1,:,:,:)+n_d(l_d1+1)*(Policy(l_d1+2,:,:,:)-1);
elseif l_d2==3
    Policy_dsemiexo=Policy(l_d1+1,:,:,:)+n_d(l_d1+1)*(Policy(l_d1+2,:,:,:)-1)+n_d(l_d1+1)*n_d(l_d1+2)*(Policy(l_d1+3,:,:,:)-1); 
elseif l_d2==4
    Policy_dsemiexo=Policy(l_d1+1,:,:,:)+n_d(l_d1+1)*(Policy(l_d1+2,:,:,:)-1)+n_d(l_d1+1)*n_d(l_d1+2)*(Policy(l_d1+3,:,:,:)-1)+n_d(l_d1+1)*n_d(l_d1+2)*n_d(l_d1+3)*(Policy(l_d1+4,:,:,:)-1);
end
Policy_dsemiexo=shiftdim(Policy_dsemiexo,1); % [N_a,N_bothze,N_j]


%%
if simoptions.gridinterplayer==0
    if N_z==0 && N_e==0
        StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_noz_raw(jequaloneDist,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,N_a,N_semiz,N_j,pi_semiz_J,Parameters);
    elseif N_e==0 % just z
        StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_raw(jequaloneDist,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,N_a,N_semiz,N_z,N_j,pi_semiz_J,pi_z_J,Parameters);
    elseif N_z==0 % just e
        StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_noz_e_raw(jequaloneDist,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,N_a,N_semiz,N_e,N_j,pi_semiz_J,simoptions.pi_e_J,Parameters);
    else % both z and e
        StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_e_raw(jequaloneDist,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,N_a,N_semiz,N_z,N_e,N_j,pi_semiz_J,pi_z_J,simoptions.pi_e_J,Parameters);
    end
elseif simoptions.gridinterplayer==1
    % (a,z,1,j)
    Policy_aprime=reshape(Policy_aprime,[N_a,N_bothze,1,N_j]);
    Policy_aprime=repmat(Policy_aprime,1,1,2,1);
    PolicyProbs=ones([N_a,N_bothze,2,N_j],'gpuArray');
    % Policy_aprime(:,:,1,:) lower grid point for a1 is unchanged 
    Policy_aprime(:,:,2,:)=Policy_aprime(:,:,2,:)+1; % add one to a1, to get upper grid point

    aprimeProbs_upper=reshape(shiftdim((Policy(end,:,:,:)-1)/(simoptions.ngridinterp+1),1),[N_a,N_bothze,1,N_j]); % probability of upper grid point (from L2 index)
    PolicyProbs(:,:,1,:)=PolicyProbs(:,:,1,:).*(1-aprimeProbs_upper); % lower a1
    PolicyProbs(:,:,2,:)=PolicyProbs(:,:,2,:).*aprimeProbs_upper; % upper a1

    if N_z==0 && N_e==0
        StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_nProbs_noz_raw(jequaloneDist,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,2,N_a,N_semiz,N_j,pi_semiz_J,Parameters);    
    elseif N_e==0 % just z
        StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_nProbs_raw(jequaloneDist,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,2,N_a,N_semiz,N_z,N_j,pi_semiz_J,pi_z_J,Parameters);
    elseif N_z==0 % just e
        StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_nProbs_noz_e_raw(jequaloneDist,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,2,N_a,N_semiz,N_e,N_j,pi_semiz_J,simoptions.pi_e_J,Parameters);
    else % both z and e
        StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_nProbs_e_raw(jequaloneDist,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,2,N_a,N_semiz,N_z,N_e,N_j,pi_semiz_J,pi_z_J,simoptions.pi_e_J,Parameters);
    end
end



if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,n_bothze,N_j]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDist,[N_a,N_bothze,N_j]);
end

end
