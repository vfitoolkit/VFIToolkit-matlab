function StationaryDist=StationaryDist_FHorz_Case1_SemiExo(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions)

if n_z(1)==0
    error('Not yet implemented n_z=0 with SemiExo, email me and I will do it (or you can just pretend by using n_z=1 and pi_z=1, not using the value of z anywhere)')
end

%% Setup related to semi-exogenous state (an exogenous state whose transition probabilities depend on a decision variable)
if ~isfield(simoptions,'n_semiz')
    error('When using simoptions.SemiExoShockFn you must declare simoptions.n_semiz')
end
if ~isfield(simoptions,'semiz_grid')
    error('When using simoptions.SemiExoShockFn you must declare simoptions.semiz_grid')
end
n_d1=n_d(1:end-1);
n_d2=n_d(end); % n_d2 is the decision variable that influences the transition probabilities of the semi-exogenous state
% d1_grid=simoptions.d_grid(1:sum(n_d1));
d2_grid=gpuArray(simoptions.d_grid(sum(n_d1)+1:end));
% Create the transition matrix in terms of (d,zprime,z) for the semi-exogenous states for each age
l_semiz=length(simoptions.n_semiz);
temp=getAnonymousFnInputNames(simoptions.SemiExoStateFn);
if length(temp)>(1+l_semiz+l_semiz) % This is largely pointless, the SemiExoShockFn is always going to have some parameters
    SemiExoStateFnParamNames={temp{1+l_semiz+l_semiz+1:end}}; % the first inputs will always be (d,semizprime,semiz)
else
    SemiExoStateFnParamNames={};
end
n_semiz=simoptions.n_semiz;
N_semiz=prod(n_semiz);
pi_semiz_J=zeros(N_semiz,N_semiz,n_d2,N_j);
for jj=1:N_j
    SemiExoStateFnParamValues=CreateVectorFromParams(Parameters,SemiExoStateFnParamNames,jj);
    pi_semiz_J(:,:,:,jj)=CreatePiSemiZ(n_d2,simoptions.n_semiz,d2_grid,simoptions.semiz_grid,simoptions.SemiExoStateFn,SemiExoStateFnParamValues);
end

%%
N_a=prod(n_a);
N_z=prod(n_z);
% N_semiz=prod(n_semiz);

jequaloneDist=reshape(jequaloneDist,[N_a*N_z*N_semiz,1]);
Policy=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, [n_z,simoptions.n_semiz],N_j);
if simoptions.iterate==0
    Policy=gather(Policy);
    jequaloneDist=gather(jequaloneDist);    
end
pi_z_J=gather(pi_z_J);

if simoptions.iterate==0
    StationaryDistKron=StationaryDist_FHorz_Case1_SemiExo_Simulation_raw(jequaloneDist,AgeWeightParamNames,Policy,n_d1,n_d2,N_a,N_z,N_semiz,N_j,pi_z_J,pi_semiz_J,Parameters,simoptions);
elseif simoptions.iterate==1
    StationaryDistKron=StationaryDist_FHorz_Case1_SemiExo_Iteration_raw(jequaloneDist,AgeWeightParamNames,Policy,n_d1,n_d2,N_a,N_z,N_semiz,N_j,pi_z_J,pi_semiz_J,Parameters,simoptions);
end

if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDistKron,[n_a,n_z,simoptions.n_semiz,N_j]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDistKron,[N_a,N_z,N_semiz,N_j]);
end

end
