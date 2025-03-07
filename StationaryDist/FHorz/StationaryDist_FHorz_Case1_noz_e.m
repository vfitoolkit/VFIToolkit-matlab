function StationaryDist=StationaryDist_FHorz_Case1_noz_e(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,N_j,Parameters,simoptions)

n_e=simoptions.n_e;

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

if ~isfield(simoptions,'loopovere')
    simoptions.loopovere=0; % default is parallel over e, 1 will loop over e, 2 will parfor loop over e
end

%% Set up pi_e_J (transition matrix for iid exogenous state e, depending on age)
if isfield(simoptions,'pi_e')
    pi_e_J=simoptions.pi_e.*ones(1,N_j);
elseif isfield(simoptions,'pi_e_J')
    pi_e_J=simoptions.pi_e_J;
elseif isfield(simoptions,'EiidShockFn')
    pi_e_J=zeros(N_e,N_e,N_j);
    for jj=1:N_j
        EiidShockFnParamNames=getAnonymousFnInputNames(simoptions.EiidShockFn);
        EiidShockFnParamsVec=CreateVectorFromParams(Parameters, EiidShockFnParamNames,jj);
        EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
        for ii=1:length(EiidShockFnParamsVec)
            EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
        end
        [~,pi_e]=simoptions.EiidShockFn(EiidShockFnParamsCell{:});
        pi_e_J(:,jj)=pi_e;
    end
end

%%
jequaloneDist=reshape(jequaloneDist,[N_a*N_e,1]);
Policy=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_e,N_j);

StationaryDist=StationaryDist_FHorz_Case1_Iteration_noz_e_raw(jequaloneDist,AgeWeightParamNames,Policy,N_d,N_a,N_e,N_j,pi_e_J,Parameters,simoptions);

if simoptions.parallel==2
    StationaryDist=gpuArray(StationaryDist); % move output to gpu
end
if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,n_e,N_j]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDist,[N_a,N_e,N_j]);
end

end
