function StationaryDist=StationaryDist_FHorz_noz_e(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_e,N_j,pi_e_J,Parameters,simoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

if ~isfield(simoptions,'loopovere')
    simoptions.loopovere=0; % default is parallel over e, 1 will loop over e, 2 will parfor loop over e
end
%%
jequaloneDist=reshape(jequaloneDist,[N_a*N_e,1]);
Policy=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_e,N_j,simoptions);

StationaryDist=StationaryDist_FHorz_Iteration_noz_e_raw(jequaloneDist,AgeWeightParamNames,Policy,N_d,N_a,N_e,N_j,pi_e_J,Parameters,simoptions);

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
