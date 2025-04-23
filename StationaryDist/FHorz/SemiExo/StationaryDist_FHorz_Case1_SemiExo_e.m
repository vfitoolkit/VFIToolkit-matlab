function StationaryDist=StationaryDist_FHorz_Case1_SemiExo_e(jequaloneDistKron,AgeWeightParamNames,Policy,n_d,n_a,n_semiz,n_z,n_e,N_j,pi_semiz_J,pi_z_J,pi_e_J,Parameters,simoptions)

if length(n_d)>simoptions.l_dsemiz
    n_d1=n_d(1:end-simoptions.l_dsemiz);
else
    n_d1=0;
end

%%
N_d1=prod(n_d1);
N_a=prod(n_a);
N_z=prod(n_z);
N_semiz=prod(n_semiz);
N_e=prod(n_e);

if ~isfield(simoptions,'loopovere')
    simoptions.loopovere=0; % default is parallel over e, 1 will loop over e, 2 will parfor loop over e
end

%%
jequaloneDistKron=reshape(jequaloneDistKron,[N_a*N_semiz*N_z*N_e,1]);
Policy=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, [simoptions.n_semiz,n_z],N_j,n_e);

Policy=gather(Policy);
jequaloneDistKron=gather(jequaloneDistKron);

pi_z_J=gather(pi_z_J);
pi_e_J=gather(pi_e_J);

StationaryDist=StationaryDist_FHorz_Case1_SemiExo_Iteration_e_raw(jequaloneDistKron,AgeWeightParamNames,Policy,N_d1,N_a,N_z,N_semiz,N_e,N_j,pi_z_J,pi_semiz_J,pi_e_J,Parameters,simoptions);

if simoptions.parallel==2
    StationaryDist=gpuArray(StationaryDist); % move output to gpu
end
if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,simoptions.n_semiz,n_z,n_e,N_j]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDist,[N_a,N_semiz,N_z,N_e,N_j]);
end

end
