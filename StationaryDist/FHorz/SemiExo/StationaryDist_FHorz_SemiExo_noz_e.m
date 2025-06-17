function StationaryDist=StationaryDist_FHorz_SemiExo_noz_e(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_semiz,n_e,N_j,pi_semiz_J,pi_e_J,Parameters,simoptions)

if length(n_d)>simoptions.l_dsemiz
    n_d1=n_d(1:end-simoptions.l_dsemiz);
else
    n_d1=0;
end

%%
N_d1=prod(n_d1);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_e=prod(n_e);

jequaloneDist=reshape(jequaloneDist,[N_a*N_semiz*N_e,1]);
Policy=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, [simoptions.n_semiz,n_e],N_j);

StationaryDist=StationaryDist_FHorz_SemiExo_Iteration_noz_e_raw(jequaloneDist,AgeWeightParamNames,Policy,N_d1,N_a,N_semiz,N_e,N_j,pi_semiz_J,pi_e_J,Parameters,simoptions);

if simoptions.parallel==2
    StationaryDist=gpuArray(StationaryDist); % move output to gpu
end
if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,simoptions.n_semiz,n_e,N_j]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDist,[N_a,N_semiz,N_e,N_j]);
end

end
