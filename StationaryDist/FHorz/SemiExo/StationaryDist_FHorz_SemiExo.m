function StationaryDist=StationaryDist_FHorz_SemiExo(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_semiz,n_z,N_j,pi_semiz_J,pi_z_J,Parameters,simoptions)

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

%%
jequaloneDist=reshape(jequaloneDist,[N_a*N_semiz*N_z,1]);

Policy=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, [simoptions.n_semiz,n_z],N_j,simoptions);

pi_z_J=gather(pi_z_J);

%%
StationaryDist=StationaryDist_FHorz_SemiExo_Iteration_raw(jequaloneDist,AgeWeightParamNames,Policy,N_d1,N_a,N_z,N_semiz,N_j,pi_z_J,pi_semiz_J,Parameters,simoptions);

if simoptions.parallel==2
    StationaryDist=gpuArray(StationaryDist); % move output to gpu
end
if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,simoptions.n_semiz,n_z,N_j]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDist,[N_a,N_semiz,N_z,N_j]);
end

end
