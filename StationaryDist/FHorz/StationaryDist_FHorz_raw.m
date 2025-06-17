function StationaryDist=StationaryDist_FHorz_raw(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

%% Solve the baseline case
jequaloneDist=reshape(jequaloneDist,[N_a*N_z,1]);
Policy=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z,N_j);
pi_z_J=gather(pi_z_J);

StationaryDist=StationaryDist_FHorz_Iteration_raw(jequaloneDist,AgeWeightParamNames,Policy,N_d,N_a,N_z,N_j,pi_z_J,Parameters,simoptions);

if simoptions.parallel==2
    StationaryDist=gpuArray(StationaryDist); % move output to gpu
end
if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,n_z,N_j]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDist,[N_a,N_z,N_j]);
end


end
