function StationaryDist=StationaryDist_FHorz_Case1_noz(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,N_j,Parameters,simoptions)

if isempty(n_d)
    n_d=0;
end
N_d=prod(n_d);
N_a=prod(n_a);

jequaloneDist=reshape(jequaloneDist,[N_a,1]);
Policy=KronPolicyIndexes_FHorz_Case1_noz(Policy, n_d, n_a,N_j);
if simoptions.iterate==0
    Policy=gather(Policy);
    jequaloneDist=gather(jequaloneDist);    
end

if simoptions.iterate==0
    StationaryDist=StationaryDist_FHorz_Case1_Simulation_noz_raw(jequaloneDist,AgeWeightParamNames,Policy,N_d,N_a,N_j,Parameters,simoptions);
elseif simoptions.iterate==1
    StationaryDist=StationaryDist_FHorz_Case1_Iteration_noz_raw(jequaloneDist,AgeWeightParamNames,Policy,N_d,N_a,N_j,Parameters,simoptions);
end

if simoptions.parallel==2
    StationaryDist=gpuArray(StationaryDist); % move output to gpu
end
if simoptions.outputkron==0 % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDist,[n_a,N_j]);
end

end