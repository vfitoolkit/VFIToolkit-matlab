function StationaryDist=StationaryDist_FHorz_Case1_Iteration(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,pi_z,simoptions)

if nargin<7
    simoptions.parallel=2;
    simoptions.lowmemory=0;
else
    eval('fieldexists=1;simoptions.parallel;','fieldexists=0;')
    if fieldexists==0
        simoptions.parallel=2;
    end
    eval('fieldexists=1;simoptions.lowmemory;','fieldexists=0;')
    if fieldexists==0
        simoptions.lowmemory=0;
    end
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

PolicyKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z,N_j,simoptions);

jequaloneDistKron=reshape(jequaloneDist,[N_a*N_z,1]);

StationaryDistKron=StationaryDist_FHorz_Case1_Iteration_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,N_d,N_a,N_z,pi_z,simoptions);

StationaryDist=reshape(StationaryDistKron,[n_a,n_z,N_j]);

end
