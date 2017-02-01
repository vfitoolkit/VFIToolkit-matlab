function StationaryDist=StationaryDist_FHorz_Case2(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid,pi_z,Phi_aprimeFn,Case2_Type,Params,PhiaprimeParamNames,simoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if nargin<12
    simoptions.nsims=10^4;
    simoptions.parallel=2;
    simoptions.verbose=0;
    try 
        PoolDetails=gcp;
        simoptions.ncores=PoolDetails.NumWorkers;
    catch
        simoptions.ncores=1;
    end
    simoptions.iterate=0;
    simoptions.tolerance=10^(-9);
else
    %Check vfoptions for missing fields, if there are some fill them with
    %the defaults
    eval('fieldexists=1;simoptions.tolerance;','fieldexists=0;')
    if fieldexists==0
        simoptions.tolerance=10^(-9);
    end
    eval('fieldexists=1;simoptions.nsims;','fieldexists=0;')
    if fieldexists==0
        simoptions.nsims=10^4;
    end
    eval('fieldexists=1;simoptions.parallel;','fieldexists=0;')
    if fieldexists==0
        simoptions.parallel=2;
    end
    eval('fieldexists=1;simoptions.verbose;','fieldexists=0;')
    if fieldexists==0
        simoptions.verbose=0;
    end
    eval('fieldexists=1;simoptions.ncores;','fieldexists=0;')
    if fieldexists==0
        try
            PoolDetails=gcp;
            simoptions.ncores=PoolDetails.NumWorkers;
        catch
            simoptions.ncores=1;
        end
    end
    eval('fieldexists=1;simoptions.iterate;','fieldexists=0;')
    if fieldexists==0
        simoptions.iterate=0;
    end
end

PolicyKron=KronPolicyIndexes_FHorz_Case2(Policy, n_d, n_a, n_z,N_j,simoptions);

jequaloneDistKron=reshape(jequaloneDist,[N_a*N_z,1]);

if simoptions.iterate==0
    StationaryDistKron=StationaryDist_FHorz_Case2_Simulation_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid,pi_z,Phi_aprimeFn,Case2_Type,Params,PhiaprimeParamNames,simoptions);
elseif simoptions.iterate==1
    StationaryDistKron=StationaryDist_FHorz_Case2_Iteration_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid,pi_z,Phi_aprimeFn,Case2_Type,Params,PhiaprimeParamNames,simoptions);
end


StationaryDist=reshape(StationaryDistKron,[n_a,n_z,N_j]);

end
