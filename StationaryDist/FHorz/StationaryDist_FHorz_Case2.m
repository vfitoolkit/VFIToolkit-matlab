function StationaryDist=StationaryDist_FHorz_Case2(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid,pi_z,Phi_aprimeFn,Case2_Type,Params,PhiaprimeParamNames,simoptions)

if ~exist('simoptions','var')
    simoptions.nsims=10^4;
    simoptions.parallel=1+(gpuDeviceCount>0);
    simoptions.verbose=0;
    try 
        PoolDetails=gcp;
        simoptions.ncores=PoolDetails.NumWorkers;
    catch
        simoptions.ncores=1;
    end
    simoptions.iterate=0;
    simoptions.tolerance=10^(-9);
    simoptions.dynasty=0;
else
    %Check simoptions for missing fields, if there are some fill them with
    %the defaults
    if ~isfield(simoptions,'tolerance')
        simoptions.tolerance=10^(-9);
    end
    if ~isfield(simoptions,'nsims')
        simoptions.nsims=10^4;
    end
    if ~isfield(simoptions,'parallel')
            simoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(simoptions,'verbose')
            simoptions.verbose=0;
    end
    if ~isfield(simoptions,'ncores')
        try
            PoolDetails=gcp;
            simoptions.ncores=PoolDetails.NumWorkers;
        catch
            simoptions.ncores=1;
        end
    end
    if ~isfield(simoptions,'iterate')
        simoptions.iterate=0;
    end
    if ~isfield(simoptions,'dynasty')
        simoptions.dynasty=0;
    elseif simoptions.dynasty==1
        if ~isfield(simoptions,'dynasty_storeP')
            simoptions.dynasty_storeP=1; % Implents a more memory intensive but faster approach to iterating stationary agent distribution with dynasty
        end
    end
    if ~isfield(simoptions,'agedependentgrids')
        simoptions.agedependentgrids=0;
    end
end

if prod(simoptions.agedependentgrids)~=0
    % Note d_grid is actually d_gridfn
    % Note a_grid is actually a_gridfn
    % Note z_grid is actually z_gridfn
    % Note pi_z is actually AgeDependentGridParamNames
    StationaryDist=StationaryDist_FHorz_Case2_AgeDepGrids(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z,Phi_aprimeFn,Case2_Type,Params,PhiaprimeParamNames,simoptions);
    return
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
        
jequaloneDistKron=reshape(jequaloneDist,[N_a*N_z,1]);

PolicyKron=KronPolicyIndexes_FHorz_Case2(Policy, n_d, n_a, n_z,N_j);%,simoptions);

if simoptions.parallel~=2 % To cover the case when using gpu to solve value fn, but cpu to solve agent dist
    PolicyKron=gather(PolicyKron);
end

if simoptions.iterate==0
    fprintf('Simulating the stationary agents distribution has not yet been implemented for Case2 of FHorz, \n please email me if you have a need for it, otherwise use simoptions.iterate=1 to iterate the stationary distribution \n')
%     StationaryDistKron=StationaryDist_FHorz_Case2_Simulation_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid,pi_z,Phi_aprimeFn,Case2_Type,Params,PhiaprimeParamNames,simoptions);
elseif simoptions.iterate==1
    StationaryDistKron=StationaryDist_FHorz_Case2_Iteration_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid,pi_z,Phi_aprimeFn,Case2_Type,Params,PhiaprimeParamNames,simoptions);
end


StationaryDist=reshape(StationaryDistKron,[n_a,n_z,N_j]);

end
