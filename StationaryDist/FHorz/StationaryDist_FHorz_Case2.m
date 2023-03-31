function StationaryDist=StationaryDist_FHorz_Case2(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid,pi_z,Phi_aprime,Case2_Type,Params,PhiaprimeParamNames,simoptions)
% Note: Case2 stationary distribution commands require grids to be able to evaluate Phi_aprime (not for the actual stationary distribution calculations)

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
    simoptions.iterate=1;
    simoptions.tolerance=10^(-9);
    simoptions.dynasty=0;
    simoptions.phiaprimedependsonage=0;
    simoptions.lowmemory=0; % used to evaluate Phi_aprime
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
        simoptions.iterate=1;
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
    if ~isfield(simoptions,'phiaprimedependsonage')
        simoptions.phiaprimedependsonage=0;
    end
    if ~isfield(simoptions,'lowmemory')
        simoptions.lowmemory=0; % used to evaluate Phi_aprime
    end
end

if prod(simoptions.agedependentgrids)~=0
    % Note d_grid is actually d_gridfn
    % Note a_grid is actually a_gridfn
    % Note z_grid is actually z_gridfn
    % Note pi_z is actually AgeDependentGridParamNames
    StationaryDist=StationaryDist_FHorz_Case2_AgeDepGrids(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z,Phi_aprime,Case2_Type,Params,PhiaprimeParamNames,simoptions);
    return
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
        
if isfield(simoptions,'n_e')
    N_e=prod(simoptions.n_e);
    jequaloneDistKron=reshape(jequaloneDist,[N_a*N_z*N_e,1]);
    PolicyKron=KronPolicyIndexes_FHorz_Case2(Policy, n_d, n_a, n_z,N_j,simoptions.n_e); % Just pretend e is another z
else
    jequaloneDistKron=reshape(jequaloneDist,[N_a*N_z,1]);
    PolicyKron=KronPolicyIndexes_FHorz_Case2(Policy, n_d, n_a, n_z,N_j);
end


if simoptions.parallel~=2 % To cover the case when using gpu to solve value fn, but cpu to solve agent dist
    PolicyKron=gather(PolicyKron);
end

%%
l_d=length(n_d);
l_a=length(n_a);
if n_z(1)>0
    l_z=length(n_z);
else
    l_z=0;
end
% Figure out PhiaprimeParamNames from Phi_aprime
if isempty(PhiaprimeParamNames)
    temp=getAnonymousFnInputNames(Phi_aprime);

    % NOTE: FOLLOWING LARGELY OMITS POSSIBILITY OF e VARIABLES
    if Case2_Type==1 % phi_a'(d,a,z,z')
        if length(temp)>(l_d+l_a+l_z+l_z) % This is largely pointless, the Phi_aprime is always going to have some parameters
            PhiaprimeParamNames={temp{l_d+l_a+l_z+l_z+1:end}}; % the first inputs will always be (d,a,z) for Case2
        else
            PhiaprimeParamNames={};
        end
    elseif Case2_Type==11 || Case2_Type==12 % phi_a'(d,a,z') OR phi_a'(d,a,z)
        if length(temp)>(l_d+l_a+l_z) % This is largely pointless, the Phi_aprime is always going to have some parameters
            PhiaprimeParamNames={temp{l_d+l_a+l_z+1:end}}; % the first inputs will always be (d,a,z) for Case2
        else
            PhiaprimeParamNames={};
        end
    elseif Case2_Type==2  % phi_a'(d,z,z')
        if length(temp)>(l_d+l_z+l_z) % This is largely pointless, the Phi_aprime is always going to have some parameters
            PhiaprimeParamNames={temp{l_d+l_z+l_z+1:end}}; % the first inputs will always be (d,a,z) for Case2
        else
            PhiaprimeParamNames={};
        end
    elseif Case2_Type==3  % phi_a'(d,z')
        if length(temp)>(l_d+l_z) % This is largely pointless, the Phi_aprime is always going to have some parameters
            PhiaprimeParamNames={temp{l_d+l_z+1:end}}; % the first inputs will always be (d,a,z) for Case2
        else
            PhiaprimeParamNames={};
        end
    elseif Case2_Type==4  % phi_a'(d,a)
        if length(temp)>(l_d+l_a) % This is largely pointless, the Phi_aprime is always going to have some parameters
            PhiaprimeParamNames={temp{l_d+l_a+1:end}}; % the first inputs will always be (d,a,z) for Case2
        else
            PhiaprimeParamNames={};
        end
    end
end

%%
if simoptions.iterate==0
    if Case2_Type==3
        if isfield(simoptions,'n_e')
            StationaryDistKron=StationaryDist_FHorz_Case2_3_Simulation_e_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,n_d,n_a,n_z,n_e,N_j,d_grid, a_grid, z_grid,pi_z,simoptions.pi_e,Phi_aprime,Case2_Type,Params,PhiaprimeParamNames,simoptions);
        else
            StationaryDistKron=StationaryDist_FHorz_Case2_3_Simulation_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid,pi_z,Phi_aprime,Case2_Type,Params,PhiaprimeParamNames,simoptions);
        end
    else
        error('Current Case2_Type has not been implemented for simulating stationary dist in FHorz, please contact me if you need this')
    end
elseif simoptions.iterate==1
    if Case2_Type==1 || Case2_Type==11 || Case2_Type==12 || Case2_Type==2
        StationaryDistKron=StationaryDist_FHorz_Case2_Iteration_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid,pi_z,Phi_aprime,Case2_Type,Params,PhiaprimeParamNames,simoptions);
    elseif Case2_Type==3
        if isfield(simoptions,'n_e')
            StationaryDistKron=StationaryDist_FHorz_Case2_3_Iteration_e_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,n_d,n_a,n_z,simoptions.n_e,N_j,d_grid, a_grid, z_grid,pi_z,simoptions.pi_e,Phi_aprime,Case2_Type,Params,PhiaprimeParamNames,simoptions);
        else
            StationaryDistKron=StationaryDist_FHorz_Case2_3_Iteration_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid,pi_z,Phi_aprime,Case2_Type,Params,PhiaprimeParamNames,simoptions);
        end
    else
        error('Current Case2_Type has not been implemented for iterating stationary dist in FHorz, please contact me if you need this')
    end
end

if isfield(simoptions,'n_e')
    StationaryDist=reshape(StationaryDistKron,[n_a,n_z,simoptions.n_e,N_j]);    
else
    StationaryDist=reshape(StationaryDistKron,[n_a,n_z,N_j]);
end

end
