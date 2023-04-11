function StationaryDist=StationaryDist_FHorz_Case3(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,n_u,N_j,d_grid, a_grid, z_grid,u_grid,pi_z,pi_u,aprimeFn,Params,simoptions)
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
    simoptions.aprimedependsonage=0;
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
    if ~isfield(simoptions,'aprimedependsonage')
        simoptions.aprimedependsonage=0;
    end
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_u=prod(n_u);
        
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
% l_a=length(n_a);
% if n_z(1)>0
%     l_z=length(n_z);
% else
%     l_z=0;
% end
% if isfield(vfoptions,'n_e')
%     l_e=length(vfoptions.n_e);
% else
%     l_e=0;
% end
l_u=length(n_a);

% aprimeFnParamNames in same fashion
l_u=length(n_u);
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d+l_u)
    aprimeFnParamNames={temp{l_d+l_u+1:end}}; % the first inputs will always be (d,u)
else
    aprimeFnParamNames={};
end



%%
if simoptions.iterate==0
    error('Simulation of agent distribution not yet implemented for Case3, contact me if you want/need')
    if isfield(simoptions,'n_e')
        StationaryDistKron=StationaryDist_FHorz_Case3_Simulation_e_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,n_d,n_a,n_z,n_e,n_u,N_j,d_grid,a_grid,u_grid,pi_z,simoptions.pi_e,pi_u,aprimeFn,Parameters,aprimeFnParamNames, simoptions)
    else
        StationaryDistKron=StationaryDist_FHorz_Case3_Simulation_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,n_d,n_a,n_z,n_u,N_j,d_grid,a_grid,u_grid,pi_z,pi_u,aprimeFn,Params,aprimeFnParamNames,simoptions);
    end
elseif simoptions.iterate==1
    if isfield(simoptions,'n_e')
        StationaryDistKron=StationaryDist_FHorz_Case3_Iteration_e_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,n_d,n_a,n_z,n_e,n_u,N_j,d_grid,a_grid,u_grid,pi_z,pi_e,pi_u,aprimeFn,Parameters,aprimeFnParamNames,simoptions);
    else
%         StationaryDistKron=StationaryDist_FHorz_Case2_Iteration_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid,pi_z,Phi_aprime,Case2_Type,Params,PhiaprimeParamNames,simoptions);
    end
end

if isfield(simoptions,'n_e')
    StationaryDist=reshape(StationaryDistKron,[n_a,n_z,simoptions.n_e,N_j]);    
else
    StationaryDist=reshape(StationaryDistKron,[n_a,n_z,N_j]);
end

end
