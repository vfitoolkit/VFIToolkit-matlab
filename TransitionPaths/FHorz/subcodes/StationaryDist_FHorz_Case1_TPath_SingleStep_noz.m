function AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_noz(AgentDist,AgeWeightParamNames,Policy,n_d,n_a,N_j,Parameters,simoptions)

N_d=prod(n_d);
N_a=prod(n_a);

% if exist('simoptions','var')==0
%     simoptions.nsims=10^4;
%     simoptions.parallel=2;
%     simoptions.verbose=0;
%     try 
%         PoolDetails=gcp;
%         simoptions.ncores=PoolDetails.NumWorkers;
%     catch
%         simoptions.ncores=1;
%     end
%     simoptions.iterate=1;
%     simoptions.tolerance=10^(-9);
% else
%     %Check vfoptions for missing fields, if there are some fill them with
%     %the defaults
%     if isfield(simoptions,'tolerance')==0
%         simoptions.tolerance=10^(-9);
%     end
%         if isfield(simoptions,'nsims')==0
%         simoptions.nsims=10^4;
%     end
%         if isfield(simoptions,'parallel')==0
%         simoptions.parallel=2;
%     end
%         if isfield(simoptions,'verbose')==0
%         simoptions.verbose=0;
%     end
%     if isfield(simoptions,'ncores')==0
%         try
%             PoolDetails=gcp;
%             simoptions.ncores=PoolDetails.NumWorkers;
%         catch
%             simoptions.ncores=1;
%         end
%     end
%     if isfield(simoptions,'iterate')==0
%         simoptions.iterate=1;
%     end
% end

% PolicyKron=KronPolicyIndexes_FHorz_Case1_noz(Policy, n_d, n_a,N_j,simoptions);
% 
% jequaloneDistKron=reshape(AgentDist,[N_a,1]);

if simoptions.iterate==0
    AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_Simulation_noz_raw(AgentDist,AgeWeightParamNames,Policy,N_d,N_a,N_j,Parameters,simoptions);
elseif simoptions.iterate==1
    AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_Iteration_noz_raw(AgentDist,AgeWeightParamNames,Policy,N_d,N_a,N_j,Parameters,simoptions);
end

% AgentDist=reshape(StationaryDistKron,[n_a,N_j]);

end
