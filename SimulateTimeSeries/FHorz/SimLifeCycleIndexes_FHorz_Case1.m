function SimLifeCycle=SimLifeCycleIndexes_FHorz_Case1(Policy,n_d,n_a,n_z,N_j,pi_z, simoptions)
% Simulates a path based on PolicyIndexes of length 'periods' beginning from point 'seedpoint' (this is not just left
% as being random since some random points may be ones that never 'exist' in eqm).

N_a=prod(n_a);
N_z=prod(n_z);
N_d=prod(n_d);

%% Check which simoptions have been declared, set all others to defaults 
if exist('simoptions','var')==1    
    %Check vfoptions for missing fields, if there are some fill them with
    %the defaults
    if ~isfield(simoptions,'polindorval')
        simoptions.polindorval=1;
    end
    if ~isfield(simoptions,'seedpoint')
        simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2),1];
    end
    if ~isfield(simoptions,'simperiods')
        simoptions.simperiods=N_j;
    end
    if ~isfield(simoptions,'parallel')
        simoptions.parallel=2;
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    if isfield(simoptions,'ExogShockFn') % If using ExogShockFn then figure out the parameter names
        simoptions.ExogShockFnParamNames=getAnonymousFnInputNames(simoptions.ExogShockFn);
    end
else
    %If simoptions is not given, just use all the defaults
    simoptions.polindorval=1;
    simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2),1];
    simoptions.simperiods=N_j;
    simoptions.parallel=2;
    simoptions.verbose=0;
end


l_a=length(n_a);
l_z=length(n_z);

eval('fieldexists_ExogShockFn=1;simoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;simoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')
eval('fieldexists_pi_z_J=1;simoptions.pi_z_J;','fieldexists_pi_z_J=0;')

%Policy is [l_d+l_a,n_a,n_s,n_z]
PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z, N_j);%,simoptions);

%seedtemp=sub2ind_homemade([n_a,n_z],simoptions.seedpoint);
%seedpoint=ind2sub_homemade([N_a,N_z],seedtemp);

if fieldexists_pi_z_J==1
    cumsumpi_z=cumsum(simoptions.pi_z_J,2);
elseif fieldexists_ExogShockFn==1
    cumsumpi_z=nan(N_z,N_z,N_j);
    for jj=1:N_j
        if fieldexists_ExogShockFnParamNames==1
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
            ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
            for kk=1:length(ExogShockFnParamsVec)
                ExogShockFnParamsCell(kk,1)={ExogShockFnParamsVec(kk)};
            end
            [~,pi_z_jj]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
        else
            [~,pi_z_jj]=simoptions.ExogShockFn(jj);
        end
        cumsumpi_z(:,:,jj)=cumsum(pi_z_jj,2);
    end
    fieldexists_pi_z_J=1; % Needed as input for SimLifeCycleIndexes_FHorz_Case1_raw()
else
    fieldexists_pi_z_J=0; % Needed as input for SimLifeCycleIndexes_FHorz_Case1_raw()
    cumsumpi_z=cumsum(pi_z,2);
end

MoveOutputtoGPU=0;
if simoptions.parallel==2
    % Simulation on GPU is really slow. So instead, switch to CPU, and then switch
    % back. For anything but ridiculously short simulations it is more than worth the overhead.
    PolicyIndexesKron=gather(PolicyIndexesKron);
    cumsumpi_z=gather(cumsumpi_z);
    MoveOutputtoGPU=1;
end

SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_raw(PolicyIndexesKron,N_d,N_a,N_z,N_j,cumsumpi_z, simoptions.seedpoint, simoptions.simperiods,fieldexists_pi_z_J);

SimLifeCycle=zeros(l_a+l_z,N_j);
for jj=1:N_j
    % IS LIKELY THIS (within for loop) COULD BE VECTORIZED IN MUCH MORE EFFICIENT MANNER
    temp=SimLifeCycleKron(:,jj);
    if ~isnan(temp)
        a_c_vec=ind2sub_homemade([n_a],temp(1));
        z_c_vec=ind2sub_homemade([n_z],temp(2));
        for i=1:l_a
            SimLifeCycle(i,jj)=a_c_vec(i);
        end
        for i=1:l_z
            SimLifeCycle(l_a+i,jj)=z_c_vec(i);
        end
    end
end

if MoveOutputtoGPU==1
    SimLifeCycle=gpuArray(SimLifeCycle);
end


end



