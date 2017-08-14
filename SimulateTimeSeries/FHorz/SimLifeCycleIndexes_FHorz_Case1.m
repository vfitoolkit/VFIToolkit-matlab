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
    eval('fieldexists=1;simoptions.polindorval;','fieldexists=0;')
    if fieldexists==0
        simoptions.polindorval=1;
    end
    eval('fieldexists=1;simoptions.seedpoint;','fieldexists=0;')
    if fieldexists==0
        simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2),1];
    end
    eval('fieldexists=1;simoptions.simperiods;','fieldexists=0;')
    if fieldexists==0
        simoptions.simperiods=N_j;
    end
    eval('fieldexists=1;simoptions.parallel;','fieldexists=0;')
    if fieldexists==0
        simoptions.parallel=2;
    end
    eval('fieldexists=1;simoptions.verbose;','fieldexists=0;')
    if fieldexists==0
        simoptions.verbose=0;
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


%Policy is [l_d+l_a,n_a,n_s,n_z]
PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z, N_j,simoptions);

%seedtemp=sub2ind_homemade([n_a,n_z],simoptions.seedpoint);
%seedpoint=ind2sub_homemade([N_a,N_z],seedtemp);

if isfield(simoptions,'ExogShockFn')==1
    fieldexists_ExogShockFn=1; % Needed as input for SimLifeCycleIndexes_FHorz_Case1_raw()
    cumsumpi_z=nan(N_z,N_z,N_j);
    for jj=1:N_j
        if isfield(simoptions,'ExogShockFnParamNames')==1
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
            [~,pi_z_jj]=vfoptions.ExogShockFn(ExogShockFnParamsVec);
        else
            [~,pi_z_jj]=vfoptions.ExogShockFn(jj);
        end
        cumsumpi_z(:,:,jj)=cumsum(pi_z_jj,2);
    end
else
    fieldexists_ExogShockFn=0; % Needed as input for SimLifeCycleIndexes_FHorz_Case1_raw()
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

SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_raw(PolicyIndexesKron,N_d,N_a,N_z,N_j,cumsumpi_z, simoptions.seedpoint, simoptions.simperiods,fieldexists_ExogShockFn);

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



