function SimPanel=SimPanelIndexes_InfHorz_SemiEndog(InitialDist,Policy,n_d,n_a,n_z,pi_z, simoptions, CondlProbOfSurvival, Parameters)
% Simulates a panel based on PolicyIndexes of 'numbersims' agents of length
% 'simperiods' beginning from randomly drawn InitialDist. 
% CondlProbOfSurvival is an optional input. (only needed when using: simoptions.exitinpanel=1, there there is exit, either exog, endog or mix of both)
%
% InitialDist can be inputed as over the finite time-horizon (j), or
% without a time-horizon in which case it is assumed to be an InitialDist
% for time j=1. (So InitialDist is either n_a-by-n_z-by-n_j, or n_a-by-n_z)
%
% Parameters is only needed as an input when you have mixed (endogenous and
% exogenous) exit. It is otherwise not required to be inputed.

N_a=prod(n_a);
N_z=prod(n_z);
N_d=prod(n_d);

%% Check which simoptions have been declared, set all others to defaults 
if exist('simoptions','var')==1
    %Check simoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(simoptions, 'polindorval')
        simoptions.polindorval=1;
    end
    if ~isfield(simoptions, 'simperiods')
        simoptions.simperiods=50;
    end
    if ~isfield(simoptions, 'numbersims')
        simoptions.numbersims=10^3;
    end
    if ~isfield(simoptions, 'parallel')
        simoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(simoptions, 'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions, 'verbose')
        simoptions.verbose=0;
    end
    % Note: SemiEndogShockFn does not presently allow entry/exit
else
    %If simoptions is not given, just use all the defaults
    simoptions.polindorval=1;
    simoptions.simperiods=50;
    simoptions.numbersims=10^3;
    simoptions.parallel=1+(gpuDeviceCount>0);
    simoptions.verbose=0;
    % Note: SemiEndogShockFn does not presently allow entry/exit
end

if exist('CondlProbOfSurvival','var')==1
    simoptions.exitinpanel=1;
    CondlProbOfSurvival=reshape(CondlProbOfSurvival,[N_a,N_z]);
    if ~isfield(simoptions, 'endogenousexit')
        simoptions.endogenousexit=0;  % Note: this will only be relevant if exitinpanel=1
    end
end

l_a=length(n_a);
l_z=length(n_z);

PolicyIndexesKron=gather(KronPolicyIndexes_Case1(Policy, n_d, n_a, n_z,simoptions));

if simoptions.parallel==2
    % Get seedpoints from InitialDist while on gpu
    seedpoints=nan(simoptions.numbersims,2,'gpuArray'); % 2 as a,z (vectorized)
    cumsumInitialDistVec=cumsum(reshape(InitialDist,[N_a*N_z,1]));
    [~,seedpointvec]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims,1,'gpuArray'));
    for ii=1:simoptions.numbersims
        seedpoints(ii,:)=ind2sub_homemade_gpu([N_a,N_z],seedpointvec(ii));
    end
    seedpoints=round(seedpoints); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.
else
    InitialDist=gather(InitialDist); % Make sure it is not on gpu
    numbersims=gather(simoptions.numbersims); % This is just to deal with weird error that matlab decided simoptions.numbersims was on gpu and so couldn't be an input to rand()
    % Get seedpoints from InitialDist
    seedpoints=nan(simoptions.numbersims,2); % 3 as a,z (vectorized)
    cumsumInitialDistVec=cumsum(reshape(InitialDist,[N_a*N_z,1]));
    [~,seedpointvec]=max(cumsumInitialDistVec>rand(1,numbersims,1));
    for ii=1:simoptions.numbersims
        seedpoints(ii,:)=ind2sub_homemade([N_a,N_z],seedpointvec(ii));
    end
    seedpoints=round(seedpoints); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.
end

%% Semi-endogenous state
% The transition matrix of the exogenous shocks depends on the value of the endogenous state.
if isa(simoptions.SemiEndogShockFn,'function_handle')==0
    pi_z_semiendog=simoptions.SemiEndogShockFn;
else
    if ~isfield(simoptions,'SemiEndogShockFnParamNames')
        fprintf('ERROR: simoptions.SemiEndogShockFnParamNames is missing (is needed for simoptions.SemiEndogShockFn) \n')
        dbstack
        return
    end
    pi_z_semiendog=zeros(N_a,N_z,N_z);
    a_gridvals=CreateGridvals(n_a,a_grid,2);
    SemiEndogParamsVec=CreateVectorFromParams(Parameters, simoptions.SemiEndogShockFnParamNames);
    SemiEndogParamsCell=cell(length(SemiEndogParamsVec),1);
    for ii=1:length(SemiEndogParamsVec)
        SemiEndogParamsCell(ii,1)={SemiEndogParamsVec(ii)};
    end
    parfor ii=1:N_a
        a_ii=a_gridvals(ii,:)';
        a_ii_SemiEndogParamsCell=[a_ii;SemiEndogParamsCell];
        [~,temp_pi_z]=SemiEndogShockFn(a_ii_SemiEndogParamsCell{:});
        pi_z_semiendog(ii,:,:)=temp_pi_z;
        % Note that temp_z_grid is just the same things for all k, and same as
        % z_grid created about 10 lines above, so I don't bother keeping it.
        % I only create it so you can double-check it is same as z_grid
    end
end

% cumsumpi_z=cumsum(pi_z,2);

cumsumpi_z_semiendog=cumsum(permute(pi_z_semiendog,[2,3,1]),2); % cumulative some over zprime; has dimensions z-zprime-k

%%
MoveOutputtoGPU=0;
if simoptions.parallel==2
    % Simulation on GPU is really slow. So instead, switch to parallel CPU, and then switch back. 
    % For anything but ridiculously short simulations it is more than worth the overhead.
    simoptions.parallel=1;
    PolicyIndexesKron=gather(PolicyIndexesKron);
    cumsumpi_z_semiendog=gather(cumsumpi_z_semiendog);
    seedpoints=gather(seedpoints);
    MoveOutputtoGPU=1;
    simoptions.simperiods=gather(simoptions.simperiods);
end


%%
SimPanel=nan(l_a+l_z,simoptions.simperiods,simoptions.numbersims); % (a,z)
if simoptions.parallel==0
    for ii=1:simoptions.numbersims
        seedpoint=seedpoints(ii,:);
        SimTimeSeriesKron=SimTimeSeriesIndexes_InfHorz_SemiEndog_raw(PolicyIndexesKron,N_d,N_a,N_z,cumsumpi_z_semiendog,0,seedpoint,simoptions.simperiods,0); % 0: burnin, 0: use single CPU
        
        SimPanel_ii=nan(l_a+l_z,simoptions.simperiods);
        
        for t=1:simoptions.simperiods
            temp=SimTimeSeriesKron(:,t);
            if ~isnan(temp)
                a_c_vec=ind2sub_homemade([n_a],temp(1));
                z_c_vec=ind2sub_homemade([n_z],temp(2));
                for kk=1:l_a
                    SimPanel_ii(kk,t)=a_c_vec(kk);
                end
                for kk=1:l_z
                    SimPanel_ii(l_a+kk,t)=z_c_vec(kk);
                end
            end
        end
        SimPanel(:,:,ii)=SimPanel_ii;
    end
else
    simperiods=simoptions.simperiods; % reduce overhead with parfor
    parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
        seedpoint=seedpoints(ii,:);
        SimTimeSeriesKron=SimTimeSeriesIndexes_InfHorz_SemiEndog_raw(PolicyIndexesKron,N_d,N_a,N_z,cumsumpi_z_semiendog,0,seedpoint,simperiods,0); % 0: burnin, 0: use single CPU
        
        SimPanel_ii=nan(l_a+l_z,simperiods);
        
        for t=1:simperiods
            temp=SimTimeSeriesKron(:,t);
            if ~isnan(temp)
                a_c_vec=ind2sub_homemade([n_a],temp(1));
                z_c_vec=ind2sub_homemade([n_z],temp(2));
                for kk=1:l_a
                    SimPanel_ii(kk,t)=a_c_vec(kk);
                end
                for kk=1:l_z
                    SimPanel_ii(l_a+kk,t)=z_c_vec(kk);
                end
            end
        end
        SimPanel(:,:,ii)=SimPanel_ii;
    end
end

if MoveOutputtoGPU==1
    SimPanel=gpuArray(SimPanel);
end

end



