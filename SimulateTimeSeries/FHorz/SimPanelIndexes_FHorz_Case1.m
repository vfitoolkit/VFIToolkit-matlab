function SimPanel=SimPanelIndexes_FHorz_Case1(InitialDist,Policy,n_d,n_a,n_z,N_j,pi_z, simoptions)
% Simulates a panel based on PolicyIndexes of 'numbersims' agents of length
% 'simperiods' beginning from randomly drawn InitialDist.
%
% InitialDist can be inputed as over the finite time-horizon (j), or
% without a time-horizon in which case it is assumed to be an InitialDist
% for time j=1. (So InitialDist is either n_a-by-n_z-by-n_j, or n_a-by-n_z)

N_a=prod(n_a);
N_z=prod(n_z);
N_d=prod(n_d);

%% Check which simoptions have been declared, set all others to defaults 
if exist('simoptions','var')==1
    %Check simoptions for missing fields, if there are some fill them with
    %the defaults
    eval('fieldexists=1;simoptions.polindorval;','fieldexists=0;')
    if fieldexists==0
        simoptions.polindorval=1;
    end
    eval('fieldexists=1;simoptions.simperiods;','fieldexists=0;')
    if fieldexists==0
        simoptions.simperiods=N_j;
    end
    eval('fieldexists=1;simoptions.numbersims;','fieldexists=0;')
    if fieldexists==0
        simoptions.numbersims=10^3;
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
    simoptions.simperiods=N_j;
    simoptions.numbersims=10^3;
    simoptions.parallel=2;
    simoptions.verbose=0;
end

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

% Check if the inputted Policy is already in form of PolicyIndexesKron. If
% so this saves a big chunk of the run time of 'SimPanelIndexes_FHorz_Case1',
% Since this command is often called as a subcommand by functions where
% PolicyIndexesKron it saves a lot of run time.
%Policy is [l_d+l_a,n_a,n_z,N_j]
if (l_d==0 && ndims(Policy)==3) || ndims(Policy)==4
%     disp('Policy is alread Kron')
    PolicyIndexesKron=Policy;
else %    size(Policy)==[l_d+l_a,n_a,n_z,N_j]
    PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z, N_j,simoptions);
end

if simoptions.parallel==2
    % Get seedpoints from InitialDist while on gpu
    seedpoints=nan(simoptions.numbersims,3,'gpuArray'); % 3 as a,z,j (vectorized)
    if numel(InitialDist)==N_a*N_z % Has just been given for age j=1
        cumsumInitialDistVec=cumsum(reshape(InitialDist,[N_a*N_z,1]));
        [~,seedpointvec]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims,1,'gpuArray'));
        for ii=1:simoptions.numbersims
            seedpoints(ii,:)=[ind2sub_homemade_gpu([N_a,N_z],seedpointvec(ii)),1];
        end
    else % Distribution across ages as well
        cumsumInitialDistVec=cumsum(reshape(InitialDist,[N_a*N_z*N_j,1]));
        [~,seedpointvec]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims,1,'gpuArray'));
        for ii=1:simoptions.numbersims
            seedpoints(ii,:)=ind2sub_homemade_gpu([N_a,N_z,N_j],seedpointvec(ii));
        end
    end
    seedpoints=floor(seedpoints); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.
else
    % Get seedpoints from InitialDist while on gpu
    seedpoints=nan(simoptions.numbersims,3); % 3 as a,z,j (vectorized)
    if numel(InitialDist)==N_a*N_z % Has just been given for age j=1
        cumsumInitialDistVec=cumsum(reshape(InitialDist,[N_a*N_z,1]));
        [~,seedpointvec]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims,1));
        for ii=1:simoptions.numbersims
            seedpoints(ii,:)=[ind2sub_homemade([N_a,N_z],seedpointvec(ii)),1];
        end
    else % Distribution across ages as well
        cumsumInitialDistVec=cumsum(reshape(InitialDist,[N_a*N_z*N_j,1]));
        [~,seedpointvec]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims,1));
        for ii=1:simoptions.numbersims
            seedpoints(ii,:)=ind2sub_homemade([N_a,N_z,N_j],seedpointvec(ii));
        end
    end
    seedpoints=floor(seedpoints); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.
end

eval('fieldexists_ExogShockFn=1;simoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;simoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')
if fieldexists_ExogShockFn==1
    cumsumpi_z=nan(N_z,N_z,N_j);
    for jj=1:N_j
        if fieldexists_ExogShockFnParamNames==1
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
            [~,pi_z_jj]=simoptions.ExogShockFn(ExogShockFnParamsVec);
        else
            [~,pi_z_jj]=simoptions.ExogShockFn(jj);
        end
        cumsumpi_z(:,:,jj)=gather(cumsum(pi_z_jj,2));
    end
else
    cumsumpi_z=cumsum(pi_z,2);
end

MoveOutputtoGPU=0;
if simoptions.parallel==2
    % Simulation on GPU is really slow. So instead, switch to CPU, and then switch
    % back. For anything but ridiculously short simulations it is more than worth the overhead.
    PolicyIndexesKron=gather(PolicyIndexesKron);
    cumsumpi_z=gather(cumsumpi_z);
    seedpoints=gather(seedpoints);
    MoveOutputtoGPU=1;
end

SimPanel=nan(l_a+l_z+1,simoptions.simperiods,simoptions.numbersims); % (a,z,j)
for ii=1:simoptions.numbersims
    seedpoint=seedpoints(ii,:);
    SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_raw(PolicyIndexesKron,N_d,N_a,N_z,N_j,cumsumpi_z, seedpoint, simoptions.simperiods,fieldexists_ExogShockFn);
    
    SimPanel_ii=nan(l_a+l_z+1,simoptions.simperiods);

    j1=seedpoint(3);
    j2=min(N_j,j1+simoptions.simperiods);
    for t=1:(j2-j1+1)
        jj=t+j1-1;
        temp=SimLifeCycleKron(:,jj);
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
        SimPanel_ii(l_a+l_z+1,t)=jj;
    end
    SimPanel(:,:,ii)=SimPanel_ii;
end

if MoveOutputtoGPU==1
    SimPanel=gpuArray(SimPanel);
end


end



