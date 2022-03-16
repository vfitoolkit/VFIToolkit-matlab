function [SimPanel,PolicyIndexesKron]=SimPanelIndexes_TransPath_Case1(PricePath, PricePathNames, ParamPath, ParamPathNames, T, V_final, AgentDist_initial, n_d, n_a, n_z, pi_z, d_grid,a_grid,z_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, transpathoptions,simoptions)
% Intended for internal use, not by user.
%
% Simulates a panel based on PolicyIndexes of 'numbersims' agents of length
% 'simperiods' beginning from randomly drawn AgentDist_initial. (If you use the
% newbirths option you will get more than 'numbersims', due to the extra births)
%
% AgentDist_initial can be inputed as over the finite time-horizon (j), or
% without a time-horizon in which case it is assumed to be an AgentDist_initial
% for time j=1. (So AgentDist_initial is n_a-by-n_z)

N_a=prod(n_a);
N_z=prod(n_z);
N_d=prod(n_d);

%% Check which simoptions have been declared, set all others to defaults 
if exist('simoptions','var')==1
    %Check simoptions for missing fields, if there are some fill them with
    %the defaults
    if ~isfield(simoptions, 'polindorval')
        simoptions.polindorval=1;
    end
    if ~isfield(simoptions, 'simperiods')
        simoptions.simperiods=T; % Have left it as an option so you can do shorter if wanted
    end
    if ~isfield(simoptions, 'numbersims')
        simoptions.numbersims=10^3;
    end
    if ~isfield(simoptions, 'parallel')
        simoptions.parallel=2;
    end
    if ~isfield(simoptions, 'verbose')
        simoptions.verbose=0;
    end
    simoptions.newbirths=0; % It is assumed you do not want to add 'new births' to panel as you go. If you do you just tell it the 'birstdist' (sometimes just the same as InitialDist, but not often)
    if isfield(simoptions,'birthdist')
        simoptions.newbirths=1;
        % if you input simoptions.birthdist, you must also input
        % simoptions.birthrate (can be scalar, or vector of length
        % simoptions.simperiods)
        % I do not allow for the birthdist to change over time, only the
        % birthrate.
    end
else
    %If simoptions is not given, just use all the defaults
    simoptions.polindorval=1;
    simoptions.simperiods=T;
    simoptions.numbersims=10^3;
    simoptions.parallel=2;
    simoptions.verbose=0;
    simoptions.newbirths=0;
end

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

%%
unkronoptions.parallel=2;

beta=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames)); % It is possible but unusual with infinite horizon that there is more than one discount factor and that these should be multiplied together
IndexesForPathParamsInDiscountFactor=CreateParamVectorIndexes(DiscountFactorParamNames, ParamPathNames);
ReturnFnParamsVec=gpuArray(CreateVectorFromParams(Parameters, ReturnFnParamNames));
[IndexesForPricePathInReturnFnParams, IndexesPricePathUsedInReturnFn]=CreateParamVectorIndexes(ReturnFnParamNames, PricePathNames);
[IndexesForPathParamsInReturnFnParams, IndexesParamPathUsedInReturnFn]=CreateParamVectorIndexes(ReturnFnParamNames, ParamPathNames);

% Transition paths do not currently allow for the exogenous shock process to differ based on time period.
fieldexists_ExogShockFn=0;

%%
V_final=reshape(V_final,[N_a,N_z]);
AgentDist_initial=reshape(AgentDist_initial,[N_a*N_z,1]);
V=zeros(size(V_final),'gpuArray');
Policy=zeros(N_a,N_z,'gpuArray');

PolicyIndexesPath=zeros(N_a,N_z,T,'gpuArray'); %Periods 1 to T-1

%First, go from T-1 to 1 calculating the Value function and Optimal
%policy function at each step. Since we won't need to keep the value
%functions for anything later we just store the next period one in
%Vnext, and the current period one to be calculated in V
Vnext=V_final;
for ttr=0:T-1 %so tt=T-ttr
    
    if ~isnan(IndexesForPathParamsInDiscountFactor)
        beta(IndexesForPathParamsInDiscountFactor)=ParamPath(T-ttr,:); % This step could be moved outside all the loops
    end
    if ~isnan(IndexesForPricePathInReturnFnParams)
        ReturnFnParamsVec(IndexesForPricePathInReturnFnParams)=PricePath(T-ttr,IndexesPricePathUsedInReturnFn);
    end
    if ~isnan(IndexesForPathParamsInReturnFnParams)
        ReturnFnParamsVec(IndexesForPathParamsInReturnFnParams)=ParamPath(T-ttr,IndexesParamPathUsedInReturnFn); % This step could be moved outside all the loops by using BigReturnFnParamsVec idea
    end
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid,ReturnFnParamsVec);
    
    for z_c=1:N_z
        ReturnMatrix_z=ReturnMatrix(:,:,z_c);
        %Calc the condl expectation term (except beta), which depends on z but
        %not on control variables
        EV_z=Vnext.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);
        
        entireEV_z=kron(EV_z,ones(N_d,1));
        entireRHS=ReturnMatrix_z+beta*entireEV_z*ones(1,N_a,1);
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,z_c)=Vtemp;
        Policy(:,z_c)=maxindex;
        
    end
    
    PolicyIndexesPath(:,:,T-ttr)=Policy;
    Vnext=V;
end
% Free up space on GPU by deleting things no longer needed
clear ReturnMatrix ReturnMatrix_z entireRHS entireEV_z EV_z maxindex V Vnext V_final


% Get seedpoints from InitialDist while on gpu
seedpoints=nan(simoptions.numbersims,2,'gpuArray'); % 2 as a,z (vectorized)
cumsumInitialDistVec=cumsum(reshape(AgentDist_initial,[N_a*N_z,1]));
[~,seedpointvec]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims,1,'gpuArray'));
for ii=1:simoptions.numbersims
    seedpoints(ii,:)=ind2sub_homemade_gpu([N_a,N_z],seedpointvec(ii));
end
seedpoints=floor(seedpoints); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.

cumsumpi_z=cumsum(pi_z,2);

PolicyIndexesKron=zeros(2,N_a,N_z,T,'gpuArray'); %Periods 1 to T
PolicyIndexesKron(2,:,:,:)=ceil(shiftdim(PolicyIndexesPath,-1)./N_d);
% We will never use PolicyIndexesKron(1,:,:,:), so just leave it as zeros
% for speed, if we did want it then it would be. Not true for TransPath, it
% gets returned as an output as it is needed to evaluate the functions when calculating values from functions.
PolicyIndexesKron(1,:,:,:)=rem(shiftdim(PolicyIndexesPath,-1),N_d);
% PolicyIndexesKron(1,:,:,:)=shiftdim(PolicyIndexesPath,-1)-N_d*(PolicyIndexesKron(2,:,:,:)-1); % This seems slower than rem, but they are essentially same runtime
% WOULD PROBABLY BE BETTER TO JUST INCLUDE d VARIABLE IN SimPanel (the
% indexes) and then return that. But not inclined to check this out just now.

MoveOutputtoGPU=0;
if simoptions.parallel==2
    % Simulation on GPU is really slow. So instead, switch to CPU, and then switch
    % back. For anything but ridiculously short simulations it is more than worth the overhead.
    PolicyIndexesKron=gather(PolicyIndexesKron);
    cumsumpi_z=gather(cumsumpi_z);
    seedpoints=gather(seedpoints);
    MoveOutputtoGPU=1;
    simoptions.simperiods=gather(simoptions.simperiods);
end

include_daprime=1;
SimPanel=zeros(l_d+l_a+l_a+l_z,simoptions.simperiods,simoptions.numbersims); % (a,z)
if simoptions.parallel==0
    for ii=1:simoptions.numbersims
        seedpoint=seedpoints(ii,:);
        % Since a finite-horizon value fn problem and a transition path are
        % much the same thing we can just piggy back on the codes for FHorz.
        SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_raw(PolicyIndexesKron,N_d,N_a,N_z,T,cumsumpi_z, [seedpoint,1], simoptions.simperiods,fieldexists_ExogShockFn,include_daprime);
                
        SimPanel(1:l_d,:,ii)=ind2sub_vec_homemadet(n_d, SimLifeCycleKron(1,:)); % d
        SimPanel(l_d+1:l_d+l_a,:,ii)=ind2sub_vec_homemadet(n_a, SimLifeCycleKron(2,:)); % aprime
        SimPanel(l_d+l_a+1:l_d+l_a+l_a,:,ii)=ind2sub_vec_homemadet(n_a, SimLifeCycleKron(3,:)); % a
        SimPanel(l_d+l_a+l_a+1:l_d+l_a+l_a+l_z,:,ii)=ind2sub_vec_homemadet(n_d, SimLifeCycleKron(4,:)); % z
    end
else
    parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
        seedpoint=seedpoints(ii,:);
        SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_raw(PolicyIndexesKron,N_d,N_a,N_z,T,cumsumpi_z, [seedpoint,1], simoptions.simperiods,fieldexists_ExogShockFn,include_daprime);
        
        SimPanel_ii=zeros(l_d+l_a+l_a+l_z,simoptions.simperiods);
         
        SimPanel_ii(1:l_d,:)=ind2sub_vec_homemadet(n_d, SimLifeCycleKron(1,:)); % d
        SimPanel_ii(l_d+1:l_d+l_a,:)=ind2sub_vec_homemadet(n_a, SimLifeCycleKron(2,:)); % aprime
        SimPanel_ii(l_d+l_a+1:l_d+l_a+l_a,:)=ind2sub_vec_homemadet(n_a, SimLifeCycleKron(3,:)); % a
        SimPanel_ii(l_d+l_a+l_a+1:l_d+l_a+l_a+l_z,:)=ind2sub_vec_homemadet(n_d, SimLifeCycleKron(4,:)); % z

        SimPanel(:,:,ii)=SimPanel_ii;
    end
end


if MoveOutputtoGPU==1
    SimPanel=gpuArray(SimPanel);
    PolicyIndexesKron=gpuArray(PolicyIndexesKron);
end

end



