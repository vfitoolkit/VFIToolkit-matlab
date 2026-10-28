function VPath=ValueFnFromPolicyOnTransPath_InfHorz(PolicyPath,V_final,ParamPath,PricePath,T,n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% No z is n_z=0 (z_grid=[], pi_z=[]). iid e shocks are given in vfoptions (n_e, e_grid, pi_e), as for
% ValueFnOnTransPath_InfHorz. PolicyPath is [rows,n_a,(n_z),(n_e),T], V_final is [n_a,(n_z),(n_e)], and the output
% VPath is [n_a,(n_z),(n_e),T], as from ValueFnOnTransPath_InfHorz.
% Without z, or with e, the z_grid, pi_z, e_grid and pi_e may vary over the path (as in ValueFnOnTransPath_InfHorz:
% pi_z(:,:,t) is the transition from t to t+1, pi_e(:,t) the distribution of the e realized IN period t).
% Semi-exogenous states are given in vfoptions (n_semiz, semiz_grid, SemiExoStateFn or pi_semiz, l_dsemiz), as for
% ValueFnOnTransPath_InfHorz; then PolicyPath is [rows,n_a,n_semiz,(n_z),(n_e),T] (rows d1, d2, aprime (, L2, L2flag)),
% V_final [n_a,n_semiz,(n_z),(n_e)] and VPath [n_a,n_semiz,(n_z),(n_e),T]; z_grid, pi_z, e_grid, pi_e and pi_semiz may
% vary over the path (pi_semiz(:,:,:,t), or SemiExoStateFn at the period-t ParamPath, is the transition from t to t+1).

if ~exist('vfoptions','var')
    vfoptions.gridinterplayer=0;
    % divide-and-conquer is not relevant for ValueFnFromPolicy
    vfoptions.n_e=0;
    vfoptions.n_semiz=0;
else
    if gpuDeviceCount==0
        error('ValueFnFromPolicyOnTransPath_InfHorz is only available on GPU')
    end
    if ~isfield(vfoptions,'gridinterplayer')
        vfoptions.gridinterplayer=0;
    end
    if isfield(vfoptions,'exoticpreferences') && ~strcmp(vfoptions.exoticpreferences,'None') % 'None' is the default that the other commands set, so a vfoptions passed on from them carries it
        error('ValueFnFromPolicyOnTransPath_InfHorz() does not yet work with exotic preferences. Please ask on forum if you want/need this feature. \n');
    end
    if isfield(vfoptions,'experienceasset')
        if vfoptions.experienceasset>=1
            error('ValueFnFromPolicyOnTransPath_InfHorz() does not yet work with experienceasset. Please ask on forum if you want/need this feature. \n');
        end
    end
    % divide-and-conquer is not relevant for ValueFnFromPolicy
    if ~isfield(vfoptions,'n_e')
        vfoptions.n_e=0;
    end
    if ~isfield(vfoptions,'n_semiz')
        vfoptions.n_semiz=0;
    end
end
if vfoptions.gridinterplayer==1
    if ~isfield(vfoptions,'ngridinterp')
        error('When using vfoptions.gridinterplayer=1 you must set vfoptions.ngridinterp')
    end
end

N_z=prod(n_z);
N_e=prod(vfoptions.n_e);
N_semiz=prod(vfoptions.n_semiz);

%% Semi-exogenous states (vfoptions.n_semiz), with/without z, with/without e (and with the grid interpolation layer)
% The last vfoptions.l_dsemiz decision variables (d2) drive the semiz transitions pi_semiz(semiz,semiz',d2), the
% exogenous states are (semiz,z,e), semiz fastest. The expectation is the one ValueFnOnTransPath_InfHorz takes in its
% SemiExo single steps: e' first, then z' conditional on z, then semiz' conditional on semiz and the policy d2, with NaN
% (from -Inf times a zero probability) set to zero after each; with the grid interpolation layer it is interpolated
% between the lower grid point and the one above it, with the second layer weights (a zero weight on a -Inf point
% contributes nothing), as in ValueFnFromPolicy_InfHorz_SemiExo. pi_z_T(:,:,t) and pi_semiz_T(:,:,:,t) are the
% transitions from t to t+1, pi_e_T(:,t) the distribution of the e realized IN period t (so V_t reads column t+1).
if N_semiz>0
    if n_d(1)==0
        error('ValueFnFromPolicyOnTransPath_InfHorz: semi-exogenous states (vfoptions.n_semiz) require a decision variable: the last vfoptions.l_dsemiz decision variables determine the semiz transitions')
    end
    N_a=prod(n_a);
    l_a=length(n_a);
    l_d=length(n_d);
    n_semiz=vfoptions.n_semiz;
    % Read here, because ExogShockSetup_InfHorz_TPath below removes n_e, e_grid and pi_e from vfoptions
    n_e=vfoptions.n_e;
    N_zr=max(N_z,1);
    N_er=max(N_e,1);
    N_zeonly=N_zr*N_er; % (z,e) alone, without semiz
    N_semizze=N_semiz*N_zeonly; % (semiz,z,e), semiz fastest
    if N_z==0 && N_e==0
        n_semizze=n_semiz;
    elseif N_z>0 && N_e==0
        n_semizze=[n_semiz,n_z];
    elseif N_z==0 && N_e>0
        n_semizze=[n_semiz,n_e];
    else % N_z>0 && N_e>0
        n_semizze=[n_semiz,n_z,n_e];
    end

    d_grid=gpuArray(d_grid);
    a_grid=gpuArray(a_grid);
    a_gridvals=CreateGridvals(n_a,a_grid,1);

    % Note: Internally PricePath is matrix of size T-by-'number of prices'.
    % ParamPath is matrix of size T-by-'number of parameters that change over the transition path'.
    [PricePath,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec]=PricePathParamPath_StructToMatrix(PricePath,ParamPath,T);
    PricePath=gpuArray(PricePath);

    % ReturnFn inputs are (d,aprime,a,semiz,z,e,...), so this needs vfoptions.n_semiz and vfoptions.n_e (read before the
    % shock setup removes n_e)
    ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,0,vfoptions,Parameters);

    % z and e: the setup sets transpathoptions.zpathtrivial/epathtrivial, and when z or e varies over the path the paths
    % are in transpathoptions.z_gridvals_T, pi_z_T, e_gridvals_T and pi_e_T (z_gridvals, pi_z, e_gridvals and pi_e
    % returned here are then only the t=1 placeholders)
    transpathoptions=struct();
    [z_gridvals, pi_z, ~, e_gridvals, pi_e, ~, ~, transpathoptions, vfoptions]=ExogShockSetup_InfHorz_TPath(n_z,z_grid,pi_z,Parameters,PricePathNames,ParamPath,ParamPathNames,ParamPathSizeVec,T,transpathoptions,vfoptions,3);
    % semiz: semiz_gridvals [N_semiz,l_semiz] (fixed over the path) and pi_semiz [N_semiz,N_semiz,N_dsemiz] (gpu); when
    % transpathoptions.semizpathtrivial=0 the path is in transpathoptions.pi_semiz_T [N_semiz,N_semiz,N_dsemiz,T]
    [semiz_gridvals, pi_semiz, transpathoptions, vfoptions]=SemiExogShockSetup_InfHorz_TPath(n_d,d_grid,Parameters,PricePathNames,ParamPath,ParamPathNames,ParamPathSizeVec,T,transpathoptions,vfoptions,3);
    l_dsemiz=vfoptions.setup_semiexo.l_d2;
    n_dsemiz=vfoptions.setup_semiexo.n_d2;

    % Policy values (the decision and next period endogenous state values) for every period at once
    PolicyPath=gpuArray(reshape(PolicyPath,[size(PolicyPath,1),N_a,N_semizze,T]));
    PolicyValuesPath=PolicyInd2Val_InfHorz_TPath(PolicyPath,n_d,n_a,n_semizze,T,d_grid,a_grid,vfoptions,1); % [l_d+l_a,N_a,N_semizze,T]
    l_daprime=size(PolicyValuesPath,1);

    % Policy indexes for every period: the joint index of d2 (the last l_dsemiz decision variable rows), and of aprime
    % (the lower grid point with the grid interpolation layer)
    PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a*N_semizze,T]);
    Policy_dsemizPath=reshape(PolicyPath(l_d-l_dsemiz+1,:,:),[N_a*N_semizze,T]);
    for ii=2:l_dsemiz
        Policy_dsemizPath=Policy_dsemizPath+prod(n_dsemiz(1:ii-1))*(reshape(PolicyPath(l_d-l_dsemiz+ii,:,:),[N_a*N_semizze,T])-1);
    end
    Policy_aprimePath=reshape(PolicyPath(l_d+1,:,:),[N_a*N_semizze,T]);
    for ii=2:l_a
        Policy_aprimePath=Policy_aprimePath+prod(n_a(1:ii-1))*(reshape(PolicyPath(l_d+ii,:,:),[N_a*N_semizze,T])-1);
    end
    if vfoptions.gridinterplayer==1
        % probability of the upper grid point, from the L2 index. The L2flag (the last row) is NOT applied: it is for the
        % agent distribution, the value of the policy uses the L2 weights as they are (as ValueFnFromPolicy_InfHorz_SemiExo)
        PolicyProbUpperPath=(reshape(PolicyPath(end-1,:,:),[N_a*N_semizze,T])-1)/(vfoptions.ngridinterp+1); % L2 index (end-1 because end is L2flag)
    end

    semizind=gpuArray(repmat(repelem((1:1:N_semiz)',N_a,1),N_zeonly,1)); % semiz of each state (a,semiz,z,e)
    zind=gpuArray(repmat(repelem((1:1:N_zr)',N_a*N_semiz,1),N_er,1)); % z of each state (a,semiz,z,e)

    VPath=zeros(N_a,N_semizze,T,'gpuArray');
    VPath(:,:,T)=reshape(V_final,[N_a,N_semizze]);

    for ttr=1:T-1
        tt=T-ttr; % T-1 to 1

        for kk=1:length(PricePathNames)
            Parameters.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
        end
        for kk=1:length(ParamPathNames)
            Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
        end

        % This period's grids, the transitions of z and semiz to next period, and NEXT period's e distribution
        if N_z>0 && transpathoptions.zpathtrivial==0
            z_gridvals=transpathoptions.z_gridvals_T(:,:,tt);
            pi_z=transpathoptions.pi_z_T(:,:,tt);
        end
        if N_e>0 && transpathoptions.epathtrivial==0
            e_gridvals=transpathoptions.e_gridvals_T(:,:,tt);
            pi_e=transpathoptions.pi_e_T(:,tt+1);
        end
        if transpathoptions.semizpathtrivial==0
            pi_semiz=transpathoptions.pi_semiz_T(:,:,:,tt);
        end

        %% Calculate FofPolicy (the return fn evaluated at the Policy), on the (semiz,z,e) grid, semiz fastest
        if N_z==0 && N_e==0
            semizze_gridvals=semiz_gridvals;
        else
            if N_z==0
                ze_gridvals=e_gridvals;
            elseif N_e==0
                ze_gridvals=z_gridvals;
            else
                ze_gridvals=[repmat(z_gridvals,N_e,1),repelem(e_gridvals,N_z,1)];
            end
            semizze_gridvals=[repmat(semiz_gridvals,N_zeonly,1),repelem(ze_gridvals,N_semiz,1)];
        end
        ReturnFnParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames);
        PolicyValuesPermute=permute(reshape(PolicyValuesPath(:,:,:,tt),[l_daprime,N_a,N_semizze]),[2,3,1]); %[N_a,N_semiz*N_z*N_e,l_d+l_a]
        FofPolicy=reshape(EvalFnOnAgentDist_Grid(ReturnFn, ReturnFnParamsCell,PolicyValuesPermute,l_daprime,n_a,n_semizze,a_gridvals,semizze_gridvals),[N_a*N_semizze,1]);

        DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames));

        %% The expectation of next period value fn over e' and z', as a function of (aprime,semizprime,z)
        EVz=reshape(VPath(:,:,tt+1),[N_a,N_semiz,N_zr,N_er]);
        if N_e>0
            EVz=EVz.*shiftdim(pi_e,-3); % [N_a,N_semiz,N_zr,N_e], times pi_e(e')
            EVz(isnan(EVz))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
            EVz=sum(EVz,4); % sum over e', [N_a,N_semiz,N_zr]
        end
        if N_z>0
            EVz=EVz.*shiftdim(pi_z',-2); % [N_a,N_semiz,N_zprime,N_z]
            EVz(isnan(EVz))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EVz=reshape(sum(EVz,3),[N_a,N_semiz,N_z]); % sum over z', aprime-by-semizprime-by-z
        end

        %% Evaluate at the policy aprime and d2, and take the expectation over semiz' (pi_semiz at the policy d2)
        Pdz=pi_semiz(semizind+N_semiz*gpuArray(0:1:N_semiz-1)+N_semiz*N_semiz*(Policy_dsemizPath(:,tt)-1)); % [N_a*N_semizze,N_semiz], pi_semiz(semiz,semizprime,d2) at the policy d2
        EVindex=Policy_aprimePath(:,tt)+N_a*gpuArray(0:1:N_semiz-1)+N_a*N_semiz*(zind-1); % [N_a*N_semizze,N_semiz], index for (aprime,semizprime,z) in EVz
        if vfoptions.gridinterplayer==0
            EVatPolicy=EVz(EVindex).*Pdz;
        else % vfoptions.gridinterplayer==1
            PolicyProbUpper=PolicyProbUpperPath(:,tt);
            EVlower=(1-PolicyProbUpper).*EVz(EVindex);
            EVlower(PolicyProbUpper==1,:)=0; % zero weight times -Inf gives NaN, a zero weight contributes nothing (the solver raws do the same)
            EVupper=PolicyProbUpper.*EVz(EVindex+1); % interpolation is on a1 only, so the upper point is one step up in a1
            EVupper(PolicyProbUpper==0,:)=0;
            EVatPolicy=(EVlower+EVupper).*Pdz;
        end
        EVatPolicy(isnan(EVatPolicy))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EVatPolicy=sum(EVatPolicy,2); % sum over semiz', [N_a*N_semizze,1]

        VPath(:,:,tt)=reshape(FofPolicy+DiscountFactorParamsVec*EVatPolicy,[N_a,N_semizze]);
    end

    if N_z==0 && N_e==0
        VPath=reshape(VPath,[n_a,n_semiz,T]);
    elseif N_z>0 && N_e==0
        VPath=reshape(VPath,[n_a,n_semiz,n_z,T]);
    elseif N_z==0 && N_e>0
        VPath=reshape(VPath,[n_a,n_semiz,n_e,T]);
    else % N_z>0 && N_e>0
        VPath=reshape(VPath,[n_a,n_semiz,n_z,n_e,T]);
    end
    return
end

%% Without semiz: with/without z, with/without e (with and without the grid interpolation layer)
% Every model without semiz comes here, including the plain z model. That used to have a branch of its own (and
% ValueFnFromPolicyOnTransPath_InfHorz_GI for the grid interpolation layer), which read a fixed z_grid and pi_z, so
% with z varying over the path it evaluated every period at the period-1 process and could not reproduce VPath;
% it also read the path prices and parameters as PricePath(t,kk), which is wrong as soon as one is vector-valued.
% The expectation is the one ValueFnOnTransPath_InfHorz takes in its single steps: e' first, then z' conditional
% on z, with NaN (from -Inf times a zero probability) set to zero. With the grid interpolation layer the expectation
% is then interpolated between the lower grid point and the one above it, with the second layer weights (a zero
% weight on a -Inf point again gives NaN, set to zero).
if N_semiz==0
    N_d=prod(n_d);
    N_a=prod(n_a);
    l_a=length(n_a);
    % Read here, because ExogShockSetup_InfHorz_TPath below removes n_e, e_grid and pi_e from vfoptions
    n_e=vfoptions.n_e;
    % (z,e) are kept as one exogenous dimension, z first, which is a singleton when there is neither
    N_ze=max(N_z,1)*max(N_e,1);
    if N_z==0 && N_e==0
        n_ze=0;
    elseif N_z==0 && N_e>0
        n_ze=n_e;
    elseif N_z>0 && N_e==0
        n_ze=n_z;
    else % N_z>0 && N_e>0
        n_ze=[n_z,n_e];
    end

    d_grid=gpuArray(d_grid);
    a_grid=gpuArray(a_grid);
    a_gridvals=CreateGridvals(n_a,a_grid,1);

    % Note: Internally PricePath is matrix of size T-by-'number of prices'.
    % ParamPath is matrix of size T-by-'number of parameters that change over the transition path'.
    [PricePath,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec]=PricePathParamPath_StructToMatrix(PricePath,ParamPath,T);
    PricePath=gpuArray(PricePath);

    % ReturnFn inputs are (d,aprime,a,z,e,...), so this needs vfoptions.n_e (read before the shock setup removes it)
    ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,0,vfoptions,Parameters);

    % The setup sets transpathoptions.zpathtrivial/epathtrivial, and when z or e varies over the path the paths are
    % in transpathoptions.z_gridvals_T, pi_z_T, e_gridvals_T and pi_e_T (z_gridvals, pi_z, e_gridvals and pi_e
    % returned here are then only the t=1 placeholders)
    transpathoptions=struct();
    [z_gridvals, pi_z, ~, e_gridvals, pi_e, ~, ~, transpathoptions, vfoptions]=ExogShockSetup_InfHorz_TPath(n_z,z_grid,pi_z,Parameters,PricePathNames,ParamPath,ParamPathNames,ParamPathSizeVec,T,transpathoptions,vfoptions,3);

    % Policy values (the decision and next period endogenous state values) for every period at once
    PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_ze,T]);
    PolicyValuesPath=PolicyInd2Val_InfHorz_TPath(PolicyPath,n_d,n_a,n_ze,T,d_grid,a_grid,vfoptions,1); % [l_d+l_a,N_a,N_ze,T]
    l_daprime=size(PolicyValuesPath,1);

    % Row of a (a1 with the grid interpolation layer) in the Kron'd policy (comes after d, when there is a d)
    if N_d==0
        index_a1=1;
    else
        index_a1=2;
    end

    VPath=zeros(N_a,N_ze,T,'gpuArray');
    VPath(:,:,T)=reshape(V_final,[N_a,N_ze]);

    for ttr=1:T-1
        tt=T-ttr; % T-1 to 1

        % The size vectors give each price/parameter's columns, as a price or parameter can be vector-valued
        for kk=1:length(PricePathNames)
            Parameters.(PricePathNames{kk})=PricePath(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
        end
        for kk=1:length(ParamPathNames)
            Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
        end

        % This period's grids, the transition of z to next period, and NEXT period's e distribution (pi_e_T(:,t) is
        % the distribution of the e realized IN period t, and V_t takes its expectation over e_{t+1})
        if N_z>0 && transpathoptions.zpathtrivial==0
            z_gridvals=transpathoptions.z_gridvals_T(:,:,tt);
            pi_z=transpathoptions.pi_z_T(:,:,tt);
        end
        if N_e>0 && transpathoptions.epathtrivial==0
            e_gridvals=transpathoptions.e_gridvals_T(:,:,tt);
            pi_e=transpathoptions.pi_e_T(:,tt+1);
        end

        %% Calculate FofPolicy (the return fn evaluated at the Policy)
        ReturnFnParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames);
        if N_z==0 && N_e==0
            PolicyValuesPermute=permute(reshape(PolicyValuesPath(:,:,:,tt),[l_daprime,N_a]),[2,1]); %[N_a,l_d+l_a]
            FofPolicy=reshape(EvalFnOnAgentDist_Grid(ReturnFn, ReturnFnParamsCell,PolicyValuesPermute,l_daprime,n_a,0,a_gridvals,[]),[N_a,1]);
        else
            % To evaluate the return fn, just treat e as part of z
            if N_z==0
                ze_gridvals=e_gridvals;
            elseif N_e==0
                ze_gridvals=z_gridvals;
            else
                ze_gridvals=[repmat(z_gridvals,N_e,1),repelem(e_gridvals,N_z,1)];
            end
            PolicyValuesPermute=permute(reshape(PolicyValuesPath(:,:,:,tt),[l_daprime,N_a,N_ze]),[2,3,1]); %[N_a,N_z*N_e,l_d+l_a]
            FofPolicy=reshape(EvalFnOnAgentDist_Grid(ReturnFn, ReturnFnParamsCell,PolicyValuesPermute,l_daprime,n_a,n_ze,a_gridvals,ze_gridvals),[N_a,N_ze]);
        end

        DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames));

        %% The expectation of next period value fn, as a function of (aprime,z)
        Vnext=VPath(:,:,tt+1);
        if N_z==0 && N_e==0
            EV=Vnext; % [N_a,1], no exogenous state so nothing to take expectations over
        elseif N_z==0 && N_e>0
            EV=Vnext.*shiftdim(pi_e,-1); % [N_a,N_e], aprime-by-eprime
            EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
            EV=sum(EV,2); % sum over e', [N_a,1]
        elseif N_z>0 && N_e==0
            EV=Vnext.*shiftdim(pi_z',-1); % [N_a,N_zprime,N_z]
            EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=reshape(sum(EV,2),[N_a,N_z]); % sum over z', aprime-by-z
        else % N_z>0 && N_e>0
            EV=reshape(Vnext,[N_a,N_z,N_e]).*shiftdim(pi_e,-2); % [N_a,N_z,N_e], aprime-by-zprime-by-eprime
            EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
            EV=sum(EV,3); % sum over e', [N_a,N_z], aprime-by-zprime
            EV=EV.*shiftdim(pi_z',-1); % [N_a,N_zprime,N_z]
            EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=reshape(sum(EV,2),[N_a,N_z]); % sum over z', aprime-by-z
        end

        %% Evaluate EV at the policy, and calculate V
        % Kron the policy, e passed as part of z (KronPolicyIndexes_forValueFnFromPolicy only uses the number of
        % exogenous states). It drops the L2flag, which plays no part in the value (it only moves the agent
        % distribution off a -Inf return point)
        Policy=KronPolicyIndexes_forValueFnFromPolicy(PolicyPath(:,:,:,tt), n_d, n_a, N_ze, 0, vfoptions);
        if vfoptions.gridinterplayer==0
            Policy_a=reshape(ceil(Policy(index_a1,:,:)),[N_a,N_ze]); % aprime index
            if N_z>0
                Policy_a=Policy_a+repmat(N_a*(0:1:N_z-1),1,max(N_e,1)); % add the index for z (for indexing EV, which is aprime-by-z), z varies first
            end
            EVatPolicy=EV(Policy_a); % [N_a,N_ze]
        else % vfoptions.gridinterplayer==1
            alowerindex=reshape(ceil(Policy(index_a1,:,:)),[1,N_a*N_ze]); % a1 lower grid point, 1 to n_a(1)-1
            if l_a>=2
                % GI2A: add the offset for the remaining endogenous state(s), which Kron returns separately.
                % Interpolation is on a1 only, so the upper point (alowerindex+1) steps one point in a1.
                alowerindex=alowerindex+n_a(1)*(reshape(ceil(Policy(index_a1+1,:,:)),[1,N_a*N_ze])-1);
            end
            aprimeindex=[alowerindex; alowerindex+1]; % [2,N_a*N_ze]
            if N_z>0
                aprimeindex=aprimeindex+reshape(repmat(N_a*(0:1:N_z-1),N_a,1,max(N_e,1)),[1,N_a*N_ze]); % add the index for z (for indexing EV, which is aprime-by-z)
            end
            PolicyProbs=reshape(ceil(Policy(end,:,:)),[1,N_a*N_ze]); % L2 (Kron drops L2flag)
            PolicyProbs=(PolicyProbs-1)/(vfoptions.ngridinterp+1); % prob of upper point
            PolicyProbs=[1-PolicyProbs; PolicyProbs]; % [2,N_a*N_ze]

            EVatPolicy=reshape(EV(aprimeindex),[2,N_a*N_ze]);
            EVatPolicy=PolicyProbs.*EVatPolicy;
            EVatPolicy(isnan(EVatPolicy))=0; % a zero interpolation weight on a grid point where EV is -Inf gives 0*(-Inf)=NaN, it should contribute nothing
            EVatPolicy=reshape(sum(EVatPolicy,1),[N_a,N_ze]);
        end

        VPath(:,:,tt)=FofPolicy+DiscountFactorParamsVec*EVatPolicy;
    end

    if N_z==0 && N_e==0
        VPath=reshape(VPath,[n_a,T]);
    elseif N_z==0 && N_e>0
        VPath=reshape(VPath,[n_a,n_e,T]);
    elseif N_z>0 && N_e==0
        VPath=reshape(VPath,[n_a,n_z,T]);
    else % N_z>0 && N_e>0
        VPath=reshape(VPath,[n_a,n_z,n_e,T]);
    end
    return
end


end
