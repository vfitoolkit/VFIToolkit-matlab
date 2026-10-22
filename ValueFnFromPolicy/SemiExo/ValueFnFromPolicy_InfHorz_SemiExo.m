function V=ValueFnFromPolicy_InfHorz_SemiExo(Policy,n_d,n_a,n_z,d_grid,a_grid,z_gridvals,pi_z,ReturnFn,Parameters,DiscountFactorParamNames,ReturnFnParamNames,vfoptions)
% Value fn from policy, infinite horizon, with semi-exogenous states semiz.
% The last vfoptions.l_dsemiz decision variables (d2) determine the transition probabilities of semiz:
% vfoptions.pi_semiz(semiz,semizprime,d2). The exogenous states are (semiz,z), with semiz varying first, and then
% the iid e if there is one (vfoptions.n_e).
% Policy is [d (of which the last l_dsemiz are d2); aprime], over (a,semiz,z,e)
% With the grid interpolation layer, Policy also has the L2 index and L2flag rows at the end, and
% next period is the two aprime points (the lower grid index and the one above it) with the L2 weights

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);
l_a=length(n_a);
if N_d==0
    l_d=0;
else
    l_d=length(n_d);
end

%% Semi-exogenous states
vfoptions=SemiExogShockSetup_InfHorz(n_d,d_grid,Parameters,vfoptions,3);
% output: vfoptions.semiz_gridvals, vfoptions.pi_semiz, vfoptions.l_dsemiz
n_semiz=vfoptions.n_semiz;
N_semiz=prod(n_semiz);
l_dsemiz=vfoptions.l_dsemiz;
n_dsemiz=n_d(end-l_dsemiz+1:end);

if N_z==0
    n_bothz=n_semiz;
    bothz_gridvals=vfoptions.semiz_gridvals;
else
    n_bothz=[n_semiz,n_z];
    bothz_gridvals=[repmat(vfoptions.semiz_gridvals,N_z,1),repelem(z_gridvals,N_semiz,1)];
end
N_bothz=prod(n_bothz);
% With e, the functions are evaluated on (semiz,z,e)
if N_e==0
    n_allz=n_bothz;
    allz_gridvals=bothz_gridvals;
else
    n_allz=[n_bothz,vfoptions.n_e];
    allz_gridvals=[repmat(bothz_gridvals,N_e,1),repelem(vfoptions.e_gridvals,N_bothz,1)];
end
N_allz=prod(n_allz);

a_gridvals=CreateGridvals(n_a,a_grid,1);

%% Calculate FofPolicy (the return fn evaluated at the Policy)
PolicyValues=PolicyInd2Val_InfHorz(Policy,n_d,n_a,n_z,d_grid,a_grid, vfoptions); % (PolicyInd2Val_InfHorz treats semiz and e as part of z)
PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a,N_allz]),[2,3,1]); %[N_a,N_semiz*N_z*N_e,l_d+l_a]
ReturnFnParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames);
FofPolicy=EvalFnOnAgentDist_Grid(ReturnFn, ReturnFnParamsCell,PolicyValuesPermute,l_d+l_a,n_a,n_allz,a_gridvals,allz_gridvals);
FofPolicy=reshape(FofPolicy,[N_a*N_allz,1]);

%% Policy indexes: the (joint) index of d2, and of aprime
Policy=reshape(Policy,[size(Policy,1),N_a*N_allz]);
Policy_dsemiz=Policy(l_d-l_dsemiz+1,:)';
for ii=2:l_dsemiz
    Policy_dsemiz=Policy_dsemiz+prod(n_dsemiz(1:ii-1))*(Policy(l_d-l_dsemiz+ii,:)'-1);
end
Policy_aprime=Policy(l_d+1,:)';
for ii=2:l_a
    Policy_aprime=Policy_aprime+prod(n_a(1:ii-1))*(Policy(l_d+ii,:)'-1);
end

if vfoptions.gridinterplayer==1
    % probability of the upper grid point, from the L2 index. The L2flag (the last row) is NOT applied here: it is for the agent
    % distribution, which must not put mass on a coarse node where the return function is -Inf. The value of the policy is the
    % one the solver computed, whose continuation uses the L2 weights as they are, so this matches ValueFnFromPolicy_InfHorz_GI
    % (where the flag row is dropped) and the solver. Applying the flag moved the continuation to one node wherever the other
    % coarse node was unaffordable from the chosen interior point, which on a coarsely spaced grid is most of the poor states.
    L2index=Policy(end-1,:)'; % L2 index (end-1 because end is L2flag)
    PolicyProbUpper=gpuArray((L2index-1)/(vfoptions.ngridinterp+1)); % [N_a*N_allz,1]
end

semizind=gpuArray(repmat(repelem((1:1:N_semiz)',N_a,1),max(N_z,1)*max(N_e,1),1)); % semiz of each state (a,semiz,z,e)
Pdz=vfoptions.pi_semiz(semizind+N_semiz*gpuArray(0:1:N_semiz-1)+N_semiz*N_semiz*(gpuArray(Policy_dsemiz)-1)); % [N_a*N_allz,N_semiz], pi_semiz(semiz,semizprime,d2) at the policy d2

%% Now that we have FofPolicy, calculate V.
DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames));

currdist=Inf;
itercount=1;
VKron=FofPolicy/(1-DiscountFactorParamsVec); % rough guess

if vfoptions.gridinterplayer==1 && N_e>0 && N_z==0
    pi_e_alt=shiftdim(vfoptions.pi_e,-2); % 1-by-1-by-eprime
    EVindex=gpuArray(Policy_aprime)+N_a*gpuArray(0:1:N_semiz-1); % [N_a*N_semiz*N_e,N_semiz], index for (aprime lower,semizprime) in EVe

    while currdist>vfoptions.tolerance && itercount<vfoptions.maxiter
        VKronold=VKron;

        % First the expectation over eprime (iid), then the two aprime points with the L2 weights, then over semizprime
        % conditional on semiz and the policy d2
        EVe=reshape(VKron,[N_a,N_semiz,N_e]).*pi_e_alt;
        EVe(isnan(EVe))=0;
        EVe=sum(EVe,3); % aprime-by-semizprime
        EVlower=(1-PolicyProbUpper).*EVe(EVindex);
        EVlower(PolicyProbUpper==1,:)=0; % zero weight times -Inf gives NaN, a zero weight contributes nothing (the solver raws do the same)
        EVupper=PolicyProbUpper.*EVe(EVindex+1);
        EVupper(PolicyProbUpper==0,:)=0;
        EVKrontemp=(EVlower+EVupper).*Pdz;
        EVKrontemp(isnan(EVKrontemp))=0; % now only Pdz==0 against -Inf
        EVKrontemp=sum(EVKrontemp,2); % sum over semizprime
        VKron=FofPolicy+DiscountFactorParamsVec*EVKrontemp;

        currdist=max(abs(VKron(:)-VKronold(:)));
        itercount=itercount+1;
    end

    V=reshape(VKron,[n_a,n_semiz,vfoptions.n_e]);
elseif vfoptions.gridinterplayer==1 && N_e>0
    pi_e_alt=shiftdim(vfoptions.pi_e,-3); % 1-by-1-by-1-by-eprime
    pi_z_alt=shiftdim(pi_z',-2); % 1-by-1-by-zprime-by-z
    zind=gpuArray(repmat(repelem((1:1:N_z)',N_a*N_semiz,1),N_e,1)); % z of each state (a,semiz,z,e)
    EVindex=gpuArray(Policy_aprime)+N_a*gpuArray(0:1:N_semiz-1)+N_a*N_semiz*(zind-1); % [N_a*N_allz,N_semiz], index for (aprime lower,semizprime,z) in EVz

    while currdist>vfoptions.tolerance && itercount<vfoptions.maxiter
        VKronold=VKron;

        % First the expectation over eprime (iid), then over zprime, then the two aprime points with the L2 weights, then
        % over semizprime conditional on semiz and the policy d2
        EVe=reshape(VKron,[N_a,N_semiz,N_z,N_e]).*pi_e_alt;
        EVe(isnan(EVe))=0;
        EVe=sum(EVe,4); % aprime-by-semizprime-by-zprime
        EVz=EVe.*pi_z_alt;
        EVz(isnan(EVz))=0;
        EVz=reshape(sum(EVz,3),[N_a,N_semiz,N_z]); % aprime-by-semizprime-by-z
        EVlower=(1-PolicyProbUpper).*EVz(EVindex);
        EVlower(PolicyProbUpper==1,:)=0; % zero weight times -Inf gives NaN, a zero weight contributes nothing (the solver raws do the same)
        EVupper=PolicyProbUpper.*EVz(EVindex+1);
        EVupper(PolicyProbUpper==0,:)=0;
        EVKrontemp=(EVlower+EVupper).*Pdz;
        EVKrontemp(isnan(EVKrontemp))=0; % now only Pdz==0 against -Inf
        EVKrontemp=sum(EVKrontemp,2); % sum over semizprime
        VKron=FofPolicy+DiscountFactorParamsVec*EVKrontemp;

        currdist=max(abs(VKron(:)-VKronold(:)));
        itercount=itercount+1;
    end

    V=reshape(VKron,[n_a,n_semiz,n_z,vfoptions.n_e]);
elseif N_e>0 && N_z==0
    pi_e_alt=shiftdim(vfoptions.pi_e,-2); % 1-by-1-by-eprime
    EVindex=gpuArray(Policy_aprime)+N_a*gpuArray(0:1:N_semiz-1); % [N_a*N_semiz*N_e,N_semiz], index for (aprime,semizprime) in EVe

    while currdist>vfoptions.tolerance && itercount<vfoptions.maxiter
        VKronold=VKron;

        % First the expectation over eprime (iid), then over semizprime conditional on semiz and the policy d2
        EVe=reshape(VKron,[N_a,N_semiz,N_e]).*pi_e_alt;
        EVe(isnan(EVe))=0;
        EVe=sum(EVe,3); % aprime-by-semizprime
        EVKrontemp=EVe(EVindex).*Pdz;
        EVKrontemp(isnan(EVKrontemp))=0;
        EVKrontemp=sum(EVKrontemp,2); % sum over semizprime
        VKron=FofPolicy+DiscountFactorParamsVec*EVKrontemp;

        currdist=max(abs(VKron(:)-VKronold(:)));
        itercount=itercount+1;
    end

    V=reshape(VKron,[n_a,n_semiz,vfoptions.n_e]);
elseif N_e>0
    pi_e_alt=shiftdim(vfoptions.pi_e,-3); % 1-by-1-by-1-by-eprime
    pi_z_alt=shiftdim(pi_z',-2); % 1-by-1-by-zprime-by-z
    zind=gpuArray(repmat(repelem((1:1:N_z)',N_a*N_semiz,1),N_e,1)); % z of each state (a,semiz,z,e)
    EVindex=gpuArray(Policy_aprime)+N_a*gpuArray(0:1:N_semiz-1)+N_a*N_semiz*(zind-1); % [N_a*N_allz,N_semiz], index for (aprime,semizprime,z) in EVz

    while currdist>vfoptions.tolerance && itercount<vfoptions.maxiter
        VKronold=VKron;

        % First the expectation over eprime (iid), then over zprime, then over semizprime conditional on semiz and the policy d2
        EVe=reshape(VKron,[N_a,N_semiz,N_z,N_e]).*pi_e_alt;
        EVe(isnan(EVe))=0;
        EVe=sum(EVe,4); % aprime-by-semizprime-by-zprime
        EVz=EVe.*pi_z_alt;
        EVz(isnan(EVz))=0;
        EVz=reshape(sum(EVz,3),[N_a,N_semiz,N_z]); % aprime-by-semizprime-by-z
        EVKrontemp=EVz(EVindex).*Pdz;
        EVKrontemp(isnan(EVKrontemp))=0;
        EVKrontemp=sum(EVKrontemp,2); % sum over semizprime
        VKron=FofPolicy+DiscountFactorParamsVec*EVKrontemp;

        currdist=max(abs(VKron(:)-VKronold(:)));
        itercount=itercount+1;
    end

    V=reshape(VKron,[n_a,n_semiz,n_z,vfoptions.n_e]);
elseif vfoptions.gridinterplayer==1 && N_z==0
    EVindex=gpuArray(Policy_aprime)+N_a*gpuArray(0:1:N_semiz-1); % [N_a*N_semiz,N_semiz], index for (aprime lower,semizprime) in V

    while currdist>vfoptions.tolerance && itercount<vfoptions.maxiter
        VKronold=VKron;

        % The two aprime points with the L2 weights, then the expectation over semizprime conditional on semiz and the policy d2
        EVlower=(1-PolicyProbUpper).*VKron(EVindex);
        EVlower(PolicyProbUpper==1,:)=0; % zero weight times -Inf gives NaN, a zero weight contributes nothing (the solver raws do the same)
        EVupper=PolicyProbUpper.*VKron(EVindex+1);
        EVupper(PolicyProbUpper==0,:)=0;
        EVKrontemp=(EVlower+EVupper).*Pdz;
        EVKrontemp(isnan(EVKrontemp))=0; % now only Pdz==0 against -Inf
        EVKrontemp=sum(EVKrontemp,2); % sum over semizprime
        VKron=FofPolicy+DiscountFactorParamsVec*EVKrontemp;

        currdist=max(abs(VKron(:)-VKronold(:)));
        itercount=itercount+1;
    end

    V=reshape(VKron,[n_a,n_semiz]);
elseif vfoptions.gridinterplayer==1
    pi_z_alt=shiftdim(pi_z',-2); % 1-by-1-by-zprime-by-z
    zind=gpuArray(repelem((1:1:N_z)',N_a*N_semiz,1)); % z of each state (a,semiz,z)
    EVindex=gpuArray(Policy_aprime)+N_a*gpuArray(0:1:N_semiz-1)+N_a*N_semiz*(zind-1); % [N_a*N_bothz,N_semiz], index for (aprime lower,semizprime,z) in EVz

    while currdist>vfoptions.tolerance && itercount<vfoptions.maxiter
        VKronold=VKron;

        % First the expectation over zprime, then the two aprime points with the L2 weights, then over semizprime
        % conditional on semiz and the policy d2
        EVz=reshape(VKron,[N_a,N_semiz,N_z]).*pi_z_alt;
        EVz(isnan(EVz))=0;
        EVz=reshape(sum(EVz,3),[N_a,N_semiz,N_z]); % aprime-by-semizprime-by-z
        EVlower=(1-PolicyProbUpper).*EVz(EVindex);
        EVlower(PolicyProbUpper==1,:)=0; % zero weight times -Inf gives NaN, a zero weight contributes nothing (the solver raws do the same)
        EVupper=PolicyProbUpper.*EVz(EVindex+1);
        EVupper(PolicyProbUpper==0,:)=0;
        EVKrontemp=(EVlower+EVupper).*Pdz;
        EVKrontemp(isnan(EVKrontemp))=0; % now only Pdz==0 against -Inf
        EVKrontemp=sum(EVKrontemp,2); % sum over semizprime
        VKron=FofPolicy+DiscountFactorParamsVec*EVKrontemp;

        currdist=max(abs(VKron(:)-VKronold(:)));
        itercount=itercount+1;
    end

    V=reshape(VKron,[n_a,n_semiz,n_z]);
elseif N_z==0
    EVindex=gpuArray(Policy_aprime)+N_a*gpuArray(0:1:N_semiz-1); % [N_a*N_semiz,N_semiz], index for (aprime,semizprime) in V

    while currdist>vfoptions.tolerance && itercount<vfoptions.maxiter
        VKronold=VKron;

        EVKrontemp=VKron(EVindex).*Pdz;
        EVKrontemp(isnan(EVKrontemp))=0;
        EVKrontemp=sum(EVKrontemp,2); % sum over semizprime
        VKron=FofPolicy+DiscountFactorParamsVec*EVKrontemp;

        currdist=max(abs(VKron(:)-VKronold(:)));
        itercount=itercount+1;
    end

    V=reshape(VKron,[n_a,n_semiz]);
else
    pi_z_alt=shiftdim(pi_z',-2); % 1-by-1-by-zprime-by-z
    zind=gpuArray(repelem((1:1:N_z)',N_a*N_semiz,1)); % z of each state (a,semiz,z)
    EVindex=gpuArray(Policy_aprime)+N_a*gpuArray(0:1:N_semiz-1)+N_a*N_semiz*(zind-1); % [N_a*N_bothz,N_semiz], index for (aprime,semizprime,z) in EVz

    while currdist>vfoptions.tolerance && itercount<vfoptions.maxiter
        VKronold=VKron;

        % First the expectation over zprime, then over semizprime conditional on semiz and the policy d2
        EVz=reshape(VKron,[N_a,N_semiz,N_z]).*pi_z_alt;
        EVz(isnan(EVz))=0;
        EVz=reshape(sum(EVz,3),[N_a,N_semiz,N_z]); % aprime-by-semizprime-by-z
        EVKrontemp=EVz(EVindex).*Pdz;
        EVKrontemp(isnan(EVKrontemp))=0;
        EVKrontemp=sum(EVKrontemp,2); % sum over semizprime
        VKron=FofPolicy+DiscountFactorParamsVec*EVKrontemp;

        currdist=max(abs(VKron(:)-VKronold(:)));
        itercount=itercount+1;
    end

    V=reshape(VKron,[n_a,n_semiz,n_z]);
end

if currdist>vfoptions.tolerance
    warning(['ValueFnFromPolicy_InfHorz_SemiExo: the policy-evaluation fixed point stopped on reaching ', ...
             'the maximum number of iterations, not on convergence (set by vfoptions.maxiter). ', ...
             'The returned V is not the value of the given Policy. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], currdist, vfoptions.tolerance)
end

end
