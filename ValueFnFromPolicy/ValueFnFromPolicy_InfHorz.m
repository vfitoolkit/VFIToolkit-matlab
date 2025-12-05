function V=ValueFnFromPolicy_InfHorz(Policy,n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)

if ~exist('vfoptions','var')
    if ~isfield(vfoptions,'tolerance')
        vfoptions.tolerance=10^(-9);
    end
else
    vfoptions.tolerance=10^(-9);
    if isfield(vfoptions,'exoticpreferences')
        error('ValueFnFromPolicy_Case1() does not yet work with exotic preferences. Please ask on forum if you want/need this feature. \n');
    end
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if N_d==0 && isscalar(n_a) && vfoptions.gridinterplayer==0
    l_daprime=1;
else
    l_daprime=size(Policy,1);
    if vfoptions.gridinterplayer==1
        l_daprime=l_daprime-1;
    end
end
a_gridvals=CreateGridvals(n_a,a_grid,1);
% Switch to z_gridvals
[z_gridvals, pi_z, vfoptions]=ExogShockSetup(n_z,z_grid,pi_z,Parameters,vfoptions,3);

%% Implement new way of handling ReturnFn inputs
ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,0,vfoptions,Parameters);
% Basic setup: the first inputs of ReturnFn will be (d,aprime,a,z,..) and everything after this is a parameter, so we get the names of all these parameters.
% But this changes if you have e, semiz, or just multiple d, and if you use riskyasset, expasset, etc.
% So figure out which setup we have, and get the relevant ReturnFnParamNames

%% Calculate FofPolicy (the return fn evaluated at the Policy)
PolicyValues=PolicyInd2Val_Case1(Policy,n_d,n_a,n_z,d_grid,a_grid, vfoptions);
PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a,N_z]),[2,3,1]); %[N_a,N_z,l_d+l_a]

ReturnFnParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames);
FofPolicy=EvalFnOnAgentDist_Grid(ReturnFn, ReturnFnParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);

%% Now that we have FofPolicy, calculate V.
DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames));
Policy=KronPolicyIndexes_Case1(Policy, n_d, n_a, n_z, vfoptions);

pi_z_howards=repelem(pi_z,N_a,1);

currdist=Inf;
VKron=FofPolicy/(1-DiscountFactorParamsVec); % rough guess
if vfoptions.gridinterplayer==0
    if N_d==0
        Policy_a=Policy;
    else
        Policy_a=shiftdim(ceil(Policy(2,:,:)),1);
    end

    while currdist>vfoptions.tolerance
        VKronold=VKron;

        EVKrontemp=VKron(Policy_a,:);

        EVKrontemp=EVKrontemp.*pi_z_howards;
        EVKrontemp(isnan(EVKrontemp))=0;
        EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
        VKron=FofPolicy+DiscountFactorParamsVec*EVKrontemp;

        currdist=max(max(abs(VKron-VKronold)));
    end

elseif vfoptions.gridinterplayer==1
    if N_d==0
        alowerindex=reshape(Policy(1,:,:),[1,N_a*N_z]);
        aprimeindex=[alowerindex; alowerindex+1]; % [2,N_a*N_z]
        PolicyProbs=reshape(ceil(Policy(end,:,:)),[1,N_a*N_z]);
        PolicyProbs=(PolicyProbs-1)/(vfoptions.ngridinterp+1); % prob of upper point
        PolicyProbs=[1-PolicyProbs; PolicyProbs]; % [2,N_a*N_z]
    else
        alowerindex=reshape(ceil(Policy(2,:,:)),[1,N_a*N_z]);
        aprimeindex=[alowerindex; alowerindex+1]; % [2,N_a*N_z]
        PolicyProbs=reshape(ceil(Policy(end,:,:)),[1,N_a*N_z]);
        PolicyProbs=(PolicyProbs-1)/(vfoptions.ngridinterp+1); % prob of upper point
        PolicyProbs=[1-PolicyProbs; PolicyProbs]; % [2,N_a*N_z]
    end
    
    while currdist>vfoptions.tolerance
        VKronold=VKron;

        EVKrontemp=reshape(VKron(aprimeindex,:),[2,N_a*N_z,N_z]); % last dimension is zprime
        EVKrontemp=shiftdim(sum(PolicyProbs.*EVKrontemp,1),1); % [N_a*N_z,N_z]

        EVKrontemp=EVKrontemp.*pi_z_howards;
        EVKrontemp(isnan(EVKrontemp))=0;
        EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
        VKron=FofPolicy+DiscountFactorParamsVec*EVKrontemp;

        currdist=max(max(abs(VKron-VKronold)));
    end
end


V=reshape(VKron,[n_a,n_z]);

end