function PricePath=TransitionPath_Case1_EntryExit(PricePathOld, PricePathNames, ParamPath, ParamPathNames, T, V_final, AgentDist_initial, n_d, n_a, n_z, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, EntryExitParamNames, transpathoptions, vfoptions, simoptions)
% This code will work for all transition paths except those that involve at
% change in the transition matrix pi_z (can handle a change in pi_z, but
% only if it is a 'surprise', not anticipated changes) 

% PricePathOld is matrix of size T-by-'number of prices'
% ParamPath is matrix of size T-by-'number of parameters that change over path'

%% Check which transpathoptions have been used, set all others to defaults 
% if exist('transpathoptions','var')==0
%     disp('No transpathoptions given, using defaults')
%     %If transpathoptions is not given, just use all the defaults
%     transpathoptions.tolerance=10^(-5);
%     transpathoptions.parallel=2;
%     transpathoptions.lowmemory=0;
%     transpathoptions.exoticpreferences=0;
%     transpathoptions.oldpathweight=0.9; % default =0.9
%     transpathoptions.weightscheme=1; % default =1
%     transpathoptions.Ttheta=1;
%     transpathoptions.maxiterations=1000;
%     transpathoptions.verbose=0;
%     transpathoptions.GEnewprice=0;
%     transpathoptions.weightsforpath=ones(size(PricePathOld));
% else
%     %Check transpathoptions for missing fields, if there are some fill them with the defaults
%     if isfield(transpathoptions,'tolerance')==0
%         transpathoptions.tolerance=10^(-5);
%     end
%     if isfield(transpathoptions,'parallel')==0
%         transpathoptions.parallel=2;
%     end
%     if isfield(transpathoptions,'lowmemory')==0
%         transpathoptions.lowmemory=0;
%     end
%     if isfield(transpathoptions,'exoticpreferences')==0
%         transpathoptions.exoticpreferences=0;
%     end
%     if isfield(transpathoptions,'oldpathweight')==0
%         transpathoptions.oldpathweight=0.9;
%     end
%     if isfield(transpathoptions,'weightscheme')==0
%         transpathoptions.weightscheme=1;
%     end
%     if isfield(transpathoptions,'Ttheta')==0
%         transpathoptions.Ttheta=1;
%     end
%     if isfield(transpathoptions,'maxiterations')==0
%         transpathoptions.maxiterations=1000;
%     end
%     if isfield(transpathoptions,'verbose')==0
%         transpathoptions.verbose=0;
%     end
%     if isfield(transpathoptions,'GEnewprice')==0
%         transpathoptions.GEnewprice=0;
%     end
%     if isfield(transpathoptions,'weightsforpath')==0
%         transpathoptions.weightsforpath=ones(T,length(GeneralEqmEqns));
%     end
% end
% 
if exist('vfoptions','var')==0
    vfoptions.lowmemory=0;
    vfoptions.verbose=0;
    vfoptions.tolerance=10^(-9);
    vfoptions.howards=80;
    vfoptions.maxhowards=500;
    vfoptions.endogenousexit=0;
    vfoptions.exoticpreferences=0;
    vfoptions.polindorval=1;
    vfoptions.policy_forceintegertype=0;
    vfoptions.piz_strictonrowsaddingtoone=0;
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if isfield(vfoptions,'parallel')==0
        vfoptions.parallel=2;
    end
    if vfoptions.parallel==2
        vfoptions.returnmatrix=2; % On GPU, must use this option
    end
    if isfield(vfoptions,'returnmatrix')==0
        if isa(ReturnFn,'function_handle')==1
            vfoptions.returnmatrix=0;
        else
            vfoptions.returnmatrix=1;
        end
    end
    if isfield(vfoptions,'lowmemory')==0
        vfoptions.lowmemory=0;
    end
    if isfield(vfoptions,'verbose')==0
        vfoptions.verbose=0;
    end
     if isfield(vfoptions,'tolerance')==0
        vfoptions.tolerance=10^(-9);
    end
    if isfield(vfoptions,'endogenousexit')==0
        vfoptions.endogenousexit=0;
    end
    if isfield(vfoptions,'policy_forceintegertype')==0
        vfoptions.policy_forceintegertype=0;
    end
end

% Make sure that the inputs specifically required for endogenous exit have been included.
if vfoptions.endogenousexit==1
    if isfield(vfoptions,'ReturnToExitFn')==0
        fprintf('ERROR: vfoptions.endogenousexit=1 requires that you specify vfoptions.ReturnToExitFn \n');
        return
    end
    if isfield(vfoptions,'ReturnToExitFnParamNames')==0
        fprintf('ERROR: vfoptions.endogenousexit=1 requires that you specify vfoptions.ReturnToExitFnParamNames \n');
        return
    end
end

%%

if transpathoptions.exoticpreferences~=0
    disp('ERROR: Only transpathoptions.exoticpreferences==0 is supported by TransitionPath_Case1_EntryExit')
    dbstack
% else
%     if length(DiscountFactorParamNames)~=1
%         disp('WARNING: DiscountFactorParamNames should be of length one')
%         dbstack
%     end
end

if transpathoptions.verbose==1
    transpathoptions
end

% if transpathoptions.parallel~=2
%     disp('ERROR: Only transpathoptions.parallel==2 is supported by TransitionPath_Case1')
% else
if transpathoptions.parallel==2
    % Make sure things are on gpu where appropriate.
    d_grid=gpuArray(d_grid); a_grid=gpuArray(a_grid); z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
    PricePathOld=gpuArray(PricePathOld);
end

if transpathoptions.lowmemory==1
    fprintf('transpathoptions.lowmemory=1 is not yet implemented for entry/exit, please contact robertdkirkby@gmail.com if you want it \n')
    dbstack
    return
    %     % The lowmemory option is going to use gpu (but loop over z instead of
%     % parallelize) for value fn, and then use sparse matrices on cpu when iterating on agent dist.
%     PricePathOld=TransitionPath_Case1_EntryExit_lowmem(PricePathOld, PricePathNames, ParamPath, ParamPathNames, T, V_final, AgentDist_initial, n_d, n_a, n_z, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames,transpathoptions);
%     return
end

%% Shooting algorithm
if transpathoptions.GEnewprice==1  % Shooting algorithm
    if n_d(1)==0
        PricePath=TransitionPath_Case1_EntryExit_no_d_shooting(PricePathOld, PricePathNames, ParamPath, ParamPathNames, T, V_final, AgentDist_initial, n_a, n_z, pi_z, a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames,EntryExitParamNames,transpathoptions,vfoptions,simoptions);
        return
    else
        PricePath=TransitionPath_Case1_EntryExit_shooting(PricePathOld, PricePathNames, ParamPath, ParamPathNames, T, V_final, AgentDist_initial, n_a, n_z, pi_z, a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames,EntryExitParamNames,transpathoptions,vfoptions,simoptions);
        return
    end
end

%% Set up transition path as minimization of a function (default is to use as objective the weighted sum of squares of the general eqm conditions)
PricePathVec=gather(reshape(PricePathOld,[T*length(PricePathNames),1])); % Has to be vector of fminsearch. Additionally, provides a double check on sizes.

if transpathoptions.GEnewprice==2 % Function minimization
    if n_d(1)==0
        GeneralEqmConditionsPathFn=@(pricepath) TransitionPath_Case1_EntryExit_no_d_subfn(pricepath, PricePathNames, ParamPath, ParamPathNames, T, V_final, AgentDist_initial, n_a, n_z, pi_z, a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames,EntryExitParamNames,transpathoptions,vfoptions,simoptions);
    else
        GeneralEqmConditionsPathFn=@(pricepath) TransitionPath_Case1_EntryExit_subfn(pricepath, PricePathNames, ParamPath, ParamPathNames, T, V_final, AgentDist_initial, n_a, n_z, pi_z, a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames,EntryExitParamNames,transpathoptions,vfoptions,simoptions);
    end
end

% if transpathoptions.GEnewprice2algo==0
[PricePath,~]=fminsearch(GeneralEqmConditionsPathFn,PricePathVec);
% else
%     [PricePath,~]=fminsearch(GeneralEqmConditionsPathFn,PricePathOld);
% end

% LOOK INTO USING 'SURROGATE OPTIMIZATION'

PricePath=gpuArray(reshape(PricePath,[T,length(PricePathNames)])); % Switch back to appropriate shape (out of the vector required to use fminsearch)

end