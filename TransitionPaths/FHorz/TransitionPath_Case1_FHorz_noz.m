function PricePath=TransitionPath_Case1_FHorz_noz(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, StationaryDist_init, n_d, n_a, N_j, d_grid,a_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, AgeWeightsParamNames, transpathoptions, simoptions, vfoptions)
% This code will work for all transition paths
%
% PricePathOld is a structure with fields names being the Prices and each field containing a T-by-1 path.
% ParamPath is a structure with fields names being the parameter names of those parameters which change over the path and each field containing a T-by-1 path.
%
% Only works for v2

% Remark to self: No real need for T as input, as this is anyway the length of PricePathOld

%% Check the sizes of some of the inputs
if isempty(n_d)
    N_d=0;
else
    N_d=prod(n_d);
end
% N_a=prod(n_a);

%%
if transpathoptions.parallel==2 
   % If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
   if N_d>0
       d_grid=gpuArray(d_grid);
   end
   a_grid=gpuArray(a_grid);
   PricePathOld=gpuArray(PricePathOld);
else
   % If using CPU make sure all the relevant inputs are CPU arrays (not standard arrays)
   if N_d>0
       d_grid=gather(d_grid);
   end
   a_grid=gather(a_grid);
   PricePathOld=gather(PricePathOld);
end

%%
if transpathoptions.usestockvars==1 
    % Get the stock variable objects out of transpathoptions.
    StockVariable_init=transpathoptions.stockvarinit;
    StockVariableEqns=transpathoptions.stockvareqns.lawofmotion;
    StockVarsPathNames=fieldnames(StockVariableEqns);
    StockVarsPathOld=zeros(T,length(StockVarsPathNames));
    for ss=1:length(StockVarsPathNames)
%         temp=getAnonymousFnInputNames(StockVariableEqns.(StockVarsPathNames{ss}));
%         StockVariableEqnParamNames(ss).Names={temp{:}}; % the first inputs will always be (d,aprime,a,z)
%         StockVariableEqns2{ss}=StockVariableEqnsStruct.(StockVarsPathNames{ss});
        StockVarsPathOld(:,ss)=transpathoptions.stockvarpath0.(StockVarsPathNames{ss});
    end    
%     StockVariableEqns=StockVariableEqns2;
    if transpathoptions.parallel==2
        StockVarsPathOld=gpuArray(StockVarsPathOld);
    end
end

%% Handle ReturnFn and FnsToEvaluate structures
l_d=length(n_d);
if n_d(1)==0
    l_d=0;
end
l_a=length(n_a);
l_a_temp=l_a;
if max(vfoptions.endotype)==1
    l_a_temp=l_a-sum(vfoptions.endotype);
end
% Create ReturnFnParamNames
temp=getAnonymousFnInputNames(ReturnFn);
if length(temp)>(l_d+l_a_temp+l_a_temp)
    ReturnFnParamNames={temp{l_d+l_a_temp+l_a_temp+1:end}}; % the first inputs will always be (d,aprime,a,z)
else
    ReturnFnParamNames={};
end

if ~isstruct(FnsToEvaluate)
    error('Transition paths only work with version 2+ (FnsToEvaluate has to be a structure)')
end


%%
transpathoptions
if transpathoptions.GEnewprice~=2
    if transpathoptions.parallel==2
        if transpathoptions.usestockvars==0
            if transpathoptions.fastOLG==0
                PricePathOld=TransitionPath_Case1_FHorz_shooting_noz(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, StationaryDist_init, n_d, n_a, N_j, d_grid,a_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, AgeWeightsParamNames, ReturnFnParamNames, vfoptions, simoptions,transpathoptions);
            else % use fastOLG setting
                PricePathOld=TransitionPath_Case1_FHorz_shooting_fastOLG_noz(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, StationaryDist_init, n_d, n_a, N_j, d_grid,a_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, AgeWeightsParamNames, ReturnFnParamNames, vfoptions, simoptions,transpathoptions);
            end
        else % transpathoptions.usestockvars==1
            warning('StockVars does not yet work correctly')
            error('Not yet implemented StockVars without z varialbes')
%             if transpathoptions.fastOLG==0                
%                 [PricePathOld,StockVarsPathOld]=TransitionPath_Case1_FHorz_StockVar_shooting(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, StockVarsPathOld, StockVarsPathNames, T, V_final, StationaryDist_init, StockVariable_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, StockVariableEqns, Parameters, DiscountFactorParamNames, AgeWeightsParamNames, ReturnFnParamNames, vfoptions, simoptions,transpathoptions);
%             else % use fastOLG setting
%                 [PricePathOld,StockVarsPathOld]=TransitionPath_Case1_FHorz_StockVar_shooting_fastOLG(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, StockVarsPathOld, StockVarsPathNames, T, V_final, StationaryDist_init, StockVariable_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, StockVariableEqns, Parameters, DiscountFactorParamNames, AgeWeightsParamNames, ReturnFnParamNames, vfoptions, simoptions,transpathoptions);
%             end
        end
    else
        error('VFI Toolkit does not offer transition path without gpu. Would be too slow to be useful.')
    end
    % Switch the solution into structure for output.
    for ii=1:length(PricePathNames)
        PricePath.(PricePathNames{ii})=PricePathOld(:,ii);
    end
    if transpathoptions.usestockvars==1
        for ii=1:length(StockVarsPathNames)
            PricePath.(StockVarsPathNames{ii})=StockVarsPathOld(:,ii);
        end
    end
    return
end


%% Set up transition path as minimization of a function (default is to use as objective the weighted sum of squares of the general eqm conditions)
l_p=size(PricePathOld,2);
if transpathoptions.verbose==1
    transpathoptions
end
if transpathoptions.verbose==1
    DiscountFactorParamNames
    ReturnFnParamNames
    ParamPathNames
    PricePathNames
end

PricePathVec=gather(reshape(PricePathOld,[T*length(PricePathNames),1])); % Has to be vector of fminsearch. Additionally, provides a double check on sizes.

% I HAVEN'T GOTTEN THIS TO WORK WELL ENOUGH THAT I AM COMFORTABLE LEAVING IT ENABLED
if transpathoptions.GEnewprice==2 % Function minimization
    error('transpathoptions.GEnewprice==2 not currently enabled')
%     if n_d(1)==0
%         GeneralEqmConditionsPathFn=@(pricepath) TransitionPath_Case1_FHorz_no_d_subfn(pricepathPricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, AgeWeightsParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, vfoptions, simoptions,transpathoptions);
%     else
%         GeneralEqmConditionsPathFn=@(pricepath) TransitionPath_Case1_FHorz_subfn(pricepath, PricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, AgeWeightsParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, vfoptions, simoptions,transpathoptions);
%     end
end

% if transpathoptions.GEnewprice2algo==0
% [PricePath,~]=fminsearch(GeneralEqmConditionsPathFn,PricePathVec);
% else
%     [PricePath,~]=fminsearch(GeneralEqmConditionsPathFn,PricePathOld);
% end
%
% LOOK INTO USING 'SURROGATE OPTIMIZATION'

if transpathoptions.parallel==2
    PricePath=gpuArray(reshape(PricePath,[T,length(PricePathNames)])); % Switch back to appropriate shape (out of the vector required to use fminsearch)
else
    PricePath=gather(reshape(PricePath,[T,length(PricePathNames)])); % Switch back to appropriate shape (out of the vector required to use fminsearch)    
end

for ii=1:length(PricePathNames)
    PricePath.(PricePathNames{ii})=PricePathOld(:,ii);
end


end