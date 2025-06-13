function [V,Policy]=ValueFnIter_SeparableReturnFn(V0,n_d,n_a,n_z,d_grid,a_grid,z_gridvals,pi_z,ReturnFn,Parameters,DiscountFactorParamNames,ReturnFnParamNames,vfoptions)

if isempty(ReturnFnParamNames)
    ReturnFnParamNames.R1=ReturnFnParamNamesFn(ReturnFn.R1,n_d+n_a,0,n_z,0,vfoptions,Parameters);
    % (d,a,z,..)
    ReturnFnParamNames.R2=ReturnFnParamNamesFn(ReturnFn.R2,[n_a,1],0,0,0,vfoptions,Parameters);
    % (aprime,R1result,..)
end

ReturnFnParamNames.R1
ReturnFnParamNames.R2

%% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec.R1=CreateVectorFromParams(Parameters, ReturnFnParamNames.R1);
ReturnFnParamsVec.R2=CreateVectorFromParams(Parameters, ReturnFnParamNames.R2);

ParamCell.R1=cell(length(ReturnFnParamsVec.R1),1);
for ii=1:length(ReturnFnParamsVec.R1)
    ParamCell.R1(ii,1)={ReturnFnParamsVec.R1(ii)};
end

ParamCell.R2=cell(length(ReturnFnParamsVec.R2),1);
for ii=1:length(ReturnFnParamsVec.R2)
    ParamCell.R2(ii,1)={ReturnFnParamsVec.R2(ii)};
end

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec); % Infinite horizon, so just do this once.

%% Solve 
% % (I still don't know if I want to use refine, for now there is no d variable)
% if strcmp(vfoptions.solnmethod,'purediscretization')
%     % TO BE IMPLEMENTED
% end
% 
% if strcmp(vfoptions.solnmethod,'purediscretization_refinement') 
%     % TO BE IMPLEMENTED
% end

if vfoptions.verbose==1
    disp('Creating return fn matrix')
end

aprime1vals = a_grid;              %(a',1,1)
a1vals      = shiftdim(a_grid,-1); %(1,a,1)

ReturnFn.R1
ReturnFn.R2

l_z = length(n_z);

if l_z==1
    z1vals = shiftdim(z_gridvals(:,1),-2);
elseif l_z==2
    z1vals = shiftdim(z_gridvals(:,1),-2);
    z2vals = shiftdim(z_gridvals(:,2),-2);
elseif l_z==3
    z1vals = shiftdim(z_gridvals(:,1),-2);
    z2vals = shiftdim(z_gridvals(:,2),-2);
    z3vals = shiftdim(z_gridvals(:,3),-2);
elseif l_z==4
    z1vals = shiftdim(z_gridvals(:,1),-2);
    z2vals = shiftdim(z_gridvals(:,2),-2);
    z3vals = shiftdim(z_gridvals(:,3),-2);
    z4vals = shiftdim(z_gridvals(:,4),-2);
else
    error('ERROR: SeparableReturnFn does not allow for more than four of z variable (you have length(n_z)>4)')
end

% Cash is (1,a,z)
if l_z==1
    cash_on_hand=arrayfun(ReturnFn.R1, a1vals,z1vals,ParamCell.R1{:});
elseif l_z==2
    cash_on_hand=arrayfun(ReturnFn.R1, a1vals,z1vals,z2vals,ParamCell.R1{:});
elseif l_z==3
    cash_on_hand=arrayfun(ReturnFn.R1, a1vals,z1vals,z2vals,z3vals,ParamCell.R1{:});
elseif l_z==4
    cash_on_hand=arrayfun(ReturnFn.R1, a1vals,z1vals,z2vals,z3vals,z4vals,ParamCell.R1{:});
end
% ReturnMatrix is (a',a,z)
ReturnMatrix=arrayfun(ReturnFn.R2, aprime1vals,cash_on_hand,ParamCell.R2{:});

if vfoptions.verbose==1
    fprintf('Starting Value Function \n')
end

if n_d(1)==0
    [VKron,Policy]=ValueFnIter_Case1_NoD_Par2_raw(V0,n_a,n_z,pi_z,DiscountFactorParamsVec,ReturnMatrix,vfoptions.howards,vfoptions.maxhowards,vfoptions.tolerance,vfoptions.maxiter); 
else
    [VKron,Policy]=ValueFnIter_Case1_Refine(V0,n_d,n_a,n_z,d_grid,a_grid,z_gridvals,pi_z,ReturnFn,ReturnFnParamsVec,DiscountFactorParamsVec,vfoptions);
end

V      = reshape(VKron,[n_a,n_z]);
Policy = UnKronPolicyIndexes_Case1(Policy, n_d, n_a, n_z,vfoptions);


end %end function