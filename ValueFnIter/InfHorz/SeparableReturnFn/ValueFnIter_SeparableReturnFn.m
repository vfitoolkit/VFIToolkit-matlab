function [V,Policy]=ValueFnIter_SeparableReturnFn(V0,n_d,n_a,n_z,d_grid,a_grid,z_gridvals,pi_z,ReturnFn,Params,DiscountFactorParamNames,ReturnFnParamNames,vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

pi_z=gpuArray(pi_z);
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
z_gridvals=gpuArray(z_gridvals);

asset_grid = a_grid(1:n_a(1));
e_grid     = a_grid(n_a(1)+1:n_a(1)+n_a(2));

V0=zeros([N_a,N_z], 'gpuArray');

% If z_grid is not already a joint grid
% z_gridvals has size [prod(n_z),length(n_z)]
%[z_gridvals, pi_z, vfoptions]=ExogShockSetup(n_z,z_grid,pi_z,Params,vfoptions,3);
%z_gridvals = z_grid;

% Generate a cell array of strings with names of parameters in the Return Function
% if isempty(ReturnFnParamNames)
% 	ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,0,vfoptions,Params);
% end

% % Create a vector containing all the return function parameters (in order)
% ReturnFnParamsVec = struct2vec(Params, ReturnFnParamNames);
% ParamCell=cell(length(ReturnFnParamsVec),1);
% for ii=1:length(ReturnFnParamsVec)
%     ParamCell(ii,1)={ReturnFnParamsVec(ii)};
% end

DiscountFactorParamsVec = Params.(DiscountFactorParamNames{1});


%% Compute ReturnMatrix
% ReturnMatrix is (a',a,z)
%ReturnMatrix

d_gridvals  = d_grid;
aprime_vals = shiftdim(asset_grid,-1);
e_vals = shiftdim(e_grid,-2);
a_vals      = shiftdim(asset_grid,-3);
eminus_vals      = shiftdim(e_grid,-4);

eps_vals = shiftdim(z_gridvals(:,1),-5);
theta_vals = shiftdim(z_gridvals(:,2),-5);
xi_vals = shiftdim(z_gridvals(:,3),-5);
age_vals = shiftdim(z_gridvals(:,4),-5);

% Step1: Compute cash on hand

% Input arguments of f_ReturnFn1:
% l_val,e_val,a_val,eps_val,theta_val,age,K_to_L,alpha,vi,delta,lambda_P,lambda_C,gamma,lam_hsv,tau_hsv,pen,tau_c,tau_d,tau_a,le,kappa
% cash(d,1,e,a,1,z)
CashMatrix = arrayfun(@f_ReturnFn1,d_gridvals,e_vals,a_vals,eps_vals,theta_vals,age_vals,...
    Params.K_to_L,Params.alpha,Params.vi,Params.delta,Params.lambda_P,Params.lambda_C,Params.gamma,Params.tau_l0,Params.ttau_0,Params.tau_1,Params.pen,Params.tau_c,Params.tau_a,Params.le,Params.kappa);

% Input arguments of f_ReturnFn2:
% cash,l_val,aprime_val,e_val,eminus_val,xi_val,age,le,crra,sigma2,chi,eta
% ReturnMatrix(d,a',e,a,eminus,z)
ReturnMatrix=arrayfun(@f_ReturnFn2,CashMatrix,d_gridvals,aprime_vals,e_vals,eminus_vals,xi_vals,age_vals,...
    Params.le,Params.crra,Params.sigma2,Params.chi,Params.eta);

ReturnMatrix=reshape(ReturnMatrix,[N_d,N_a,N_a,N_z]); % (d,a',a,z)

%% Now max over d --> dstar(a',a,z) and ReturnMatrix(a',a,z)
[ReturnMatrix,dstar] = max(ReturnMatrix,[],1);
ReturnMatrix         = shiftdim(ReturnMatrix,1);

%% Pass ReturnMatrix(a',a,z) to standard VFI function
[VKron,Policy_a]=ValueFnIter_NoD_raw(V0, n_a, n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance,vfoptions.maxiter); 

% Policy(2,a,z) is Policy_a
% Policy(1,a,z) is dstar(Policy_a(a,z),a,z)

Policy=zeros(2,N_a,N_z);
Policy(2,:,:)=shiftdim(Policy_a,-1);
temppolicyindex=reshape(Policy_a,[1,N_a*N_z])+(0:1:N_a*N_z-1)*N_a;
Policy(1,:,:)=reshape(dstar(temppolicyindex),[N_a,N_z]);

% OUTPUTs of ValueFnIter_Case1_Refine are VKron,Policy
% VKron has size (N_a,N_z) and Policy has size (2,N_a,N_z)

V=reshape(VKron,[n_a,n_z]);

vfoptions.parallel=2;
Policy=UnKronPolicyIndexes_Case1(Policy, n_d, n_a, n_z,vfoptions);

% OUTPUTs of ValueFnIter_Case1 are V,Policy
% V has size (a1,a2,z) and Policy(2,a1,a2,z)





% if isempty(ReturnFnParamNames)
%     ReturnFnParamNames.R1=ReturnFnParamNamesFn(ReturnFn.R1,n_d+n_a,0,n_z,0,vfoptions,Parameters);
%     % (d,a,z,..)
%     ReturnFnParamNames.R2=ReturnFnParamNamesFn(ReturnFn.R2,[n_a,1],0,0,0,vfoptions,Parameters);
%     % (aprime,R1result,..)
% end
% 
% ReturnFnParamNames.R1
% ReturnFnParamNames.R2
% 
% %% Create a vector containing all the return function parameters (in order)
% ReturnFnParamsVec.R1=CreateVectorFromParams(Parameters, ReturnFnParamNames.R1);
% ReturnFnParamsVec.R2=CreateVectorFromParams(Parameters, ReturnFnParamNames.R2);
% 
% ParamCell.R1=cell(length(ReturnFnParamsVec.R1),1);
% for ii=1:length(ReturnFnParamsVec.R1)
%     ParamCell.R1(ii,1)={ReturnFnParamsVec.R1(ii)};
% end
% 
% ParamCell.R2=cell(length(ReturnFnParamsVec.R2),1);
% for ii=1:length(ReturnFnParamsVec.R2)
%     ParamCell.R2(ii,1)={ReturnFnParamsVec.R2(ii)};
% end
% 
% DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
% DiscountFactorParamsVec=prod(DiscountFactorParamsVec); % Infinite horizon, so just do this once.
% 
% %% Solve 
% % % (I still don't know if I want to use refine, for now there is no d variable)
% % if strcmp(vfoptions.solnmethod,'purediscretization')
% %     % TO BE IMPLEMENTED
% % end
% % 
% % if strcmp(vfoptions.solnmethod,'purediscretization_refinement') 
% %     % TO BE IMPLEMENTED
% % end
% 
% if vfoptions.verbose==1
%     disp('Creating return fn matrix')
% end
% 
% aprime1vals = a_grid;              %(a',1,1)
% a1vals      = shiftdim(a_grid,-1); %(1,a,1)
% 
% ReturnFn.R1
% ReturnFn.R2
% 
% l_z = length(n_z);
% 
% if l_z==1
%     z1vals = shiftdim(z_gridvals(:,1),-2);
% elseif l_z==2
%     z1vals = shiftdim(z_gridvals(:,1),-2);
%     z2vals = shiftdim(z_gridvals(:,2),-2);
% elseif l_z==3
%     z1vals = shiftdim(z_gridvals(:,1),-2);
%     z2vals = shiftdim(z_gridvals(:,2),-2);
%     z3vals = shiftdim(z_gridvals(:,3),-2);
% elseif l_z==4
%     z1vals = shiftdim(z_gridvals(:,1),-2);
%     z2vals = shiftdim(z_gridvals(:,2),-2);
%     z3vals = shiftdim(z_gridvals(:,3),-2);
%     z4vals = shiftdim(z_gridvals(:,4),-2);
% else
%     error('ERROR: SeparableReturnFn does not allow for more than four of z variable (you have length(n_z)>4)')
% end
% 
% % Cash is (1,a,z)
% if l_z==1
%     cash_on_hand=arrayfun(ReturnFn.R1, a1vals,z1vals,ParamCell.R1{:});
% elseif l_z==2
%     cash_on_hand=arrayfun(ReturnFn.R1, a1vals,z1vals,z2vals,ParamCell.R1{:});
% elseif l_z==3
%     cash_on_hand=arrayfun(ReturnFn.R1, a1vals,z1vals,z2vals,z3vals,ParamCell.R1{:});
% elseif l_z==4
%     cash_on_hand=arrayfun(ReturnFn.R1, a1vals,z1vals,z2vals,z3vals,z4vals,ParamCell.R1{:});
% end
% % ReturnMatrix is (a',a,z)
% ReturnMatrix=arrayfun(ReturnFn.R2, aprime1vals,cash_on_hand,ParamCell.R2{:});
% 
% if vfoptions.verbose==1
%     fprintf('Starting Value Function \n')
% end
% 
% if n_d(1)==0
%     [VKron,Policy]=ValueFnIter_Case1_NoD_Par2_raw(V0,n_a,n_z,pi_z,DiscountFactorParamsVec,ReturnMatrix,vfoptions.howards,vfoptions.maxhowards,vfoptions.tolerance,vfoptions.maxiter); 
% else
%     [VKron,Policy]=ValueFnIter_Case1_Refine(V0,n_d,n_a,n_z,d_grid,a_grid,z_gridvals,pi_z,ReturnFn,ReturnFnParamsVec,DiscountFactorParamsVec,vfoptions);
% end
% 
% V      = reshape(VKron,[n_a,n_z]);
% Policy = UnKronPolicyIndexes_Case1(Policy, n_d, n_a, n_z,vfoptions);


end %end function