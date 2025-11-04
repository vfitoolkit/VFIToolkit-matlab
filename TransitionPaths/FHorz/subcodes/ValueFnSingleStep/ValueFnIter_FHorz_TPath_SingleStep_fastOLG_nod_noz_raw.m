function [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_nod_noz_raw(V,n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames)

N_a=prod(n_a);

% Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%% First, create the big 'next period (of transition path) expected value fn.

% VfastOLG will be N_d*N_aprime by N_a*N_j (note: N_aprime is just equal to N_a)

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

DiscountFactorParamsVec=CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec,2);
DiscountFactorParamsVec=shiftdim(DiscountFactorParamsVec,-2);

EV=zeros(N_a,1,N_j,'gpuArray');
EV(:,1,1:N_j-1)=V(:,2:end); % Leave N_j as zeros

ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_noz_Par2(ReturnFn, N_j, a_grid, a_grid, ReturnFnParamsAgeMatrix,2);

entireRHS=ReturnMatrix+DiscountFactorParamsVec.*EV; %(aprime)-by-(a,j)

% Calc the max and it's index
[V,Policy]=max(entireRHS,[],1);
V=squeeze(V);
Policy=squeeze(Policy);

%%
Policy=shiftdim(Policy,-1); % So first dim is just one point


end
