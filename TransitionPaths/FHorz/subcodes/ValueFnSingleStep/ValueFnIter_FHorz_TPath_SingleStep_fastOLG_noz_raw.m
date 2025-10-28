function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_noz_raw(V,n_d,n_a,N_j, d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames)
% fastOLG just means parallelize over "age" (j)

N_d=prod(n_d);
N_a=prod(n_a);

% Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%% First, create the big 'next period (of transition path) expected value fn.

% VfastOLG will be N_d*N_aprime by N_a*N_j (note: N_aprime is just equal to N_a)

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

DiscountFactorParamsVec=CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec,2);
DiscountFactorParamsVec=shiftdim(DiscountFactorParamsVec,-2);

EV=zeros(N_a,1,N_j,'gpuArray');
EV(:,1,1:N_j-1)=V(:,2:end);

DiscountedEV=repelem(DiscountFactorParamsVec.*EV,N_d,1,1); % [N_d*N_aprime,1,N_j]

ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_noz_Par2(ReturnFn, n_d, N_j, d_grid, a_grid, a_grid, ReturnFnParamsAgeMatrix,2);

entireRHS=ReturnMatrix+DiscountedEV; %(d,aprime)-by-(a,j)

% Calc the max and it's index
[V,Policy]=max(entireRHS,[],1);
V=squeeze(V); % V is over (a,j)
Policy=squeeze(Policy); % Policy is over (a,j)

end
