function [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep_fastOLG_nod_noz_raw(V,n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_a=prod(n_a);

% Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%% First, create the big 'next period (of transition path) expected value fn.

% VfastOLG will be N_d*N_aprime by N_a*N_j (note: N_aprime is just equal to N_a)

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2_fastOLG_noz(ReturnFn, 0, n_a, N_j, [], a_grid, ReturnFnParamsAgeMatrix);

ReturnMatrix=reshape(ReturnMatrix,[N_a,N_a*N_j]);

DiscountFactorParamsVec=CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec,2);
DiscountFactorParamsVec=DiscountFactorParamsVec';
% DiscountFactorParamsVec=kron(ones(N_a,1),DiscountFactorParamsVec);

VKronNext=zeros(N_a,N_j,'gpuArray');
VKronNext(:,1:N_j-1)=V(:,2:end); % Swap j and z

entirediscountedEV=ReturnMatrix+DiscountFactorParamsVec*kron(VKronNext,ones(1,N_a)); %(aprime)-by-(a,j)

%Calc the max and it's index
[Vtemp,maxindex]=max(entirediscountedEV,[],1);
V=reshape(Vtemp,[N_a,N_j]); % V is over (a,j)
Policy=reshape(maxindex,[N_a,N_j]); % Policy is over (a,j)


end