function [V,Policy2]=ValueFnIter_Case1_FHorz_TPath_SingleStep_fastOLG_noz_raw(V,n_d,n_a,N_j, d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames)
% fastOLG just means parallelize over "age" (j)

N_d=prod(n_d);
N_a=prod(n_a);

% Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%% First, create the big 'next period (of transition path) expected value fn.

% VfastOLG will be N_d*N_aprime by N_a*N_j (note: N_aprime is just equal to N_a)

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2_fastOLG_noz(ReturnFn, n_d, n_a, N_j, d_grid, a_grid, ReturnFnParamsAgeMatrix);

DiscountFactorParamsVec=CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec,2);
DiscountFactorParamsVec=DiscountFactorParamsVec';

VKronNext=zeros(N_a,N_j,'gpuArray');
VKronNext(:,1:N_j-1)=V(:,2:end);

RHS=ReturnMatrix+kron(DiscountFactorParamsVec.*VKronNext,ones(N_d,N_a)); %(d,aprime)-by-(a,j)

%Calc the max and it's index
[Vtemp,maxindex]=max(RHS,[],1);
V=reshape(Vtemp,[N_a,N_j]); % V is over (a,j)
Policy=reshape(maxindex,[N_a,N_j]); % Policy is over (a,j)

%%
Policy2=zeros(2,N_a,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
Policy2(1,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:)=shiftdim(ceil(Policy/N_d),-1);

end