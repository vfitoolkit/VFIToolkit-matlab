function [V,Policy,Vhat]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_noz_raw(V,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% V is (a,j) (V carries Vunderbar for Sophisticated); Vhat is agent's-perspective (beta0*beta) value

N_d=prod(n_d);
N_a=prod(n_a);

%% First, create the big 'next period (of transition path) expected value fn.

% VfastOLG will be N_d*N_aprime by N_a*N_j (note: N_aprime is just equal to N_a)

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

beta_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);
beta0_J=CreateAgeMatrixFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
beta0beta_J=beta0_J.*beta_J; % Discount factor between today and tomorrow.

EV=zeros(N_a,1,N_j,'gpuArray');
EV(:,1,1:N_j-1)=V(:,2:end);

DiscountedEV_under=repelem(reshape(beta_J,[1,1,N_j]).*EV,N_d,1,1); % [N_d*N_aprime,1,N_j]
DiscountedEV=repelem(reshape(beta0beta_J,[1,1,N_j]).*EV,N_d,1,1); % [N_d*N_aprime,1,N_j]

ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_noz(ReturnFn, n_d, n_a, N_j, d_gridvals, a_grid, a_grid, ReturnFnParamsAgeMatrix);

% First Policy (QH-optimal); capture Vhat (beta0*beta-step max value, agent's perspective)
entireRHS=ReturnMatrix+DiscountedEV; %(d,aprime)-by-(a,j)
[Vhat,Policy]=max(entireRHS,[],1);
% Now Vunderbar: evaluate at QH-optimal index with two-future-periods discount factor
entireRHS_under=ReturnMatrix+DiscountedEV_under; %(d,aprime)-by-(a,j)
maxindexfull=Policy+(N_d*N_a)*(0:1:N_a-1)+(N_d*N_a*N_a)*shiftdim((0:1:N_j-1),-1);
V=entireRHS_under(maxindexfull);
V=squeeze(V); % V is over (a,j)
Policy=squeeze(Policy);
Vhat=squeeze(Vhat); % (a,j)

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point


end
