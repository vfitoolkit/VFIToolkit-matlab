function [V, Policy, Vhat]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_nod_noz_raw(V,n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% V is (a,j) (V carries Vunderbar for Sophisticated); Vhat is agent's-perspective (beta0*beta) value

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
EV(:,1,1:N_j-1)=V(:,2:end); % Leave N_j as zeros

ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_nod_noz(ReturnFn, n_a, N_j, a_grid, a_grid, ReturnFnParamsAgeMatrix);

% First Policy (QH-optimal); capture Vhat (beta0*beta-step max value, agent's perspective)
entireRHS=ReturnMatrix+reshape(beta0beta_J,[1,1,N_j]).*EV; %(aprime)-by-(a,j)
[Vhat,Policy]=max(entireRHS,[],1);
% Now Vunderbar: evaluate at QH-optimal index with two-future-periods discount factor
entireRHS_under=ReturnMatrix+reshape(beta_J,[1,1,N_j]).*EV; %(aprime)-by-(a,j)
maxindexfull=Policy+N_a*(0:1:N_a-1)+(N_a*N_a)*shiftdim((0:1:N_j-1),-1);
V=entireRHS_under(maxindexfull);
V=squeeze(V);
Policy=squeeze(Policy);
Vhat=squeeze(Vhat); % (a,j)

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point


end
