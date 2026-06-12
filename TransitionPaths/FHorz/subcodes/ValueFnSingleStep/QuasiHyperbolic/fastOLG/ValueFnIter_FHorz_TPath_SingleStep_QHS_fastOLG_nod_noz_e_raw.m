function [V, Policy, Vhat]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_nod_noz_e_raw(V,n_a,n_e,N_j, a_grid,e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,e), rather than standard (a,e,j)
% V is (a,j)-by-e (V carries Vunderbar for Sophisticated); Vhat is agent's-perspective (beta0*beta) value

N_a=prod(n_a);
N_e=prod(n_e);

e_gridvals_J=shiftdim(e_gridvals_J,-2);

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_z (note: N_aprime is just equal to N_a)

beta_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);
beta0_J=CreateAgeMatrixFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
beta0beta_J=beta0_J.*beta_J; % Discount factor between today and tomorrow.

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

EV=[sum(V(N_a+1:end,:).*pi_e_J(1:end-N_a,:),2); zeros(N_a,1,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_e_J to create expectations

DiscountedEV_under=reshape(beta_J,[1,1,N_j]).*reshape(EV,[N_a,1,N_j]); % [N_aprime,1,N_j] % 2nd dim will be autofilled with a
DiscountedEV=reshape(beta0beta_J,[1,1,N_j]).*reshape(EV,[N_a,1,N_j]); % [N_aprime,1,N_j] % 2nd dim will be autofilled with a

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_nod(ReturnFn, n_a, n_e, N_j, a_grid, a_grid, e_gridvals_J, ReturnFnParamsAgeMatrix);
    % fastOLG: ReturnMatrix is [aprime,a,j,e]

    % First Policy (QH-optimal); capture Vhat (beta0*beta-step max value, agent's perspective)
    entireRHS=ReturnMatrix+DiscountedEV; % [aprime,a,j,e]
    [Vhat,Policy]=max(entireRHS,[],1);
    % Now Vunderbar: evaluate at QH-optimal index with two-future-periods discount factor
    entireRHS_under=ReturnMatrix+DiscountedEV_under; % [aprime,a,j,e]
    maxindexfull=Policy+N_a*(0:1:N_a-1)+(N_a*N_a)*shiftdim((0:1:N_j-1),-1)+(N_a*N_a*N_j)*shiftdim((0:1:N_e-1),-2);
    V=entireRHS_under(maxindexfull);
    V=reshape(V,[N_a*N_j,N_e]);
    Vhat=reshape(Vhat,[N_a*N_j,N_e]);

elseif vfoptions.lowmemory==1

    n_e_special=ones(1,length(n_e));
    V=zeros(N_a*N_j,N_e,'gpuArray');
    Policy=zeros(N_a,N_j,N_e,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z
    Vhat=zeros(N_a*N_j,N_e,'gpuArray');

    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,:,e_c,:); % e_gridvals_J has shape (j,prod(n_e),l_e) for fastOLG with no z
        ReturnMatrix_e=CreateReturnFnMatrix_fastOLG_Disc_nod(ReturnFn, n_a, n_e_special, N_j, a_grid, a_grid, e_vals, ReturnFnParamsAgeMatrix);
        % fastOLG: ReturnMatrix is [aprime,a,j]

        % First Policy; capture Vhat
        entireRHS_e=ReturnMatrix_e+DiscountedEV; % [aprime,a,j]
        [Vhattmp,maxindex]=max(entireRHS_e,[],1);
        Policy(:,:,e_c)=reshape(maxindex,[N_a,N_j]);
        Vhat(:,e_c)=reshape(Vhattmp,[N_a*N_j,1]);
        % Now Vunderbar
        entireRHS_under_e=ReturnMatrix_e+DiscountedEV_under; % [aprime,a,j]
        maxindexfull=maxindex+N_a*(0:1:N_a-1)+(N_a*N_a)*shiftdim((0:1:N_j-1),-1);
        V(:,e_c)=reshape(entireRHS_under_e(maxindexfull),[N_a*N_j,1]);
    end

    Policy=shiftdim(Policy,-1);

end

%% fastOLG with e, so need to output to take certain shapes
% V=reshape(V,[N_a*N_j,N_e]);
% Policy=reshape(Policy,[N_a,N_j,N_e]);


end
