function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_noz_e_raw(V,n_d,n_a,n_e,N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,e), rather than standard (a,e,j)
% V is (a,j)-by-e

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

e_gridvals_J=shiftdim(e_gridvals_J,-3); % [1,1,1,N_j,N_e,l_e]

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_z (note: N_aprime is just equal to N_a)

DiscountFactorParamsVec=CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec,2);
DiscountFactorParamsVec=shiftdim(DiscountFactorParamsVec,-2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

EV=[sum(V(N_a+1:end,:).*pi_e_J(N_a+1:end,:),2); zeros(N_a,1,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_e_J to create expectations

discountedEV=repelem(DiscountFactorParamsVec.*reshape(EV,[N_a,1,N_j]),N_d,1,1); % [d & aprime, 1, j]

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_Par2(ReturnFn, n_d, n_e, N_j, d_gridvals, a_grid', a_grid, e_gridvals_J, ReturnFnParamsAgeMatrix,2);
    % fastOLG: ReturnMatrix is [d,aprime,a,j,e]
    
    entirediscountedEV=ReturnMatrix+discountedEV; %(d,aprime)-by-(a,j,e)

    %Calc the max and it's index
    [V,Policy]=max(entirediscountedEV,[],1);

    V=reshape(V,[N_a*N_j,N_e]);
    Policy=squeeze(Policy);

elseif vfoptions.lowmemory==1

    special_n_e=ones(1,length(n_e));

    Policy=zeros(N_a*N_j,N_e,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,1,:,e_c,:); % e_gridvals_J has shape (j,prod(n_e),l_e) for fastOLG with no z

        ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_Par2(ReturnFn, n_d, special_n_e, N_j, d_gridvals, a_grid', a_grid, e_vals, ReturnFnParamsAgeMatrix,2);
        % fastOLG: ReturnMatrix is [d,aprime,a,j,e]

        entirediscountedEV_e=ReturnMatrix_e+discountedEV; %(d,aprime)-by-(a,j,e)

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entirediscountedEV_e,[],1);
        V(:,e_c)=reshape(Vtemp,[N_a*N_j,1]);
        Policy(:,e_c)=reshape(maxindex,[N_a*N_j,1]);
    end
end

%% fastOLG with e, so need output to take certain shapes
% V=reshape(V,[N_a*N_j,N_e]);
Policy=reshape(Policy,[N_a,N_j,N_e]);
% Note that in fastOLG, we do not separate d from aprime in Policy


end
