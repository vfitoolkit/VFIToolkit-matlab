function [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_nod_noz_e_raw(V,n_a,n_e,N_j, a_grid,e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,e), rather than standard (a,e,j)
% V is (a,j)-by-e

N_a=prod(n_a);
N_e=prod(n_e);

Policy=zeros(N_a*N_j,N_e,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_z (note: N_aprime is just equal to N_a)

DiscountFactorParamsVec=CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec,2);
DiscountFactorParamsVec=shiftdim(DiscountFactorParamsVec,-1);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

EV=[sum(V(N_a+1:end,:).*pi_e_J(N_a+1:end,:),2); zeros(N_a,1,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_z_J to create expectations

if vfoptions.lowmemory==0

    discountedEV=repelem(reshape(DiscountFactorParamsVec.*EV,[N_a,N_j]),1,N_a); % aprime-by-(a,j)

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2_fastOLG(ReturnFn, 0, n_a, n_e, N_j, [], a_grid, e_gridvals_J, ReturnFnParamsAgeMatrix);
    % fastOLG: ReturnMatrix is [aprime,a,j]

    entireRHS=ReturnMatrix+discountedEV; %(aprime)-by-(a,j)-by-e

    %Calc the max and it's index
    [V,Policy]=max(entireRHS,[],1);
    V=shiftdim(V,1);

elseif vfoptions.lowmemory==1

    n_e_special=ones(1,length(n_e));

    discountedEV=repelem(reshape(DiscountFactorParamsVec.*EV,[N_a,N_j]),1,N_a); % aprime-by-(a,j)
    
    for e_c=1:N_e
        e_vals=e_gridvals_J(:,e_c,:); % e_gridvals_J has shape (j,prod(n_e),l_e) for fastOLG with no z
        ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2_fastOLG(ReturnFn, 0, n_a, n_e_special, N_j, [], a_grid, e_vals, ReturnFnParamsAgeMatrix);
        % fastOLG: ReturnMatrix is [aprime,a,j] (e)

        entireRHS_e=ReturnMatrix_e+discountedEV; %(aprime)-by-(a,j)-by-e

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS_e,[],1);
        V(:,e_c)=Vtemp;
        Policy(:,e_c)=maxindex;
    end

end


%% fastOLG with e, so need to output to take certain shapes
% V=reshape(V,[N_a*N_j,N_e]);
Policy=reshape(Policy,[N_a,N_j,N_e]);
% Note that in fastOLG, we do not separate d from aprime in Policy


end
