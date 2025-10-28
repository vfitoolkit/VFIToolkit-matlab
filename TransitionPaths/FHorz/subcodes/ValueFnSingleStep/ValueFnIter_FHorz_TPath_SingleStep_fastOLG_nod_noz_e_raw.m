function [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_nod_noz_e_raw(V,n_a,n_e,N_j, a_grid,e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,e), rather than standard (a,e,j)
% V is (a,j)-by-e

N_a=prod(n_a);
N_e=prod(n_e);

e_gridvals_J=shiftdim(e_gridvals_J,-2);

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_z (note: N_aprime is just equal to N_a)

DiscountFactorParamsVec=CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec,2);
DiscountFactorParamsVec=shiftdim(DiscountFactorParamsVec,-2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

EV=[sum(V(N_a+1:end,:).*pi_e_J(N_a+1:end,:),2); zeros(N_a,1,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_z_J to create expectations

discountedEV=DiscountFactorParamsVec.*reshape(EV,[N_a,1,N_j]); % [N_aprime,1,N_j] % 2nd dim will be autofilled with a

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2(ReturnFn, n_e, N_j, a_grid, a_grid, e_gridvals_J, ReturnFnParamsAgeMatrix,1);
    % fastOLG: ReturnMatrix is [aprime,a,j,e]

    entireRHS=ReturnMatrix+discountedEV; % [aprime,a,j,e]

    %Calc the max and it's index
    [V,Policy]=max(entireRHS,[],1);
    V=reshape(V,[N_a*N_j,N_e]);

elseif vfoptions.lowmemory==1

    n_e_special=ones(1,length(n_e));
    Policy=zeros(N_a*N_j,N_e,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z
    
    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,:,e_c,:); % e_gridvals_J has shape (j,prod(n_e),l_e) for fastOLG with no z
        ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2(ReturnFn, n_e_special, N_j, a_grid, a_grid, e_vals, ReturnFnParamsAgeMatrix,1);
        % fastOLG: ReturnMatrix is [aprime,a,j]

        entireRHS_e=ReturnMatrix_e+discountedEV; % [aprime,a,j]

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS_e,[],1);
        V(:,e_c)=Vtemp;
        Policy(:,e_c)=maxindex;
    end

    Policy=reshape(Policy,[N_a,N_j,N_e]);

end


%% fastOLG with e, so need to output to take certain shapes
% V=reshape(V,[N_a*N_j,N_e]);
% Policy=reshape(Policy,[N_a,N_j,N_e]);
% Note that in fastOLG, we do not separate d from aprime in Policy


end
