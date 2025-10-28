function [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_nod_e_raw(V,n_a,n_z,n_e,N_j, a_grid, z_gridvals_J,e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z,e), rather than standard (a,z,e,j)
% V is (a,j)-by-z-by-e

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

Policy=zeros(N_a*N_j,N_z,N_e,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_z (note: N_aprime is just equal to N_a)

DiscountFactorParamsVec=CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec,2);
DiscountFactorParamsVec=shiftdim(DiscountFactorParamsVec,-1);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

VKronNext=[sum(V(N_a+1:end,:,:).*pi_e_J(N_a+1:end,1,:),3); zeros(N_a,N_z,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_z_J to create expectations

EVpre=zeros(N_a,1,N_j,N_z);
EVpre(:,1,1:N_j-1,:)=reshape(V(N_a+1:end,:),[N_a,1,N_j-1,N_z]); % I use zeros in j=N_j so that can just use pi_z_J to create expectations
EV=EVpre.*shiftdim(pi_z_J,-2);
EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a

DiscountedEV=DiscountFactorParamsVec.*EV;


if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2e(ReturnFn, 0, n_a, n_z, n_e, N_j, [], a_grid, z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix);
    % fastOLG: ReturnMatrix is [aprime,a,j,z]

    %Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=VKronNext.*repelem(pi_z_J,N_a,1,1);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2);

    discountedEV=repelem(reshape(DiscountFactorParamsVec.*EV,[N_a,N_j,N_z]),1,N_a,1); %(aprime)-by-(a,j)-by-z

    entireRHS=ReturnMatrix+discountedEV; %(aprime)-by-(a,j)-by-z-by-e

    %Calc the max and it's index
    [V,Policy]=max(entireRHS,[],1);
    V=shiftdim(V,1);

elseif vfoptions.lowmemory==1

    n_e_special=ones(1,length(n_e));

    %Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=VKronNext.*repelem(pi_z_J,N_a,1,1);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2);

    discountedEV=repelem(reshape(DiscountFactorParamsVec.*EV,[N_a,N_j,N_z]),1,N_a,1); % aprime-j
    
    for e_c=1:N_e
        e_vals=e_gridvals_J(:,:,e_c,:); % e_gridvals_J has shape (j,1,prod(n_e),l_e) for fastOLG
        ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2_fastOLGe(ReturnFn, 0, n_a, n_z, n_e_special, N_j, [], a_grid, z_grid, e_vals, ReturnFnParamsAgeMatrix);
        % fastOLG: ReturnMatrix is [aprime,a,j,z] (e)

        entireRHS_e=ReturnMatrix_e+discountedEV; %(aprime)-by-(a,j)-by-z

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS_e,[],1);
        V(:,:,e_c)=Vtemp;
        Policy(:,:,e_c)=maxindex;
    end

elseif vfoptions.lowmemory==2

    n_e_special=ones(1,length(n_e));
    n_z_special=ones(1,length(n_z));

    %Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=VKronNext.*repelem(pi_z_J,N_a,1,1);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2);

    discountedEV=repelem(reshape(DiscountFactorParamsVec.*EV,[N_a,N_j,N_z]),1,N_a,1); % aprime-j

    for z_c=1:N_z
        z_vals=z_gridvals_J(:,z_c,:); % z_gridvals_J has shape (j,prod(n_z),l_z) for fastOLG
        discountedEV_z=discountedEV(:,:,z_c);

        for e_c=1:N_e
            e_vals=e_gridvals_J(:,:,e_c,:); % e_gridvals_J has shape (j,1,prod(n_e),l_e) for fastOLG

            ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2_fastOLGe(ReturnFn, 0, n_a, n_z_special, n_e_special, N_j, [], a_grid, z_vals, e_vals, ReturnFnParamsAgeMatrix);
            % fastOLG: ReturnMatrix is [aprime,a,j] (z,e)

            entireRHS_ze=ReturnMatrix_ze+discountedEV_z; %(aprime)-by-(a,j)

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_ze,[],1);
            V(:,z_c,e_c)=Vtemp;
            Policy(:,z_c,e_c)=maxindex;
        end
    end

end


%% fastOLG with z & e, so need to output to take certain shapes
% V=reshape(V,[N_a*N_j,N_z,N_e]);
Policy=reshape(Policy,[N_a,N_j,N_z,N_e]);
% Note that in fastOLG, we do not separate d from aprime in Policy


end
