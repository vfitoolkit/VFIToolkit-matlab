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

VKronNext=[sum(V(N_a+1:end,:,:).*pi_e_J(N_a+1:end,:,:),3); zeros(N_a,N_z,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_z_J to create expectations

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2_fastOLGe(ReturnFn, 0, n_a, n_z, n_e, N_j, [], a_grid, z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix);
    % fastOLG: ReturnMatrix is [aprime,a,j,z]

    %Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=VKronNext.*repelem(pi_z_J,N_a,1,1);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2);

    discountedEV=DiscountFactorParamsVec.*reshape(EV,[N_a,N_j,N_z]); % aprime-j-z

    entireRHS=ReturnMatrix+repelem(discountedEV,1,N_a,1,N_e); %(aprime)-by-(a,j)-by-z-by-e

    %Calc the max and it's index
    [V,Policy]=max(entireRHS,[],1);
    V=shiftdim(V,1);

elseif vfoptions.lowmemory==1

    n_e_special=ones(1,length(n_e));
    
    for e_c=1:N_e
        e_vals=e_gridvals_J(:,:,e_c,:); % e_gridvals_J has shape (j,1,prod(n_e),l_e) for fastOLG
        ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2_fastOLGe(ReturnFn, 0, n_a, n_z, n_e_special, N_j, [], a_grid, z_grid, e_vals, ReturnFnParamsAgeMatrix);
        % fastOLG: ReturnMatrix is [aprime,a,j,z] (e)

        %Calc the condl expectation term (except beta), which depends on z but not on control variables
        EV=VKronNext.*repelem(pi_z_J,N_a,1,1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2);

        discountedEV=DiscountFactorParamsVec.*reshape(EV,[N_a,N_j,N_z]); % aprime-j-z

        entireRHS_e=ReturnMatrix_e+repelem(discountedEV,1,N_a,1); %(aprime)-by-(a,j)-by-z

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS_e,[],1);
        V(:,:,e_c)=Vtemp;
        Policy(:,:,e_c)=maxindex;
    end

elseif vfoptions.lowmemory==2

    n_e_special=ones(1,length(n_e));
    n_z_special=ones(1,length(n_z));

    for e_c=1:N_e
        e_vals=e_gridvals_J(:,:,e_c,:); % e_gridvals_J has shape (j,1,prod(n_e),l_e) for fastOLG
        for z_c=1:N_z
            z_vals=z_gridvals_J(:,z_c,:); % z_gridvals_J has shape (j,prod(n_z),l_z) for fastOLG
            ReturnMatrix_ez=CreateReturnFnMatrix_Case1_Disc_Par2_fastOLGe(ReturnFn, 0, n_a, n_z_special, n_e_special, N_j, [], a_grid, z_vals, e_vals, ReturnFnParamsAgeMatrix);
            % fastOLG: ReturnMatrix is [aprime,a,j] (z,e)

            %Calc the condl expectation term (except beta), which depends on z but not on control variables
            EV_z=VKronNext.*repelem(pi_z_J(:,:,z_c),N_a,1,1);
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);

            discountedEV_z=DiscountFactorParamsVec.*reshape(EV_z,[N_a,N_j]); % aprime-j

            entireRHS_ez=ReturnMatrix_ez+repelem(discountedEV_z,1,N_a); %(aprime)-by-(a,j)

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_ez,[],1);
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
