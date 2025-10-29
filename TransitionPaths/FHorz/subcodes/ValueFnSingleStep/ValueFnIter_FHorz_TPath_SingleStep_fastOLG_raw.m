function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_raw(V,n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z), rather than standard (a,z,j)
% V is (a,j)-by-z
% Policy is (a,j,z)
% pi_z_J is (j,z',z) for fastOLG
% z_gridvals_J is (j,N_z,l_z) for fastOLG


N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

z_gridvals_J=shiftdim(z_gridvals_J,-3); % [1,1,1,N_j,N_z,l_z]

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_z (note: N_aprime is just equal to N_a)

DiscountFactorParamsVec=CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec,2);
DiscountFactorParamsVec=shiftdim(DiscountFactorParamsVec,-2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

EVpre=zeros(N_a,1,N_j,N_z);
EVpre(:,1,1:N_j-1,:)=reshape(V(N_a+1:end,:),[N_a,1,N_j-1,N_z]); % I use zeros in j=N_j so that can just use pi_z_J to create expectations
EV=EVpre.*shiftdim(pi_z_J,-2);
EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a

DiscountedEV=DiscountFactorParamsVec.*EV;
DiscountedEV=repelem(DiscountedEV,N_d,1,1,1); % [d & aprime,1,j,z]

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_Par2(ReturnFn, n_d, n_z, N_j, d_gridvals, a_grid', a_grid, z_gridvals_J, ReturnFnParamsAgeMatrix,2);
    % fastOLG: ReturnMatrix is [d & aprime,a,j,z]

    entireRHS=ReturnMatrix+DiscountedEV; %  [d & aprime,a,j,z]

    %Calc the max and it's index
    [V,Policy]=max(entireRHS,[],1);

    V=reshape(V,[N_a*N_j,N_z]);
    Policy=squeeze(Policy);

elseif vfoptions.lowmemory==1

    Policy=zeros(N_a,N_j,N_z,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

    special_n_z=ones(1,length(n_z));

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,1,:,z_c,:); % z_gridvals_J has shape (j,prod(n_z),l_z) for fastOLG
        DiscountedEV_z=DiscountedEV(:,:,:,z_c);

        ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_Par2(ReturnFn, n_d, special_n_z, N_j, d_gridvals, a_grid', a_grid, z_vals, ReturnFnParamsAgeMatrix,2);
        % fastOLG: ReturnMatrix is [d,aprime,a,j]

        entireRHS_z=ReturnMatrix_z+DiscountedEV_z; %(d,aprime)-by-(a,j)

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS_z,[],1);
        V(:,z_c)=reshape(Vtemp,[N_a*N_j,1]);
        Policy(:,:,z_c)=maxindex;
    end
end

%% fastOLG with z, so need to output to take certain shapes
% V=reshape(V,[N_a*N_j,N_z]);
% Policy=reshape(Policy,[N_a,N_j,N_z]);
% Note that in fastOLG, we do not separate d from aprime in Policy

end
