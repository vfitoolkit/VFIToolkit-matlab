function [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_nod_e_raw(V,n_a,n_z,n_e,N_j, a_grid, z_gridvals_J,e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z,e), rather than standard (a,z,e,j)
% V is (a,j)-by-z-by-e

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

z_gridvals_J=shiftdim(z_gridvals_J,-2); % [1,1,N_j,N_z,l_z]
e_gridvals_J=reshape(e_gridvals_J,[1,1,N_j,1,N_e,length(n_e)]); % [1,1,N_j,1,N_e,l_e]


%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_z (note: N_aprime is just equal to N_a)

DiscountFactorParamsVec=CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec,2);
DiscountFactorParamsVec=shiftdim(DiscountFactorParamsVec,-2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

EVpre=[sum(V(N_a+1:end,:,:).*pi_e_J(N_a+1:end,:,:),3); zeros(N_a,N_z,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_z_J to create expectations
EVpre=reshape(EVpre,[N_a,1,N_j,N_z]);
EV=EVpre.*shiftdim(pi_z_J,-2);
EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a

DiscountedEV=DiscountFactorParamsVec.*EV;

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2e(ReturnFn, n_z, n_e, N_j, a_grid, a_grid, z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,1);
    % fastOLG: ReturnMatrix is [aprime,a,j,z]

    entireRHS=ReturnMatrix+DiscountedEV; %(aprime)-by-(a,j)-by-z-by-e

    %Calc the max and it's index
    [V,Policy]=max(entireRHS,[],1);
    V=reshape(V,[N_a*N_j,N_z,N_e]);

elseif vfoptions.lowmemory==1

    special_n_e=ones(1,length(n_e));
    V=zeros(N_a*N_j,N_z,N_e,'gpuArray');
    Policy=zeros(N_a,N_j,N_z,N_e,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z
    
    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,:,1,e_c,:); % e_gridvals_J has shape (1,1,N_j,1,prod(n_e),l_e)

        ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2e(ReturnFn, n_z, special_n_e, N_j, a_grid, a_grid, z_gridvals_J, e_vals, ReturnFnParamsAgeMatrix,1);
        % fastOLG: ReturnMatrix is [aprime,a,j,z] (e)

        entireRHS_e=ReturnMatrix_e+DiscountedEV; %(aprime)-by-(a,j)-by-z

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS_e,[],1);
        V(:,:,e_c)=reshape(Vtemp,[N_a*N_j,N_z]);
        Policy(:,:,:,e_c)=reshape(maxindex,[N_a,N_j,N_z]);
    end

elseif vfoptions.lowmemory==2

    special_n_e=ones(1,length(n_e));
    special_n_z=ones(1,length(n_z));
    V=zeros(N_a*N_j,N_z,N_e,'gpuArray');
    Policy=zeros(N_a,N_j,N_z,N_e,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,:,z_c,:); % z_gridvals_J has shape (1,1,j,prod(n_z),l_z)
        DiscountedEV_z=DiscountedEV(:,:,:,z_c);

        for e_c=1:N_e
            e_vals=e_gridvals_J(1,1,:,1,e_c,:); % e_gridvals_J has shape (1,1,N_j,1,prod(n_e),l_e)

            ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2e(ReturnFn, special_n_z, special_n_e, N_j, a_grid, a_grid, z_vals, e_vals, ReturnFnParamsAgeMatrix,1);
            % fastOLG: ReturnMatrix is [aprime,a,j] (z,e)

            entireRHS_ze=ReturnMatrix_ze+DiscountedEV_z; %(aprime)-by-(a,j)

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_ze,[],1);
            V(:,z_c,e_c)=reshape(Vtemp,[N_a*N_j,1]);
            Policy(:,:,z_c,e_c)=reshape(maxindex,[N_a,N_j]);
        end
    end

end


%% fastOLG with z & e, so need to output to take certain shapes
% V=reshape(V,[N_a*N_j,N_z,N_e]);
% Policy=reshape(Policy,[N_a,N_j,N_z,N_e]);

%% Output shape for policy
if vfoptions.lowmemory>0
    Policy=shiftdim(Policy,-1); % so first dim is just one point
end

end
