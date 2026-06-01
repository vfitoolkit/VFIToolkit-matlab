function [V,Policy,Policyalt,Vtilde]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_e_raw(V,n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z), rather than standard (a,z,j)
% V is (a,j)-by-z-by-e (V carries Valt for Naive); Vtilde is agent's-perspective (beta0*beta) value

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

z_gridvals_J=shiftdim(z_gridvals_J,-3);
e_gridvals_J=reshape(e_gridvals_J,[1,1,1,N_j,1,N_e,length(n_e)]);

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_z (note: N_aprime is just equal to N_a)

beta_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);
beta0_J=CreateAgeMatrixFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
beta0beta_J=beta0_J.*beta_J; % Discount factor between today and tomorrow.

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

EVpre=[sum(V(N_a+1:end,:,:).*pi_e_J(N_a+1:end,:,:),3); zeros(N_a,N_z,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_z_J to create expectations
EVpre=reshape(EVpre,[N_a,1,N_j,N_z]);
EV=EVpre.*shiftdim(pi_z_J,-2);
EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a

DiscountedEV_alt=repelem(reshape(beta_J,[1,1,N_j]).*EV,N_d,1,1,1); % [N_d*N_aprime,1,N_j,N_z]
DiscountedEV=repelem(reshape(beta0beta_J,[1,1,N_j]).*EV,N_d,1,1,1); % [N_d*N_aprime,1,N_j,N_z]

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_e(ReturnFn, n_d, n_a, n_z, n_e, N_j, d_gridvals, a_grid, a_grid, z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix);
    % fastOLG: ReturnMatrix is [d,aprime,a,j,z,e]

    % First Valt
    entireRHS_alt=ReturnMatrix+DiscountedEV_alt; %(d,aprime)-by-(a,j,z,e)
    [V,Policyalt]=max(entireRHS_alt,[],1);
    % Now Policy; capture Vtilde (beta0*beta-step max value, agent's perspective)
    entireRHS=ReturnMatrix+DiscountedEV; %(d,aprime)-by-(a,j,z,e)
    [Vtilde,Policy]=max(entireRHS,[],1);

    V=reshape(V,[N_a*N_j,N_z,N_e]);
    Vtilde=reshape(Vtilde,[N_a*N_j,N_z,N_e]);
    Policy=squeeze(Policy);
    Policyalt=squeeze(Policyalt);

elseif vfoptions.lowmemory==1

    special_n_e=ones(1,length(n_e));
    V=zeros(N_a*N_j,N_z,N_e,'gpuArray');
    Policy=zeros(N_a,N_j,N_z,N_e,'gpuArray');
    Policyalt=zeros(N_a,N_j,N_z,N_e,'gpuArray');
    Vtilde=zeros(N_a*N_j,N_z,N_e,'gpuArray');

    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,1,:,1,e_c,:); % e_gridvals_J has shape (1,1,1,j,1,prod(n_e),l_e) for fastOLG with d

        ReturnMatrix_e=CreateReturnFnMatrix_fastOLG_Disc_e(ReturnFn, n_d, n_a, n_z, special_n_e, N_j, d_gridvals, a_grid, a_grid, z_gridvals_J, e_vals, ReturnFnParamsAgeMatrix);
        % fastOLG: ReturnMatrix is [d,aprime,a,j,z]

        % First Valt
        entireRHS_alt_e=ReturnMatrix_e+DiscountedEV_alt; %(d,aprime)-by-(a,j,z)
        [Vtemp,maxindex_alt]=max(entireRHS_alt_e,[],1);
        V(:,:,e_c)=reshape(Vtemp,[N_a*N_j,N_z]);
        Policyalt(:,:,:,e_c)=maxindex_alt;
        % Now Policy; capture Vtilde
        entireRHS_e=ReturnMatrix_e+DiscountedEV; %(d,aprime)-by-(a,j,z)
        [Vtildetmp,maxindex]=max(entireRHS_e,[],1);
        Policy(:,:,:,e_c)=maxindex;
        Vtilde(:,:,e_c)=reshape(Vtildetmp,[N_a*N_j,N_z]);
    end
elseif vfoptions.lowmemory==2

    special_n_e=ones(1,length(n_e));
    special_n_z=ones(1,length(n_z));
    V=zeros(N_a*N_j,N_z,N_e,'gpuArray');
    Policy=zeros(N_a,N_j,N_z,N_e,'gpuArray');
    Policyalt=zeros(N_a,N_j,N_z,N_e,'gpuArray');
    Vtilde=zeros(N_a*N_j,N_z,N_e,'gpuArray');

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,1,:,z_c,:); % z_gridvals_J has shape (1,1,1,j,prod(n_z),l_z) for fastOLG with d
        DiscountedEV_alt_z=DiscountedEV_alt(:,:,:,z_c);
        DiscountedEV_z=DiscountedEV(:,:,:,z_c);
        for e_c=1:N_e
            e_vals=e_gridvals_J(1,1,1,:,1,e_c,:); % e_gridvals_J has shape (1,1,1,j,1,prod(n_e),l_e) for fastOLG with d

            ReturnMatrix_ze=CreateReturnFnMatrix_fastOLG_Disc_e(ReturnFn, n_d, n_a, special_n_z, special_n_e, N_j, d_gridvals, a_grid, a_grid, z_vals, e_vals, ReturnFnParamsAgeMatrix);
            % fastOLG: ReturnMatrix is [d,aprime,a,j,z]

            % First Valt
            entireRHS_alt_ze=ReturnMatrix_ze+DiscountedEV_alt_z; %(d,aprime)-by-(a,j)
            [Vtemp,maxindex_alt]=max(entireRHS_alt_ze,[],1);
            V(:,z_c,e_c)=reshape(Vtemp,[N_a*N_j,1]);
            Policyalt(:,:,z_c,e_c)=maxindex_alt;
            % Now Policy; capture Vtilde
            entireRHS_ze=ReturnMatrix_ze+DiscountedEV_z; %(d,aprime)-by-(a,j)
            [Vtildetmp,maxindex]=max(entireRHS_ze,[],1);
            Policy(:,:,z_c,e_c)=maxindex;
            Vtilde(:,z_c,e_c)=reshape(Vtildetmp,[N_a*N_j,1]);
        end
     end
end

%% fastOLG with z & e, so need output to take certain shapes
% V=reshape(V,[N_a*N_j,N_z,N_e]);
% Policy=reshape(Policy,[N_a,N_j,N_z,N_e]);

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point
Policyalt=shiftdim(Policyalt,-1);


end
