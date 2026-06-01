function [V,Policy,Policyalt,Vtilde]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_raw(V,n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z), rather than standard (a,z,j)
% V is (a,j)-by-z (V carries Valt for Naive); Vtilde is agent's-perspective (beta0*beta) value
% Policy is (a,j,z)
% pi_z_J is (j,z',z) for fastOLG
% z_gridvals_J is (j,N_z,l_z) for fastOLG


N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

z_gridvals_J=shiftdim(z_gridvals_J,-3); % [1,1,1,N_j,N_z,l_z]

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_z (note: N_aprime is just equal to N_a)

beta_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);
beta0_J=CreateAgeMatrixFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
beta0beta_J=beta0_J.*beta_J; % Discount factor between today and tomorrow.

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

if vfoptions.EVpre==0
    EVpre=zeros(N_a,1,N_j,N_z);
    EVpre(:,1,1:N_j-1,:)=reshape(V(N_a+1:end,:),[N_a,1,N_j-1,N_z]); % I use zeros in j=N_j so that can just use pi_z_J to create expectations
    EV=EVpre.*shiftdim(pi_z_J,-2);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a
elseif vfoptions.EVpre==1
    % This is used for 'Matched Expecations Path'
    EV=reshape(V,[N_a,1,N_j,N_z]).*shiftdim(pi_z_J,-2); % input V is already of size [N_a,N_j,N_z] and we want to use the whole thing
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a
end

DiscountedEV_alt=reshape(beta_J,[1,1,N_j]).*EV;
DiscountedEV_alt=repelem(DiscountedEV_alt,N_d,1,1,1); % [d & aprime,1,j,z]
DiscountedEV=reshape(beta0beta_J,[1,1,N_j]).*EV;
DiscountedEV=repelem(DiscountedEV,N_d,1,1,1); % [d & aprime,1,j,z]

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc(ReturnFn, n_d, n_a, n_z, N_j, d_gridvals, a_grid, a_grid, z_gridvals_J, ReturnFnParamsAgeMatrix);
    % fastOLG: ReturnMatrix is [d & aprime,a,j,z]

    % First Valt (exponential-discounter value)
    entireRHS_alt=ReturnMatrix+DiscountedEV_alt; %  [d & aprime,a,j,z]
    [V,Policyalt]=max(entireRHS_alt,[],1);
    % Now Policy (QH-optimal, drives agent dist); capture Vtilde (beta0*beta-step max value, agent's perspective)
    entireRHS=ReturnMatrix+DiscountedEV; %  [d & aprime,a,j,z]
    [Vtilde,Policy]=max(entireRHS,[],1);

    V=reshape(V,[N_a*N_j,N_z]);
    Vtilde=reshape(Vtilde,[N_a*N_j,N_z]);
    Policy=squeeze(Policy);
    Policyalt=squeeze(Policyalt);

elseif vfoptions.lowmemory==1

    Policy=zeros(N_a,N_j,N_z,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z
    Policyalt=zeros(N_a,N_j,N_z,'gpuArray');
    Vtilde=zeros(N_a*N_j,N_z,'gpuArray');

    special_n_z=ones(1,length(n_z));

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,1,:,z_c,:); % z_gridvals_J has shape (j,prod(n_z),l_z) for fastOLG
        DiscountedEV_alt_z=DiscountedEV_alt(:,:,:,z_c);
        DiscountedEV_z=DiscountedEV(:,:,:,z_c);

        ReturnMatrix_z=CreateReturnFnMatrix_fastOLG_Disc(ReturnFn, n_d, n_a, special_n_z, N_j, d_gridvals, a_grid, a_grid, z_vals, ReturnFnParamsAgeMatrix);
        % fastOLG: ReturnMatrix is [d,aprime,a,j]

        % First Valt
        entireRHS_alt_z=ReturnMatrix_z+DiscountedEV_alt_z; %(d,aprime)-by-(a,j)
        [Vtemp,maxindex_alt]=max(entireRHS_alt_z,[],1);
        V(:,z_c)=reshape(Vtemp,[N_a*N_j,1]);
        Policyalt(:,:,z_c)=maxindex_alt;
        % Now Policy; capture Vtilde
        entireRHS_z=ReturnMatrix_z+DiscountedEV_z; %(d,aprime)-by-(a,j)
        [Vtildetmp,maxindex]=max(entireRHS_z,[],1);
        Policy(:,:,z_c)=maxindex;
        Vtilde(:,z_c)=reshape(Vtildetmp,[N_a*N_j,1]);
    end
end

%% fastOLG with z, so need to output to take certain shapes
% V=reshape(V,[N_a*N_j,N_z]);
% Policy=reshape(Policy,[1,N_a,N_j,N_z]);

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point
Policyalt=shiftdim(Policyalt,-1);


end
