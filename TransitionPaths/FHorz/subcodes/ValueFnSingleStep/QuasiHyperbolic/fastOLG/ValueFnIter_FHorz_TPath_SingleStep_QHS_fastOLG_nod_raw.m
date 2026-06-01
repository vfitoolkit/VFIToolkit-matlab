function [V, Policy, Vhat]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_nod_raw(V,n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z), rather than standard (a,z,j)
% V is (a,j)-by-z (V carries Vunderbar for Sophisticated); Vhat is agent's-perspective (beta0*beta) value
% pi_z_J is (j,z',z) for fastOLG
% z_gridvals_J is (j,N_z,l_z) for fastOLG

N_a=prod(n_a);
N_z=prod(n_z);

z_gridvals_J=shiftdim(z_gridvals_J,-2); % [1,1,N_j,N_z,l_z]

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

DiscountedEV_under=reshape(beta_J,[1,1,N_j]).*EV;
DiscountedEV=reshape(beta0beta_J,[1,1,N_j]).*EV;

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_nod(ReturnFn, n_a, n_z, N_j, a_grid, a_grid, z_gridvals_J, ReturnFnParamsAgeMatrix);
    % fastOLG: ReturnMatrix is [aprime,a,j,z]

    % First Policy (QH-optimal); capture Vhat (beta0*beta-step max value, agent's perspective)
    entireRHS=ReturnMatrix+DiscountedEV; % [aprime,a,j,z]
    [Vhat,Policy]=max(entireRHS,[],1);
    % Now Vunderbar: evaluate at QH-optimal index with two-future-periods discount factor
    entireRHS_under=ReturnMatrix+DiscountedEV_under; % [aprime,a,j,z]
    maxindexfull=Policy+N_a*(0:1:N_a-1)+(N_a*N_a)*shiftdim((0:1:N_j-1),-1)+(N_a*N_a*N_j)*shiftdim((0:1:N_z-1),-2);
    V=entireRHS_under(maxindexfull);

    V=reshape(V,[N_a*N_j,N_z]);
    Vhat=reshape(Vhat,[N_a*N_j,N_z]);
    Policy=squeeze(Policy);

elseif vfoptions.lowmemory==1

    Policy=zeros(N_a,N_j,N_z,'gpuArray'); %first dim indexes the optimal choice for aprime
    Vhat=zeros(N_a*N_j,N_z,'gpuArray');

    special_n_z=ones(1,length(n_z));

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,:,z_c,:); % z_gridvals_J has shape (j,prod(n_z),l_z) for fastOLG
        DiscountedEV_under_z=DiscountedEV_under(:,:,:,z_c);
        DiscountedEV_z=DiscountedEV(:,:,:,z_c);

        ReturnMatrix_z=CreateReturnFnMatrix_fastOLG_Disc_nod(ReturnFn, n_a, special_n_z, N_j, a_grid, a_grid, z_vals, ReturnFnParamsAgeMatrix);
        % fastOLG: ReturnMatrix_z is [aprime,a,j]

        % First Policy; capture Vhat
        entireRHS_z=ReturnMatrix_z+DiscountedEV_z; % [aprime,a,j]
        [Vhattmp,maxindex]=max(entireRHS_z,[],1);
        Policy(:,:,z_c)=maxindex;
        Vhat(:,z_c)=reshape(Vhattmp,[N_a*N_j,1]);
        % Now Vunderbar
        entireRHS_under_z=ReturnMatrix_z+DiscountedEV_under_z; % [aprime,a,j]
        maxindexfull=maxindex+N_a*(0:1:N_a-1)+(N_a*N_a)*shiftdim((0:1:N_j-1),-1);
        V(:,z_c)=reshape(entireRHS_under_z(maxindexfull),[N_a*N_j,1]);
    end

end


%% fastOLG with z, so need to output to take certain shapes
% V=reshape(V,[N_a*N_j,N_z]);
% Policy=reshape(Policy,[1,N_a,N_j,N_z]);

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point


end
