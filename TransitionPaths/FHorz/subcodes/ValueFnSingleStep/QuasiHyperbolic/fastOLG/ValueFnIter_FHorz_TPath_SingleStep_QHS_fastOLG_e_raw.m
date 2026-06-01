function [V,Policy,Vhat]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_e_raw(V,n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z), rather than standard (a,z,j)
% V is (a,j)-by-z-by-e (V carries Vunderbar for Sophisticated); Vhat is agent's-perspective (beta0*beta) value

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

DiscountedEV_under=repelem(reshape(beta_J,[1,1,N_j]).*EV,N_d,1,1,1); % [N_d*N_aprime,1,N_j,N_z]
DiscountedEV=repelem(reshape(beta0beta_J,[1,1,N_j]).*EV,N_d,1,1,1); % [N_d*N_aprime,1,N_j,N_z]

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_e(ReturnFn, n_d, n_a, n_z, n_e, N_j, d_gridvals, a_grid, a_grid, z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix);
    % fastOLG: ReturnMatrix is [d,aprime,a,j,z,e]

    % First Policy (QH-optimal); capture Vhat (beta0*beta-step max value, agent's perspective)
    entireRHS=ReturnMatrix+DiscountedEV; %(d,aprime)-by-(a,j,z,e)
    [Vhat,Policy]=max(entireRHS,[],1);
    % Now Vunderbar: evaluate at QH-optimal index with two-future-periods discount factor
    entireRHS_under=ReturnMatrix+DiscountedEV_under; %(d,aprime)-by-(a,j,z,e)
    maxindexfull=Policy+(N_d*N_a)*(0:1:N_a-1)+(N_d*N_a*N_a)*shiftdim((0:1:N_j-1),-1)+(N_d*N_a*N_a*N_j)*shiftdim((0:1:N_z-1),-2)+(N_d*N_a*N_a*N_j*N_z)*shiftdim((0:1:N_e-1),-3);
    V=entireRHS_under(maxindexfull);

    V=reshape(V,[N_a*N_j,N_z,N_e]);
    Vhat=reshape(Vhat,[N_a*N_j,N_z,N_e]);
    Policy=squeeze(Policy);

elseif vfoptions.lowmemory==1

    special_n_e=ones(1,length(n_e));
    V=zeros(N_a*N_j,N_z,N_e,'gpuArray');
    Policy=zeros(N_a,N_j,N_z,N_e,'gpuArray');
    Vhat=zeros(N_a*N_j,N_z,N_e,'gpuArray');

    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,1,:,1,e_c,:); % e_gridvals_J has shape (1,1,1,j,1,prod(n_e),l_e) for fastOLG with d

        ReturnMatrix_e=CreateReturnFnMatrix_fastOLG_Disc_e(ReturnFn, n_d, n_a, n_z, special_n_e, N_j, d_gridvals, a_grid, a_grid, z_gridvals_J, e_vals, ReturnFnParamsAgeMatrix);
        % fastOLG: ReturnMatrix is [d,aprime,a,j,z]

        % First Policy; capture Vhat
        entireRHS_e=ReturnMatrix_e+DiscountedEV; %(d,aprime)-by-(a,j,z)
        [Vhattmp,maxindex]=max(entireRHS_e,[],1);
        Policy(:,:,:,e_c)=maxindex;
        Vhat(:,:,e_c)=reshape(Vhattmp,[N_a*N_j,N_z]);
        % Now Vunderbar
        entireRHS_under_e=ReturnMatrix_e+DiscountedEV_under; %(d,aprime)-by-(a,j,z)
        maxindexfull=maxindex+(N_d*N_a)*(0:1:N_a-1)+(N_d*N_a*N_a)*shiftdim((0:1:N_j-1),-1)+(N_d*N_a*N_a*N_j)*shiftdim((0:1:N_z-1),-2);
        V(:,:,e_c)=reshape(entireRHS_under_e(maxindexfull),[N_a*N_j,N_z]);
    end
elseif vfoptions.lowmemory==2

    special_n_e=ones(1,length(n_e));
    special_n_z=ones(1,length(n_z));
    V=zeros(N_a*N_j,N_z,N_e,'gpuArray');
    Policy=zeros(N_a,N_j,N_z,N_e,'gpuArray');
    Vhat=zeros(N_a*N_j,N_z,N_e,'gpuArray');

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,1,:,z_c,:); % z_gridvals_J has shape (1,1,1,j,prod(n_z),l_z) for fastOLG with d
        DiscountedEV_under_z=DiscountedEV_under(:,:,:,z_c);
        DiscountedEV_z=DiscountedEV(:,:,:,z_c);
        for e_c=1:N_e
            e_vals=e_gridvals_J(1,1,1,:,1,e_c,:); % e_gridvals_J has shape (1,1,1,j,1,prod(n_e),l_e) for fastOLG with d

            ReturnMatrix_ze=CreateReturnFnMatrix_fastOLG_Disc_e(ReturnFn, n_d, n_a, special_n_z, special_n_e, N_j, d_gridvals, a_grid, a_grid, z_vals, e_vals, ReturnFnParamsAgeMatrix);
            % fastOLG: ReturnMatrix is [d,aprime,a,j,z]

            % First Policy; capture Vhat
            entireRHS_ze=ReturnMatrix_ze+DiscountedEV_z; %(d,aprime)-by-(a,j)
            [Vhattmp,maxindex]=max(entireRHS_ze,[],1);
            Policy(:,:,z_c,e_c)=maxindex;
            Vhat(:,z_c,e_c)=reshape(Vhattmp,[N_a*N_j,1]);
            % Now Vunderbar
            entireRHS_under_ze=ReturnMatrix_ze+DiscountedEV_under_z; %(d,aprime)-by-(a,j)
            maxindexfull=maxindex+(N_d*N_a)*(0:1:N_a-1)+(N_d*N_a*N_a)*shiftdim((0:1:N_j-1),-1);
            V(:,z_c,e_c)=reshape(entireRHS_under_ze(maxindexfull),[N_a*N_j,1]);
        end
     end
end

%% fastOLG with z & e, so need output to take certain shapes
% V=reshape(V,[N_a*N_j,N_z,N_e]);
% Policy=reshape(Policy,[N_a,N_j,N_z,N_e]);

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point


end
