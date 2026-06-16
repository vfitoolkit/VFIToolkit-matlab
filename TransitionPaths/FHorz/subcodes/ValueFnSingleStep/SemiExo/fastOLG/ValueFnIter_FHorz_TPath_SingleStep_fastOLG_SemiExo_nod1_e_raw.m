function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_SemiExo_nod1_e_raw(V,n_d2,n_a,n_z,n_semiz,n_e,N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,bothz), rather than standard (a,bothz,j), where bothz=(semiz,z) with semiz indexing fastest
% V is (a,j)-by-bothz-by-e
% Policy is (a,j,bothz,e)
% pi_z_J is (j,z',z) for fastOLG
% z_gridvals_J is (j,N_z,l_z) for fastOLG
% semiz_gridvals_J is (j,N_semiz,l_semiz) for fastOLG
% pi_semiz_J is (semiz,semiz',d2,j) [standard form, transitions depend on d2]
% pi_e_J is (a,j)-by-1-by-e for fastOLG

n_d=n_d2;
n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=N_semiz*N_z;
N_e=prod(n_e);

d_gridvals=d2_gridvals;

bothz_gridvals_J=cat(3,repmat(semiz_gridvals_J,1,N_z,1),repelem(z_gridvals_J,1,N_semiz,1)); % (j,N_bothz,l_semiz+l_z), semiz indexes fastest
bothz_gridvals_J=shiftdim(bothz_gridvals_J,-3); % [1,1,1,N_j,N_bothz,l_bothz]
e_gridvals_J=reshape(e_gridvals_J,[1,1,1,N_j,1,N_e,length(n_e)]);

pi_semiz_J=permute(pi_semiz_J,[4,2,1,3]); % (j,semiz',semiz,d2)

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_bothz (note: N_aprime is just equal to N_a)

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

% Take expectations over e (Vnext for age jj is V at age jj+1 weighted by pi_e_J at age jj)
EVpre=[sum(V(N_a+1:end,:,:).*pi_e_J(1:end-N_a,:,:),3); zeros(N_a,N_bothz,'gpuArray')]; % I use zeros in j=N_j so that can just use the transition probabilities to create expectations
EVpre=reshape(EVpre,[N_a,1,N_j,N_bothz]);

% Expectations over the semi-exogenous state depend on d2: compute them for each d2 and stack over d2 (d2 indexes fastest, then aprime)
EV=zeros(N_a,1,N_j,N_bothz,N_d2,'gpuArray');
for d2_c=1:N_d2
    pi_bothz=reshape(reshape(pi_z_J,[N_j,1,N_z,1,N_z]).*reshape(pi_semiz_J(:,:,:,d2_c),[N_j,N_semiz,1,N_semiz,1]),[N_j,N_bothz,N_bothz]); % (j,bothz',bothz) [semiz indexes fastest]
    EV_d2=EVpre.*shiftdim(pi_bothz,-2);
    EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV(:,1,:,:,d2_c)=reshape(sum(EV_d2,4),[N_a,1,N_j,N_bothz]); % (aprime,1,j,bothz)
end
EV=reshape(permute(EV,[5,1,2,3,4]),[N_d2*N_a,1,N_j,N_bothz]); % (d2 & aprime,1,j,bothz), d2 indexes fastest

DiscountedEV=reshape(DiscountFactor_J,[1,1,N_j]).*EV;

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_e(ReturnFn, n_d, n_a, n_bothz, n_e, N_j, d_gridvals, a_grid, a_grid, bothz_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix);
    % fastOLG: ReturnMatrix is [d & aprime,a,j,bothz,e]

    entireRHS=ReturnMatrix+DiscountedEV; % [d & aprime,a,j,bothz,e]

    %Calc the max and it's index
    [V,Policy]=max(entireRHS,[],1);

    V=reshape(V,[N_a*N_j,N_bothz,N_e]);
    Policy=squeeze(Policy);

elseif vfoptions.lowmemory==1

    special_n_e=ones(1,length(n_e));
    V=zeros(N_a*N_j,N_bothz,N_e,'gpuArray');
    Policy=zeros(N_a,N_j,N_bothz,N_e,'gpuArray');

    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,1,:,1,e_c,:); % e_gridvals_J has shape [1,1,1,N_j,1,N_e,l_e] for fastOLG

        ReturnMatrix_e=CreateReturnFnMatrix_fastOLG_Disc_e(ReturnFn, n_d, n_a, n_bothz, special_n_e, N_j, d_gridvals, a_grid, a_grid, bothz_gridvals_J, e_vals, ReturnFnParamsAgeMatrix);
        % fastOLG: ReturnMatrix is [d & aprime,a,j,bothz]

        entireRHS_e=ReturnMatrix_e+DiscountedEV; %(d,aprime)-by-(a,j,bothz)

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS_e,[],1);
        V(:,:,e_c)=reshape(Vtemp,[N_a*N_j,N_bothz]);
        Policy(:,:,:,e_c)=maxindex;
    end
elseif vfoptions.lowmemory>=2
    error('vfoptions.lowmemory=2 not supported with semi-exogenous states')
end

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point
% Policy contains the joint (d2,aprime) index: d2 indexes fastest, then aprime


end
