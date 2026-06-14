function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_SemiExo_nod1_noz_raw(V,n_d2,n_a,n_semiz,N_j, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,semiz), rather than standard (a,semiz,j)
% V is (a,j)-by-semiz
% Policy is (a,j,semiz)
% semiz_gridvals_J is (j,N_semiz,l_semiz) for fastOLG
% pi_semiz_J is (semiz,semiz',d2,j) [standard form, transitions depend on d2]

n_d=n_d2;
n_bothz=n_semiz; % These are the return function arguments

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_bothz=N_semiz;

d_gridvals=d2_gridvals;

bothz_gridvals_J=shiftdim(semiz_gridvals_J,-3); % [1,1,1,N_j,N_semiz,l_semiz]

pi_semiz_J=permute(pi_semiz_J,[4,2,1,3]); % (j,semiz',semiz,d2)

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_bothz (note: N_aprime is just equal to N_a)

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

if vfoptions.EVpre==0
    EVpre=zeros(N_a,1,N_j,N_bothz);
    EVpre(:,1,1:N_j-1,:)=reshape(V(N_a+1:end,:),[N_a,1,N_j-1,N_bothz]); % I use zeros in j=N_j so that can just use the transition probabilities to create expectations
elseif vfoptions.EVpre==1
    % This is used for 'Matched Expecations Path'
    EVpre=reshape(V,[N_a,1,N_j,N_bothz]); % input V is already of size [N_a*N_j,N_bothz] and we want to use the whole thing
end

% Expectations over the semi-exogenous state depend on d2: compute them for each d2 and stack over d2 (d2 indexes fastest, then aprime)
EV=zeros(N_a,1,N_j,N_bothz,N_d2,'gpuArray');
for d2_c=1:N_d2
    pi_bothz=pi_semiz_J(:,:,:,d2_c); % (j,semiz',semiz)
    EV_d2=EVpre.*shiftdim(pi_bothz,-2);
    EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV(:,1,:,:,d2_c)=reshape(sum(EV_d2,4),[N_a,1,N_j,N_bothz]); % (aprime,1,j,bothz)
end
EV=reshape(permute(EV,[5,1,2,3,4]),[N_d2*N_a,1,N_j,N_bothz]); % (d2 & aprime,1,j,bothz), d2 indexes fastest

DiscountedEV=reshape(DiscountFactor_J,[1,1,N_j]).*EV;

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc(ReturnFn, n_d, n_a, n_bothz, N_j, d_gridvals, a_grid, a_grid, bothz_gridvals_J, ReturnFnParamsAgeMatrix);
    % fastOLG: ReturnMatrix is [d & aprime,a,j,bothz]

    entireRHS=ReturnMatrix+DiscountedEV; % [d & aprime,a,j,bothz]

    %Calc the max and it's index
    [V,Policy]=max(entireRHS,[],1);

    V=reshape(V,[N_a*N_j,N_bothz]);
    Policy=squeeze(Policy);

elseif vfoptions.lowmemory==1

    Policy=zeros(N_a,N_j,N_bothz,'gpuArray');

    special_n_bothz=ones(1,length(n_bothz));

    for z_c=1:N_bothz
        bothz_vals=bothz_gridvals_J(1,1,1,:,z_c,:); % bothz_gridvals_J has shape [1,1,1,N_j,N_bothz,l_bothz] for fastOLG
        DiscountedEV_z=DiscountedEV(:,:,:,z_c);

        ReturnMatrix_z=CreateReturnFnMatrix_fastOLG_Disc(ReturnFn, n_d, n_a, special_n_bothz, N_j, d_gridvals, a_grid, a_grid, bothz_vals, ReturnFnParamsAgeMatrix);
        % fastOLG: ReturnMatrix is [d & aprime,a,j]

        entireRHS_z=ReturnMatrix_z+DiscountedEV_z; %(d,aprime)-by-(a,j)

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS_z,[],1);
        V(:,z_c)=reshape(Vtemp,[N_a*N_j,1]);
        Policy(:,:,z_c)=maxindex;
    end
elseif vfoptions.lowmemory>=2
    error('vfoptions.lowmemory=2 not supported with semi-exogenous states')
end

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point
% Policy contains the joint (d2,aprime) index: d2 indexes fastest, then aprime


end
