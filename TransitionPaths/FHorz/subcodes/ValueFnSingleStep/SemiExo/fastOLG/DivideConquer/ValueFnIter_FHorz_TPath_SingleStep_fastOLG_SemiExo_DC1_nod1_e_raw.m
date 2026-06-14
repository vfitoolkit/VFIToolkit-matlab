function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_SemiExo_DC1_nod1_e_raw(V,n_d2,n_a,n_z,n_semiz,n_e,N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z,e), rather than standard (a,z,e,j)
% V is (a,j)-by-z-by-e

n_d=n_d2;
N_d2=prod(n_d2);
N_d=N_d2;
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
n_bothz=[n_semiz,n_z]; % These are the return function arguments
N_bothz=N_semiz*N_z;
N_e=prod(n_e);

% fastOLG, so a-j-z
d_gridvals=d2_gridvals;
bothz_gridvals_J=cat(3,repmat(semiz_gridvals_J,1,N_z,1),repelem(z_gridvals_J,1,N_semiz,1)); % (j,N_bothz,l_semiz+l_z), semiz indexes fastest
bothz_gridvals_J=shiftdim(bothz_gridvals_J,-3); % [1,1,1,N_j,N_bothz,l_bothz]
pi_semiz_J=permute(pi_semiz_J,[4,2,1,3]); % (j,semiz',semiz,d2)
e_gridvals_J=reshape(e_gridvals_J,[1,1,1,N_j,1,N_e,length(n_e)]);

Policy=zeros(N_a,N_j,N_bothz,N_e,'gpuArray'); % first dim indexes the optimal choice for d and aprime

%%

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

jind=shiftdim(gpuArray(0:1:N_j-1),-2);
zind=shiftdim(gpuArray(0:1:N_bothz-1),-3);

jBind=shiftdim(gpuArray(0:1:N_j-1),-1);
zBind=shiftdim(gpuArray(0:1:N_bothz-1),-2);
eBind=shiftdim(gpuArray(0:1:N_e-1),-3);

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_bothz*N_e (note: N_aprime is just equal to N_a)

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

if vfoptions.EVpre==0
    EVpre=[sum(V(N_a+1:end,:,:).*pi_e_J(1:end-N_a,:,:),3); zeros(N_a,N_bothz,'gpuArray')]; % I use zeros in j=N_j so that can just use the transition probabilities to create expectations
    EVpre=reshape(EVpre,[N_a,1,N_j,N_bothz]);
elseif vfoptions.EVpre==1
    % This is used for 'Matched Expecations Path'
    EVpre=sum(reshape(V,[N_a*N_j,N_bothz,N_e]).*pi_e_J,3); % input V is already of size [N_a*N_j,N_bothz,N_e] and we want to use the whole thing
    EVpre=reshape(EVpre,[N_a,1,N_j,N_bothz]);
end

% Expectations over the semi-exogenous state depend on d2: compute them for each d2 and stack over d2
EV=zeros(N_a,1,N_j,N_bothz,N_d2,'gpuArray');
for d2_c=1:N_d2
    pi_bothz=reshape(reshape(pi_z_J,[N_j,1,N_z,1,N_z]).*reshape(pi_semiz_J(:,:,:,d2_c),[N_j,N_semiz,1,N_semiz,1]),[N_j,N_bothz,N_bothz]); % (j,bothz',bothz) [semiz indexes fastest]
    EV_d2=EVpre.*shiftdim(pi_bothz,-2);
    EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV(:,1,:,:,d2_c)=reshape(sum(EV_d2,4),[N_a,1,N_j,N_bothz]); % (aprime,1,j,bothz)
end
V=zeros(N_a,N_j,N_bothz,N_e,'gpuArray'); % V is over (a,j)

DiscountedEV=reshape(DiscountFactor_J,[1,1,N_j]).*EV;
DiscountedEV=permute(DiscountedEV,[5,1,2,3,4]); % [N_d2,N_aprime,1,N_j,N_bothz] (d2 indexes the first dimension)

if vfoptions.lowmemory==0

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_bothz, n_e, N_j, d_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,1);

    entireRHS_ii=ReturnMatrix_ii+DiscountedEV; % (d,aprime,a and j,z,e), autofills a and e for expectation term

    % First, we want aprime conditional on (d,1,a,j,z,e)
    [RMtemp_ii,maxindex1]=max(entireRHS_ii,[],2);
    % Now, we get the d and we store the (d,aprime) and the

    %Calc the max and it's index
    [Vtempii,maxindex2]=max(RMtemp_ii,[],1);
    maxindex2=shiftdim(maxindex2,2); % d
    maxindex1d=maxindex1(maxindex2+N_d*(0:1:vfoptions.level1n-1)'+N_d*vfoptions.level1n*(0:1:N_j-1)+N_d*vfoptions.level1n*N_j*shiftdim((0:1:N_bothz-1),-1)+N_d*vfoptions.level1n*N_j*N_bothz*shiftdim((0:1:N_e-1),-2)); % aprime

    % Store
    V(level1ii,:,:,:)=shiftdim(Vtempii,2);
    Policy(level1ii,:,:,:)=maxindex2+N_d*(maxindex1d-1); % d,aprime

    % Attempt for improved version
    maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
    for ii=1:(vfoptions.level1n-1)
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:,:,:),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-N_j-by-N_bothz-by-N_e
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_bothz, n_e, N_j, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,2);
            daprimejz=(1:1:N_d)'+N_d*repelem(aprimeindexes-1,1,1,level1iidiff(ii),1,1)+N_d*N_a*jind+N_d*N_a*N_j*zind; % all the d, with the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV(daprimejz(:)),[N_d*(maxgap(ii)+1),level1iidiff(ii),N_j,N_bothz,N_e]); % autofill e for the expectations
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,:,:)=shiftdim(Vtempii,1);
            d_ind=rem(maxindex-1,N_d)+1;
            allind=d_ind+N_d*jBind+N_d*N_j*zBind+N_d*N_j*N_bothz*eBind;
            Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,:,:)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1); % loweredge(given the d)
        else
            loweredge=maxindex1(:,1,ii,:,:,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_bothz, n_e, N_j, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,2);
            daprimejz=(1:1:N_d)'+N_d*repelem(loweredge-1,1,1,level1iidiff(ii),1,1)+N_d*N_a*jind+N_d*N_a*N_j*zind; % all the d, with the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV(daprimejz(:)),[N_d,level1iidiff(ii),N_j,N_bothz,N_e]); % autofill e for the expectations
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,:,:)=shiftdim(Vtempii,1);
            d_ind=rem(maxindex-1,N_d)+1;
            allind=d_ind+N_d*jBind+N_d*N_j*zBind+N_d*N_j*N_bothz*eBind;
            Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,:,:)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1); % loweredge(given the d)
        end
    end

elseif vfoptions.lowmemory==1

    special_n_e=ones(1,length(n_e));

    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,1,:,1,e_c,:); % e_gridvals_J has shape (1,1,1,j,1,prod(n_e),l_e)

        % n-Monotonicity
        ReturnMatrix_ii_e=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_bothz, special_n_e, N_j, d_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J, e_vals, ReturnFnParamsAgeMatrix,1);

        entireRHS_ii=ReturnMatrix_ii_e+DiscountedEV; % (d,aprime,a and j,z), autofills a for expectation term

        % First, we want aprime conditional on (d,1,a,j)
        [RMtemp_ii,maxindex1]=max(entireRHS_ii,[],2);
        % Now, we get the d and we store the (d,aprime) and the

        %Calc the max and it's index
        [Vtempii,maxindex2]=max(RMtemp_ii,[],1);
        maxindex2=shiftdim(maxindex2,2); % d
        maxindex1d=maxindex1(maxindex2+N_d*(0:1:vfoptions.level1n-1)'+N_d*vfoptions.level1n*(0:1:N_j-1)+N_d*vfoptions.level1n*N_j*shiftdim((0:1:N_bothz-1),-1)); % aprime

        % Store
        V(level1ii,:,:,e_c)=shiftdim(Vtempii,2);
        Policy(level1ii,:,:,e_c)=maxindex2+N_d*(maxindex1d-1); % d,aprime

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-N_j-by-N_bothz
                ReturnMatrix_ii_e=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_bothz, special_n_e, N_j, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J, e_vals, ReturnFnParamsAgeMatrix,2);
                daprimejz=(1:1:N_d)'+N_d*repelem(aprimeindexes-1,1,1,level1iidiff(ii),1,1)+N_d*N_a*jind+N_d*N_a*N_j*zind; % all the d, with the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii_e+reshape(DiscountedEV(daprimejz(:)),[N_d*(maxgap(ii)+1),level1iidiff(ii),N_j,N_bothz]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(level1ii(ii)+1:level1ii(ii+1)-1,:,:,e_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindex-1,N_d)+1;
                allind=d_ind+N_d*jBind+N_d*N_j*zBind;
                Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,:,e_c)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1); % loweredge(given the d)
            else
                loweredge=maxindex1(:,1,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii_e=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_bothz, special_n_e, N_j, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J, e_vals, ReturnFnParamsAgeMatrix,2);
                daprimejz=(1:1:N_d)'+N_d*repelem(loweredge-1,1,1,level1iidiff(ii),1,1)+N_d*N_a*jind+N_d*N_a*N_j*zind; % all the d, with the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii_e+reshape(DiscountedEV(daprimejz(:)),[N_d,level1iidiff(ii),N_j,N_bothz]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(level1ii(ii)+1:level1ii(ii+1)-1,:,:,e_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindex-1,N_d)+1;
                allind=d_ind+N_d*jBind+N_d*N_j*zBind;
                Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,:,e_c)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1); % loweredge(given the d)
            end
        end
    end
elseif vfoptions.lowmemory>=2
    error('vfoptions.lowmemory=2 not supported with semi-exogenous states')
end

%% fastOLG with z & e, so need output to take certain shapes
V=reshape(V,[N_a*N_j,N_bothz,N_e]);
% Policy=reshape(Policy,[N_a,N_j,N_bothz,N_e]);

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point


end