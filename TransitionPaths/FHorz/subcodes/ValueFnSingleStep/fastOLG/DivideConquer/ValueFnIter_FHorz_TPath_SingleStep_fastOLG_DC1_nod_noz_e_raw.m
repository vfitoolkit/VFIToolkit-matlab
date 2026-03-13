function [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_nod_noz_e_raw(V,n_a,n_e,N_j, a_grid,e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,e), rather than standard (a,e,j)
% V is (a,j)-by-e
% e_gridvals_J has shape (j,prod(n_e),l_e) for fastOLG

N_a=prod(n_a);
N_e=prod(n_e);

% fastOLG, so a-j-e
Policy=zeros(N_a,N_j,N_e,'gpuArray'); % first dim indexes the optimal choice for d and aprime

e_gridvals_J=shiftdim(e_gridvals_J,-2); % needed shape for ReturnFnMatrix with fastOLG and DC1

%%

% n-Monotonicity
% vfoptions.level1n=5;
level1ii=round(linspace(1,n_a,vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_e (note: N_aprime is just equal to N_a)

DiscountFactorParamsVec=CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec,2);
DiscountFactorParamsVec=shiftdim(DiscountFactorParamsVec,-2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

if vfoptions.EVpre==0
    % pi_e_J is (a,j)-by-e
    EV=[sum(V(N_a+1:end,:).*pi_e_J(N_a+1:end,:),2); zeros(N_a,1,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_z_J to create expectations
elseif vfoptions.EVpre==1
    % This is used for 'Matched Expecations Path'
    EV=[reshape(V,[N_a*N_j,N_e].*pi_e_J,2)];  % input V is already of size [N_a,N_j] and we want to use the whole thing
end
V=zeros(N_a,N_j,N_e,'gpuArray'); % V is over (a,j)

discountedEV=DiscountFactorParamsVec.*reshape(EV,[N_a,1,N_j]); % [aprime]



if vfoptions.lowmemory==0

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2(ReturnFn, n_e, N_j, a_grid, a_grid(level1ii), e_gridvals_J, ReturnFnParamsAgeMatrix,1);

    entireRHS_ii=ReturnMatrix_ii+discountedEV; % (d,aprime,a and j,z,e), autofills a and e for expectation term
     
    [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

    % Store
    V(level1ii,:,:)=Vtempii;
    Policy(level1ii,:,:)=maxindex1; % d,aprime

    % Attempt for improved version
    maxgap=squeeze(max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3));
    for ii=1:(vfoptions.level1n-1)
        if maxgap(ii)>0
            loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
            aprimeindexes=loweredge+(0:1:maxgap(ii))'; % ' due to no d
            % aprime possibilities are maxgap(ii)+1-by-1-by-N_j-by-N_e
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2(ReturnFn, n_e, N_j, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J, ReturnFnParamsAgeMatrix,2);
            aprime=aprimeindexes+N_a*shiftdim((0:1:N_j-1),-1); % with the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+discountedEV(aprime); % autofill e for the expectations
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(Vtempii,1);
            Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(maxindex+loweredge-1,1); % loweredge(given the d)
        else
            loweredge=maxindex1(1,ii,:,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2(ReturnFn, n_e, N_j, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J, ReturnFnParamsAgeMatrix,2);
            aprime=loweredge+N_a*shiftdim((0:1:N_j-1),-1); % with the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+discountedEV(aprime); % autofill e for the expectations
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(Vtempii,1);
            Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(maxindex+loweredge-1,1); % loweredge(given the d)
        end
    end

elseif vfoptions.lowmemory==1

    special_n_e=ones(1,length(n_e));

    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,:,e_c,:); % e_gridvals_J has shape (1,1,j,prod(n_e),l_e)

        % n-Monotonicity
        ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2(ReturnFn, special_n_e, N_j, a_grid, a_grid(level1ii), e_vals, ReturnFnParamsAgeMatrix,1);

        entireRHS_ii=ReturnMatrix_ii_e+discountedEV; % (d,aprime,a and j,z), autofills a for expectation term

        [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
 
        % Store
        V(level1ii,:,e_c)=Vtempii;
        Policy(level1ii,:,e_c)=maxindex1; % d,aprime

        % Attempt for improved version
        maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
        for ii=1:(vfoptions.level1n-1)
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
                aprimeindexes=loweredge+(0:1:maxgap(ii))'; % ' due to no d
                % aprime possibilities are maxgap(ii)+1-by-1-by-N_j
                ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2(ReturnFn, special_n_e, N_j, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_vals, ReturnFnParamsAgeMatrix,2);
                aprime=aprimeindexes+N_a*shiftdim((0:1:N_j-1),-1); % with the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii_e+discountedEV(aprime);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(level1ii(ii)+1:level1ii(ii+1)-1,:,e_c)=shiftdim(Vtempii,1);
                Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,e_c)=shiftdim(maxindex+loweredge-1,1); % loweredge
            else
                loweredge=maxindex1(1,ii,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2(ReturnFn, special_n_e, N_j, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_vals, ReturnFnParamsAgeMatrix,2);
                aprime=loweredge+N_a*shiftdim((0:1:N_j-1),-1); % with the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii_e+discountedEV(aprime);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(level1ii(ii)+1:level1ii(ii+1)-1,:,e_c)=shiftdim(Vtempii,1);
                Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,e_c)=shiftdim(maxindex+loweredge-1,1); % loweredge
            end
        end
    end
end

%% fastOLG with e, so need output to take certain shapes
V=reshape(V,[N_a*N_j,N_e]);
% Policy=reshape(Policy,[N_a,N_j,N_e]);

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point


end