function [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_nod_e_raw(V,n_a,n_z,n_e,N_j, a_grid, z_gridvals_J,e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z,e), rather than standard (a,z,e,j)
% V is (a,j)-by-z-by-e

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

% pi_e_J is (a,j)-by-1-by-e
VKronNext=[sum(V(N_a+1:end,:,:).*pi_e_J(N_a+1:end,1,:),3); zeros(N_a,N_z,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_z_J to create expectations

% fastOLG, so a-j-z
V=zeros(N_a,N_j,N_z,N_e,'gpuArray'); % V is over (a,j)
Policy=zeros(N_a,N_j,N_z,N_e,'gpuArray'); % first dim indexes the optimal choice for d and aprime

% z_gridvals_J has shape (j,prod(n_z),l_z) for fastOLG
z_gridvals_J=reshape(z_gridvals_J,[1,1,1,N_j,N_z,length(n_z)]); % needed shape for ReturnFnMatrix with fastOLG and DC1
% pi_z_J=permute(pi_z_J,[3,2,1]); % Give it the size best for the loop below: (j,z',z)

% e_gridvals_J has shape (j,prod(n_e),l_e) for fastOLG
e_gridvals_J=reshape(e_gridvals_J,[1,1,1,N_j,1,N_e,length(n_e)]); % needed shape for ReturnFnMatrix with fastOLG and DC1
% pi_e_J=repelem(permute(pi_e_J,[2,1]),N_a,1) % Give it the size best for expectations with fastOLG (already taken above)

%%
a_grid=gpuArray(a_grid);

% n-Monotonicity
% vfoptions.level1n=5;
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_z*N_e (note: N_aprime is just equal to N_a)

DiscountFactorParamsVec=CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec,2);
DiscountFactorParamsVec=repelem(DiscountFactorParamsVec,N_a,1);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

if vfoptions.lowmemory==0
    %Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=VKronNext.*repelem(pi_z_J,N_a,1,1);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2);

    discountedentireEV=reshape(DiscountFactorParamsVec,[N_a,1,N_j,1]).*reshape(EV,[N_a,1,N_j,N_z]); % [aprime]

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2e(ReturnFn, n_z, n_e, N_j, a_grid, a_grid(level1ii), z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,1);

    entireRHS_ii=ReturnMatrix_ii+discountedentireEV; % (d,aprime,a and j,z,e), autofills a and e for expectation term
     
    [Vtempii,maxindex1]=max(entireRHS_ii,[],2);

    % Store
    V(level1ii,:,:,:)=Vtempii;
    Policy(level1ii,:,:,:)=maxindex1; % d,aprime

    % Attempt for improved version
    maxgap=squeeze(max(max(max(maxindex1(1,2:end,:,:,:)-maxindex1(1,1:end-1,:,:,:),[],6),[],5),[],4));
    for ii=1:(vfoptions.level1n-1)
        if maxgap(ii)>0
            loweredge=min(maxindex1(1,ii,:,:,:),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are maxgap(ii)+1-by-1-by-N_j-by-N_z-by-N_e
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2e(ReturnFn, n_z, n_e, N_j, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,2);
            aprimez=repelem(reshape(aprimeindexes-1,[(maxgap(ii)+1),1,N_j,N_z,N_e]),1,level1iidiff(ii))+N_a*shiftdim((0:1:N_j-1),-1)+N_a*N_j*shiftdim((0:1:N_z-1),-2); % with the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+discountedentireEV(aprimez); % autofill e for the expectations
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,:,:)=shiftdim(Vtempii,1);
            Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,:,:)=shiftdim(maxindex+loweredge(maxindex+shiftdim((0:1:N_j-1),-1)+N_j*shiftdim((0:1:N_z-1),-2)+N_j*N_z*shiftdim((0:1:N_e-1),-3))-1,1); % loweredge(given the d)
        else
            loweredge=maxindex1(1,ii,:,:,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2e(ReturnFn, n_z, n_e, N_j, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,2);
            aprimez=repelem(reshape(loweredge-1,[1,1,N_j,N_z,N_e]),1,level1iidiff(ii))+N_a*shiftdim((0:1:N_j-1),-1)+N_a*N_j*shiftdim((0:1:N_z-1),-2); % with the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+discountedentireEV(aprimez); % autofill e for the expectations
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,:,:)=shiftdim(Vtempii,1);
            Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,:,:)=shiftdim(maxindex+loweredge(maxindex+shiftdim((0:1:N_j-1),-1)+N_j*shiftdim((0:1:N_z-1),-2)+N_j*N_z*shiftdim((0:1:N_e-1),-2))-1,1); % loweredge(given the d)
        end
    end

elseif vfoptions.lowmemory==1

    special_n_e=ones(1,length(n_e));

    %Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=VKronNext.*repelem(pi_z_J,N_a,1,1);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2);

    discountedentireEV=reshape(DiscountFactorParamsVec,[N_a,1,N_j,1]).*reshape(EV,[N_a,1,N_j,N_z]); % [aprime]

    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,1,:,1,e_c,:); % e_gridvals_J has shape (1,1,1,j,1,prod(n_e),l_e)

        % n-Monotonicity
        ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_Par2e(ReturnFn, n_d, n_z, special_n_e, N_j, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J, e_vals, ReturnFnParamsAgeMatrix,1);

        entireRHS_ii=ReturnMatrix_ii_e+discountedentireEV; % (d,aprime,a and j,z), autofills a for expectation term

        [Vtempii,maxindex1]=max(entireRHS_ii,[],2);
 
        % Store
        V(level1ii,:,:,e_c)=Vtempii;
        Policy(level1ii,:,:,e_c)=maxindex1; % d,aprime

        % Attempt for improved version
        maxgap=squeeze(max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],5),[],4));
        for ii=1:(vfoptions.level1n-1)
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are maxgap(ii)+1-by-1-by-N_j-by-N_z
                ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2e(ReturnFn, n_z, special_n_e, N_j, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J, e_vals, ReturnFnParamsAgeMatrix,2);
                aprimez=repelem(reshape(aprimeindexes-1,[(maxgap(ii)+1),1,N_j,N_z]),1,level1iidiff(ii))+N_a*shiftdim((0:1:N_j-1),-1)+N_a*N_j*shiftdim((0:1:N_z-1),-2); % with the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii_e+discountedentireEV(aprimez);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(level1ii(ii)+1:level1ii(ii+1)-1,:,:,e_c)=shiftdim(Vtempii,1);
                Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,:,e_c)=shiftdim(maxindex+loweredge(maxindex+shiftdim((0:1:N_j-1),-1)+N_j*shiftdim((0:1:N_z-1),-2))-1,1); % loweredge
            else
                loweredge=maxindex1(1,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2e(ReturnFn, n_z, special_n_e, N_j, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J, e_vals, ReturnFnParamsAgeMatrix,2);
                aprimez=repelem(reshape(loweredge-1,[1,1,N_j,N_z]),1,level1iidiff(ii))+N_a*shiftdim((0:1:N_j-1),-1)+N_a*N_j*shiftdim((0:1:N_z-1),-2); % with the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii_e+discountedentireEV(aprimez);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(level1ii(ii)+1:level1ii(ii+1)-1,:,:,e_c)=shiftdim(Vtempii,1);
                Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,:,e_c)=shiftdim(maxindex+loweredge(maxindex+shiftdim((0:1:N_j-1),-1)+N_j*shiftdim((0:1:N_z-1),-2))-1,1); % loweredge
            end
        end
    end
elseif vfoptions.lowmemory==2

    special_n_e=ones(1,length(n_e));
    special_n_z=ones(1,length(n_z));

    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,1,:,1,e_c,:); % e_gridvals_J has shape (1,1,1,j,1,prod(n_e),l_e)
        for z_c=1:N_z
            z_vals=z_gridvals_J(1,1,1,:,z_c,:); % z_gridvals_J has shape (1,1,1,j,prod(n_z),l_z) for fastOLG

            %Calc the condl expectation term (except beta), which depends on z but not on control variables
            EV_z=VKronNext.*repelem(pi_z_J(:,:,z_c),N_a,1,1);
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);

            discountedentireEV_z=reshape(DiscountFactorParamsVec,[N_a,1,N_j,1]).*reshape(EV_z,[N_a,1,N_j]); % [d,aprime]

            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2e(ReturnFn, special_n_z, special_n_e, N_j, a_grid, a_grid(level1ii), z_vals, e_vals, ReturnFnParamsAgeMatrix,1);

            entireRHS_ii=ReturnMatrix_ii+discountedentireEV_z; % (d,aprime,a and j,z), autofills j for expectation term

            [Vtempii,maxindex1]=max(entireRHS_ii,[],2);

            % Store
            V(level1ii,:,z_c,e_c)=Vtempii;
            Policy(level1ii,:,z_c,e_c)=maxindex1; % aprime

            % Attempt for improved version
            maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],4));
            for ii=1:(vfoptions.level1n-1)
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are maxgap(ii)+1-by-1-by-N_j
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2e(ReturnFn, special_n_z, special_n_e, N_j, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_vals, e_vals, ReturnFnParamsAgeMatrix,2);
                    aprime=repelem(reshape(aprimeindexes-1,[(maxgap(ii)+1),1,N_j]),1,level1iidiff(ii))+N_a*shiftdim((0:1:N_j-1),-1); % with the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii+discountedentireEV_z(aprime);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c,e_c)=shiftdim(Vtempii,1);
                    Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c,e_c)=shiftdim(maxindex+loweredge(maxindex+shiftdim((0:1:N_j-1),-1))-1,1); % loweredge(given the j)
                else
                    loweredge=maxindex1(1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2e(ReturnFn, special_n_z, special_n_e, N_j, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_vals, e_vals, ReturnFnParamsAgeMatrix,2);
                    aprime=repelem(reshape(loweredge-1,[1,1,N_j]),1,level1iidiff(ii))+N_a*shiftdim((0:1:N_j-1),-1); % with the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii+discountedentireEV_z(aprime);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c,e_c)=shiftdim(Vtempii,1);
                    Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c,e_c)=shiftdim(maxindex+loweredge(maxindex+shiftdim((0:1:N_j-1),-1))-1,1); % loweredge(given the j)
                end
            end
        end
     end
end

%% fastOLG with z & e, so need output to take certain shapes
V=reshape(V,[N_a*N_j,N_z,N_e]);
% Policy=reshape(Policy,[N_a,N_j,N_z,N_e]);
% Note that in fastOLG, we do not separate d from aprime in Policy




end