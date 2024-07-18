function [V,Policy2]=ValueFnIter_Case1_FHorz_DC1_e_raw(n_d,n_a,n_z,n_e,N_j, d_grid, a_grid, z_gridvals_J, e_gridvals_J,pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
d_grid=gpuArray(d_grid);
d_gridvals=CreateGridvals(n_d,d_grid,1);
a_grid=gpuArray(a_grid);

% n-Monotonicity
% vfoptions.level1n=5;
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;
Policytemp=zeros(N_a,N_z,N_e,'gpuArray');


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

pi_e_J=shiftdim(pi_e_J,-2); % Move to third dimension

if ~isfield(vfoptions,'V_Jplus1')

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

    % size(ReturnMatrix_ii) % (d, aprime, a,z,e)
    % [n_d,n_a,vfoptions.level1n,n_z,n_e]

    % First, we want aprime conditional on (d,1,a,z,e)
    [RMtemp_ii,maxindex1]=max(ReturnMatrix_ii,[],2);
    % Now, we get the d and we store the (d,aprime) and the 

    %Calc the max and it's index
    [Vtempii,maxindex2]=max(RMtemp_ii,[],1);
    maxindex2=shiftdim(maxindex2,2); % d
    maxindex1d=maxindex1(maxindex2(:)+N_d*repmat((0:1:vfoptions.level1n-1)',N_z*N_e,1)+N_d*vfoptions.level1n*repelem((0:1:N_z*N_e-1)',vfoptions.level1n,1)); % aprime

    % Store
    V(level1ii,:,:,N_j)=shiftdim(Vtempii,2);
    Policytemp(level1ii,:,:)=maxindex2+N_d*(reshape(maxindex1d,[vfoptions.level1n,N_z,N_e])-1); % d,aprime

    % Attempt for improved version
    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1)); % max over d,z,e
    for ii=1:(vfoptions.level1n-1)
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1-by-1-by-n_z-by-n_e
            aprimeindexes=loweredge+(0:1:maxgap(ii)); 
            % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_z-by-n_e
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,:,N_j)=shiftdim(Vtempii,1);
            Policytemp(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(maxindex+N_d*(loweredge(rem(maxindex-1,N_d)+1+N_d*shiftdim((0:1:N_z-1),-1)+N_d*N_z*shiftdim((0:1:N_e-1),-2))-1),1); % loweredge(given the d and z and e)
        else
            loweredge=maxindex1(:,1,ii,:,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,:,N_j)=shiftdim(Vtempii,1);
            Policytemp(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(maxindex+N_d*(loweredge(rem(maxindex-1,N_d)+1+N_d*shiftdim((0:1:N_z-1),-1)+N_d*N_z*shiftdim((0:1:N_e-1),-2))-1),1); % loweredge(given the d and z and e)
        end
    end

    Policy(:,:,:,N_j)=Policytemp;

else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    V_Jplus1=sum(V_Jplus1.*pi_e_J(1,1,:,N_j),3);

    EV=V_Jplus1.*shiftdim(pi_z_J(:,:,N_j)',-1);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    entireEV=repmat(shiftdim(EV,-1),N_d,1,1,1); % [d,aprime,1,z]

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

    % size(ReturnMatrix_ii) % (d, aprime, a,z,e)
    % [n_d,n_a,vfoptions.level1n,n_z,n_e]

    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV; %repmat(entireEV,1,N_a,1,N_e);

    % First, we want aprime conditional on (d,1,a,z,e)
    [RMtemp_ii,maxindex1]=max(entireRHS_ii,[],2);
    % Now, we get the d and we store the (d,aprime) and the 

    %Calc the max and it's index
    [Vtempii,maxindex2]=max(RMtemp_ii,[],1);
    maxindex2=shiftdim(maxindex2,2); % d
    maxindex1d=maxindex1(maxindex2(:)+N_d*repmat((0:1:vfoptions.level1n-1)',N_z*N_e,1)+N_d*vfoptions.level1n*repelem((0:1:N_z*N_e-1)',vfoptions.level1n,1)); % aprime

    % Store
    V(level1ii,:,:,N_j)=shiftdim(Vtempii,2);
    Policytemp(level1ii,:,:)=maxindex2+N_d*(reshape(maxindex1d,[vfoptions.level1n,N_z,N_e])-1); % d,aprime

    % Attempt for improved version
    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1)); % max over d,z,e
    for ii=1:(vfoptions.level1n-1)
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1-by-1-by-n_z-by-n_e
            aprimeindexes=loweredge+(0:1:maxgap(ii)); 
            % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_z-by-n_e
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            daprimez=(1:1:N_d)'+N_d*repelem(aprimeindexes-1,1,1,level1iidiff(ii),1)+N_d*N_a*shiftdim((0:1:N_z-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV(reshape(daprimez,[N_d*(maxgap(ii)+1),level1iidiff(ii),N_z,N_e]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,:,N_j)=shiftdim(Vtempii,1);
            Policytemp(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(maxindex+N_d*(loweredge(rem(maxindex-1,N_d)+1+N_d*shiftdim((0:1:N_z-1),-1)+N_d*N_z*shiftdim((0:1:N_e-1),-2))-1),1); % loweredge(given the d and z and e)
        else
            loweredge=maxindex1(:,1,ii,:,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            daprimez=(1:1:N_d)'+N_d*repelem(loweredge-1,1,1,level1iidiff(ii),1)+N_d*N_a*shiftdim((0:1:N_z-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV(reshape(daprimez,[N_d*1,level1iidiff(ii),N_z,N_e]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,:,N_j)=shiftdim(Vtempii,1);
            Policytemp(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(maxindex+N_d*(loweredge(rem(maxindex-1,N_d)+1+N_d*shiftdim((0:1:N_z-1),-1)+N_d*N_z*shiftdim((0:1:N_e-1),-2))-1),1); % loweredge(given the d and z and e)
        end
    end

    
    Policy(:,:,:,N_j)=Policytemp;

end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end
    
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    VKronNext_j=V(:,:,:,jj+1);
    VKronNext_j=sum(VKronNext_j.*pi_e_J(1,1,:,jj),3);

    EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    entireEV=repmat(shiftdim(EV,-1),N_d,1,1,1); % [d,aprime,1,z]

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

    % size(ReturnMatrix_ii) % (d, aprime, a,z,e)
    % [n_d,n_a,vfoptions.level1n,n_z,n_e]

    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV; %repmat(entireEV,1,N_a,1,N_e);

    % First, we want aprime conditional on (d,1,a,z,e)
    [RMtemp_ii,maxindex1]=max(entireRHS_ii,[],2);
    % Now, we get the d and we store the (d,aprime) and the 

    %Calc the max and it's index
    [Vtempii,maxindex2]=max(RMtemp_ii,[],1);
    maxindex2=shiftdim(maxindex2,2); % d
    maxindex1d=maxindex1(maxindex2(:)+N_d*repmat((0:1:vfoptions.level1n-1)',N_z*N_e,1)+N_d*vfoptions.level1n*repelem((0:1:N_z*N_e-1)',vfoptions.level1n,1)); % aprime

    % Store
    V(level1ii,:,:,jj)=shiftdim(Vtempii,2);
    Policytemp(level1ii,:,:)=maxindex2+N_d*(reshape(maxindex1d,[vfoptions.level1n,N_z,N_e])-1); % d,aprime

    % Attempt for improved version
    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1)); % max over d,z,e
    for ii=1:(vfoptions.level1n-1)
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1-by-1-by-n_z-by-n_e
            aprimeindexes=loweredge+(0:1:maxgap(ii)); % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_z-by-n_e
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
            daprimez=(1:1:N_d)'+N_d*repelem(aprimeindexes-1,1,1,level1iidiff(ii),1)+N_d*N_a*shiftdim((0:1:N_z-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV(reshape(daprimez,[N_d*(maxgap(ii)+1),level1iidiff(ii),N_z,N_e]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,:,jj)=shiftdim(Vtempii,1);
            Policytemp(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(maxindex+N_d*(loweredge(rem(maxindex-1,N_d)+1+N_d*shiftdim((0:1:N_z-1),-1)+N_d*N_z*shiftdim((0:1:N_e-1),-2))-1),1); % loweredge(given the d and z and e)
        else
            loweredge=maxindex1(:,1,ii,:,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
            daprimez=(1:1:N_d)'+N_d*repelem(loweredge-1,1,1,level1iidiff(ii),1)+N_d*N_a*shiftdim((0:1:N_z-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV(reshape(daprimez,[N_d*1,level1iidiff(ii),N_z,N_e]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,:,jj)=shiftdim(Vtempii,1);
            Policytemp(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(maxindex+N_d*(loweredge(rem(maxindex-1,N_d)+1+N_d*shiftdim((0:1:N_z-1),-1)+N_d*N_z*shiftdim((0:1:N_e-1),-2))-1),1); % loweredge(given the d and z and e)
        end
    end
    
    Policy(:,:,:,jj)=Policytemp;

end

%%
Policy2=zeros(2,N_a,N_z,N_e,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
Policy2(1,:,:,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:,:,:)=shiftdim(ceil(Policy/N_d),-1);

end