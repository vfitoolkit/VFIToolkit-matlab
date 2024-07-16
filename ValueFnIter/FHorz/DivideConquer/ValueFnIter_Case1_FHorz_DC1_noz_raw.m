function [V,Policy2]=ValueFnIter_Case1_FHorz_DC1_noz_raw(n_d,n_a,N_j, d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% divide-and-conquer for length(n_a)==1

N_d=prod(n_d);
N_a=prod(n_a);

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z 


%%
d_grid=gpuArray(d_grid);
d_gridvals=CreateGridvals(n_d,d_grid,1);
a_grid=gpuArray(a_grid);

% n-Monotonicity
% vfoptions.level1n=5;
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;
Policytemp=zeros(N_a,1,'gpuArray');

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);

    % size(ReturnMatrix_ii) % (d, aprime, a)
    % [n_d,n_a,vfoptions.level1n]

    % First, we want aprime conditional on (d,1,a)
    [RMtemp_ii,maxindex1]=max(ReturnMatrix_ii,[],2);
    % Now, we get the d and we store the (d,aprime) and the 

    %Calc the max and it's index
    [Vtempii,maxindex2]=max(RMtemp_ii,[],1);
    maxindex2=shiftdim(maxindex2,2); % d
    maxindex1d=maxindex1(maxindex2+N_d*(0:1:vfoptions.level1n-1)'); % aprime

    % Store
    V(level1ii,N_j)=shiftdim(Vtempii,2);
    Policytemp(level1ii)=shiftdim(maxindex2,2)+N_d*(maxindex1d-1); % d,aprime
    
    % Attempt for improved version
    maxindex1=squeeze(maxindex1);
    maxgap=max(maxindex1(:,2:end)-maxindex1(:,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,ii),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,2);
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,N_j)=shiftdim(Vtempii,1);
            Policytemp(level1ii(ii)+1:level1ii(ii+1)-1)=shiftdim(maxindex,1)+N_d*(loweredge(rem(maxindex-1,N_d)+1)-1); % maxindex is (d,aprime): rem(maxindex-1,N_d)+1 is the d
        else
            loweredge=maxindex1(:,ii);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,2);
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,N_j)=shiftdim(Vtempii,1);
            Policytemp(level1ii(ii)+1:level1ii(ii+1)-1)=shiftdim(maxindex,1)+N_d*(loweredge(maxindex)-1);
        end
    end
    
    Policy(:,N_j)=Policytemp;

else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,1]);    % First, switch V_Jplus1 into Kron form
    entireEV=repmat(V_Jplus1',N_d,1); % [d,aprime]

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);

    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV; % (d,aprime,a)

    % First, we want aprime conditional on (d,1,a)
    [RMtemp_ii,maxindex1]=max(entireRHS_ii,[],2);
    % Now, we get the d and we store the (d,aprime) and the 

    %Calc the max and it's index
    [Vtempii,maxindex2]=max(RMtemp_ii,[],1);
    maxindex2=shiftdim(maxindex2,2); % d
    maxindex1d=maxindex1(maxindex2+N_d*(0:1:vfoptions.level1n-1)'); % aprime

    % Store
    V(level1ii,N_j)=shiftdim(Vtempii,2);
    Policytemp(level1ii)=shiftdim(maxindex2,2)+N_d*(maxindex1d-1); % d,aprime
    
    % Attempt for improved version
    maxindex1=squeeze(maxindex1);
    maxgap=max(maxindex1(:,2:end)-maxindex1(:,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,ii),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1
            aprimeindexes=loweredge+(0:1:maxgap(ii)); 
            % aprime possibilities are n_d-by-maxgap(ii)+1
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(loweredge+(0:1:maxgap(ii))), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,2);
            daprime=(repmat(1:1:N_d,1,maxgap(ii)+1))'+N_d*repelem(aprimeindexes-1,1,level1iidiff(ii)); % all the d, with the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV(daprime);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,N_j)=shiftdim(Vtempii,1);
            Policytemp(level1ii(ii)+1:level1ii(ii+1)-1)=shiftdim(maxindex,1)+N_d*(loweredge(rem(maxindex-1,N_d)+1)-1);
        else
            loweredge=maxindex1(:,ii);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,2);
            daprime=(repmat(1:1:N_d,1,maxgap(ii)+1))'+N_d*repelem(loweredge-1,1,level1iidiff(ii)); % all the d, with the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV(daprime);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,N_j)=shiftdim(Vtempii,1);
            Policytemp(level1ii(ii)+1:level1ii(ii+1)-1)=shiftdim(maxindex,1)+N_d*(loweredge(maxindex)-1);
        end
    end

    Policy(:,N_j)=Policytemp;
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
    
    VKronNext_j=V(:,jj+1);
    entireEV=repmat(VKronNext_j',N_d,1); % [d,aprime]

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);
    
    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV; % (d,aprime,a)
    % First, we want aprime conditional on (d,1,a)
    [RMtemp_ii,maxindex1]=max(entireRHS_ii,[],2);
    % Now, we get the d and we store the (d,aprime) and the 

    %Calc the max and it's index
    [Vtempii,maxindex2]=max(RMtemp_ii,[],1);
    maxindex2=shiftdim(maxindex2,2); % d
    maxindex1d=maxindex1(maxindex2+N_d*(0:1:vfoptions.level1n-1)'); % aprime

    % Store
    V(level1ii,jj)=shiftdim(Vtempii,2);
    Policytemp(level1ii)=shiftdim(maxindex2,2)+N_d*(maxindex1d-1); % d,aprime
    
    % Attempt for improved version
    maxindex1=squeeze(maxindex1);
    maxgap=max(maxindex1(:,2:end)-maxindex1(:,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,ii),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1
            aprimeindexes=loweredge+(0:1:maxgap(ii)); 
            % aprime possibilities are n_d-by-maxgap(ii)+1
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,2);
            daprime=(repmat(1:1:N_d,1,maxgap(ii)+1))'+N_d*repelem(aprimeindexes(:)-1,1,level1iidiff(ii)); % all the d, with the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV(daprime);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,jj)=shiftdim(Vtempii,1);
            Policytemp(level1ii(ii)+1:level1ii(ii+1)-1)=shiftdim(maxindex,1)+N_d*(loweredge(rem(maxindex-1,N_d)+1)-1);
        else
            loweredge=maxindex1(:,ii);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,2);
            daprime=(repmat(1:1:N_d,1,maxgap(ii)+1))'+N_d*repelem(loweredge,1,level1iidiff(ii)); % all the d, with the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV(daprime);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,jj)=shiftdim(Vtempii,1);
            Policytemp(level1ii(ii)+1:level1ii(ii+1)-1)=shiftdim(maxindex,1)+N_d*(loweredge(maxindex)-1);
        end
    end

    Policy(:,jj)=Policytemp;

end

%%
Policy2=zeros(2,N_a,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
Policy2(1,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:)=shiftdim(ceil(Policy/N_d),-1);

end