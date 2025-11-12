function [V,Policy2]=ValueFnIter_FHorz_DC1_noz_raw(n_d,n_a,N_j, d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% divide-and-conquer for length(n_a)==1

N_d=prod(n_d);
N_a=prod(n_a);

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z 


%%
d_gridvals=CreateGridvals(n_d,d_grid,1);

% n-Monotonicity
% vfoptions.level1n=5;
level1ii=round(linspace(1,n_a,vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);

    % First, we want aprime conditional on (d,1,a,z)
    [~,maxindex1]=max(ReturnMatrix_ii,[],2);

    % Now, get and store the full (d,aprime)
    [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a,vfoptions.level1n]),[],1);
    
    % Store
    V(level1ii,N_j)=shiftdim(Vtempii,1);
    Policy(level1ii,N_j)=shiftdim(maxindex2,1); % d,aprime
    
    % Attempt for improved version
    maxgap=max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,2);
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            V(curraindex,N_j)=shiftdim(Vtempii,1);
            Policy(curraindex,N_j)=shiftdim(maxindex,1)+N_d*(loweredge(rem(maxindex-1,N_d)+1)-1); % loweredge(given the d)
        else
            loweredge=maxindex1(:,1,ii);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,2);
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            V(curraindex,N_j)=shiftdim(Vtempii,1);
            Policy(curraindex,N_j)=shiftdim(maxindex,1)+N_d*(loweredge(rem(maxindex-1,N_d)+1)-1); % loweredge(given the d)
        end
    end
    
else
    % Using V_Jplus1
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=reshape(vfoptions.V_Jplus1,[N_a,1]);    % First, switch V_Jplus1 into Kron form
    
    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);

    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*shiftdim(EV,-1); % (d,aprime,a)

    % First, we want aprime conditional on (d,1,a,z)
    [~,maxindex1]=max(entireRHS_ii,[],2);

    % Now, get and store the full (d,aprime)
    [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);

    % Store
    V(level1ii,N_j)=shiftdim(Vtempii,1);
    Policy(level1ii,N_j)=shiftdim(maxindex2,1); % d,aprime

    % Attempt for improved version
    maxgap=max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1
            aprimeindexes=loweredge+(0:1:maxgap(ii)); 
            % aprime possibilities are n_d-by-maxgap(ii)+1
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(loweredge+(0:1:maxgap(ii))), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,2);
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV(aprimeindexes),[N_d,(maxgap(ii)+1),1]); % autoexpand level1iidiff(ii) in 2nd-dim
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,N_j)=shiftdim(Vtempii,1);
            Policy(curraindex,N_j)=shiftdim(maxindex,1)+N_d*(loweredge(rem(maxindex-1,N_d)+1)-1); % loweredge(given the d)
        else
            loweredge=maxindex1(:,1,ii);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,2);
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV(loweredge); % autoexpand level1iidiff(ii) in 2nd-dim
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,N_j)=shiftdim(Vtempii,1);
            Policy(curraindex,N_j)=shiftdim(maxindex,1)+N_d*(loweredge(rem(maxindex-1,N_d)+1)-1); % loweredge(given the d)
        end
    end

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
    
    EV=V(:,jj+1);
    
    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);
    
    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*shiftdim(EV,-1); % (d,aprime,a)

    % First, we want aprime conditional on (d,1,a,z)
    [~,maxindex1]=max(entireRHS_ii,[],2);

    % Now, get and store the full (d,aprime)
    [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);

    % Store
    V(level1ii,jj)=shiftdim(Vtempii,1);
    Policy(level1ii,jj)=shiftdim(maxindex2,1); % d,aprime
    
    % Attempt for improved version
    maxgap=max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1
            aprimeindexes=loweredge+(0:1:maxgap(ii)); 
            % aprime possibilities are n_d-by-maxgap(ii)+1
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,2);
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV(aprimeindexes),[N_d*(maxgap(ii)+1),1]); % autoexpand level1iidiff(ii) in 2nd-dim
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,jj)=shiftdim(Vtempii,1);
            Policy(curraindex,jj)=shiftdim(maxindex,1)+N_d*(loweredge(rem(maxindex-1,N_d)+1)-1); % loweredge(given the d)
        else
            loweredge=maxindex1(:,1,ii);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,2);
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV(loweredge); % autoexpand level1iidiff(ii) in 2nd-dim
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,jj)=shiftdim(Vtempii,1);
            Policy(curraindex,jj)=shiftdim(maxindex,1)+N_d*(loweredge(rem(maxindex-1,N_d)+1)-1); % loweredge(given the d)
        end
    end

end

%%
Policy2=zeros(2,N_a,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
Policy2(1,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:)=shiftdim(ceil(Policy/N_d),-1);

end
