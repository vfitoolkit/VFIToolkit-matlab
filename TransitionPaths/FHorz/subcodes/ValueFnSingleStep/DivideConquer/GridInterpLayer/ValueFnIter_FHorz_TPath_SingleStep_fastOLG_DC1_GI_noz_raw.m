function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_GI_noz_raw(V,n_d,n_a,N_j, d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames,vfoptions)
% fastOLG just means parallelize over "age" (j)

N_d=prod(n_d);
N_a=prod(n_a);

VKronNext=zeros(N_a,N_j,'gpuArray');
VKronNext(:,1:N_j-1)=V(:,2:end);

V=zeros(N_a,N_j,'gpuArray'); % V is over (a,j)
Policy=zeros(N_a,N_j,'gpuArray'); % first dim indexes the optimal choice for d and aprime

%%
d_grid=gpuArray(d_grid);
d_gridvals=CreateGridvals(n_d,d_grid,1);
a_grid=gpuArray(a_grid);

% n-Monotonicity
% vfoptions.level1n=5;
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j (note: N_aprime is just equal to N_a)

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

DiscountFactorParamsVec=CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec,2);
DiscountFactorParamsVec=shiftdim(DiscountFactorParamsVec,-2);

entireEV=repmat(reshape(VKronNext,[1,N_a,1,N_j]),N_d,1); % [d,aprime]

% n-Monotonicity
ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_noz_Par2(ReturnFn, n_d, N_j, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsAgeMatrix,1);

entireRHS_ii=ReturnMatrix_ii+shiftdim(DiscountFactorParamsVec,-1).*entireEV; % (d,aprime,a and j), autofills j for expectation term

% First, we want aprime conditional on (d,1,a,j)
[RMtemp_ii,maxindex1]=max(entireRHS_ii,[],2);
% Now, we get the d and we store the (d,aprime) and the

%Calc the max and it's index
[Vtempii,maxindex2]=max(RMtemp_ii,[],1);
maxindex2=shiftdim(maxindex2,2); % d
maxindex1d=maxindex1(maxindex2+N_d*(0:1:vfoptions.level1n-1)'+N_d*vfoptions.level1n*(0:1:N_j-1)); % aprime

% Store
V(level1ii,:)=shiftdim(Vtempii,2);
Policy(level1ii,:)=maxindex2+N_d*(maxindex1d-1); % d,aprime

% Attempt for improved version
maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
for ii=1:(vfoptions.level1n-1)
    if maxgap(ii)>0
        loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
        % loweredge is n_d-by-1
        aprimeindexes=loweredge+(0:1:maxgap(ii));
        % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-N_j
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_noz_Par2(ReturnFn, n_d, N_j, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsAgeMatrix,2);
        daprime=(repmat(1:1:N_d,1,maxgap(ii)+1))'+N_d*repelem(reshape(aprimeindexes-1,[N_d*(maxgap(ii)+1),1,N_j]),1,level1iidiff(ii))+N_d*N_a*shiftdim((0:1:N_j-1),-1); % all the d, with the current aprimeii(ii):aprimeii(ii+1)
        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec.*entireEV(daprime);
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        V(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(Vtempii,1);
        Policy(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(maxindex+N_d*(loweredge(rem(maxindex-1,N_d)+1+N_d*shiftdim((0:1:N_j-1),-1))-1),1); % loweredge(given the d)
        
    else
        loweredge=maxindex1(:,1,ii,:);        
        % Just use aprime(ii) for everything
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_noz_Par2(ReturnFn, n_d, N_j, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsAgeMatrix,2);
        daprime=(1:1:N_d)'+N_d*repelem(reshape(loweredge-1,[N_d,1,N_j]),1,level1iidiff(ii))+N_d*N_a*shiftdim((0:1:N_j-1),-1); % all the d, with the current aprimeii(ii):aprimeii(ii+1)
        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec.*entireEV(daprime);
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        V(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(Vtempii,1);
        Policy(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(maxindex+N_d*(loweredge(rem(maxindex-1,N_d)+1+N_d*shiftdim((0:1:N_j-1),-1))-1),1); % loweredge(given the d)
    end
end


end