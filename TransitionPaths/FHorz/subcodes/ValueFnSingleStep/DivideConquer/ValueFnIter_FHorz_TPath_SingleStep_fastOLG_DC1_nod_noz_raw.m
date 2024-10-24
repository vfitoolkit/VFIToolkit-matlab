function [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_nod_noz_raw(V,n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames)

% fastOLG just means parallelize over "age" (j)
N_a=prod(n_a);

VKronNext=zeros(N_a,N_j,'gpuArray');
VKronNext(:,1:N_j-1)=V(:,2:end);

V=zeros(N_a,N_j,'gpuArray'); % V is over (a,j)
Policy=zeros(N_a,N_j,'gpuArray'); % first dim indexes the optimal choice for d and aprime

%%
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

entireEV=reshape(VKronNext,[N_a,1,N_j]); % [aprime]

% n-Monotonicity
ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_noz_Par2(ReturnFn, N_j, a_grid, a_grid(level1ii), ReturnFnParamsAgeMatrix,1);

entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec.*entireEV; % (aprime,a and j), autofills j for expectation term

%Calc the max and it's index
[Vtempii,maxindex1]=max(entireRHS_ii,[],1);

V(level1ii,jj)=shiftdim(Vtempii,1);
Policy(level1ii,jj)=shiftdim(maxindex1,1);

% Attempt for improved version
maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
for ii=1:(vfoptions.level1n-1)
    if maxgap(ii)>0
        loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
        % loweredge is n_d-by-1
        aprimeindexes=loweredge+(0:1:maxgap(ii));
        % aprime possibilities are maxgap(ii)+1-by-1-by-N_j
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_noz_Par2(ReturnFn, N_j, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsAgeMatrix,2);
        aprime=repelem(reshape(aprimeindexes-1,[(maxgap(ii)+1),1,N_j]),1,level1iidiff(ii))+N_a*shiftdim((0:1:N_j-1),-1); % with the current aprimeii(ii):aprimeii(ii+1)
        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec.*entireEV(aprime);
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        V(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(Vtempii,1);
        Policy(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(maxindex+loweredge(maxindex+shiftdim((0:1:N_j-1),-1))-1,1); % loweredge
    else
        loweredge=maxindex1(1,ii,:);        
        % Just use aprime(ii) for everything
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_noz_Par2(ReturnFn, N_j, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsAgeMatrix,2);
        aprime=reshape(loweredge-1,[1,1,N_j])+N_a*shiftdim((0:1:N_j-1),-1); % with the current aprimeii(ii):aprimeii(ii+1)
        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec.*entireEV(aprime);
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        V(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(Vtempii,1);
        Policy(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(maxindex+loweredge(maxindex+shiftdim((0:1:N_j-1),-1))-1,1); % loweredge
    end
end


end