function [V,Policy2]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC2A_noz_raw(V,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames,vfoptions)
% fastOLG just means parallelize over "age" (j)
% DC2A: divide-and-conquer in the first endogenous state only (a1), iterate
%       (vectorize) over the second endogenous state (a2).

N_d=prod(n_d);
N_a=prod(n_a);

Policy=zeros(N_a,N_j,'gpuArray'); % first dim indexes the optimal choice for d and (a1prime,a2prime)

%%
a_grid=gpuArray(a_grid);

n_a1=n_a(1);
n_a2=n_a(2:end);
N_a1=n_a1;
N_a2=n_a2;
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% n-Monotonicity (a1 only)
level1ii=round(linspace(1,n_a(1),vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% precompute indices
a2ind=gpuArray(0:1:N_a2-1); % already includes -1
a2Bind=shiftdim(gpuArray(0:1:N_a2-1),-1); % sits at dim 3 (a2prime) of a1primeindexes
jind=shiftdim(gpuArray(0:1:N_j-1),-1); % sits at dim 3 (N_j) of maxindex, used in allind
jBind=shiftdim(gpuArray(0:1:N_j-1),-4); % sits at dim 6 (N_j) of a1primeindexes, used in aprimej


%% First, create the big 'next period (of transition path) expected value fn'
% spanning all ages simultaneously.

ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j);

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

if vfoptions.EVpre==0
    EV=zeros(N_a,N_j,'gpuArray');
    EV(:,1:N_j-1)=V(:,2:end); % shift next-age V down into current-age slot; j=N_j gets zero (terminal)
    EV=reshape(EV,[N_a1,N_a2,1,1,N_j]);
elseif vfoptions.EVpre==1
    % This is used for 'Matched Expectations Path'
    EV=reshape(V,[N_a1,N_a2,1,1,N_j]);
end
V=zeros(N_a,N_j,'gpuArray'); % V is over (a,j)

DiscountedEV=shiftdim(reshape(DiscountFactor_J,[1,1,1,1,N_j]).*EV,-1); % [1,N_a1,N_a2,1,1,N_j] — 1st dim autofills d


% n-Monotonicity
ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_noz(ReturnFn, n_d, N_j, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, ReturnFnParamsAgeMatrix,1);

entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

% First, we want a1prime conditional on (d,1,a2prime,a1,a2,j)
[~,maxindex1]=max(entireRHS_ii,[],2);

% Now, get and store the full (d,a1prime,a2prime)
[Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2,N_j]),[],1);
% Store
curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
V(curraindex,:)=shiftdim(Vtempii,1);
Policy(curraindex,:)=shiftdim(maxindex2,1);

% Attempt for improved version
maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
for ii=1:(vfoptions.level1n-1)
    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
    if maxgap(ii)>0
        loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
        % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-N_j
        a1primeindexes=loweredge+(0:1:maxgap(ii));
        % a1prime possibilities are n_d-by-maxgap(ii)+1-by-n_a2-by-1-by-n_a2-by-N_j
        ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_noz(ReturnFn, n_d, N_j, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, ReturnFnParamsAgeMatrix,2);
        aprimej=repelem(a1primeindexes,1,1,1,level1iidiff(ii),1,1)+N_a1*a2Bind+N_a*jBind;
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprimej,[N_d*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_j]));
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        V(curraindex,:)=shiftdim(Vtempii,1);
        dind=(rem(maxindex-1,N_d)+1);
        a1primeind=rem(ceil(maxindex/N_d)-1,maxgap(ii)+1)+1-1;
        a2primeind=ceil(maxindex/(N_d*(maxgap(ii)+1)))-1;
        maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind;
        allind=dind+N_d*a2primeind+N_d*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d*N_a2*N_a2*jind;
        Policy(curraindex,:)=shiftdim(maxindexfix+N_d*(loweredge(allind)-1),1);
    else
        loweredge=maxindex1(:,1,:,ii,:,:);
        ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_noz(ReturnFn, n_d, N_j, d_gridvals, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, ReturnFnParamsAgeMatrix,2);
        aprimej=repelem(loweredge,1,1,1,level1iidiff(ii),1,1)+N_a1*a2Bind+N_a*jBind;
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprimej,[N_d*1*N_a2,level1iidiff(ii)*N_a2,N_j]));
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        V(curraindex,:)=shiftdim(Vtempii,1);
        dind=(rem(maxindex-1,N_d)+1);
        a1primeind=0;
        a2primeind=ceil(maxindex/N_d)-1;
        maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind;
        allind=dind+N_d*a2primeind+N_d*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d*N_a2*N_a2*jind;
        Policy(curraindex,:)=shiftdim(maxindexfix+N_d*(loweredge(allind)-1),1);
    end
end

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point


end
