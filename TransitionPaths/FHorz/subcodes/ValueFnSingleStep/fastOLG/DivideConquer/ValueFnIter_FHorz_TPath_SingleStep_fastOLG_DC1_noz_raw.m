function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_noz_raw(V,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames,vfoptions)
% fastOLG just means parallelize over "age" (j)

N_d=prod(n_d);
N_a=prod(n_a);

Policy=zeros(N_a,N_j,'gpuArray'); % first dim indexes the optimal choice for d and aprime
% THE THIRD OUTPUT is for vfoptions.localsearch, on a SEED sweep (vfoptions.localsearchseed, set by
% Step1): the aprime argmax FOR EACH d, [N_d,N_a,N_j], as LocalSearch/..._fastOLG_LSseed_noz_raw returns
% it. At the level-1 points that is maxindex1. Between them it is an EXTRA per-d max over the level-2 RHS
% (fold is d fastest, so reshape to [N_d,maxgap+1,...]); when maxgap==0 it is loweredge. See
% slowOLG/DivideConquer/ValueFnIter_FHorz_TPath_SingleStep_DC1_raw. The max that sets V/Policy is NOT
% changed. Only built when asked for.
if nargout>2
    aprimeReferencePolicyNew=zeros(N_d,N_a,N_j,'gpuArray');
end

%%
a_grid=gpuArray(a_grid);

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

jind=shiftdim(gpuArray(0:1:N_j-1),-2);
jBind=shiftdim(gpuArray(0:1:N_j-1),-1);

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j (note: N_aprime is just equal to N_a)

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

if vfoptions.EVpre==0
    EV=zeros(N_a,N_j,'gpuArray');
    EV(:,1:N_j-1)=V(:,2:end);
    EV=reshape(EV,[1,N_a,1,N_j]);
elseif vfoptions.EVpre==1
    % This is used for 'Matched Expecations Path'
    EV=reshape(V,[1,N_a,1,N_j]); % input V is of size [N_a,N_j] and we want to use the whole thing
end
V=zeros(N_a,N_j,'gpuArray'); % V is over (a,j)

DiscountedEV=shiftdim(reshape(DiscountFactor_J,[1,1,N_j]),-1).*EV; % [1,aprime,1,j] — pre-discounted; broadcasts over d (and level1n)


% n-Monotonicity
ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_noz(ReturnFn, n_d, N_j, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsAgeMatrix,1);

entireRHS_ii=ReturnMatrix_ii+DiscountedEV; % d, a (level1) broadcast

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
if nargout>2
    aprimeReferencePolicyNew(:,level1ii,:)=reshape(maxindex1,[N_d,vfoptions.level1n,N_j]); % seed reference, see top
end

% Attempt for improved version
maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
for ii=1:(vfoptions.level1n-1)
    if maxgap(ii)>0
        loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
        % loweredge is n_d-by-1
        aprimeindexes=loweredge+(0:1:maxgap(ii));
        % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-N_j
        ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_noz(ReturnFn, n_d, N_j, d_gridvals, reshape(a_grid(aprimeindexes),size(aprimeindexes)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsAgeMatrix,2);
        aprime=aprimeindexes+N_a*jind; % the current aprimeii(ii):aprimeii(ii+1)
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV(aprime(:)),[N_d*(maxgap(ii)+1),1,N_j]); % broadcast level1iidiff at dim 2
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        V(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(Vtempii,1);
        Policy(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(maxindex+N_d*(loweredge(rem(maxindex-1,N_d)+1+N_d*jBind)-1),1); % loweredge(given the d)
        if nargout>2
            [~,aprimeoffset]=max(reshape(entireRHS_ii,[N_d,maxgap(ii)+1,level1iidiff(ii),N_j]),[],2); % per-d max, see top
            aprimeReferencePolicyNew(:,level1ii(ii)+1:level1ii(ii+1)-1,:)=reshape(loweredge+aprimeoffset-1,[N_d,level1iidiff(ii),N_j]);
        end

    else
        loweredge=maxindex1(:,1,ii,:);
        % Just use aprime(ii) for everything
        ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_noz(ReturnFn, n_d, N_j, d_gridvals, reshape(a_grid(loweredge),size(loweredge)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsAgeMatrix,2);
        aprime=loweredge+N_a*jind; % the current aprimeii(ii):aprimeii(ii+1)
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV(aprime(:)),[N_d,1,N_j]); % broadcast level1iidiff at dim 2
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        V(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(Vtempii,1);
        Policy(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(maxindex+N_d*(loweredge(rem(maxindex-1,N_d)+1+N_d*jBind)-1),1); % loweredge(given the d)
        if nargout>2
            aprimeReferencePolicyNew(:,level1ii(ii)+1:level1ii(ii+1)-1,:)=repmat(reshape(loweredge,[N_d,1,N_j]),1,level1iidiff(ii)); % one aprime per d, see top
        end
    end
end

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point


end