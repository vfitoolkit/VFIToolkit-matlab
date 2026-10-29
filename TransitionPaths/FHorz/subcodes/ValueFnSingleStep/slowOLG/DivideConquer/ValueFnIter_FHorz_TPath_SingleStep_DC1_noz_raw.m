function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_DC1_noz_raw(V,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The V input is next period value fn (across all ages), the V output is this period.

N_d=prod(n_d);
N_a=prod(n_a);

Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a
% THE THIRD OUTPUT is for vfoptions.localsearch, on a SEED sweep (vfoptions.localsearchseed, set by
% Step1): the aprime argmax FOR EACH d, [N_d,N_a,N_j], as LocalSearch/..._LSseed_noz_raw returns it.
% At the level-1 points that is maxindex1. Between them it is an EXTRA per-d max over the level-2 RHS
% (fold is d fastest, so reshape to [N_d,maxgap+1,...]); when maxgap==0 it is loweredge. Under DC's own
% monotonicity this is the unrestricted per-d argmax, as the interval is built over all d. The max that
% sets V/Policy is NOT changed (its tie-breaking is seen by every bank). Only built when asked for.
if nargout>2
    aprimeReferencePolicyNew=zeros(N_d,N_a,N_j,'gpuArray');
end

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% j=N_j: terminal age has no continuation in TPath
% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,N_j);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

% n-Monotonicity
ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);

% First, we want aprime conditional on (d,1,a)
[~,maxindex1]=max(ReturnMatrix_ii,[],2);

% Now, get and store the full (d,aprime)
[Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a,vfoptions.level1n]),[],1);

% Store
V(level1ii,N_j)=shiftdim(Vtempii,1);
Policy(level1ii,N_j)=shiftdim(maxindex2,1); % d,aprime
if nargout>2
    aprimeReferencePolicyNew(:,level1ii,N_j)=reshape(maxindex1,[N_d,vfoptions.level1n]); % seed reference, see top
end

% Attempt for improved version
maxgap=max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1);
for ii=1:(vfoptions.level1n-1)
    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
    if maxgap(ii)>0
        loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
        % loweredge is n_d-by-1
        aprimeindexes=loweredge+(0:1:maxgap(ii));
        % aprime possibilities are n_d-by-maxgap(ii)+1
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, reshape(a_grid(aprimeindexes),size(aprimeindexes)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,2);
        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
        V(curraindex,N_j)=shiftdim(Vtempii,1);
        Policy(curraindex,N_j)=shiftdim(maxindex,1)+N_d*(reshape(loweredge(rem(maxindex-1,N_d)+1),[],1)-1); % loweredge(given the d), forced to a column: with N_d=1 loweredge is a scalar and indexing it by a row gives a row
        if nargout>2
            [~,aprimeoffset]=max(reshape(ReturnMatrix_ii,[N_d,maxgap(ii)+1,length(curraindex)]),[],2); % per-d max, see top
            aprimeReferencePolicyNew(:,curraindex,N_j)=reshape(loweredge+aprimeoffset-1,[N_d,length(curraindex)]);
        end
    else
        loweredge=maxindex1(:,1,ii);
        % Just use aprime(ii) for everything
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, reshape(a_grid(loweredge),size(loweredge)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,2);
        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
        V(curraindex,N_j)=shiftdim(Vtempii,1);
        Policy(curraindex,N_j)=shiftdim(maxindex,1)+N_d*(reshape(loweredge(rem(maxindex-1,N_d)+1),[],1)-1); % loweredge(given the d), forced to a column: with N_d=1 loweredge is a scalar and indexing it by a row gives a row
        if nargout>2
            aprimeReferencePolicyNew(:,curraindex,N_j)=repmat(loweredge,1,length(curraindex)); % one aprime per d, see top
        end
    end
end


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    VKronNext_j=Vtemp_j; % Has been presaved before it was replaced
    Vtemp_j=V(:,jj); % Grab this before it is replaced/updated

    EV=VKronNext_j;

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);

    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*shiftdim(EV,-1); % (d,aprime,a)

    % First, we want aprime conditional on (d,1,a)
    [~,maxindex1]=max(entireRHS_ii,[],2);

    % Now, get and store the full (d,aprime)
    [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);

    % Store
    V(level1ii,jj)=shiftdim(Vtempii,1);
    Policy(level1ii,jj)=shiftdim(maxindex2,1); % d,aprime
    if nargout>2
        aprimeReferencePolicyNew(:,level1ii,jj)=reshape(maxindex1,[N_d,vfoptions.level1n]); % seed reference, see top
    end

    % Attempt for improved version
    maxgap=max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, reshape(a_grid(aprimeindexes),size(aprimeindexes)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,2);
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV(aprimeindexes),[N_d*(maxgap(ii)+1),1]); % autoexpand level1iidiff(ii) in 2nd-dim
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,jj)=shiftdim(Vtempii,1);
            Policy(curraindex,jj)=shiftdim(maxindex,1)+N_d*(reshape(loweredge(rem(maxindex-1,N_d)+1),[],1)-1); % loweredge(given the d), forced to a column: with N_d=1 loweredge is a scalar and indexing it by a row gives a row
            if nargout>2
                [~,aprimeoffset]=max(reshape(entireRHS_ii,[N_d,maxgap(ii)+1,length(curraindex)]),[],2); % per-d max, see top
                aprimeReferencePolicyNew(:,curraindex,jj)=reshape(loweredge+aprimeoffset-1,[N_d,length(curraindex)]);
            end
        else
            loweredge=maxindex1(:,1,ii);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, reshape(a_grid(loweredge),size(loweredge)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,2);
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV(loweredge); % autoexpand level1iidiff(ii) in 2nd-dim
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,jj)=shiftdim(Vtempii,1);
            Policy(curraindex,jj)=shiftdim(maxindex,1)+N_d*(reshape(loweredge(rem(maxindex-1,N_d)+1),[],1)-1); % loweredge(given the d), forced to a column: with N_d=1 loweredge is a scalar and indexing it by a row gives a row
            if nargout>2
                aprimeReferencePolicyNew(:,curraindex,jj)=repmat(loweredge,1,length(curraindex)); % one aprime per d, see top
            end
        end
    end

end

%% Output shape for policy
Policy=shiftdim(Policy,-1);

end
