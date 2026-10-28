function [V,Policy]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_noz_e_raw(Vnext,n_d,n_a,n_e, d_gridvals, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Single Bellman step, divide-and-conquer (one endogenous state), with d, no z, with iid e.
% Vnext is [N_a,N_e]; pi_e is [N_e,1], the distribution of next period e.
% lowmemory: 0 is everything at once; 1 and 2 both loop over e.

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

if vfoptions.lowmemory>=1
    special_n_e=ones(1,length(n_e));
else
    eind=shiftdim(gpuArray(0:1:N_e-1),-1); % [1,1,N_e], already includes -1
end

V=zeros(N_a,N_e,'gpuArray');
Policy=zeros(N_a,N_e,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,e

%%
% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));


%%
% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

% Calc the expectation term (except beta), computed once and used by every lowmemory branch
EV=Vnext.*shiftdim(pi_e,-1); % [N_a,N_e], aprime-by-eprime
EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
EV=sum(EV,2); % sum over e', [N_a,1], aprime-by-1
DiscountedEV=DiscountFactorParamsVec*shiftdim(EV,-1); % [1,N_a], 1-by-aprime — pre-discounted; broadcasts over d (and level1n, and e); does not depend on e, so is indexed by aprime alone

if vfoptions.lowmemory==0

    % n-Monotonicity
    % e plays the role of z in the return fn matrix
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_e, d_gridvals, a_grid, a_grid(level1ii), e_gridvals, ReturnFnParamsVec,1); % [N_d,N_a,level1n,N_e], d-by-aprime-by-a-by-e

    entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

    % First, we want aprime conditional on (d,1,a,e)
    [~,maxindex1]=max(entireRHS_ii,[],2); % [N_d,1,level1n,N_e]

    % Now, get and store the full (d,aprime)
    [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n,N_e]),[],1); % [1,level1n,N_e]

    % Store
    V(level1ii,:)=shiftdim(Vtempii,1);
    Policy(level1ii,:)=shiftdim(maxindex2,1); % d,aprime

    % Attempt for improved version
    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1)); % max over d,e
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1-by-1-by-n_e
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_e
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_e, d_gridvals, reshape(a_grid(aprimeindexes),[N_d,maxgap(ii)+1,1,N_e]), a_grid(curraindex), e_gridvals, ReturnFnParamsVec,2); % [N_d*(maxgap(ii)+1),npts,N_e]
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV(aprimeindexes),[N_d*(maxgap(ii)+1),1,N_e]); % autoexpand level1iidiff(ii) in 2nd-dim
            [Vtempii,maxindex]=max(entireRHS_ii,[],1); % [1,npts,N_e], maxindex=d+N_d*(aprime-loweredge(d,e))
            V(curraindex,:)=shiftdim(Vtempii,1);
            dind=(rem(maxindex-1,N_d)+1);
            allind=dind+N_d*eind; % loweredge is n_d-by-1-by-1-by-n_e, so linear index is d+N_d*(e-1)
            Policy(curraindex,:)=shiftdim(maxindex+N_d*(reshape(loweredge(allind),size(allind))-1),1); % loweredge(given the d and e)
        else
            loweredge=maxindex1(:,1,ii,:);
            % loweredge is n_d-by-1-by-1-by-n_e
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_e, d_gridvals, reshape(a_grid(loweredge),[N_d,1,1,N_e]), a_grid(curraindex), e_gridvals, ReturnFnParamsVec,2); % [N_d,npts,N_e]
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV(loweredge),[N_d,1,N_e]); % autoexpand level1iidiff(ii) in 2nd-dim
            [Vtempii,maxindex]=max(entireRHS_ii,[],1); % [1,npts,N_e], maxindex=d
            V(curraindex,:)=shiftdim(Vtempii,1);
            dind=(rem(maxindex-1,N_d)+1);
            allind=dind+N_d*eind; % loweredge is n_d-by-1-by-1-by-n_e, so linear index is d+N_d*(e-1)
            Policy(curraindex,:)=shiftdim(maxindex+N_d*(reshape(loweredge(allind),size(allind))-1),1); % loweredge(given the d and e)
        end
    end

elseif vfoptions.lowmemory>=1
    for e_c=1:N_e
        e_val=e_gridvals(e_c,:);

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_e, d_gridvals, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec,1); % [N_d,N_a,level1n], d-by-aprime-by-a

        entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

        % First, we want aprime conditional on (d,1,a)
        [~,maxindex1]=max(entireRHS_ii,[],2); % [N_d,1,level1n]

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1); % [1,level1n]

        % Store
        V(level1ii,e_c)=Vtempii';
        Policy(level1ii,e_c)=maxindex2'; % d,aprime

        % Attempt for improved version
        maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1)); % max over d
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_e, d_gridvals, reshape(a_grid(aprimeindexes),[N_d,maxgap(ii)+1]), a_grid(curraindex), e_val, ReturnFnParamsVec,2); % [N_d*(maxgap(ii)+1),npts]
                entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV(aprimeindexes),[N_d*(maxgap(ii)+1),1]); % autoexpand level1iidiff(ii) in 2nd-dim
                [Vtempii,maxindex]=max(entireRHS_ii,[],1); % [1,npts], maxindex=d+N_d*(aprime-loweredge(d))
                V(curraindex,e_c)=Vtempii';
                dind=rem(maxindex(:)-1,N_d)+1; % column
                Policy(curraindex,e_c)=maxindex(:)+N_d*(loweredge(dind)-1); % loweredge(given the d)
            else
                loweredge=maxindex1(:,1,ii);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_e, d_gridvals, a_grid(loweredge), a_grid(curraindex), e_val, ReturnFnParamsVec,2); % [N_d,npts]
                entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV(loweredge),[N_d,1]); % autoexpand level1iidiff(ii) in 2nd-dim
                [Vtempii,maxindex]=max(entireRHS_ii,[],1); % [1,npts], maxindex=d
                V(curraindex,e_c)=Vtempii';
                dind=rem(maxindex(:)-1,N_d)+1; % column
                Policy(curraindex,e_c)=maxindex(:)+N_d*(loweredge(dind)-1); % loweredge(given the d)
            end
        end
    end
end


%% Policy in transition paths
Policy=reshape(ind2sub_vec_homemade([n_d,n_a],Policy(:))',[length(n_d)+length(n_a),N_a,N_e]);


end
