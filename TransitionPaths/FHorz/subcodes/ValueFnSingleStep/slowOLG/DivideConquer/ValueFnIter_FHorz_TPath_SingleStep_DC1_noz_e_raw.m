function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_DC1_noz_e_raw(V,n_d,n_a,n_e,N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The V input is next period value fn (across all ages), the V output is this period.

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

Policy=zeros(N_a,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,e
% THE THIRD OUTPUT is for vfoptions.localsearch, on a SEED sweep (vfoptions.localsearchseed, set by
% Step1): the aprime argmax FOR EACH d, [N_d,N_a,N_e,N_j]. At the level-1 points it is maxindex1, already
% a per-d argmax over the whole grid. Between them it is an EXTRA per-d max over the same level-2 RHS
% (the level-2 fold is d fastest), or loweredge itself when maxgap is 0. Within the interval this is the
% unrestricted per-d argmax under DC's own monotonicity assumption, as the interval is built over all d.
% The max that sets V and Policy is left untouched, so tie-breaking does not move. Only built when
% asked for, as it is N_d times Policy.
if nargout>2
    aprimeReferencePolicyNew=zeros(N_d,N_a,N_e,N_j,'gpuArray');
end

Vnext=sum(V.*shiftdim(pi_e_J,-1),2); % Take expectations over e: Vnext(...,jj+1) is read for current age jj, so weight V at age jj+1 by pi_e_J(:,jj+1), the distribution of the e realized in period jj+1; first column is never read

%%

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory>=2
    error('vfoptions.lowmemory>=2 not supported for ValueFnIter_FHorz_TPath_SingleStep_DC1_noz_e_raw')
else
    eind=shiftdim((0:1:N_e-1),-1); % already includes -1
end

%% j=N_j: terminal age has no continuation in TPath

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if vfoptions.lowmemory==0
    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_e, d_gridvals, a_grid, a_grid(level1ii), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

    % First, we want aprime conditional on (d,1,a,e)
    [~,maxindex1]=max(ReturnMatrix_ii,[],2);

    % Now, get and store the full (d,aprime)
    [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a,vfoptions.level1n,N_e]),[],1);

    % Store
    V(level1ii,:,N_j)=shiftdim(Vtempii,1);
    Policy(level1ii,:,N_j)=shiftdim(maxindex2,1); % d,aprime
    if nargout>2
        aprimeReferencePolicyNew(:,level1ii,:,N_j)=reshape(maxindex1,[N_d,vfoptions.level1n,N_e]); % seed reference, see top
    end

    % Attempt for improved version
    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1)); % max over d,e
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1-by-1-by-n_e
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_e
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_e, d_gridvals, reshape(a_grid(aprimeindexes),size(aprimeindexes)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            V(curraindex,:,N_j)=shiftdim(Vtempii,1);
            dind=(rem(maxindex-1,N_d)+1);
            allind=dind+N_d*eind; % loweredge is n_d-by-1-by-1-by-n_e
            Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            if nargout>2 % per-d max over the same RHS, seed reference, see top
                [~,maxindexd]=max(reshape(ReturnMatrix_ii,[N_d,maxgap(ii)+1,length(curraindex),N_e]),[],2);
                aprimeReferencePolicyNew(:,curraindex,:,N_j)=reshape(maxindexd+(loweredge-1),[N_d,length(curraindex),N_e]);
            end
        else
            loweredge=maxindex1(:,1,ii,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_e, d_gridvals, reshape(a_grid(loweredge),size(loweredge)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            V(curraindex,:,N_j)=shiftdim(Vtempii,1);
            dind=(rem(maxindex-1,N_d)+1);
            allind=dind+N_d*eind; % loweredge is n_d-by-1-by-1-by-n_e
            Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            if nargout>2 % seed reference, see top
                aprimeReferencePolicyNew(:,curraindex,:,N_j)=repmat(reshape(loweredge,[N_d,1,N_e]),1,length(curraindex),1);
            end
        end
    end
elseif vfoptions.lowmemory==1
    for e_c=1:N_e
        e_val=e_gridvals_J(e_c,:,N_j);
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_e, d_gridvals, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec,1);

        % First, we want aprime conditional on (d,1,a,e)
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a,vfoptions.level1n]),[],1);

        % Store
        V(level1ii,e_c,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,e_c,N_j)=shiftdim(maxindex2,1); % d,aprime
        if nargout>2
            aprimeReferencePolicyNew(:,level1ii,e_c,N_j)=reshape(maxindex1,[N_d,vfoptions.level1n]); % seed reference, see top
        end

        % Attempt for improved version
        maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1)); % max over d
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-1
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_e, d_gridvals, reshape(a_grid(aprimeindexes),size(aprimeindexes)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind; % loweredge is n_d-by-1-by-1
                Policy(curraindex,e_c,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
                if nargout>2 % per-d max over the same RHS, seed reference, see top
                    [~,maxindexd]=max(reshape(ReturnMatrix_ii,[N_d,maxgap(ii)+1,length(curraindex)]),[],2);
                    aprimeReferencePolicyNew(:,curraindex,e_c,N_j)=reshape(maxindexd+(loweredge-1),[N_d,length(curraindex)]);
                end
            else
                loweredge=maxindex1(:,1,ii);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_e, d_gridvals, reshape(a_grid(loweredge),size(loweredge)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                % allind=dind; % loweredge is n_d-by-1-by-1
                Policy(curraindex,e_c,N_j)=maxindex'+N_d*(reshape(loweredge(dind),[],1)-1); % forced to a column: with N_d=1 loweredge is a scalar and indexing it by a row gives a row
                if nargout>2 % seed reference, see top
                    aprimeReferencePolicyNew(:,curraindex,e_c,N_j)=repmat(loweredge,1,length(curraindex));
                end
            end
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

    EV=Vnext(:,1,jj+1); % e-expectation pre-computed

    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_e, d_gridvals, a_grid, a_grid(level1ii), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*shiftdim(EV,-1);

        % First, we want aprime conditional on (d,1,a,e)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n,N_e]),[],1);

        % Store
        V(level1ii,:,jj)=shiftdim(Vtempii,1);
        Policy(level1ii,:,jj)=shiftdim(maxindex2,1); % d,aprime
        if nargout>2
            aprimeReferencePolicyNew(:,level1ii,:,jj)=reshape(maxindex1,[N_d,vfoptions.level1n,N_e]); % seed reference, see top
        end

        % Attempt for improved version
        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1)); % max over d,e
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-1-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_e, d_gridvals, reshape(a_grid(aprimeindexes),size(aprimeindexes)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV(aprimeindexes),[N_d*(maxgap(ii)+1),1,N_e]); % autoexpand level1iidiff(ii) in 2nd-dim
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,jj)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*eind; % loweredge is n_d-by-1-by-1-by-n_e
                Policy(curraindex,:,jj)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
                if nargout>2 % per-d max over the same RHS, seed reference, see top
                    [~,maxindexd]=max(reshape(entireRHS_ii,[N_d,maxgap(ii)+1,length(curraindex),N_e]),[],2);
                    aprimeReferencePolicyNew(:,curraindex,:,jj)=reshape(maxindexd+(loweredge-1),[N_d,length(curraindex),N_e]);
                end
            else
                loweredge=maxindex1(:,1,ii,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_e, d_gridvals, reshape(a_grid(loweredge),size(loweredge)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV(loweredge),[N_d*1,1,N_e]); % autoexpand level1iidiff(ii) in 2nd-dim
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,jj)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*eind; % loweredge is n_d-by-1-by-1-by-n_e
                Policy(curraindex,:,jj)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
                if nargout>2 % seed reference, see top
                    aprimeReferencePolicyNew(:,curraindex,:,jj)=repmat(reshape(loweredge,[N_d,1,N_e]),1,length(curraindex),1);
                end
            end
        end
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);

            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_e, d_gridvals, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec,1);

            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*shiftdim(EV,-1);

            % First, we want aprime conditional on (d,1,a,e)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);

            % Store
            V(level1ii,e_c,jj)=shiftdim(Vtempii,1);
            Policy(level1ii,e_c,jj)=shiftdim(maxindex2,1); % d,aprime
            if nargout>2
                aprimeReferencePolicyNew(:,level1ii,e_c,jj)=reshape(maxindex1,[N_d,vfoptions.level1n]); % seed reference, see top
            end

            % Attempt for improved version
            maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1)); % max over d
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-1
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_e, d_gridvals, reshape(a_grid(aprimeindexes),size(aprimeindexes)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV(aprimeindexes),[N_d*(maxgap(ii)+1),1]); % autoexpand level1iidiff(ii) in 2nd-dim
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curraindex,e_c,jj)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    % allind=dind; % loweredge is n_d-by-1-by-1
                    Policy(curraindex,e_c,jj)=maxindex'+N_d*(reshape(loweredge(dind),[],1)-1); % forced to a column: with N_d=1 loweredge is a scalar and indexing it by a row gives a row
                    if nargout>2 % per-d max over the same RHS, seed reference, see top
                        [~,maxindexd]=max(reshape(entireRHS_ii,[N_d,maxgap(ii)+1,length(curraindex)]),[],2);
                        aprimeReferencePolicyNew(:,curraindex,e_c,jj)=reshape(maxindexd+(loweredge-1),[N_d,length(curraindex)]);
                    end
                else
                    loweredge=maxindex1(:,1,ii);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_e, d_gridvals, reshape(a_grid(loweredge),size(loweredge)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV(loweredge); % autoexpand level1iidiff(ii) in 2nd-dim
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curraindex,e_c,jj)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    % allind=dind; % loweredge is n_d-by-1-by-1
                    Policy(curraindex,e_c,jj)=maxindex'+N_d*(reshape(loweredge(dind),[],1)-1); % forced to a column: with N_d=1 loweredge is a scalar and indexing it by a row gives a row
                    if nargout>2 % seed reference, see top
                        aprimeReferencePolicyNew(:,curraindex,e_c,jj)=repmat(loweredge,1,length(curraindex));
                    end
                end
            end
        end
    end
end

Policy=shiftdim(Policy,-1);

end
