function [V,Policy2]=ValueFnIter_FHorz_DC1_noz_e_raw(n_d,n_a,n_e,N_j, d_grid, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
d_gridvals=CreateGridvals(n_d,d_grid,1);

% n-Monotonicity
% vfoptions.level1n=5;
level1ii=round(linspace(1,n_a,vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
else
    eind=shiftdim((0:1:N_e-1),-1); % already includes -1
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

pi_e_J=shiftdim(pi_e_J,-1); % Move to second dimension (normally -2, but no z so -1)

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_e, d_grid, a_grid, a_grid(level1ii), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        % First, we want aprime conditional on (d,1,a,e)
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a,vfoptions.level1n,N_e]),[],1);

        % Store
        V(level1ii,:,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,:,N_j)=shiftdim(maxindex2,1); % d,aprime

        % Attempt for improved version
        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1)); % max over d,e
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-1-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*eind; % loweredge is n_d-by-1-by-1-by-n_e
                Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*eind; % loweredge is n_d-by-1-by-1-by-n_e
                Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            end
        end
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_e, d_grid, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec,1);

            % First, we want aprime conditional on (d,1,a,e)
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a,vfoptions.level1n]),[],1);

            % Store
            V(level1ii,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(level1ii,e_c,N_j)=shiftdim(maxindex2,1); % d,aprime

            % Attempt for improved version
            maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1)); % max over d,e
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-1
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    allind=dind; % loweredge is n_d-by-1-by-1
                    Policy(curraindex,e_c,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    % allind=dind; % loweredge is n_d-by-1-by-1
                    Policy(curraindex,e_c,N_j)=maxindex'+N_d*(loweredge(dind)-1);
                end
            end
        end
    end

else
    
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    EV=sum(reshape(vfoptions.V_Jplus1,[N_a,N_e]).*pi_e_J(1,:,N_j),2); % Using V_Jplus1
    
    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_e, d_grid, a_grid, a_grid(level1ii), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*shiftdim(EV,-1);

        % First, we want aprime conditional on (d,1,a,z)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n,N_e]),[],1);

        % Store
        V(level1ii,:,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,:,N_j)=shiftdim(maxindex2,1); % d,aprime

        % Attempt for improved version
        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1)); % max over d,e
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-1-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV(aprimeindexes),[N_d*(maxgap(ii)+1),1,N_e]); % autoexpand level1iidiff(ii) in 2nd-dim
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*eind; % loweredge is n_d-by-1-by-1-by-n_e
                Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV(loweredge),[N_d*1,1,N_e]); % autoexpand level1iidiff(ii) in 2nd-dim
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*eind; % loweredge is n_d-by-1-by-1-by-n_e
                Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            end
        end
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);

            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_e, d_grid, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec,1);

            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*shiftdim(EV,-1);

            % First, we want aprime conditional on (d,1,a,z)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);

            % Store
            V(level1ii,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(level1ii,e_c,N_j)=shiftdim(maxindex2,1); % d,aprime

            % Attempt for improved version
            maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1)); % max over d,e
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-1
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV(aprimeindexes),[N_d*(maxgap(ii)+1),1]);  % autoexpand level1iidiff(ii) in 2nd-dim
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    % allind=dind; % loweredge is n_d-by-1-by-1
                    Policy(curraindex,e_c,N_j)=maxindex'+N_d*(loweredge(dind)-1);
                else
                    loweredge=maxindex1(:,1,ii);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV(loweredge); % autoexpand level1iidiff(ii) in 2nd-dim
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    % allind=dind; % loweredge is n_d-by-1-by-1
                    Policy(curraindex,e_c,N_j)=maxindex'+N_d*(loweredge(dind)-1);
                end
            end
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
    
    EV=sum(V(:,:,jj+1).*pi_e_J(1,:,jj),2);

    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_e, d_grid, a_grid, a_grid(level1ii), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*shiftdim(EV,-1);

        % First, we want aprime conditional on (d,1,a,z)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n,N_e]),[],1);

        % Store
        V(level1ii,:,jj)=shiftdim(Vtempii,1);
        Policy(level1ii,:,jj)=shiftdim(maxindex2,1); % d,aprime

        % Attempt for improved version
        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1)); % max over d,e
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-1-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV(aprimeindexes),[N_d*(maxgap(ii)+1),1,N_e]);  % autoexpand level1iidiff(ii) in 2nd-dim
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,jj)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*eind; % loweredge is n_d-by-1-by-1-by-n_e
                Policy(curraindex,:,jj)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV(loweredge),[N_d*1,1,N_e]); % autoexpand level1iidiff(ii) in 2nd-dim
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,jj)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*eind; % loweredge is n_d-by-1-by-1-by-n_e
                Policy(curraindex,:,jj)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            end
        end
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);

            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_e, d_grid, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec,1);

            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*shiftdim(EV,-1);

            % First, we want aprime conditional on (d,1,a,z)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);

            % Store
            V(level1ii,e_c,jj)=shiftdim(Vtempii,1);
            Policy(level1ii,e_c,jj)=shiftdim(maxindex2,1); % d,aprime

            % Attempt for improved version
            maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1)); % max over d,e
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-1
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV(aprimeindexes),[N_d*(maxgap(ii)+1),1]);  % autoexpand level1iidiff(ii) in 2nd-dim
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curraindex,e_c,jj)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    % allind=dind; % loweredge is n_d-by-1-by-1
                    Policy(curraindex,e_c,jj)=maxindex'+N_d*(loweredge(dind)-1);
                else
                    loweredge=maxindex1(:,1,ii);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV(loweredge);  % autoexpand level1iidiff(ii) in 2nd-dim
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curraindex,e_c,jj)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    % allind=dind; % loweredge is n_d-by-1-by-1
                    Policy(curraindex,e_c,jj)=maxindex'+N_d*(loweredge(dind)-1);
                end
            end
        end
    end
end

%%
Policy2=zeros(2,N_a,N_e,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
Policy2(1,:,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:,:)=shiftdim(ceil(Policy/N_d),-1);

end
