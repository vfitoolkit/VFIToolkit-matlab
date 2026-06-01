function [V, Policy, Policyalt]=ValueFnIter_FHorz_TPath_SingleStep_QHN_DC1_nod_noz_e_raw(V,n_a,n_e,N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The V input is next period value fn (across all ages), the V output is this period.
% Naive quasi-hyperbolic: V carries Valt (exp-discounter value).

N_a=prod(n_a);
N_e=prod(n_e);

Policy=zeros(N_a,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,e
Policyalt=zeros(N_a,N_e,N_j,'gpuArray'); % exponential discounter optimal choice (Valt is computed at this)

Vnext=sum(V.*shiftdim(pi_e_J,-1),2); % Take expectations over e

%%
if vfoptions.lowmemory==0
    loweredgesize=[1,1,N_e];
elseif vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory>=2
    error('vfoptions.lowmemory>=2 not supported for ValueFnIter_FHorz_TPath_SingleStep_QHN_DC1_nod_noz_e_raw')
end

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% j=N_j: terminal age has no continuation in TPath

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if vfoptions.lowmemory==0
    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_e, a_grid, a_grid(level1ii), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

    [Vtempii,maxindex1]=max(ReturnMatrix_ii,[],1);

    V(level1ii,:,N_j)=shiftdim(Vtempii,1);
    Policy(level1ii,:,N_j)=shiftdim(maxindex1,1);

    maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
            aprimeindexes=loweredge+(0:1:maxgap(ii))';
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_e, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            V(curraindex,:,N_j)=shiftdim(Vtempii,1);
            Policy(curraindex,:,N_j)=shiftdim(maxindex+loweredge-1,1);
        else
            loweredge=maxindex1(1,ii,:);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_e, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            V(curraindex,:,N_j)=shiftdim(ReturnMatrix_ii,1);
            Policy(curraindex,:,N_j)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
        end
    end
elseif vfoptions.lowmemory==1
    for e_c=1:N_e
        e_val=e_gridvals_J(e_c,:,N_j);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, special_n_e, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec,1);

        [Vtempii,maxindex1]=max(ReturnMatrix_ii,[],1);

        V(level1ii,e_c,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,e_c,N_j)=shiftdim(maxindex1,1);

        maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, special_n_e, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                Policy(curraindex,e_c,N_j)=shiftdim(maxindex+loweredge-1,1);
            else
                loweredge=maxindex1(1,ii);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, special_n_e, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,2);
                V(curraindex,e_c,N_j)=shiftdim(ReturnMatrix_ii,1);
                Policy(curraindex,e_c,N_j)=loweredge;
            end
        end
    end
end
Policyalt(:,:,N_j)=Policy(:,:,N_j); % terminal: QH and exp discounter coincide


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    EV=Vnext(:,1,jj+1); % e-expectation pre-computed

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_e, a_grid, a_grid(level1ii), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        %% Valt (beta)
        entireRHS_ii=ReturnMatrix_ii+beta*EV;
        [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
        V(level1ii,:,jj)=shiftdim(Vtempii,1);
        Policyalt(level1ii,:,jj)=shiftdim(maxindex1,1);
        maxgap_V=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap_V(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap_V(ii));
                aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_e, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                entireRHS_ii=ReturnMatrix_ii_dc+beta*reshape(EV(aprimeindexes),[maxgap_V(ii)+1,1,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,jj)=shiftdim(Vtempii,1);
                Policyalt(curraindex,:,jj)=shiftdim(maxindex+loweredge-1,1);
            else
                loweredge=maxindex1(1,ii,:);
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_e, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                entireRHS_ii=ReturnMatrix_ii_dc+beta*reshape(EV(loweredge),[1,1,N_e]);
                V(curraindex,:,jj)=shiftdim(entireRHS_ii,1);
                Policyalt(curraindex,:,jj)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
            end
        end
        %% Policy (beta0*beta)
        entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
        [~,maxindex1]=max(entireRHS_ii,[],1);
        Policy(level1ii,:,jj)=shiftdim(maxindex1,1);
        maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_e, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*reshape(EV(aprimeindexes),[maxgap(ii)+1,1,N_e]);
                [~,maxindex]=max(entireRHS_ii,[],1);
                Policy(curraindex,:,jj)=shiftdim(maxindex+loweredge-1,1);
            else
                loweredge=maxindex1(1,ii,:);
                Policy(curraindex,:,jj)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
            end
        end
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, special_n_e, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec,1);

            %% Valt (beta)
            entireRHS_ii=ReturnMatrix_ii+beta*EV;
            [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
            V(level1ii,e_c,jj)=shiftdim(Vtempii,1);
            Policyalt(level1ii,e_c,jj)=shiftdim(maxindex1,1);
            maxgap_V=maxindex1(1,2:end)-maxindex1(1,1:end-1);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(1,ii),n_a-maxgap_V(ii));
                    aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, special_n_e, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii_dc+beta*EV(aprimeindexes);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curraindex,e_c,jj)=shiftdim(Vtempii,1);
                    Policyalt(curraindex,e_c,jj)=shiftdim(maxindex+loweredge-1,1);
                else
                    loweredge=maxindex1(1,ii);
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, special_n_e, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii_dc+beta*EV(loweredge);
                    V(curraindex,e_c,jj)=shiftdim(entireRHS_ii,1);
                    Policyalt(curraindex,e_c,jj)=loweredge;
                end
            end
            %% Policy (beta0*beta)
            entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
            [~,maxindex1]=max(entireRHS_ii,[],1);
            Policy(level1ii,e_c,jj)=shiftdim(maxindex1,1);
            maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, special_n_e, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*EV(aprimeindexes);
                    [~,maxindex]=max(entireRHS_ii,[],1);
                    Policy(curraindex,e_c,jj)=shiftdim(maxindex+loweredge-1,1);
                else
                    loweredge=maxindex1(1,ii);
                    Policy(curraindex,e_c,jj)=loweredge;
                end
            end
        end
    end
end

%% Output shape for policy
Policy=shiftdim(Policy,-1);
Policyalt=shiftdim(Policyalt,-1);

end
