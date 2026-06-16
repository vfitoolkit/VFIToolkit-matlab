function [V, Policy, Vhat]=ValueFnIter_FHorz_TPath_SingleStep_QHS_DC1_nod_e_raw(V,n_a,n_z,n_e,N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The V input is next period value fn (across all ages), the V output is this period.
% Sophisticated quasi-hyperbolic: V carries Vunderbar (realised value under QH policies); Vhat is the agent's-perspective (beta0*beta) value.

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z,e
Vhat=zeros(N_a,N_z,N_e,N_j,'gpuArray'); % agent's-perspective value (beta0*beta-discounted), before the Vunderbar transform

Vnext=sum(V.*shiftdim(pi_e_J(:,[1,1:end-1]),-2),3); % Take expectations over e: Vnext(...,jj+1) is read for current age jj, so weight V at age jj+1 by pi_e_J(:,jj) [same timing as standard ValueFnIter commands]; first column is padding, never read

%%
if vfoptions.lowmemory==0
    loweredgesize=[1,1,N_z,N_e];
elseif vfoptions.lowmemory==1
    loweredgesize=[1,1,N_z];
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory==2
    special_n_e=ones(1,length(n_e));
    special_n_z=ones(1,length(n_z));
elseif vfoptions.lowmemory>=3
    error('vfoptions.lowmemory>=3 not supported for ValueFnIter_FHorz_TPath_SingleStep_QHS_DC1_nod_e_raw')
end

zind=shiftdim(gpuArray(0:1:N_z-1),-1);
zCind=gpuArray(0:1:N_z-1);

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% j=N_j: terminal age has no continuation in TPath

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if vfoptions.lowmemory==0
    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, n_z, n_e, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

    [Vtempii,maxindex1]=max(ReturnMatrix_ii,[],1);

    V(level1ii,:,:,N_j)=shiftdim(Vtempii,1);
    Policy(level1ii,:,:,N_j)=shiftdim(maxindex1,1);

    maxgap=max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii));
            aprimeindexes=loweredge+(0:1:maxgap(ii))';
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, n_z, n_e, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
            Policy(curraindex,:,:,N_j)=shiftdim(maxindex+loweredge-1,1);
        else
            loweredge=maxindex1(1,ii,:,:);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, n_z, n_e, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            V(curraindex,:,:,N_j)=shiftdim(ReturnMatrix_ii,1);
            Policy(curraindex,:,:,N_j)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1,1);
        end
    end
elseif vfoptions.lowmemory==1
    for e_c=1:N_e
        e_val=e_gridvals_J(e_c,:,N_j);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, n_z, special_n_e, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);

        [Vtempii,maxindex1]=max(ReturnMatrix_ii,[],1);

        V(level1ii,:,e_c,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,:,e_c,N_j)=shiftdim(maxindex1,1);

        maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, n_z, special_n_e, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                Policy(curraindex,:,e_c,N_j)=shiftdim(maxindex+loweredge-1,1);
            else
                loweredge=maxindex1(1,ii,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, n_z, special_n_e, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                V(curraindex,:,e_c,N_j)=shiftdim(ReturnMatrix_ii,1);
                Policy(curraindex,:,e_c,N_j)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
            end
        end
    end
elseif vfoptions.lowmemory==2
    for z_c=1:N_z
        z_val=z_gridvals_J(z_c,:,N_j);
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, special_n_z, special_n_e, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,1);

            [Vtempii,maxindex1]=max(ReturnMatrix_ii,[],1);

            V(level1ii,z_c,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(level1ii,z_c,e_c,N_j)=shiftdim(maxindex1,1);

            maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, special_n_z, special_n_e, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                    Policy(curraindex,z_c,e_c,N_j)=shiftdim(maxindex+loweredge-1,1);
                else
                    loweredge=maxindex1(1,ii);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, special_n_z, special_n_e, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,2);
                    V(curraindex,z_c,e_c,N_j)=shiftdim(ReturnMatrix_ii,1);
                    Policy(curraindex,z_c,e_c,N_j)=loweredge;
                end
            end
        end
    end
end
Vhat(:,:,:,N_j)=V(:,:,:,N_j); % terminal: Vhat coincides with V (no Vunderbar transform at terminal)


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    VKronNext_j=Vnext(:,:,1,jj+1); % e-expectation pre-computed

    EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, n_z, n_e, a_grid, a_grid(level1ii), z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
        [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
        V(level1ii,:,:,jj)=shiftdim(Vtempii,1);
        Policy(level1ii,:,:,jj)=shiftdim(maxindex1,1);
        maxgap=max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, n_z, n_e, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                aprimez=aprimeindexes+N_a*zind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV(aprimez),[maxgap(ii)+1,1,N_z,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,jj)=shiftdim(Vtempii,1);
                Policy(curraindex,:,:,jj)=shiftdim(maxindex+loweredge-1,1);
            else
                loweredge=maxindex1(1,ii,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, n_z, n_e, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                aprimez=loweredge+N_a*zind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV(aprimez),[1,1,N_z,N_e]);
                V(curraindex,:,:,jj)=shiftdim(entireRHS_ii,1);
                Policy(curraindex,:,:,jj)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1,1);
            end
        end
        % Vunderbar = Vhat + (beta - beta0*beta)*EV_at_optimal_aprime
        Vhat(:,:,:,jj)=V(:,:,:,jj); % Save Vhat before applying the Vunderbar transform
        aprime_ind=Policy(:,:,:,jj);
        EV_at_policy=EV(aprime_ind+N_a*zCind);
        V(:,:,:,jj)=V(:,:,:,jj)+(beta-beta0beta)*EV_at_policy;

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, n_z, special_n_e, a_grid, a_grid(level1ii), z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1);

            entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
            [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
            V(level1ii,:,e_c,jj)=shiftdim(Vtempii,1);
            Policy(level1ii,:,e_c,jj)=shiftdim(maxindex1,1);
            maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, n_z, special_n_e, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
                    aprimez=aprimeindexes+N_a*zind;
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV(aprimez),[maxgap(ii)+1,1,N_z]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curraindex,:,e_c,jj)=shiftdim(Vtempii,1);
                    Policy(curraindex,:,e_c,jj)=shiftdim(maxindex+loweredge-1,1);
                else
                    loweredge=maxindex1(1,ii,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, n_z, special_n_e, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
                    aprimez=loweredge+N_a*zind;
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV(aprimez),[1,1,N_z]);
                    V(curraindex,:,e_c,jj)=shiftdim(entireRHS_ii,1);
                    Policy(curraindex,:,e_c,jj)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
                end
            end
            Vhat(:,:,e_c,jj)=V(:,:,e_c,jj); % Save Vhat before applying the Vunderbar transform
            aprime_ind_e=Policy(:,:,e_c,jj);
            EV_at_policy_e=EV(aprime_ind_e+N_a*zCind);
            V(:,:,e_c,jj)=V(:,:,e_c,jj)+(beta-beta0beta)*EV_at_policy_e;
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            EV_z=EV(:,:,z_c);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, special_n_z, special_n_e, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,1);

                entireRHS_ii=ReturnMatrix_ii+beta0beta*EV_z;
                [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
                V(level1ii,z_c,e_c,jj)=shiftdim(Vtempii,1);
                Policy(level1ii,z_c,e_c,jj)=shiftdim(maxindex1,1);
                maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(1,ii),n_a-maxgap(ii));
                        aprimeindexes=loweredge+(0:1:maxgap(ii))';
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, special_n_z, special_n_e, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,2);
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*EV_z(aprimeindexes);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V(curraindex,z_c,e_c,jj)=shiftdim(Vtempii,1);
                        Policy(curraindex,z_c,e_c,jj)=shiftdim(maxindex+loweredge-1,1);
                    else
                        loweredge=maxindex1(1,ii);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, special_n_z, special_n_e, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,2);
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*EV_z(loweredge);
                        V(curraindex,z_c,e_c,jj)=shiftdim(entireRHS_ii,1);
                        Policy(curraindex,z_c,e_c,jj)=loweredge;
                    end
                end
                Vhat(:,z_c,e_c,jj)=V(:,z_c,e_c,jj); % Save Vhat before applying the Vunderbar transform
                aprime_ind_ze=Policy(:,z_c,e_c,jj);
                EV_at_policy_ze=EV_z(aprime_ind_ze);
                V(:,z_c,e_c,jj)=V(:,z_c,e_c,jj)+(beta-beta0beta)*EV_at_policy_ze;
            end
        end
    end
end

%% Output shape for policy
Policy=shiftdim(Policy,-1);

end
