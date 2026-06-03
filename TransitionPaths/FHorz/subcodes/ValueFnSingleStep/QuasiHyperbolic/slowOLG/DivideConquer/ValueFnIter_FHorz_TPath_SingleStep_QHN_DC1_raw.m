function [V,Policy,Policyalt,Vtilde]=ValueFnIter_FHorz_TPath_SingleStep_QHN_DC1_raw(V,n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The V input is next period value fn (across all ages), the V output is this period.
% Naive quasi-hyperbolic: V carries Valt (exp-discounter value); Vtilde is the agent's-perspective value (beta0*beta).

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

Policy=zeros(N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z
Policyalt=zeros(N_a,N_z,N_j,'gpuArray'); % exponential discounter optimal choice (Valt is computed at this)
Vtilde=zeros(N_a,N_z,N_j,'gpuArray'); % agent's-perspective value (beta0*beta-discounted)

%%
% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

if vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
elseif vfoptions.lowmemory>=2
    error('vfoptions.lowmemory>=2 not supported')
else
    zind=shiftdim((0:1:N_z-1),-1); % already includes -1
end

zBind=shiftdim(gpuArray(0:1:N_z-1),-2);

%% j=N_j: terminal age has no continuation in TPath
% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,:,N_j);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if vfoptions.lowmemory==0
    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

    % First, we want aprime conditional on (d,1,a,z)
    [~,maxindex1]=max(ReturnMatrix_ii,[],2);

    % Now, get and store the full (d,aprime)
    [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a,vfoptions.level1n,N_z]),[],1);

    % Store
    V(level1ii,:,N_j)=shiftdim(Vtempii,1);
    Policy(level1ii,:,N_j)=shiftdim(maxindex2,1); % d,aprime

    % Second level based on monotonicity
    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1-by-1-by-n_z
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            V(curraindex,:,N_j)=shiftdim(Vtempii,1);
            dind=(rem(maxindex-1,N_d)+1);
            allind=dind+N_d*zind; % loweredge is n_d-by-1-by-1-by-n_z
            Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
        else
            loweredge=maxindex1(:,1,ii,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            V(curraindex,:,N_j)=shiftdim(Vtempii,1);
            dind=(rem(maxindex-1,N_d)+1);
            allind=dind+N_d*zind; % loweredge is n_d-by-1-by-1-by-n_z
            Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
        end
    end
elseif vfoptions.lowmemory==1
    for z_c=1:N_z
        z_val=z_gridvals_J(z_c,:,N_j);
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_z, d_gridvals, a_grid, a_grid(level1ii), z_val, ReturnFnParamsVec,1);

        % First, we want aprime conditional on (d,1,a,z)
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a,vfoptions.level1n]),[],1);

        % Store
        V(level1ii,z_c,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,z_c,N_j)=shiftdim(maxindex2,1); % d,aprime

        % Second level based on monotonicity
        maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_z, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind;
                Policy(curraindex,z_c,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)'-1),1);
            else
                loweredge=maxindex1(:,1,ii);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_z, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind;
                Policy(curraindex,z_c,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)'-1),1);
            end
        end
    end
end
Policyalt(:,:,N_j)=Policy(:,:,N_j); % terminal: QH and exp discounter coincide
Vtilde(:,:,N_j)=V(:,:,N_j); % terminal: Vtilde coincides with Valt


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    VKronNext_j=Vtemp_j; % Has been presaved before it was replaced
    Vtemp_j=V(:,:,jj); % Grab this before it is replaced/updated

    EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        %% Valt (beta)
        entireRHS_ii=ReturnMatrix_ii+beta*shiftdim(EV,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,maxindex2alt]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n,N_z]),[],1);
        V(level1ii,:,jj)=shiftdim(Vtempii,1);
        Policyalt(level1ii,:,jj)=shiftdim(maxindex2alt,1);
        maxgap_V=max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap_V(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap_V(:,1,ii,:));
                aprimeindexes=loweredge+(0:1:maxgap_V(ii));
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                aprimez=aprimeindexes+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii_dc+beta*reshape(EV(aprimez),[N_d*(maxgap_V(ii)+1),1,N_z]);
                [Vtempii,maxindexalt]=max(entireRHS_ii,[],1);
                V(curraindex,:,jj)=shiftdim(Vtempii,1);
                dindalt=(rem(maxindexalt-1,N_d)+1);
                allindalt=dindalt+N_d*zind;
                Policyalt(curraindex,:,jj)=shiftdim(maxindexalt+N_d*(loweredge(allindalt)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:);
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                aprimez=loweredge+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii_dc+beta*reshape(EV(aprimez),[N_d*1,1,N_z]);
                [Vtempii,maxindexalt]=max(entireRHS_ii,[],1);
                V(curraindex,:,jj)=shiftdim(Vtempii,1);
                dindalt=(rem(maxindexalt-1,N_d)+1);
                allindalt=dindalt+N_d*zind;
                Policyalt(curraindex,:,jj)=shiftdim(maxindexalt+N_d*(loweredge(allindalt)-1),1);
            end
        end
        %% Policy (beta0*beta)
        entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n,N_z]),[],1);
        Vtilde(level1ii,:,jj)=shiftdim(Vtempii,1);
        Policy(level1ii,:,jj)=shiftdim(maxindex2,1);
        maxgap=max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(:,1,ii,:));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                aprimez=aprimeindexes+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*reshape(EV(aprimez),[N_d*(maxgap(ii)+1),1,N_z]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vtilde(curraindex,:,jj)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*zind;
                Policy(curraindex,:,jj)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:);
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                aprimez=loweredge+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*reshape(EV(aprimez),[N_d*1,1,N_z]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vtilde(curraindex,:,jj)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*zind;
                Policy(curraindex,:,jj)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            end
        end
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            EV_z=EV(:,:,z_c);

            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_z, d_gridvals, a_grid, a_grid(level1ii), z_val, ReturnFnParamsVec,1);

            %% Valt (beta)
            entireRHS_ii=ReturnMatrix_ii+beta*shiftdim(EV_z,-1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            [Vtempii,maxindex2alt]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);
            V(level1ii,z_c,jj)=shiftdim(Vtempii,1);
            Policyalt(level1ii,z_c,jj)=shiftdim(maxindex2alt,1);
            maxgap_V=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(:,1,ii),n_a-maxgap_V(ii));
                    aprimeindexes=loweredge+(0:1:maxgap_V(ii));
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_z, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii_dc+beta*reshape(EV_z(aprimeindexes),[N_d*(maxgap_V(ii)+1),1]);
                    [Vtempii,maxindexalt]=max(entireRHS_ii,[],1);
                    V(curraindex,z_c,jj)=shiftdim(Vtempii,1);
                    dindalt=(rem(maxindexalt-1,N_d)+1);
                    allindalt=dindalt;
                    Policyalt(curraindex,z_c,jj)=shiftdim(maxindexalt+N_d*(loweredge(allindalt)'-1),1);
                else
                    loweredge=maxindex1(:,1,ii);
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_z, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii_dc+beta*reshape(EV_z(loweredge),[N_d*1,1]);
                    [Vtempii,maxindexalt]=max(entireRHS_ii,[],1);
                    V(curraindex,z_c,jj)=shiftdim(Vtempii,1);
                    dindalt=(rem(maxindexalt-1,N_d)+1);
                    allindalt=dindalt;
                    Policyalt(curraindex,z_c,jj)=shiftdim(maxindexalt+N_d*(loweredge(allindalt)'-1),1);
                end
            end
            %% Policy (beta0*beta)
            entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV_z,-1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);
            Vtilde(level1ii,z_c,jj)=shiftdim(Vtempii,1);
            Policy(level1ii,z_c,jj)=shiftdim(maxindex2,1);
            maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_z, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*reshape(EV_z(aprimeindexes),[N_d*(maxgap(ii)+1),1]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vtilde(curraindex,z_c,jj)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    allind=dind;
                    Policy(curraindex,z_c,jj)=shiftdim(maxindex+N_d*(loweredge(allind)'-1),1);
                else
                    loweredge=maxindex1(:,1,ii);
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_z, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*reshape(EV_z(loweredge),[N_d*1,1]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vtilde(curraindex,z_c,jj)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    allind=dind;
                    Policy(curraindex,z_c,jj)=shiftdim(maxindex+N_d*(loweredge(allind)'-1),1);
                end
            end
        end
    end
end


Policy=shiftdim(Policy,-1);
Policyalt=shiftdim(Policyalt,-1);

end
