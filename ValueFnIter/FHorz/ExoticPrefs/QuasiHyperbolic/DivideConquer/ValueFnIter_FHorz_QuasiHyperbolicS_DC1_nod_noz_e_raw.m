function [Vunderbar,Policy,Vhat]=ValueFnIter_FHorz_QuasiHyperbolicS_DC1_nod_noz_e_raw(n_a,n_e,N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Sophisticated quasi-hyperbolic discounting variant of ValueFnIter_FHorz_DC1_nod_noz_e_raw.
% No d variables. No z variable. Has e variable. GPU (parallel==2 only).
%
% Sophisticated: Vhat_j = max u + beta_0*beta*E[Vunderbar_{j+1}]  (agent's choice)
%                Vunderbar_j = Vhat_j + (beta - beta_0*beta)*E[Vunderbar_{j+1}|aprime=policy]
%                EVsource = Vunderbar

N_a=prod(n_a);
N_e=prod(n_e);

Vhat=zeros(N_a,N_e,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_e,N_j,'gpuArray');

if vfoptions.lowmemory==0
    loweredgesize=[1,1,N_e];
elseif vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
end

level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

pi_e_J=shiftdim(pi_e_J,-1);

%% j=N_j (terminal period)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_e, a_grid, a_grid(level1ii), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        [Vtempii,maxindex1]=max(ReturnMatrix_ii,[],1);
        Vhat(level1ii,:,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,:,N_j)=shiftdim(maxindex1,1);
        maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_e, a_grid(aprimeindexes), a_grid(curraindex), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Vhat(curraindex,:,N_j)=shiftdim(Vtempii,1);
                Policy(curraindex,:,N_j)=shiftdim(maxindex+loweredge-1,1);
            else
                loweredge=maxindex1(1,ii,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_e, reshape(a_grid(loweredge),loweredgesize), a_grid(curraindex), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                Vhat(curraindex,:,N_j)=shiftdim(ReturnMatrix_ii,1);
                Policy(curraindex,:,N_j)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
            end
        end
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_e, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec,1);
            [Vtempii,maxindex1]=max(ReturnMatrix_ii,[],1);
            Vhat(level1ii,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(level1ii,e_c,N_j)=shiftdim(maxindex1,1);
            maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_e, a_grid(aprimeindexes), a_grid(curraindex), e_val, ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    Vhat(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    Policy(curraindex,e_c,N_j)=shiftdim(maxindex+loweredge-1,1);
                else
                    loweredge=maxindex1(1,ii);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_e, a_grid(loweredge), a_grid(curraindex), e_val, ReturnFnParamsVec,2);
                    Vhat(curraindex,e_c,N_j)=shiftdim(ReturnMatrix_ii,1);
                    Policy(curraindex,e_c,N_j)=loweredge;
                end
            end
        end
    end
    Vunderbar=Vhat;

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=sum(reshape(vfoptions.V_Jplus1,[N_a,N_e]).*pi_e_J(1,:,N_j),2);

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_e, a_grid, a_grid(level1ii), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        % --- Vhat search (beta0beta) ---
        entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
        [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
        Vhat(level1ii,:,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,:,N_j)=shiftdim(maxindex1,1);
        maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii_g=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_e, a_grid(aprimeindexes), a_grid(curraindex), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                entireRHS_ii_g=ReturnMatrix_ii_g+beta0beta*reshape(EV(aprimeindexes),[maxgap(ii)+1,1,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii_g,[],1);
                Vhat(curraindex,:,N_j)=shiftdim(Vtempii,1);
                Policy(curraindex,:,N_j)=shiftdim(maxindex+loweredge-1,1);
            else
                loweredge=maxindex1(1,ii,:);
                ReturnMatrix_ii_g=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_e, reshape(a_grid(loweredge),loweredgesize), a_grid(curraindex), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                entireRHS_ii_g=ReturnMatrix_ii_g+beta0beta*reshape(EV(loweredge),[1,1,N_e]);
                Vhat(curraindex,:,N_j)=shiftdim(entireRHS_ii_g,1);
                Policy(curraindex,:,N_j)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
            end
        end
        % --- Compute Vunderbar ---
        aprime_ind=Policy(:,:,N_j);
        EV_at_policy=EV(aprime_ind);
        Vunderbar(:,:,N_j)=Vhat(:,:,N_j)+(beta-beta0beta)*EV_at_policy;

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_e, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec,1);
            % --- Vhat search (beta0beta) ---
            entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
            [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
            Vhat(level1ii,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(level1ii,e_c,N_j)=shiftdim(maxindex1,1);
            maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    ReturnMatrix_ii_g=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_e, a_grid(aprimeindexes), a_grid(curraindex), e_val, ReturnFnParamsVec,2);
                    entireRHS_ii_g=ReturnMatrix_ii_g+beta0beta*EV(aprimeindexes);
                    [Vtempii,maxindex]=max(entireRHS_ii_g,[],1);
                    Vhat(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    Policy(curraindex,e_c,N_j)=shiftdim(maxindex+loweredge-1,1);
                else
                    loweredge=maxindex1(1,ii);
                    ReturnMatrix_ii_g=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_e, a_grid(loweredge), a_grid(curraindex), e_val, ReturnFnParamsVec,2);
                    entireRHS_ii_g=ReturnMatrix_ii_g+beta0beta*EV(loweredge);
                    Vhat(curraindex,e_c,N_j)=shiftdim(entireRHS_ii_g,1);
                    Policy(curraindex,e_c,N_j)=loweredge;
                end
            end
            % --- Compute Vunderbar for this e_c ---
            aprime_ind_e=Policy(:,e_c,N_j);
            EV_at_policy_e=EV(aprime_ind_e);
            Vunderbar(:,e_c,N_j)=Vhat(:,e_c,N_j)+(beta-beta0beta)*EV_at_policy_e;
        end
    end
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    EV=sum(Vunderbar(:,:,jj+1).*pi_e_J(1,:,jj),2);

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_e, a_grid, a_grid(level1ii), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);
        % --- Vhat search (beta0beta) ---
        entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
        [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
        Vhat(level1ii,:,jj)=shiftdim(Vtempii,1);
        Policy(level1ii,:,jj)=shiftdim(maxindex1,1);
        maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii_g=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_e, a_grid(aprimeindexes), a_grid(curraindex), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                entireRHS_ii_g=ReturnMatrix_ii_g+beta0beta*reshape(EV(aprimeindexes),[maxgap(ii)+1,1,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii_g,[],1);
                Vhat(curraindex,:,jj)=shiftdim(Vtempii,1);
                Policy(curraindex,:,jj)=shiftdim(maxindex+loweredge-1,1);
            else
                loweredge=maxindex1(1,ii,:);
                ReturnMatrix_ii_g=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_e, reshape(a_grid(loweredge),loweredgesize), a_grid(curraindex), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                entireRHS_ii_g=ReturnMatrix_ii_g+beta0beta*reshape(EV(loweredge),[1,1,N_e]);
                Vhat(curraindex,:,jj)=shiftdim(entireRHS_ii_g,1);
                Policy(curraindex,:,jj)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
            end
        end
        % --- Compute Vunderbar ---
        aprime_ind=Policy(:,:,jj);
        EV_at_policy=EV(aprime_ind);
        Vunderbar(:,:,jj)=Vhat(:,:,jj)+(beta-beta0beta)*EV_at_policy;

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_e, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec,1);
            % --- Vhat search (beta0beta) ---
            entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
            [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
            Vhat(level1ii,e_c,jj)=shiftdim(Vtempii,1);
            Policy(level1ii,e_c,jj)=shiftdim(maxindex1,1);
            maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    ReturnMatrix_ii_g=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_e, a_grid(aprimeindexes), a_grid(curraindex), e_val, ReturnFnParamsVec,2);
                    entireRHS_ii_g=ReturnMatrix_ii_g+beta0beta*EV(aprimeindexes);
                    [Vtempii,maxindex]=max(entireRHS_ii_g,[],1);
                    Vhat(curraindex,e_c,jj)=shiftdim(Vtempii,1);
                    Policy(curraindex,e_c,jj)=shiftdim(maxindex+loweredge-1,1);
                else
                    loweredge=maxindex1(1,ii);
                    ReturnMatrix_ii_g=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_e, a_grid(loweredge), a_grid(curraindex), e_val, ReturnFnParamsVec,2);
                    entireRHS_ii_g=ReturnMatrix_ii_g+beta0beta*EV(loweredge);
                    Vhat(curraindex,e_c,jj)=shiftdim(entireRHS_ii_g,1);
                    Policy(curraindex,e_c,jj)=loweredge;
                end
            end
            % --- Compute Vunderbar for this e_c ---
            aprime_ind_e=Policy(:,e_c,jj);
            EV_at_policy_e=EV(aprime_ind_e);
            Vunderbar(:,e_c,jj)=Vhat(:,e_c,jj)+(beta-beta0beta)*EV_at_policy_e;
        end
    end
end

end
