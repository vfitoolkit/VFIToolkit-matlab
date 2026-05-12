function [Vunderbar,Policy2,Vhat]=ValueFnIter_FHorz_QuasiHyperbolicS_DC1_noz_e_raw(n_d,n_a,n_e,N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Sophisticated quasi-hyperbolic discounting variant of ValueFnIter_FHorz_DC1_noz_e_raw.
% Has d and e variables. No z variable. GPU (parallel==2 only).
%
% Sophisticated: Vhat_j = max u + beta_0*beta*E[Vunderbar_{j+1}]
%                Vunderbar_j = Vhat_j + (beta - beta_0*beta)*EV_at_optimal_aprime

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

Vhat=zeros(N_a,N_e,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_e,N_j,'gpuArray');

level1ii=round(linspace(1,n_a,vfoptions.level1n));

if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
else
    eind=shiftdim((0:1:N_e-1),-1);
end

pi_e_J=shiftdim(pi_e_J,-1);

%% j=N_j (terminal period)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_e, d_gridvals, a_grid, a_grid(level1ii), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a,vfoptions.level1n,N_e]),[],1);
        Vhat(level1ii,:,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,:,N_j)=shiftdim(maxindex2,1);
        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Vhat(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*eind;
                Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_e, d_gridvals, a_grid(loweredge), a_grid(curraindex), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Vhat(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*eind;
                Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            end
        end
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_e, d_gridvals, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec,1);
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a,vfoptions.level1n]),[],1);
            Vhat(level1ii,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(level1ii,e_c,N_j)=shiftdim(maxindex2,1);
            maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), e_val, ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    Vhat(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    Policy(curraindex,e_c,N_j)=maxindex'+N_d*(loweredge(dind)-1);
                else
                    loweredge=maxindex1(:,1,ii);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_e, d_gridvals, a_grid(loweredge), a_grid(curraindex), e_val, ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    Vhat(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    Policy(curraindex,e_c,N_j)=maxindex'+N_d*(loweredge(dind)-1);
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
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_e, d_gridvals, a_grid, a_grid(level1ii), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n,N_e]),[],1);
        Vhat(level1ii,:,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,:,N_j)=shiftdim(maxindex2,1);
        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV(aprimeindexes),[N_d*(maxgap(ii)+1),1,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*eind;
                Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_e, d_gridvals, a_grid(loweredge), a_grid(curraindex), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV(loweredge),[N_d*1,1,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*eind;
                Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            end
        end
        aprime_ind=ceil(Policy(:,:,N_j)/N_d);
        EV_at_policy=EV(aprime_ind);
        Vunderbar(:,:,N_j)=Vhat(:,:,N_j)+(beta-beta0beta)*EV_at_policy;

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_e, d_gridvals, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec,1);
            entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV,-1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);
            Vhat(level1ii,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(level1ii,e_c,N_j)=shiftdim(maxindex2,1);
            maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), e_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV(aprimeindexes),[N_d*(maxgap(ii)+1),1]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vhat(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    Policy(curraindex,e_c,N_j)=maxindex'+N_d*(loweredge(dind)-1);
                else
                    loweredge=maxindex1(:,1,ii);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_e, d_gridvals, a_grid(loweredge), a_grid(curraindex), e_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*EV(loweredge);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vhat(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    Policy(curraindex,e_c,N_j)=maxindex'+N_d*(loweredge(dind)-1);
                end
            end
            aprime_ind_e=ceil(Policy(:,e_c,N_j)/N_d);
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
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_e, d_gridvals, a_grid, a_grid(level1ii), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);
        entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n,N_e]),[],1);
        Vhat(level1ii,:,jj)=shiftdim(Vtempii,1);
        Policy(level1ii,:,jj)=shiftdim(maxindex2,1);
        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV(aprimeindexes),[N_d*(maxgap(ii)+1),1,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat(curraindex,:,jj)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*eind;
                Policy(curraindex,:,jj)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_e, d_gridvals, a_grid(loweredge), a_grid(curraindex), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV(loweredge),[N_d*1,1,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat(curraindex,:,jj)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*eind;
                Policy(curraindex,:,jj)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            end
        end
        aprime_ind=ceil(Policy(:,:,jj)/N_d);
        EV_at_policy=EV(aprime_ind);
        Vunderbar(:,:,jj)=Vhat(:,:,jj)+(beta-beta0beta)*EV_at_policy;

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_e, d_gridvals, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec,1);
            entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV,-1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);
            Vhat(level1ii,e_c,jj)=shiftdim(Vtempii,1);
            Policy(level1ii,e_c,jj)=shiftdim(maxindex2,1);
            maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), e_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV(aprimeindexes),[N_d*(maxgap(ii)+1),1]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vhat(curraindex,e_c,jj)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    Policy(curraindex,e_c,jj)=maxindex'+N_d*(loweredge(dind)-1);
                else
                    loweredge=maxindex1(:,1,ii);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_e, d_gridvals, a_grid(loweredge), a_grid(curraindex), e_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*EV(loweredge);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vhat(curraindex,e_c,jj)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    Policy(curraindex,e_c,jj)=maxindex'+N_d*(loweredge(dind)-1);
                end
            end
            aprime_ind_e=ceil(Policy(:,e_c,jj)/N_d);
            EV_at_policy_e=EV(aprime_ind_e);
            Vunderbar(:,e_c,jj)=Vhat(:,e_c,jj)+(beta-beta0beta)*EV_at_policy_e;
        end
    end
end

%%
Policy2=zeros(2,N_a,N_e,N_j,'gpuArray');
Policy2(1,:,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:,:)=shiftdim(ceil(Policy/N_d),-1);

end
