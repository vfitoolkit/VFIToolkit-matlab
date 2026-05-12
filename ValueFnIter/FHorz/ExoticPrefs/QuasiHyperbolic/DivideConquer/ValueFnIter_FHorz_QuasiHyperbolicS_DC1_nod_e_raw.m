function [Vunderbar,Policy,Vhat]=ValueFnIter_FHorz_QuasiHyperbolicS_DC1_nod_e_raw(n_a,n_z,n_e,N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Sophisticated quasi-hyperbolic discounting variant of ValueFnIter_FHorz_DC1_nod_e_raw.
% No d variables. Has z and e variables. GPU (parallel==2 only).
%
% Sophisticated: Vhat_j = max u + beta_0*beta*E[Vunderbar_{j+1}]
%                Vunderbar_j = Vhat_j + (beta - beta_0*beta)*EV_at_optimal_aprime

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

Vhat=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray');

if vfoptions.lowmemory==0
    loweredgesize=[1,1,N_z,N_e];
elseif vfoptions.lowmemory==1
    loweredgesize=[1,1,N_z];
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory==2
    special_n_e=ones(1,length(n_e));
    special_n_z=ones(1,length(n_z));
end

zind=shiftdim(gpuArray((0:1:N_z-1)),-1); % 1-by-N_z
zindB=gpuArray(0:1:N_z-1);  % 1-by-N_z (for EV_at_Policy indexing)

level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

pi_e_J=shiftdim(pi_e_J,-2);

%% j=N_j (terminal period)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, n_z, n_e, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        [Vtempii,maxindex1]=max(ReturnMatrix_ii,[],1);
        Vhat(level1ii,:,:,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,:,:,N_j)=shiftdim(maxindex1,1);
        maxgap=max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, n_z, n_e, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Vhat(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                Policy(curraindex,:,:,N_j)=shiftdim(maxindex+loweredge-1,1);
            else
                loweredge=maxindex1(1,ii,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, n_z, n_e, reshape(a_grid(loweredge),loweredgesize), a_grid(curraindex), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                Vhat(curraindex,:,:,N_j)=shiftdim(ReturnMatrix_ii,1);
                Policy(curraindex,:,:,N_j)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1,1);
            end
        end
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, n_z, special_n_e, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);
            [Vtempii,maxindex1]=max(ReturnMatrix_ii,[],1);
            Vhat(level1ii,:,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(level1ii,:,e_c,N_j)=shiftdim(maxindex1,1);
            maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, n_z, special_n_e, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    Vhat(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    Policy(curraindex,:,e_c,N_j)=shiftdim(maxindex+loweredge-1,1);
                else
                    loweredge=maxindex1(1,ii,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, n_z, special_n_e, reshape(a_grid(loweredge),loweredgesize), a_grid(curraindex), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                    Vhat(curraindex,:,e_c,N_j)=shiftdim(ReturnMatrix_ii,1);
                    Policy(curraindex,:,e_c,N_j)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
                end
            end
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, special_n_z, special_n_e, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,1);
                [Vtempii,maxindex1]=max(ReturnMatrix_ii,[],1);
                Vhat(level1ii,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                Policy(level1ii,z_c,e_c,N_j)=shiftdim(maxindex1,1);
                maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(1,ii),n_a-maxgap(ii));
                        aprimeindexes=loweredge+(0:1:maxgap(ii))';
                        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, special_n_z, special_n_e, a_grid(aprimeindexes), a_grid(curraindex), z_val, e_val, ReturnFnParamsVec,2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                        Vhat(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        Policy(curraindex,z_c,e_c,N_j)=shiftdim(maxindex+loweredge-1,1);
                    else
                        loweredge=maxindex1(1,ii);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, special_n_z, special_n_e, a_grid(loweredge), a_grid(curraindex), z_val, e_val, ReturnFnParamsVec,2);
                        Vhat(curraindex,z_c,e_c,N_j)=shiftdim(ReturnMatrix_ii,1);
                        Policy(curraindex,z_c,e_c,N_j)=loweredge;
                    end
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

    EV=sum(reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]).*pi_e_J(1,1,:,N_j),3);
    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, n_z, n_e, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
        [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
        Vhat(level1ii,:,:,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,:,:,N_j)=shiftdim(maxindex1,1);
        maxgap=max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, n_z, n_e, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                aprimez=aprimeindexes+N_a*zind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV(aprimez),[maxgap(ii)+1,1,N_z,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                Policy(curraindex,:,:,N_j)=shiftdim(maxindex+loweredge-1,1);
            else
                loweredge=maxindex1(1,ii,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, n_z, n_e, reshape(a_grid(loweredge),loweredgesize), a_grid(curraindex), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                aprimez=loweredge+N_a*zind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV(aprimez),[1,1,N_z,N_e]);
                Vhat(curraindex,:,:,N_j)=shiftdim(entireRHS_ii,1);
                Policy(curraindex,:,:,N_j)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1,1);
            end
        end
        aprime_ind=Policy(:,:,:,N_j);
        EV_at_policy=EV(aprime_ind+N_a*zindB);
        Vunderbar(:,:,:,N_j)=Vhat(:,:,:,N_j)+(beta-beta0beta)*EV_at_policy;
    
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            EV_e=EV;
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, n_z, special_n_e, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);
            entireRHS_ii=ReturnMatrix_ii+beta0beta*EV_e;
            [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
            Vhat(level1ii,:,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(level1ii,:,e_c,N_j)=shiftdim(maxindex1,1);
            maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, n_z, special_n_e, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                    aprimez=aprimeindexes+N_a*zind;
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_e(aprimez),[maxgap(ii)+1,1,N_z]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vhat(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    Policy(curraindex,:,e_c,N_j)=shiftdim(maxindex+loweredge-1,1);
                else
                    loweredge=maxindex1(1,ii,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, n_z, special_n_e, reshape(a_grid(loweredge),loweredgesize), a_grid(curraindex), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                    aprimez=loweredge+N_a*zind;
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_e(aprimez),[1,1,N_z]);
                    Vhat(curraindex,:,e_c,N_j)=shiftdim(entireRHS_ii,1);
                    Policy(curraindex,:,e_c,N_j)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
                end
            end
            aprime_ind_e=Policy(:,:,e_c,N_j);
            EV_at_policy_e=EV_e(aprime_ind_e+N_a*zindB);
            Vunderbar(:,:,e_c,N_j)=Vhat(:,:,e_c,N_j)+(beta-beta0beta)*EV_at_policy_e;
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            EV_z=EV(:,:,z_c);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, special_n_z, special_n_e, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,1);
                entireRHS_ii=ReturnMatrix_ii+beta0beta*EV_z;
                [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
                Vhat(level1ii,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                Policy(level1ii,z_c,e_c,N_j)=shiftdim(maxindex1,1);
                maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(1,ii),n_a-maxgap(ii));
                        aprimeindexes=loweredge+(0:1:maxgap(ii))';
                        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, special_n_z, special_n_e, a_grid(aprimeindexes), a_grid(curraindex), z_val, e_val, ReturnFnParamsVec,2);
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*EV_z(aprimeindexes);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vhat(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        Policy(curraindex,z_c,e_c,N_j)=shiftdim(maxindex+loweredge-1,1);
                    else
                        loweredge=maxindex1(1,ii);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, special_n_z, special_n_e, a_grid(loweredge), a_grid(curraindex), z_val, e_val, ReturnFnParamsVec,2);
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*EV_z(loweredge);
                        Vhat(curraindex,z_c,e_c,N_j)=shiftdim(entireRHS_ii,1);
                        Policy(curraindex,z_c,e_c,N_j)=loweredge;
                    end
                end
                aprime_ind_ze=Policy(:,z_c,e_c,N_j);
                EV_at_policy_ze=EV_z(aprime_ind_ze);
                Vunderbar(:,z_c,e_c,N_j)=Vhat(:,z_c,e_c,N_j)+(beta-beta0beta)*EV_at_policy_ze;
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

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    EVsource=Vunderbar(:,:,:,jj+1);
    EV=sum(EVsource.*pi_e_J(1,1,:,jj),3);
    EV=EV.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, n_z, n_e, a_grid, a_grid(level1ii), z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);
        entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
        [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
        Vhat(level1ii,:,:,jj)=shiftdim(Vtempii,1);
        Policy(level1ii,:,:,jj)=shiftdim(maxindex1,1);
        maxgap=max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, n_z, n_e, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                aprimez=aprimeindexes+N_a*zind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV(aprimez),[maxgap(ii)+1,1,N_z,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat(curraindex,:,:,jj)=shiftdim(Vtempii,1);
                Policy(curraindex,:,:,jj)=shiftdim(maxindex+loweredge-1,1);
            else
                loweredge=maxindex1(1,ii,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, n_z, n_e, reshape(a_grid(loweredge),loweredgesize), a_grid(curraindex), z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                aprimez=loweredge+N_a*zind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV(aprimez),[1,1,N_z,N_e]);
                Vhat(curraindex,:,:,jj)=shiftdim(entireRHS_ii,1);
                Policy(curraindex,:,:,jj)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1,1);
            end
        end
        aprime_ind=Policy(:,:,:,jj);
        EV_at_policy=EV(aprime_ind+N_a*zindB);
        Vunderbar(:,:,:,jj)=Vhat(:,:,:,jj)+(beta-beta0beta)*EV_at_policy;

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            EV_e=EV;
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, n_z, special_n_e, a_grid, a_grid(level1ii), z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1);
            entireRHS_ii=ReturnMatrix_ii+beta0beta*EV_e;
            [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
            Vhat(level1ii,:,e_c,jj)=shiftdim(Vtempii,1);
            Policy(level1ii,:,e_c,jj)=shiftdim(maxindex1,1);
            maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, n_z, special_n_e, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
                    aprimez=aprimeindexes+N_a*zind;
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_e(aprimez),[maxgap(ii)+1,1,N_z]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vhat(curraindex,:,e_c,jj)=shiftdim(Vtempii,1);
                    Policy(curraindex,:,e_c,jj)=shiftdim(maxindex+loweredge-1,1);
                else
                    loweredge=maxindex1(1,ii,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, n_z, special_n_e, reshape(a_grid(loweredge),loweredgesize), a_grid(curraindex), z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
                    aprimez=loweredge+N_a*zind;
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_e(aprimez),[1,1,N_z]);
                    Vhat(curraindex,:,e_c,jj)=shiftdim(entireRHS_ii,1);
                    Policy(curraindex,:,e_c,jj)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
                end
            end
            aprime_ind_e=Policy(:,:,e_c,jj);
            EV_at_policy_e=EV_e(aprime_ind_e+N_a*zindB);
            Vunderbar(:,:,e_c,jj)=Vhat(:,:,e_c,jj)+(beta-beta0beta)*EV_at_policy_e;
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            EV_z=EV(:,:,z_c);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, special_n_z, special_n_e, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,1);
                entireRHS_ii=ReturnMatrix_ii+beta0beta*EV_z;
                [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
                Vhat(level1ii,z_c,e_c,jj)=shiftdim(Vtempii,1);
                Policy(level1ii,z_c,e_c,jj)=shiftdim(maxindex1,1);
                maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(1,ii),n_a-maxgap(ii));
                        aprimeindexes=loweredge+(0:1:maxgap(ii))';
                        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, special_n_z, special_n_e, a_grid(aprimeindexes), a_grid(curraindex), z_val, e_val, ReturnFnParamsVec,2);
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*EV_z(aprimeindexes);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vhat(curraindex,z_c,e_c,jj)=shiftdim(Vtempii,1);
                        Policy(curraindex,z_c,e_c,jj)=shiftdim(maxindex+loweredge-1,1);
                    else
                        loweredge=maxindex1(1,ii);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, special_n_z, special_n_e, a_grid(loweredge), a_grid(curraindex), z_val, e_val, ReturnFnParamsVec,2);
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*EV_z(loweredge);
                        Vhat(curraindex,z_c,e_c,jj)=shiftdim(entireRHS_ii,1);
                        Policy(curraindex,z_c,e_c,jj)=loweredge;
                    end
                end
                aprime_ind_ze=Policy(:,z_c,e_c,jj);
                EV_at_policy_ze=EV_z(aprime_ind_ze);
                Vunderbar(:,z_c,e_c,jj)=Vhat(:,z_c,e_c,jj)+(beta-beta0beta)*EV_at_policy_ze;
            end
        end
    end
end

end
