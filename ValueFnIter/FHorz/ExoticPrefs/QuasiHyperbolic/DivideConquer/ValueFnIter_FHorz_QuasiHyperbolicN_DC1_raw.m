function [Vtilde,Policy2,V]=ValueFnIter_FHorz_QuasiHyperbolicN_DC1_raw(n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive quasi-hyperbolic discounting variant of ValueFnIter_FHorz_DC1_raw.
% Has d (decision) variables. Uses divide-and-conquer on a (GPU, parallel==2 only).
%
% DiscountFactorParamNames is the standard discount factor beta
% vfoptions.QHadditionaldiscount gives the name of beta_0, the additional discount factor parameter
%
% The 'Naive' quasi-hyperbolic solution takes current actions as if the future agent takes actions
% as if having time-consistent (exponential discounting) preferences.
% V_naive_j = u_t + beta_0*E[V_{j+1}]

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray');  % combined Kron (d,aprime)

%%
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

if vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
else
    zind=shiftdim((0:1:N_z-1),-1);  % 1-by-N_z (for Policy indexing)
end

zBind=shiftdim(gpuArray(0:1:N_z-1),-2);  % 1-by-1-by-N_z (for EV indexing)

%% j=N_j  (terminal period)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % No discounting at terminal period.
    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_z, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a,vfoptions.level1n,N_z]),[],1);
        V(level1ii,:,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,:,N_j)=shiftdim(maxindex2,1);
        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_z, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*zind;
                Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_z, d_gridvals, a_grid(loweredge), a_grid(curraindex), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*zind;
                Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            end
        end
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_z, d_gridvals, a_grid, a_grid(level1ii), z_val, ReturnFnParamsVec,1);
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a,vfoptions.level1n]),[],1);
            V(level1ii,z_c,N_j)=shiftdim(Vtempii,1);
            Policy(level1ii,z_c,N_j)=shiftdim(maxindex2,1);
            maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_z, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_val, ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    allind=dind;
                    Policy(curraindex,z_c,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)'-1),1);
                else
                    loweredge=maxindex1(:,1,ii);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_z, d_gridvals, a_grid(loweredge), a_grid(curraindex), z_val, ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    allind=dind;
                    Policy(curraindex,z_c,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)'-1),1);
                end
            end
        end
    end

    Vtilde=V;

else
    % Using V_Jplus1
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=reshape(vfoptions.V_Jplus1,[N_a,N_z]);
    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);   % N_a-by-1-by-N_z

    Vtilde=zeros(N_a,N_z,N_j,'gpuArray');

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_z, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        % --- V search (beta) ---
        entireRHS_ii=ReturnMatrix_ii+beta*shiftdim(EV,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,~]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n,N_z]),[],1);
        V(level1ii,:,N_j)=shiftdim(Vtempii,1);
        maxgap_V=max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap_V(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap_V(:,1,ii,:));
                aprimeindexes=loweredge+(0:1:maxgap_V(ii));
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_z, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                aprimez=aprimeindexes+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii_dc+beta*reshape(EV(aprimez),[N_d*(maxgap_V(ii)+1),1,N_z]);
                [Vtempii,~]=max(entireRHS_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
            else
                loweredge=maxindex1(:,1,ii,:);
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_z, d_gridvals, a_grid(loweredge), a_grid(curraindex), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                aprimez=loweredge+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii_dc+beta*reshape(EV(aprimez),[N_d*1,1,N_z]);
                [Vtempii,~]=max(entireRHS_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
            end
        end
        % --- Vtilde search (beta0*beta) ---
        entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n,N_z]),[],1);
        Vtilde(level1ii,:,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,:,N_j)=shiftdim(maxindex2,1);
        maxgap=max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(:,1,ii,:));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_z, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                aprimez=aprimeindexes+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*reshape(EV(aprimez),[N_d*(maxgap(ii)+1),1,N_z]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vtilde(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*zind;
                Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:);
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_z, d_gridvals, a_grid(loweredge), a_grid(curraindex), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                aprimez=loweredge+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*reshape(EV(aprimez),[N_d*1,1,N_z]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vtilde(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*zind;
                Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            end
        end

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            EV_z=EV(:,:,z_c);

            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_z, d_gridvals, a_grid, a_grid(level1ii), z_val, ReturnFnParamsVec,1);

            % --- V search (beta) ---
            entireRHS_ii=ReturnMatrix_ii+beta*shiftdim(EV_z,-1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            [Vtempii,~]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);
            V(level1ii,z_c,N_j)=shiftdim(Vtempii,1);
            maxgap_V=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(:,1,ii),n_a-maxgap_V(ii));
                    aprimeindexes=loweredge+(0:1:maxgap_V(ii));
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_z, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii_dc+beta*reshape(EV_z(aprimeindexes),[N_d*(maxgap_V(ii)+1),1]);
                    [Vtempii,~]=max(entireRHS_ii,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                else
                    loweredge=maxindex1(:,1,ii);
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_z, d_gridvals, a_grid(loweredge), a_grid(curraindex), z_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii_dc+beta*reshape(EV_z(loweredge),[N_d*1,1]);
                    [Vtempii,~]=max(entireRHS_ii,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                end
            end
            % --- Vtilde search (beta0*beta) ---
            entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV_z,-1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);
            Vtilde(level1ii,z_c,N_j)=shiftdim(Vtempii,1);
            Policy(level1ii,z_c,N_j)=shiftdim(maxindex2,1);
            maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_z, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*reshape(EV_z(aprimeindexes),[N_d*(maxgap(ii)+1),1]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vtilde(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    allind=dind;
                    Policy(curraindex,z_c,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)'-1),1);
                else
                    loweredge=maxindex1(:,1,ii);
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_z, d_gridvals, a_grid(loweredge), a_grid(curraindex), z_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*reshape(EV_z(loweredge),[N_d*1,1]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vtilde(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    allind=dind;
                    Policy(curraindex,z_c,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)'-1),1);
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

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    EVsource=V(:,:,jj+1);
    EV=EVsource.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);   % N_a-by-1-by-N_z

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_z, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        % --- V search (beta) ---
        entireRHS_ii=ReturnMatrix_ii+beta*shiftdim(EV,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,~]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n,N_z]),[],1);
        V(level1ii,:,jj)=shiftdim(Vtempii,1);
        maxgap_V=max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap_V(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap_V(:,1,ii,:));
                aprimeindexes=loweredge+(0:1:maxgap_V(ii));
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_z, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                aprimez=aprimeindexes+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii_dc+beta*reshape(EV(aprimez),[N_d*(maxgap_V(ii)+1),1,N_z]);
                [Vtempii,~]=max(entireRHS_ii,[],1);
                V(curraindex,:,jj)=shiftdim(Vtempii,1);
            else
                loweredge=maxindex1(:,1,ii,:);
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_z, d_gridvals, a_grid(loweredge), a_grid(curraindex), z_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                aprimez=loweredge+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii_dc+beta*reshape(EV(aprimez),[N_d*1,1,N_z]);
                [Vtempii,~]=max(entireRHS_ii,[],1);
                V(curraindex,:,jj)=shiftdim(Vtempii,1);
            end
        end
        % --- Vtilde search (beta0*beta) ---
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
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_z, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                aprimez=aprimeindexes+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*reshape(EV(aprimez),[N_d*(maxgap(ii)+1),1,N_z]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vtilde(curraindex,:,jj)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*zind;
                Policy(curraindex,:,jj)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:);
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_z, d_gridvals, a_grid(loweredge), a_grid(curraindex), z_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
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

            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_z, d_gridvals, a_grid, a_grid(level1ii), z_val, ReturnFnParamsVec,1);

            % --- V search (beta) ---
            entireRHS_ii=ReturnMatrix_ii+beta*shiftdim(EV_z,-1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            [Vtempii,~]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);
            V(level1ii,z_c,jj)=shiftdim(Vtempii,1);
            maxgap_V=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(:,1,ii),n_a-maxgap_V(ii));
                    aprimeindexes=loweredge+(0:1:maxgap_V(ii));
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_z, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii_dc+beta*reshape(EV_z(aprimeindexes),[N_d*(maxgap_V(ii)+1),1]);
                    [Vtempii,~]=max(entireRHS_ii,[],1);
                    V(curraindex,z_c,jj)=shiftdim(Vtempii,1);
                else
                    loweredge=maxindex1(:,1,ii);
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_z, d_gridvals, a_grid(loweredge), a_grid(curraindex), z_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii_dc+beta*reshape(EV_z(loweredge),[N_d*1,1]);
                    [Vtempii,~]=max(entireRHS_ii,[],1);
                    V(curraindex,z_c,jj)=shiftdim(Vtempii,1);
                end
            end
            % --- Vtilde search (beta0*beta) ---
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
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_z, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*reshape(EV_z(aprimeindexes),[N_d*(maxgap(ii)+1),1]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vtilde(curraindex,z_c,jj)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    allind=dind;
                    Policy(curraindex,z_c,jj)=shiftdim(maxindex+N_d*(loweredge(allind)'-1),1);
                else
                    loweredge=maxindex1(:,1,ii);
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_z, d_gridvals, a_grid(loweredge), a_grid(curraindex), z_val, ReturnFnParamsVec,2);
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

%%
Policy2=zeros(2,N_a,N_z,N_j,'gpuArray');
Policy2(1,:,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:,:)=shiftdim(ceil(Policy/N_d),-1);

end
