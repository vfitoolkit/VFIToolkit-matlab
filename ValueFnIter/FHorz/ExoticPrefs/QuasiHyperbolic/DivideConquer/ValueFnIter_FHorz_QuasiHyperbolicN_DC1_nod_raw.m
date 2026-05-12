function [Vtilde,Policy,V]=ValueFnIter_FHorz_QuasiHyperbolicN_DC1_nod_raw(n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive quasi-hyperbolic discounting variant of ValueFnIter_FHorz_DC1_nod_raw.
% No d (decision) variables. Uses divide-and-conquer on a (GPU, parallel==2 only).
%
% DiscountFactorParamNames is the standard discount factor beta
% vfoptions.QHadditionaldiscount gives the name of beta_0, the additional discount factor parameter
%
% The 'Naive' quasi-hyperbolic solution takes current actions as if the future agent takes actions
% as if having time-consistent (exponential discounting) preferences.
% V_naive_j = u_t + beta_0*E[V_{j+1}]

N_a=prod(n_a);
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray');  % optimal aprime index

%%
if vfoptions.lowmemory==0
    loweredgesize=[1,1,N_z];
elseif vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
end

zind=shiftdim(gpuArray((0:1:N_z-1)),-1);  % 1-by-N_z

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% j=N_j  (terminal period)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % No discounting at terminal period.
    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        [Vtempii,maxindex1]=max(ReturnMatrix_ii,[],1);
        V(level1ii,:,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,:,N_j)=shiftdim(maxindex1,1);
        maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                Policy(curraindex,:,N_j)=shiftdim(maxindex+loweredge-1,1);
            else
                loweredge=maxindex1(1,ii,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, reshape(a_grid(loweredge),loweredgesize), a_grid(curraindex), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                V(curraindex,:,N_j)=shiftdim(ReturnMatrix_ii,1);
                Policy(curraindex,:,N_j)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
            end
        end
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_z, a_grid, a_grid(level1ii), z_val, ReturnFnParamsVec,1);
            [Vtempii,maxindex1]=max(ReturnMatrix_ii,[],1);
            V(level1ii,z_c,N_j)=shiftdim(Vtempii,1);
            Policy(level1ii,z_c,N_j)=shiftdim(maxindex1,1);
            maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_z, a_grid(aprimeindexes), a_grid(curraindex), z_val, ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    Policy(curraindex,z_c,N_j)=shiftdim(maxindex+loweredge-1,1);
                else
                    loweredge=maxindex1(1,ii);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_z, a_grid(loweredge), a_grid(curraindex), z_val, ReturnFnParamsVec,2);
                    V(curraindex,z_c,N_j)=shiftdim(ReturnMatrix_ii,1);
                    Policy(curraindex,z_c,N_j)=loweredge;
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
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        % --- V search (beta) ---
        entireRHS_ii=ReturnMatrix_ii+beta*EV;
        [Vtempii,maxindex1_V]=max(entireRHS_ii,[],1);
        V(level1ii,:,N_j)=shiftdim(Vtempii,1);
        maxgap_V=max(maxindex1_V(1,2:end,:)-maxindex1_V(1,1:end-1,:),[],3);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap_V(ii)>0
                loweredge=min(maxindex1_V(1,ii,:),n_a-maxgap_V(ii));
                aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                ReturnMatrix_ii_V=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                aprimez=aprimeindexes+N_a*zind;
                entireRHS_ii_V=ReturnMatrix_ii_V+beta*reshape(EV(aprimez),[(maxgap_V(ii)+1),1,N_z]);
                [Vtempii,~]=max(entireRHS_ii_V,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
            else
                loweredge=maxindex1_V(1,ii,:);
                ReturnMatrix_ii_V=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, reshape(a_grid(loweredge),loweredgesize), a_grid(curraindex), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                aprimez=loweredge+N_a*zind;
                entireRHS_ii_V=ReturnMatrix_ii_V+beta*reshape(EV(aprimez),[1,1,N_z]);
                V(curraindex,:,N_j)=shiftdim(entireRHS_ii_V,1);
            end
        end
        % --- Vtilde search (beta0*beta) ---
        entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
        [Vtempii,maxindex1_Vt]=max(entireRHS_ii,[],1);
        Vtilde(level1ii,:,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,:,N_j)=shiftdim(maxindex1_Vt,1);
        maxgap_Vt=max(maxindex1_Vt(1,2:end,:)-maxindex1_Vt(1,1:end-1,:),[],3);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap_Vt(ii)>0
                loweredge=min(maxindex1_Vt(1,ii,:),n_a-maxgap_Vt(ii));
                aprimeindexes=loweredge+(0:1:maxgap_Vt(ii))';
                ReturnMatrix_ii_Vt=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                aprimez=aprimeindexes+N_a*zind;
                entireRHS_ii_Vt=ReturnMatrix_ii_Vt+beta0beta*reshape(EV(aprimez),[(maxgap_Vt(ii)+1),1,N_z]);
                [Vtempii,maxindex]=max(entireRHS_ii_Vt,[],1);
                Vtilde(curraindex,:,N_j)=shiftdim(Vtempii,1);
                Policy(curraindex,:,N_j)=shiftdim(maxindex+loweredge-1,1);
            else
                loweredge=maxindex1_Vt(1,ii,:);
                ReturnMatrix_ii_Vt=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, reshape(a_grid(loweredge),loweredgesize), a_grid(curraindex), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                aprimez=loweredge+N_a*zind;
                entireRHS_ii_Vt=ReturnMatrix_ii_Vt+beta0beta*reshape(EV(aprimez),[1,1,N_z]);
                Vtilde(curraindex,:,N_j)=shiftdim(entireRHS_ii_Vt,1);
                Policy(curraindex,:,N_j)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
            end
        end

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            EV_z=EV(:,:,z_c);

            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_z, a_grid, a_grid(level1ii), z_val, ReturnFnParamsVec,1);

            % --- V search (beta) ---
            entireRHS_ii=ReturnMatrix_ii+beta*EV_z;
            [Vtempii,maxindex1_V]=max(entireRHS_ii,[],1);
            V(level1ii,z_c,N_j)=shiftdim(Vtempii,1);
            maxgap_V=maxindex1_V(1,2:end)-maxindex1_V(1,1:end-1);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1_V(1,ii),n_a-maxgap_V(ii));
                    aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                    ReturnMatrix_ii_V=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_z, a_grid(aprimeindexes), a_grid(curraindex), z_val, ReturnFnParamsVec,2);
                    entireRHS_ii_V=ReturnMatrix_ii_V+beta*EV_z(aprimeindexes);
                    [Vtempii,~]=max(entireRHS_ii_V,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                else
                    loweredge=maxindex1_V(1,ii);
                    ReturnMatrix_ii_V=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_z, a_grid(loweredge), a_grid(curraindex), z_val, ReturnFnParamsVec,2);
                    entireRHS_ii_V=ReturnMatrix_ii_V+beta*EV_z(loweredge);
                    V(curraindex,z_c,N_j)=shiftdim(entireRHS_ii_V,1);
                end
            end
            % --- Vtilde search (beta0*beta) ---
            entireRHS_ii=ReturnMatrix_ii+beta0beta*EV_z;
            [Vtempii,maxindex1_Vt]=max(entireRHS_ii,[],1);
            Vtilde(level1ii,z_c,N_j)=shiftdim(Vtempii,1);
            Policy(level1ii,z_c,N_j)=shiftdim(maxindex1_Vt,1);
            maxgap_Vt=maxindex1_Vt(1,2:end)-maxindex1_Vt(1,1:end-1);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap_Vt(ii)>0
                    loweredge=min(maxindex1_Vt(1,ii),n_a-maxgap_Vt(ii));
                    aprimeindexes=loweredge+(0:1:maxgap_Vt(ii))';
                    ReturnMatrix_ii_Vt=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_z, a_grid(aprimeindexes), a_grid(curraindex), z_val, ReturnFnParamsVec,2);
                    entireRHS_ii_Vt=ReturnMatrix_ii_Vt+beta0beta*EV_z(aprimeindexes);
                    [Vtempii,maxindex]=max(entireRHS_ii_Vt,[],1);
                    Vtilde(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    Policy(curraindex,z_c,N_j)=shiftdim(maxindex+loweredge-1,1);
                else
                    loweredge=maxindex1_Vt(1,ii);
                    ReturnMatrix_ii_Vt=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_z, a_grid(loweredge), a_grid(curraindex), z_val, ReturnFnParamsVec,2);
                    entireRHS_ii_Vt=ReturnMatrix_ii_Vt+beta0beta*EV_z(loweredge);
                    Vtilde(curraindex,z_c,N_j)=shiftdim(entireRHS_ii_Vt,1);
                    Policy(curraindex,z_c,N_j)=loweredge;
                end
            end
        end
    end
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
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
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid, a_grid(level1ii), z_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        % --- V search (beta) ---
        entireRHS_ii=ReturnMatrix_ii+beta*EV;
        [Vtempii,maxindex1_V]=max(entireRHS_ii,[],1);
        V(level1ii,:,jj)=shiftdim(Vtempii,1);
        maxgap_V=max(maxindex1_V(1,2:end,:)-maxindex1_V(1,1:end-1,:),[],3);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap_V(ii)>0
                loweredge=min(maxindex1_V(1,ii,:),n_a-maxgap_V(ii));
                aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                ReturnMatrix_ii_V=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                aprimez=aprimeindexes+N_a*zind;
                entireRHS_ii_V=ReturnMatrix_ii_V+beta*reshape(EV(aprimez),[(maxgap_V(ii)+1),1,N_z]);
                [Vtempii,~]=max(entireRHS_ii_V,[],1);
                V(curraindex,:,jj)=shiftdim(Vtempii,1);
            else
                loweredge=maxindex1_V(1,ii,:);
                ReturnMatrix_ii_V=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, reshape(a_grid(loweredge),loweredgesize), a_grid(curraindex), z_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                aprimez=loweredge+N_a*zind;
                entireRHS_ii_V=ReturnMatrix_ii_V+beta*reshape(EV(aprimez),[1,1,N_z]);
                V(curraindex,:,jj)=shiftdim(entireRHS_ii_V,1);
            end
        end
        % --- Vtilde search (beta0*beta) ---
        entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
        [Vtempii,maxindex1_Vt]=max(entireRHS_ii,[],1);
        Vtilde(level1ii,:,jj)=shiftdim(Vtempii,1);
        Policy(level1ii,:,jj)=shiftdim(maxindex1_Vt,1);
        maxgap_Vt=max(maxindex1_Vt(1,2:end,:)-maxindex1_Vt(1,1:end-1,:),[],3);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap_Vt(ii)>0
                loweredge=min(maxindex1_Vt(1,ii,:),n_a-maxgap_Vt(ii));
                aprimeindexes=loweredge+(0:1:maxgap_Vt(ii))';
                ReturnMatrix_ii_Vt=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                aprimez=aprimeindexes+N_a*zind;
                entireRHS_ii_Vt=ReturnMatrix_ii_Vt+beta0beta*reshape(EV(aprimez),[(maxgap_Vt(ii)+1),1,N_z]);
                [Vtempii,maxindex]=max(entireRHS_ii_Vt,[],1);
                Vtilde(curraindex,:,jj)=shiftdim(Vtempii,1);
                Policy(curraindex,:,jj)=shiftdim(maxindex+loweredge-1,1);
            else
                loweredge=maxindex1_Vt(1,ii,:);
                ReturnMatrix_ii_Vt=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, reshape(a_grid(loweredge),loweredgesize), a_grid(curraindex), z_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                aprimez=loweredge+N_a*zind;
                entireRHS_ii_Vt=ReturnMatrix_ii_Vt+beta0beta*reshape(EV(aprimez),[1,1,N_z]);
                Vtilde(curraindex,:,jj)=shiftdim(entireRHS_ii_Vt,1);
                Policy(curraindex,:,jj)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
            end
        end

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            EV_z=EV(:,:,z_c);

            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_z, a_grid, a_grid(level1ii), z_val, ReturnFnParamsVec,1);

            % --- V search (beta) ---
            entireRHS_ii=ReturnMatrix_ii+beta*EV_z;
            [Vtempii,maxindex1_V]=max(entireRHS_ii,[],1);
            V(level1ii,z_c,jj)=shiftdim(Vtempii,1);
            maxgap_V=maxindex1_V(1,2:end)-maxindex1_V(1,1:end-1);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1_V(1,ii),n_a-maxgap_V(ii));
                    aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                    ReturnMatrix_ii_V=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_z, a_grid(aprimeindexes), a_grid(curraindex), z_val, ReturnFnParamsVec,2);
                    entireRHS_ii_V=ReturnMatrix_ii_V+beta*EV_z(aprimeindexes);
                    [Vtempii,~]=max(entireRHS_ii_V,[],1);
                    V(curraindex,z_c,jj)=shiftdim(Vtempii,1);
                else
                    loweredge=maxindex1_V(1,ii);
                    ReturnMatrix_ii_V=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_z, a_grid(loweredge), a_grid(curraindex), z_val, ReturnFnParamsVec,2);
                    entireRHS_ii_V=ReturnMatrix_ii_V+beta*EV_z(loweredge);
                    V(curraindex,z_c,jj)=shiftdim(entireRHS_ii_V,1);
                end
            end
            % --- Vtilde search (beta0*beta) ---
            entireRHS_ii=ReturnMatrix_ii+beta0beta*EV_z;
            [Vtempii,maxindex1_Vt]=max(entireRHS_ii,[],1);
            Vtilde(level1ii,z_c,jj)=shiftdim(Vtempii,1);
            Policy(level1ii,z_c,jj)=shiftdim(maxindex1_Vt,1);
            maxgap_Vt=maxindex1_Vt(1,2:end)-maxindex1_Vt(1,1:end-1);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap_Vt(ii)>0
                    loweredge=min(maxindex1_Vt(1,ii),n_a-maxgap_Vt(ii));
                    aprimeindexes=loweredge+(0:1:maxgap_Vt(ii))';
                    ReturnMatrix_ii_Vt=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_z, a_grid(aprimeindexes), a_grid(curraindex), z_val, ReturnFnParamsVec,2);
                    entireRHS_ii_Vt=ReturnMatrix_ii_Vt+beta0beta*EV_z(aprimeindexes);
                    [Vtempii,maxindex]=max(entireRHS_ii_Vt,[],1);
                    Vtilde(curraindex,z_c,jj)=shiftdim(Vtempii,1);
                    Policy(curraindex,z_c,jj)=shiftdim(maxindex+loweredge-1,1);
                else
                    loweredge=maxindex1_Vt(1,ii);
                    ReturnMatrix_ii_Vt=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_z, a_grid(loweredge), a_grid(curraindex), z_val, ReturnFnParamsVec,2);
                    entireRHS_ii_Vt=ReturnMatrix_ii_Vt+beta0beta*EV_z(loweredge);
                    Vtilde(curraindex,z_c,jj)=shiftdim(entireRHS_ii_Vt,1);
                    Policy(curraindex,z_c,jj)=loweredge;
                end
            end
        end
    end
end

end
