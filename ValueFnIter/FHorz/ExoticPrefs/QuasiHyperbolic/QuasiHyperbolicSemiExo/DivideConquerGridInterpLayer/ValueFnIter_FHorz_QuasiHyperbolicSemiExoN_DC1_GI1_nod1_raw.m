function [V1, Policy, Valt]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_DC1_GI1_nod1_raw(n_d2,n_a,n_z,n_semiz,N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive QH + SemiExo + DC + GI raw, no d1, with z, no e. Output: (V1=Vtilde, Policy3, Valt=V).

n_bothz=[n_semiz,n_z];

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);

V=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
Vtilde=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
Policy=zeros(3,N_a,N_semiz*N_z,N_j,'gpuArray'); % first dim: d2, midpoint, aprimeL2ind
PolicyL2flag=2*ones(1,N_a,N_semiz*N_z,N_j,'gpuArray');

%%
special_n_d2=ones(1,length(n_d2));

aind=gpuArray(0:1:N_a-1);
bothzind=shiftdim(gpuArray(0:1:N_bothz-1),-1);

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

% Preallocate per-d2 slabs
Valt_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
V_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
midpoint_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
PolicyL2flag_ford2_jj=2*ones(N_a,N_semiz*N_z,N_d2,'gpuArray');
midpoints_jj=zeros(1,N_a,N_semiz*N_z,'gpuArray');

level1ii=round(linspace(1,n_a,vfoptions.level1n));

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);


%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    midpoints_Nj=zeros(N_d2,1,N_a,N_semiz*N_z,'gpuArray');

    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d2, n_bothz, d2_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
    [~,maxindex1]=max(ReturnMatrix_ii,[],2);
    midpoints_Nj(:,1,level1ii,:)=maxindex1;
    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d2, n_bothz, d2_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,3);
            [~,maxindex]=max(ReturnMatrix_ii,[],2);
            midpoints_Nj(:,1,curraindex,:)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(:,1,ii,:);
            midpoints_Nj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
        end
    end
    midpoints_Nj=max(min(midpoints_Nj,n_a-1),2);
    aprimeindexes=(midpoints_Nj+(midpoints_Nj-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn,n_d2,n_bothz,d2_gridvals,aprime_grid(aprimeindexes),a_grid,bothz_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
    V(:,:,N_j)=shiftdim(Vtempii,1);
    d_ind=rem(maxindexL2-1,N_d2)+1;
    allind=d_ind+N_d2*aind+N_d2*N_a*bothzind;
    Policy(1,:,:,N_j)=d_ind;
    Policy(2,:,:,N_j)=shiftdim(squeeze(midpoints_Nj(allind)),-1);
    Policy(3,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d2),-1);

    L2offset = ceil(maxindexL2/N_d2);
    linidx_lower = d_ind                   + N_d2*n2long*aind + N_d2*n2long*N_a*bothzind;
    linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*bothzind;
    isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
    isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
    PolicyL2flag(1,:,:,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

    Vtilde(:,:,N_j)=V(:,:,N_j);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_z]);

    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j));

        EV_d2=EV.*shiftdim(pi_bothz',-1);
        EV_d2(isnan(EV_d2))=0;
        EV_d2=sum(EV_d2,2);

        EVinterp_d2=interp1(a_grid,EV_d2,aprime_grid);

        ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,4);

        %% V (beta)
        entireRHS_d2ii=ReturnMatrix_d2ii+beta*EV_d2;
        [~,maxindex1]=max(entireRHS_d2ii,[],1);
        midpoints_jj(1,level1ii,:)=maxindex1;
        maxgap_V=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap_V(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap_V(ii));
                aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
                aprimez=aprimeindexes+N_a*bothzind;
                entireRHS_ii=ReturnMatrix_ii+beta*reshape(EV_d2(aprimez),[(maxgap_V(ii)+1),1,N_bothz]);
                [~,maxindex]=max(entireRHS_ii,[],1);
                midpoints_jj(1,curraindex,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(1,ii,:);
                midpoints_jj(1,curraindex,:)=repelem(loweredge,1,length(curraindex),1);
            end
        end
        midpoints_jj=max(min(midpoints_jj,n_a-1),2);
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
        aprimez=aprimeindexes+n2aprime*bothzind;
        entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp_d2(aprimez),[n2long,N_a,N_bothz]);
        [Vtemp,~]=max(entireRHS_L2,[],1);
        Valt_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);

        %% Vtilde (beta0*beta)
        entireRHS_d2ii=ReturnMatrix_d2ii+beta0beta*EV_d2;
        [~,maxindex1]=max(entireRHS_d2ii,[],1);
        midpoints_jj(1,level1ii,:)=maxindex1;
        maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
                aprimez=aprimeindexes+N_a*bothzind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[(maxgap(ii)+1),1,N_bothz]);
                [~,maxindex]=max(entireRHS_ii,[],1);
                midpoints_jj(1,curraindex,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(1,ii,:);
                midpoints_jj(1,curraindex,:)=repelem(loweredge,1,length(curraindex),1);
            end
        end
        midpoints_jj=max(min(midpoints_jj,n_a-1),2);
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
        aprimez=aprimeindexes+n2aprime*bothzind;
        entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp_d2(aprimez),[n2long,N_a,N_bothz]);
        [Vtemp,maxindex]=max(entireRHS_L2,[],1);

        V_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
        Policy_ford2_jj(:,:,d2_c)=shiftdim(maxindex,1);

        midpoint_ford2_jj(:,:,d2_c)=squeeze(midpoints_jj);

        isInfLower    = (ReturnMatrix_L2(1,     :,:) == -Inf);
        isInfUpper    = (ReturnMatrix_L2(n2long,:,:) == -Inf);
        inLowerStrict = (maxindex >= 2)         & (maxindex <= n2short+1);
        inUpperStrict = (maxindex >= n2short+3) & (maxindex <= n2long-1);
        PolicyL2flag_ford2_jj(:,:,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
    end

    [V_jj,maxindex]=max(V_ford2_jj,[],3);
    Vtilde(:,:,N_j)=V_jj;
    Policy(1,:,:,N_j)=shiftdim(maxindex,-1); % d2
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]);
    V(:,:,N_j)=max(Valt_ford2_jj,[],3); % max over d2 of V-at-V-optimal (Naive QH: V uses its own argmax, NOT the Vtilde argmax)
    aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    Policy(2,:,:,N_j)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    Policy(3,:,:,N_j)=aprimeL2_ind;
    PolicyL2flag(1,:,:,N_j)=reshape(PolicyL2flag_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
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

    EV=reshape(V(:,:,jj+1),[N_a,N_semiz,N_z]);

    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d2_c,jj));

        EV_d2=EV.*shiftdim(pi_bothz',-1);
        EV_d2(isnan(EV_d2))=0;
        EV_d2=sum(EV_d2,2);

        EVinterp_d2=interp1(a_grid,EV_d2,aprime_grid);

        ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,4);

        %% V (beta)
        entireRHS_d2ii_V=ReturnMatrix_d2ii+beta*EV_d2;
        [~,maxindex1]=max(entireRHS_d2ii_V,[],1);
        midpoints_jj(1,level1ii,:)=maxindex1;
        maxgap_V=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap_V(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap_V(ii));
                aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,5);
                aprimez=aprimeindexes+N_a*bothzind;
                entireRHS_ii=ReturnMatrix_ii+beta*reshape(EV_d2(aprimez),[(maxgap_V(ii)+1),1,N_bothz]);
                [~,maxindex]=max(entireRHS_ii,[],1);
                midpoints_jj(1,curraindex,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(1,ii,:);
                midpoints_jj(1,curraindex,:)=repelem(loweredge,1,length(curraindex),1);
            end
        end
        midpoints_jj=max(min(midpoints_jj,n_a-1),2);
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,5);
        aprimez=aprimeindexes+n2aprime*bothzind;
        entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp_d2(aprimez),[n2long,N_a,N_bothz]);
        [Vtemp,~]=max(entireRHS_L2,[],1);
        Valt_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);

        %% Vtilde (beta0*beta)
        entireRHS_d2ii_T=ReturnMatrix_d2ii+beta0beta*EV_d2;
        [~,maxindex1]=max(entireRHS_d2ii_T,[],1);
        midpoints_jj(1,level1ii,:)=maxindex1;
        maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,5);
                aprimez=aprimeindexes+N_a*bothzind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[(maxgap(ii)+1),1,N_bothz]);
                [~,maxindex]=max(entireRHS_ii,[],1);
                midpoints_jj(1,curraindex,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(1,ii,:);
                midpoints_jj(1,curraindex,:)=repelem(loweredge,1,length(curraindex),1);
            end
        end
        midpoints_jj=max(min(midpoints_jj,n_a-1),2);
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,5);
        aprimez=aprimeindexes+n2aprime*bothzind;
        entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp_d2(aprimez),[n2long,N_a,N_bothz]);
        [Vtemp,maxindex]=max(entireRHS_L2,[],1);

        V_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
        Policy_ford2_jj(:,:,d2_c)=shiftdim(maxindex,1);

        midpoint_ford2_jj(:,:,d2_c)=squeeze(midpoints_jj);

        isInfLower    = (ReturnMatrix_L2(1,     :,:) == -Inf);
        isInfUpper    = (ReturnMatrix_L2(n2long,:,:) == -Inf);
        inLowerStrict = (maxindex >= 2)         & (maxindex <= n2short+1);
        inUpperStrict = (maxindex >= n2short+3) & (maxindex <= n2long-1);
        PolicyL2flag_ford2_jj(:,:,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
    end

    [V_jj,maxindex]=max(V_ford2_jj,[],3);
    Vtilde(:,:,jj)=V_jj;
    Policy(1,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]);
    V(:,:,jj)=max(Valt_ford2_jj,[],3); % max over d2 of V-at-V-optimal (Naive QH: V uses its own argmax, NOT the Vtilde argmax)
    aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    Policy(2,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    Policy(3,:,:,jj)=aprimeL2_ind;
    PolicyL2flag(1,:,:,jj)=reshape(PolicyL2flag_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
end

%% Post-process Policy
adjust=(Policy(3,:,:,:)<1+n2short+1);
Policy(2,:,:,:)=Policy(2,:,:,:)-adjust;
Policy(3,:,:,:)=adjust.*Policy(3,:,:,:)+(1-adjust).*(Policy(3,:,:,:)-n2short-1);

Policy=[Policy;PolicyL2flag];

V1=Vtilde;
Valt=V;

end
