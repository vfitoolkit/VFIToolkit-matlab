function [Vtilde, Policy, Valt, Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_DC1_GI1_nod1_e_raw(n_d2,n_a,n_z,n_semiz, n_e,N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J,pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive QH + SemiExo + DC + GI raw, no d1, with z, with e. Output: (Vtilde=Vtilde, Policy, Valt=Valt).

n_bothz=[n_semiz,n_z];

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

Valt=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
Vtilde=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
Policy=zeros(3,N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
Policyalt=zeros(3,N_a,N_semiz*N_z,N_e,N_j,'gpuArray'); % exponential discounter optimal [d2; midpoint; aprimeL2ind]
PolicyL2flagalt=2*ones(1,N_a,N_semiz*N_z,N_e,N_j,'gpuArray');

%%
special_n_d2=ones(1,length(n_d2));

aind=gpuArray(0:1:N_a-1);
bothzind=shiftdim(gpuArray(0:1:N_bothz-1),-1);
eind=shiftdim(gpuArray(0:1:N_e-1),-2);

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

Valt_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
V_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
midpoint_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
PolicyL2flag_ford2_jj=2*ones(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
Policy_V_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
midpointV_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
PolicyL2flagV_ford2_jj=2*ones(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
midpoints_jj=zeros(1,N_a,N_semiz*N_z,N_e,'gpuArray');

pi_e_J=shiftdim(pi_e_J,-2);

level1ii=round(linspace(1,n_a,vfoptions.level1n));

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory==2
    error('vfoptions.lowmemory=2 not supported with semi-exogenous states');
end

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        midpoints_Nj=zeros(N_d2,1,N_a,N_semiz*N_z,N_e,'gpuArray');

        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d2, n_bothz, n_e, d2_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        midpoints_Nj(:,1,level1ii,:,:)=maxindex1;
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d2, n_bothz, n_e, d2_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3);
                [~,maxindex]=max(ReturnMatrix_ii,[],2);
                midpoints_Nj(:,1,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                midpoints_Nj(:,1,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1,1);
            end
        end
        midpoints_Nj=max(min(midpoints_Nj,n_a-1),2);
        aprimeindexes=(midpoints_Nj+(midpoints_Nj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d2, n_bothz, n_e, d2_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        Valt(:,:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d2)+1;
        allind=d_ind+N_d2*aind+N_d2*N_a*bothzind+N_d2*N_a*N_bothz*eind;
        Policy(1,:,:,:,N_j)=d_ind;
        Policy(2,:,:,:,N_j)=shiftdim(squeeze(midpoints_Nj(allind)),-1);
        Policy(3,:,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d2),-1);

        L2offset = ceil(maxindexL2/N_d2);
        linidx_lower = d_ind                   + N_d2*n2long*aind + N_d2*n2long*N_a*bothzind + N_d2*n2long*N_a*N_bothz*eind;
        linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*bothzind + N_d2*n2long*N_a*N_bothz*eind;
        isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag(1,:,:,:,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
    elseif vfoptions.lowmemory==1
        midpoints_Nj=zeros(N_d2,1,N_a,N_semiz*N_z,'gpuArray');
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d2, n_bothz, special_n_e, d2_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);
            midpoints_Nj(:,1,level1ii,:)=maxindex1;
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d2, n_bothz, special_n_e, d2_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3);
                    [~,maxindex]=max(ReturnMatrix_ii,[],2);
                    midpoints_Nj(:,1,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    midpoints_Nj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
                end
            end
            midpoints_Nj=max(min(midpoints_Nj,n_a-1),2);
            aprimeindexes=(midpoints_Nj+(midpoints_Nj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d2, n_bothz, special_n_e, d2_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            Valt(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind+N_d2*N_a*bothzind;
            Policy(1,:,:,e_c,N_j)=d_ind;
            Policy(2,:,:,e_c,N_j)=shiftdim(squeeze(midpoints_Nj(allind)),-1);
            Policy(3,:,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d2),-1);

            L2offset = ceil(maxindexL2/N_d2);
            linidx_lower = d_ind                   + N_d2*n2long*aind + N_d2*n2long*N_a*bothzind;
            linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*bothzind;
            isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            PolicyL2flag(1,:,:,e_c,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
        end
    end

    Vtilde(:,:,:,N_j)=Valt(:,:,:,N_j);
    % terminal: QH and exponential discounter coincide
    Policyalt(:,:,:,:,N_j)=Policy(:,:,:,:,N_j);
    PolicyL2flagalt(1,:,:,:,N_j)=PolicyL2flag(1,:,:,:,N_j);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=sum(reshape(vfoptions.V_Jplus1,[N_a,N_semiz*N_z,N_e]).*pi_e_J(1,1,:,N_j),3);
    EV=reshape(EV,[N_a,N_semiz,N_z]);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j));

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            EVinterp_d2=interp1(a_grid,EV_d2,aprime_grid);

            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,4);

            %% Valt (beta)
            entireRHS_d2ii_V=ReturnMatrix_d2ii+beta*EV_d2;
            [~,maxindex1]=max(entireRHS_d2ii_V,[],1);
            midpoints_jj(1,level1ii,:,:)=maxindex1;
            maxgap_V=squeeze(max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap_V(ii));
                    aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
                    aprimez=aprimeindexes+N_a*bothzind;
                    entireRHS_ii=ReturnMatrix_ii+beta*reshape(EV_d2(aprimez),[(maxgap_V(ii)+1),1,N_bothz,N_e]);
                    [~,maxindex]=max(entireRHS_ii,[],1);
                    midpoints_jj(1,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(1,ii,:,:);
                    midpoints_jj(1,curraindex,:,:)=repelem(loweredge,1,length(curraindex),1);
                end
            end
            midpoints_jj=max(min(midpoints_jj,n_a-1),2);
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
            ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
            aprimez=aprimeindexes+n2aprime*bothzind;
            entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp_d2(aprimez),[n2long,N_a,N_bothz,N_e]);
            [Vtemp,maxindexL2alt]=max(entireRHS_L2,[],1);
            Valt_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_V_ford2_jj(:,:,:,d2_c)=shiftdim(maxindexL2alt,1);
            midpointV_ford2_jj(:,:,:,d2_c)=squeeze(midpoints_jj);
            isInfLoweralt    = (ReturnMatrix_L2(1,     :,:,:) == -Inf);
            isInfUpperalt    = (ReturnMatrix_L2(n2long,:,:,:) == -Inf);
            inLowerStrictalt = (maxindexL2alt >= 2)         & (maxindexL2alt <= n2short+1);
            inUpperStrictalt = (maxindexL2alt >= n2short+3) & (maxindexL2alt <= n2long-1);
            PolicyL2flagV_ford2_jj(:,:,:,d2_c) = squeeze(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt));

            %% Vtilde (beta0*beta)
            entireRHS_d2ii_T=ReturnMatrix_d2ii+beta0beta*EV_d2;
            [~,maxindex1]=max(entireRHS_d2ii_T,[],1);
            midpoints_jj(1,level1ii,:,:)=maxindex1;
            maxgap=squeeze(max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
                    aprimez=aprimeindexes+N_a*bothzind;
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[(maxgap(ii)+1),1,N_bothz,N_e]);
                    [~,maxindex]=max(entireRHS_ii,[],1);
                    midpoints_jj(1,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(1,ii,:,:);
                    midpoints_jj(1,curraindex,:,:)=repelem(loweredge,1,length(curraindex),1);
                end
            end
            midpoints_jj=max(min(midpoints_jj,n_a-1),2);
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
            ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
            aprimez=aprimeindexes+n2aprime*bothzind;
            entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp_d2(aprimez),[n2long,N_a,N_bothz,N_e]);
            [Vtemp,maxindex]=max(entireRHS_L2,[],1);

            V_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,:,d2_c)=shiftdim(maxindex,1);

            midpoint_ford2_jj(:,:,:,d2_c)=squeeze(midpoints_jj);

            isInfLower    = (ReturnMatrix_L2(1,     :,:,:) == -Inf);
            isInfUpper    = (ReturnMatrix_L2(n2long,:,:,:) == -Inf);
            inLowerStrict = (maxindex >= 2)         & (maxindex <= n2short+1);
            inUpperStrict = (maxindex >= n2short+3) & (maxindex <= n2long-1);
            PolicyL2flag_ford2_jj(:,:,:,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
        end
    elseif vfoptions.lowmemory==1
        midpoints_jj_e=zeros(1,N_a,N_semiz*N_z,'gpuArray');
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j));

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            EVinterp_d2=interp1(a_grid,EV_d2,aprime_grid);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,4);

                %% Valt (beta)
                entireRHS_d2ii_V=ReturnMatrix_d2ii+beta*EV_d2;
                [~,maxindex1]=max(entireRHS_d2ii_V,[],1);
                midpoints_jj_e(1,level1ii,:)=maxindex1;
                maxgap_V=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap_V(ii)>0
                        loweredge=min(maxindex1(1,ii,:),n_a-maxgap_V(ii));
                        aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,5);
                        aprimez=aprimeindexes+N_a*bothzind;
                        entireRHS_ii=ReturnMatrix_ii+beta*reshape(EV_d2(aprimez),[(maxgap_V(ii)+1),1,N_bothz]);
                        [~,maxindex]=max(entireRHS_ii,[],1);
                        midpoints_jj_e(1,curraindex,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(1,ii,:);
                        midpoints_jj_e(1,curraindex,:)=repelem(loweredge,1,length(curraindex),1);
                    end
                end
                midpoints_jj_e=max(min(midpoints_jj_e,n_a-1),2);
                aprimeindexes=(midpoints_jj_e+(midpoints_jj_e-1)*n2short)+(-n2short-1:1:1+n2short)';
                ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,5);
                aprimez=aprimeindexes+n2aprime*bothzind;
                entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp_d2(aprimez),[n2long,N_a,N_bothz]);
                [Vtemp,maxindexL2alt]=max(entireRHS_L2,[],1);
                Valt_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_V_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindexL2alt,1);
                midpointV_ford2_jj(:,:,e_c,d2_c)=squeeze(midpoints_jj_e);
                isInfLoweralt    = (ReturnMatrix_L2(1,     :,:) == -Inf);
                isInfUpperalt    = (ReturnMatrix_L2(n2long,:,:) == -Inf);
                inLowerStrictalt = (maxindexL2alt >= 2)         & (maxindexL2alt <= n2short+1);
                inUpperStrictalt = (maxindexL2alt >= n2short+3) & (maxindexL2alt <= n2long-1);
                PolicyL2flagV_ford2_jj(:,:,e_c,d2_c) = squeeze(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt));

                %% Vtilde (beta0*beta)
                entireRHS_d2ii_T=ReturnMatrix_d2ii+beta0beta*EV_d2;
                [~,maxindex1]=max(entireRHS_d2ii_T,[],1);
                midpoints_jj_e(1,level1ii,:)=maxindex1;
                maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                        aprimeindexes=loweredge+(0:1:maxgap(ii))';
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,5);
                        aprimez=aprimeindexes+N_a*bothzind;
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[(maxgap(ii)+1),1,N_bothz]);
                        [~,maxindex]=max(entireRHS_ii,[],1);
                        midpoints_jj_e(1,curraindex,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(1,ii,:);
                        midpoints_jj_e(1,curraindex,:)=repelem(loweredge,1,length(curraindex),1);
                    end
                end
                midpoints_jj_e=max(min(midpoints_jj_e,n_a-1),2);
                aprimeindexes=(midpoints_jj_e+(midpoints_jj_e-1)*n2short)+(-n2short-1:1:1+n2short)';
                ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,5);
                aprimez=aprimeindexes+n2aprime*bothzind;
                entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp_d2(aprimez),[n2long,N_a,N_bothz]);
                [Vtemp,maxindex]=max(entireRHS_L2,[],1);

                V_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindex,1);

                midpoint_ford2_jj(:,:,e_c,d2_c)=squeeze(midpoints_jj_e);

                isInfLower    = (ReturnMatrix_L2(1,     :,:) == -Inf);
                isInfUpper    = (ReturnMatrix_L2(n2long,:,:) == -Inf);
                inLowerStrict = (maxindex >= 2)         & (maxindex <= n2short+1);
                inUpperStrict = (maxindex >= n2short+3) & (maxindex <= n2long-1);
                PolicyL2flag_ford2_jj(:,:,e_c,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
            end
        end
    end

    [V_jj,maxindex]=max(V_ford2_jj,[],4);
    Vtilde(:,:,:,N_j)=V_jj;
    Policy(1,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]);
    % Valt at exponential discounter optimum (full max over d2 and aprime)
    [V_jjalt,maxindexalt_d2]=max(Valt_ford2_jj,[],4);
    Valt(:,:,:,N_j)=V_jjalt;
    aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
    Policy(2,:,:,:,N_j)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
    Policy(3,:,:,:,N_j)=aprimeL2_ind;
    PolicyL2flag(1,:,:,:,N_j)=reshape(PolicyL2flag_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
    Policyalt(1,:,:,:,N_j)=shiftdim(maxindexalt_d2,-1);
    maxindexalt_lin=reshape(maxindexalt_d2,[N_a*N_semiz*N_z*N_e,1]);
    Policyalt(2,:,:,:,N_j)=reshape(midpointV_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindexalt_lin-1)),[1,N_a,N_semiz*N_z,N_e]);
    Policyalt(3,:,:,:,N_j)=reshape(Policy_V_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindexalt_lin-1)),[1,N_a,N_semiz*N_z,N_e]);
    PolicyL2flagalt(1,:,:,:,N_j)=reshape(PolicyL2flagV_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindexalt_lin-1)),[1,N_a,N_semiz*N_z,N_e]);
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

    EV=sum(Valt(:,:,:,jj+1).*pi_e_J(1,1,:,jj),3);
    EV=reshape(EV,[N_a,N_semiz,N_z]);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d2_c,jj));

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            EVinterp_d2=interp1(a_grid,EV_d2,aprime_grid);

            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,4);

            %% Valt (beta)
            entireRHS_d2ii_V=ReturnMatrix_d2ii+beta*EV_d2;
            [~,maxindex1]=max(entireRHS_d2ii_V,[],1);
            midpoints_jj(1,level1ii,:,:)=maxindex1;
            maxgap_V=squeeze(max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap_V(ii));
                    aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,5);
                    aprimez=aprimeindexes+N_a*bothzind;
                    entireRHS_ii=ReturnMatrix_ii+beta*reshape(EV_d2(aprimez),[(maxgap_V(ii)+1),1,N_bothz,N_e]);
                    [~,maxindex]=max(entireRHS_ii,[],1);
                    midpoints_jj(1,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(1,ii,:,:);
                    midpoints_jj(1,curraindex,:,:)=repelem(loweredge,1,length(curraindex),1);
                end
            end
            midpoints_jj=max(min(midpoints_jj,n_a-1),2);
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
            ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,5);
            aprimez=aprimeindexes+n2aprime*bothzind;
            entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp_d2(aprimez),[n2long,N_a,N_bothz,N_e]);
            [Vtemp,maxindexL2alt]=max(entireRHS_L2,[],1);
            Valt_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_V_ford2_jj(:,:,:,d2_c)=shiftdim(maxindexL2alt,1);
            midpointV_ford2_jj(:,:,:,d2_c)=squeeze(midpoints_jj);
            isInfLoweralt    = (ReturnMatrix_L2(1,     :,:,:) == -Inf);
            isInfUpperalt    = (ReturnMatrix_L2(n2long,:,:,:) == -Inf);
            inLowerStrictalt = (maxindexL2alt >= 2)         & (maxindexL2alt <= n2short+1);
            inUpperStrictalt = (maxindexL2alt >= n2short+3) & (maxindexL2alt <= n2long-1);
            PolicyL2flagV_ford2_jj(:,:,:,d2_c) = squeeze(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt));

            %% Vtilde (beta0*beta)
            entireRHS_d2ii_T=ReturnMatrix_d2ii+beta0beta*EV_d2;
            [~,maxindex1]=max(entireRHS_d2ii_T,[],1);
            midpoints_jj(1,level1ii,:,:)=maxindex1;
            maxgap=squeeze(max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,5);
                    aprimez=aprimeindexes+N_a*bothzind;
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[(maxgap(ii)+1),1,N_bothz,N_e]);
                    [~,maxindex]=max(entireRHS_ii,[],1);
                    midpoints_jj(1,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(1,ii,:,:);
                    midpoints_jj(1,curraindex,:,:)=repelem(loweredge,1,length(curraindex),1);
                end
            end
            midpoints_jj=max(min(midpoints_jj,n_a-1),2);
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
            ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,5);
            aprimez=aprimeindexes+n2aprime*bothzind;
            entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp_d2(aprimez),[n2long,N_a,N_bothz,N_e]);
            [Vtemp,maxindex]=max(entireRHS_L2,[],1);

            V_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,:,d2_c)=shiftdim(maxindex,1);

            midpoint_ford2_jj(:,:,:,d2_c)=squeeze(midpoints_jj);

            isInfLower    = (ReturnMatrix_L2(1,     :,:,:) == -Inf);
            isInfUpper    = (ReturnMatrix_L2(n2long,:,:,:) == -Inf);
            inLowerStrict = (maxindex >= 2)         & (maxindex <= n2short+1);
            inUpperStrict = (maxindex >= n2short+3) & (maxindex <= n2long-1);
            PolicyL2flag_ford2_jj(:,:,:,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
        end
    elseif vfoptions.lowmemory==1
        midpoints_jj_e=zeros(1,N_a,N_semiz*N_z,'gpuArray');
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d2_c,jj));

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            EVinterp_d2=interp1(a_grid,EV_d2,aprime_grid);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,4);

                %% Valt (beta)
                entireRHS_d2ii_V=ReturnMatrix_d2ii+beta*EV_d2;
                [~,maxindex1]=max(entireRHS_d2ii_V,[],1);
                midpoints_jj_e(1,level1ii,:)=maxindex1;
                maxgap_V=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap_V(ii)>0
                        loweredge=min(maxindex1(1,ii,:),n_a-maxgap_V(ii));
                        aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,5);
                        aprimez=aprimeindexes+N_a*bothzind;
                        entireRHS_ii=ReturnMatrix_ii+beta*reshape(EV_d2(aprimez),[(maxgap_V(ii)+1),1,N_bothz]);
                        [~,maxindex]=max(entireRHS_ii,[],1);
                        midpoints_jj_e(1,curraindex,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(1,ii,:);
                        midpoints_jj_e(1,curraindex,:)=repelem(loweredge,1,length(curraindex),1);
                    end
                end
                midpoints_jj_e=max(min(midpoints_jj_e,n_a-1),2);
                aprimeindexes=(midpoints_jj_e+(midpoints_jj_e-1)*n2short)+(-n2short-1:1:1+n2short)';
                ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,5);
                aprimez=aprimeindexes+n2aprime*bothzind;
                entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp_d2(aprimez),[n2long,N_a,N_bothz]);
                [Vtemp,maxindexL2alt]=max(entireRHS_L2,[],1);
                Valt_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_V_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindexL2alt,1);
                midpointV_ford2_jj(:,:,e_c,d2_c)=squeeze(midpoints_jj_e);
                isInfLoweralt    = (ReturnMatrix_L2(1,     :,:) == -Inf);
                isInfUpperalt    = (ReturnMatrix_L2(n2long,:,:) == -Inf);
                inLowerStrictalt = (maxindexL2alt >= 2)         & (maxindexL2alt <= n2short+1);
                inUpperStrictalt = (maxindexL2alt >= n2short+3) & (maxindexL2alt <= n2long-1);
                PolicyL2flagV_ford2_jj(:,:,e_c,d2_c) = squeeze(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt));

                %% Vtilde (beta0*beta)
                entireRHS_d2ii_T=ReturnMatrix_d2ii+beta0beta*EV_d2;
                [~,maxindex1]=max(entireRHS_d2ii_T,[],1);
                midpoints_jj_e(1,level1ii,:)=maxindex1;
                maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                        aprimeindexes=loweredge+(0:1:maxgap(ii))';
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,5);
                        aprimez=aprimeindexes+N_a*bothzind;
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[(maxgap(ii)+1),1,N_bothz]);
                        [~,maxindex]=max(entireRHS_ii,[],1);
                        midpoints_jj_e(1,curraindex,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(1,ii,:);
                        midpoints_jj_e(1,curraindex,:)=repelem(loweredge,1,length(curraindex),1);
                    end
                end
                midpoints_jj_e=max(min(midpoints_jj_e,n_a-1),2);
                aprimeindexes=(midpoints_jj_e+(midpoints_jj_e-1)*n2short)+(-n2short-1:1:1+n2short)';
                ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,5);
                aprimez=aprimeindexes+n2aprime*bothzind;
                entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp_d2(aprimez),[n2long,N_a,N_bothz]);
                [Vtemp,maxindex]=max(entireRHS_L2,[],1);

                V_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindex,1);

                midpoint_ford2_jj(:,:,e_c,d2_c)=squeeze(midpoints_jj_e);

                isInfLower    = (ReturnMatrix_L2(1,     :,:) == -Inf);
                isInfUpper    = (ReturnMatrix_L2(n2long,:,:) == -Inf);
                inLowerStrict = (maxindex >= 2)         & (maxindex <= n2short+1);
                inUpperStrict = (maxindex >= n2short+3) & (maxindex <= n2long-1);
                PolicyL2flag_ford2_jj(:,:,e_c,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
            end
        end
    end

    [V_jj,maxindex]=max(V_ford2_jj,[],4);
    Vtilde(:,:,:,jj)=V_jj;
    Policy(1,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]);
    % Valt at exponential discounter optimum (full max over d2 and aprime)
    [V_jjalt,maxindexalt_d2]=max(Valt_ford2_jj,[],4);
    Valt(:,:,:,jj)=V_jjalt;
    aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
    Policy(2,:,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
    Policy(3,:,:,:,jj)=aprimeL2_ind;
    PolicyL2flag(1,:,:,:,jj)=reshape(PolicyL2flag_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
    Policyalt(1,:,:,:,jj)=shiftdim(maxindexalt_d2,-1);
    maxindexalt_lin=reshape(maxindexalt_d2,[N_a*N_semiz*N_z*N_e,1]);
    Policyalt(2,:,:,:,jj)=reshape(midpointV_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindexalt_lin-1)),[1,N_a,N_semiz*N_z,N_e]);
    Policyalt(3,:,:,:,jj)=reshape(Policy_V_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindexalt_lin-1)),[1,N_a,N_semiz*N_z,N_e]);
    PolicyL2flagalt(1,:,:,:,jj)=reshape(PolicyL2flagV_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindexalt_lin-1)),[1,N_a,N_semiz*N_z,N_e]);
end

%% Post-process Policy
adjust=(Policy(3,:,:,:,:)<1+n2short+1);
Policy(2,:,:,:,:)=Policy(2,:,:,:,:)-adjust;
Policy(3,:,:,:,:)=adjust.*Policy(3,:,:,:,:)+(1-adjust).*(Policy(3,:,:,:,:)-n2short-1);

Policy=[Policy;PolicyL2flag];

adjustalt=(Policyalt(3,:,:,:,:)<1+n2short+1);
Policyalt(2,:,:,:,:)=Policyalt(2,:,:,:,:)-adjustalt;
Policyalt(3,:,:,:,:)=adjustalt.*Policyalt(3,:,:,:,:)+(1-adjustalt).*(Policyalt(3,:,:,:,:)-n2short-1);

Policyalt=[Policyalt;PolicyL2flagalt];


end
