function [Vhat, Policy, Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_DC1_GI1_noz_e_raw(n_d1,n_d2,n_a,n_semiz, n_e,N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Sophisticated QH + SemiExo + DC + GI raw, with d1, no z, with e. Output: (Vunderbar, Policy3, Vhat) -- matches plain QH SemiExo S raw convention.

n_d=[n_d1,n_d2];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=N_d1*N_d2;
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_e=prod(n_e);

Vhat=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Policy=zeros(4,N_a,N_semiz,N_e,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_semiz,N_e,N_j,'gpuArray');

%%
special_n_d=[n_d1,ones(1,length(n_d2))];
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];
d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_d2,length(n_d1)+length(n_d2)]),[1,3,2]);

aind=gpuArray(0:1:N_a-1);
semizind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
semizind2=shiftdim(gpuArray(0:1:N_semiz-1),-2);
eind=shiftdim(gpuArray(0:1:N_e-1),-2);

Vhat_ford2_jj=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
Vunderbar_ford2_jj=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
midpoint_ford2_jj=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
PolicyL2flag_ford2_jj=2*ones(N_a,N_semiz,N_e,N_d2,'gpuArray');
midpoints_jj=zeros(N_d1,1,N_a,N_semiz,N_e,'gpuArray');

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
        midpoints_Nj=zeros(N_d,1,N_a,N_semiz,N_e,'gpuArray');

        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_semiz, n_e, d_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        midpoints_Nj(:,1,level1ii,:,:)=maxindex1;
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_semiz, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3);
                [~,maxindex]=max(ReturnMatrix_ii,[],2);
                midpoints_Nj(:,1,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                midpoints_Nj(:,1,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1,1);
            end
        end
        midpoints_Nj=max(min(midpoints_Nj,n_a-1),2);
        aprimeindexes=(midpoints_Nj+(midpoints_Nj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_semiz, n_e, d_gridvals, aprime_grid(aprimeindexes), a_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        Vhat(:,:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*semizind+N_d*N_a*N_semiz*eind;
        Policy(1,:,:,:,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
        Policy(2,:,:,:,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
        Policy(3,:,:,:,N_j)=shiftdim(squeeze(midpoints_Nj(allind)),-1);
        Policy(4,:,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);

        L2offset = ceil(maxindexL2/N_d);
        linidx_lower = d_ind                  + N_d*n2long*aind + N_d*n2long*N_a*semizind + N_d*n2long*N_a*N_semiz*eind;
        linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*semizind + N_d*n2long*N_a*N_semiz*eind;
        isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag(1,:,:,:,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
    elseif vfoptions.lowmemory==1
        midpoints_Nj=zeros(N_d,1,N_a,N_semiz,'gpuArray');
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_semiz, special_n_e, d_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);
            midpoints_Nj(:,1,level1ii,:)=maxindex1;
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_semiz, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3);
                    [~,maxindex]=max(ReturnMatrix_ii,[],2);
                    midpoints_Nj(:,1,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    midpoints_Nj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
                end
            end
            midpoints_Nj=max(min(midpoints_Nj,n_a-1),2);
            aprimeindexes=(midpoints_Nj+(midpoints_Nj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_semiz, special_n_e, d_gridvals, aprime_grid(aprimeindexes), a_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            Vhat(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind+N_d*N_a*semizind;
            Policy(1,:,:,e_c,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
            Policy(2,:,:,e_c,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
            Policy(3,:,:,e_c,N_j)=shiftdim(squeeze(midpoints_Nj(allind)),-1);
            Policy(4,:,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);

            L2offset = ceil(maxindexL2/N_d);
            linidx_lower = d_ind                  + N_d*n2long*aind + N_d*n2long*N_a*semizind;
            linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*semizind;
            isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            PolicyL2flag(1,:,:,e_c,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
        end
    end

    Vunderbar(:,:,:,N_j)=Vhat(:,:,:,N_j);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=sum(reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]).*pi_e_J(1,1,:,N_j),3);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=pi_semiz_J(:,:,d2_c,N_j);

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            EVinterp_d2=interp1(a_grid,EV_d2,aprime_grid);

            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, n_e, d12c_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
            entireRHS_ii=ReturnMatrix_d2ii+beta0beta*shiftdim(EV_d2,-1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoints_jj(:,1,level1ii,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,6);
                    aprimez=aprimeindexes+N_a*semizind2;
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1,(maxgap(ii)+1),1,N_semiz,N_e]);
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoints_jj(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    midpoints_jj(:,1,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1,1);
                end
            end
            midpoints_jj=max(min(midpoints_jj,n_a-1),2);
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            aprimez=aprimeindexes+n2aprime*semizind2;
            EVfine=reshape(EVinterp_d2(aprimez),[N_d1*n2long,N_a,N_semiz,N_e]);
            entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine;
            [Vtemp,maxindex]=max(entireRHS_L2,[],1);

            Vhat_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,:,d2_c)=shiftdim(maxindex,1);

            d1_ind=rem(maxindex-1,N_d1)+1;
            allind=d1_ind+N_d1*aind+N_d1*N_a*semizind+N_d1*N_a*N_semiz*eind;
            midpoint_ford2_jj(:,:,:,d2_c)=squeeze(midpoints_jj(allind));

            L2offset_d2 = ceil(maxindex/N_d1);
            linidx_lower = d1_ind                  + N_d1*n2long*aind + N_d1*n2long*N_a*semizind + N_d1*n2long*N_a*N_semiz*eind;
            linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*semizind + N_d1*n2long*N_a*N_semiz*eind;
            isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
            inLowerStrict = (L2offset_d2 >= 2)         & (L2offset_d2 <= n2short+1);
            inUpperStrict = (L2offset_d2 >= n2short+3) & (L2offset_d2 <= n2long-1);
            PolicyL2flag_ford2_jj(:,:,:,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

            linidx=double(reshape(maxindex,[1,N_a*N_semiz*N_e]))+N_d1*n2long*(0:N_a*N_semiz*N_e-1);
            EV_at_policy=reshape(EVfine(linidx),[N_a,N_semiz,N_e]);
            Vunderbar_ford2_jj(:,:,:,d2_c)=Vhat_ford2_jj(:,:,:,d2_c)+(beta-beta0beta)*EV_at_policy;
        end
    elseif vfoptions.lowmemory==1
        midpoints_jj_e=zeros(N_d1,1,N_a,N_semiz,'gpuArray');
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=pi_semiz_J(:,:,d2_c,N_j);

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            EVinterp_d2=interp1(a_grid,EV_d2,aprime_grid);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, special_n_e, d12c_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);
                entireRHS_ii=ReturnMatrix_d2ii+beta0beta*shiftdim(EV_d2,-1);
                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoints_jj_e(:,1,level1ii,:)=maxindex1;
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, special_n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,6);
                        aprimez=aprimeindexes+N_a*semizind2;
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1,(maxgap(ii)+1),1,N_semiz]);
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoints_jj_e(:,1,curraindex,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        midpoints_jj_e(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
                    end
                end
                midpoints_jj_e=max(min(midpoints_jj_e,n_a-1),2);
                aprimeindexes=(midpoints_jj_e+(midpoints_jj_e-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, special_n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                aprimez=aprimeindexes+n2aprime*semizind2;
                EVfine=reshape(EVinterp_d2(aprimez),[N_d1*n2long,N_a,N_semiz]);
                entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine;
                [Vtemp,maxindex]=max(entireRHS_L2,[],1);

                Vhat_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindex,1);

                d1_ind=rem(maxindex-1,N_d1)+1;
                allind=d1_ind+N_d1*aind+N_d1*N_a*semizind;
                midpoint_ford2_jj(:,:,e_c,d2_c)=squeeze(midpoints_jj_e(allind));

                L2offset_d2 = ceil(maxindex/N_d1);
                linidx_lower = d1_ind                  + N_d1*n2long*aind + N_d1*n2long*N_a*semizind;
                linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*semizind;
                isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
                inLowerStrict = (L2offset_d2 >= 2)         & (L2offset_d2 <= n2short+1);
                inUpperStrict = (L2offset_d2 >= n2short+3) & (L2offset_d2 <= n2long-1);
                PolicyL2flag_ford2_jj(:,:,e_c,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

                linidx=double(reshape(maxindex,[1,N_a*N_semiz]))+N_d1*n2long*(0:N_a*N_semiz-1);
                EV_at_policy=reshape(EVfine(linidx),[N_a,N_semiz]);
                Vunderbar_ford2_jj(:,:,e_c,d2_c)=Vhat_ford2_jj(:,:,e_c,d2_c)+(beta-beta0beta)*EV_at_policy;
            end
        end
    end

    [V_jj,maxindex]=max(Vhat_ford2_jj,[],4);
    Vhat(:,:,:,N_j)=V_jj;
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    Vunderbar(:,:,:,N_j)=reshape(Vunderbar_ford2_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[N_a,N_semiz,N_e]);
    d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_e]);
    Policy(1,:,:,:,N_j)=reshape(rem(d1aprimeL2_ind-1,N_d1)+1,[N_a,N_semiz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(ceil(d1aprimeL2_ind/N_d1),[N_a,N_semiz,N_e]);
    Policy(3,:,:,:,N_j)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    PolicyL2flag(1,:,:,:,N_j)=reshape(PolicyL2flag_ford2_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
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

    EV=sum(Vunderbar(:,:,:,jj+1).*pi_e_J(1,1,:,jj),3);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=pi_semiz_J(:,:,d2_c,jj);

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            EVinterp_d2=interp1(a_grid,EV_d2,aprime_grid);

            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, n_e, d12c_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);
            entireRHS_ii=ReturnMatrix_d2ii+beta0beta*shiftdim(EV_d2,-1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoints_jj(:,1,level1ii,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,6);
                    aprimez=aprimeindexes+N_a*semizind2;
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1,(maxgap(ii)+1),1,N_semiz,N_e]);
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoints_jj(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    midpoints_jj(:,1,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1,1);
                end
            end
            midpoints_jj=max(min(midpoints_jj,n_a-1),2);
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
            aprimez=aprimeindexes+n2aprime*semizind2;
            EVfine=reshape(EVinterp_d2(aprimez),[N_d1*n2long,N_a,N_semiz,N_e]);
            entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine;
            [Vtemp,maxindex]=max(entireRHS_L2,[],1);

            Vhat_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,:,d2_c)=shiftdim(maxindex,1);

            d1_ind=rem(maxindex-1,N_d1)+1;
            allind=d1_ind+N_d1*aind+N_d1*N_a*semizind+N_d1*N_a*N_semiz*eind;
            midpoint_ford2_jj(:,:,:,d2_c)=squeeze(midpoints_jj(allind));

            L2offset_d2 = ceil(maxindex/N_d1);
            linidx_lower = d1_ind                  + N_d1*n2long*aind + N_d1*n2long*N_a*semizind + N_d1*n2long*N_a*N_semiz*eind;
            linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*semizind + N_d1*n2long*N_a*N_semiz*eind;
            isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
            inLowerStrict = (L2offset_d2 >= 2)         & (L2offset_d2 <= n2short+1);
            inUpperStrict = (L2offset_d2 >= n2short+3) & (L2offset_d2 <= n2long-1);
            PolicyL2flag_ford2_jj(:,:,:,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

            linidx=double(reshape(maxindex,[1,N_a*N_semiz*N_e]))+N_d1*n2long*(0:N_a*N_semiz*N_e-1);
            EV_at_policy=reshape(EVfine(linidx),[N_a,N_semiz,N_e]);
            Vunderbar_ford2_jj(:,:,:,d2_c)=Vhat_ford2_jj(:,:,:,d2_c)+(beta-beta0beta)*EV_at_policy;
        end
    elseif vfoptions.lowmemory==1
        midpoints_jj_e=zeros(N_d1,1,N_a,N_semiz,'gpuArray');
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=pi_semiz_J(:,:,d2_c,jj);

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            EVinterp_d2=interp1(a_grid,EV_d2,aprime_grid);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, special_n_e, d12c_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1);
                entireRHS_ii=ReturnMatrix_d2ii+beta0beta*shiftdim(EV_d2,-1);
                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoints_jj_e(:,1,level1ii,:)=maxindex1;
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, special_n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,6);
                        aprimez=aprimeindexes+N_a*semizind2;
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1,(maxgap(ii)+1),1,N_semiz]);
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoints_jj_e(:,1,curraindex,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        midpoints_jj_e(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
                    end
                end
                midpoints_jj_e=max(min(midpoints_jj_e,n_a-1),2);
                aprimeindexes=(midpoints_jj_e+(midpoints_jj_e-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, special_n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
                aprimez=aprimeindexes+n2aprime*semizind2;
                EVfine=reshape(EVinterp_d2(aprimez),[N_d1*n2long,N_a,N_semiz]);
                entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine;
                [Vtemp,maxindex]=max(entireRHS_L2,[],1);

                Vhat_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindex,1);

                d1_ind=rem(maxindex-1,N_d1)+1;
                allind=d1_ind+N_d1*aind+N_d1*N_a*semizind;
                midpoint_ford2_jj(:,:,e_c,d2_c)=squeeze(midpoints_jj_e(allind));

                L2offset_d2 = ceil(maxindex/N_d1);
                linidx_lower = d1_ind                  + N_d1*n2long*aind + N_d1*n2long*N_a*semizind;
                linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*semizind;
                isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
                inLowerStrict = (L2offset_d2 >= 2)         & (L2offset_d2 <= n2short+1);
                inUpperStrict = (L2offset_d2 >= n2short+3) & (L2offset_d2 <= n2long-1);
                PolicyL2flag_ford2_jj(:,:,e_c,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

                linidx=double(reshape(maxindex,[1,N_a*N_semiz]))+N_d1*n2long*(0:N_a*N_semiz-1);
                EV_at_policy=reshape(EVfine(linidx),[N_a,N_semiz]);
                Vunderbar_ford2_jj(:,:,e_c,d2_c)=Vhat_ford2_jj(:,:,e_c,d2_c)+(beta-beta0beta)*EV_at_policy;
            end
        end
    end

    [V_jj,maxindex]=max(Vhat_ford2_jj,[],4);
    Vhat(:,:,:,jj)=V_jj;
    Policy(2,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    Vunderbar(:,:,:,jj)=reshape(Vunderbar_ford2_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[N_a,N_semiz,N_e]);
    d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_e]);
    Policy(1,:,:,:,jj)=reshape(rem(d1aprimeL2_ind-1,N_d1)+1,[N_a,N_semiz,N_e]);
    Policy(4,:,:,:,jj)=reshape(ceil(d1aprimeL2_ind/N_d1),[N_a,N_semiz,N_e]);
    Policy(3,:,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    PolicyL2flag(1,:,:,:,jj)=reshape(PolicyL2flag_ford2_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
end

%% Post-process Policy
adjust=(Policy(4,:,:,:,:)<1+n2short+1);
Policy(3,:,:,:,:)=Policy(3,:,:,:,:)-adjust;
Policy(4,:,:,:,:)=adjust.*Policy(4,:,:,:,:)+(1-adjust).*(Policy(4,:,:,:,:)-n2short-1);

Policy=[Policy;PolicyL2flag];


end
