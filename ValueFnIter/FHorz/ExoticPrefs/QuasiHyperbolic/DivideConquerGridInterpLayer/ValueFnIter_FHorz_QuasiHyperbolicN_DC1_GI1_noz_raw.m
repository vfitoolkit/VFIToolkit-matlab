function [Vtilde,Policy,V,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicN_DC1_GI1_noz_raw(n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive quasi-hyperbolic discounting variant of ValueFnIter_FHorz_DC1_GI1_noz_raw.
% Has d variables. No z variable. GPU (parallel==2 only).
%
% Naive:  V_j    = max_{d,a'} u + beta*E[V_{j+1}]
%         Vtilde_j = max_{d,a'} u + beta_0*beta*E[V_{j+1}]   (agent's choice)

N_d=prod(n_d);
N_a=prod(n_a);

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(3,N_a,N_j,'gpuArray'); % [d_ind; midpoint; aprimeL2ind]
PolicyL2flag=2*ones(1,N_a,N_j,'gpuArray'); % 1=all weight to lower coarse pt, 2=usual linear weights, 3=all weight to upper coarse pt
Policyalt=zeros(3,N_a,N_j,'gpuArray'); % exponential discounter optimal [d_ind; midpoint; aprimeL2ind]
PolicyL2flagalt=2*ones(1,N_a,N_j,'gpuArray');

midpoints_jj=zeros(N_d,1,N_a,'gpuArray');

aind=gpuArray(0:1:N_a-1);

level1ii=round(linspace(1,n_a,vfoptions.level1n));

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));

%% j=N_j (terminal period)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);
    [~,maxindex1]=max(ReturnMatrix_ii,[],2);
    midpoints_jj(:,1,level1ii)=maxindex1;
    maxgap=max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), ReturnFnParamsVec,3);
            [~,maxindex]=max(ReturnMatrix_ii,[],2);
            midpoints_jj(:,1,curraindex)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(:,1,ii);
            midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1);
        end
    end
    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn,n_d,d_gridvals,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec,2);
    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
    V(:,N_j)=shiftdim(Vtempii,1);
    d_ind=rem(maxindexL2-1,N_d)+1;
    allind=d_ind+N_d*aind;
    Policy(1,:,N_j)=d_ind;
    Policy(2,:,N_j)=shiftdim(squeeze(midpoints_jj(allind)),-1);
    Policy(3,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);

    L2offset=ceil(maxindexL2/N_d);
    linidx_lower=d_ind                + N_d*n2long*aind;
    linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*aind;
    isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
    isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
    inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
    inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
    PolicyL2flag(1,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    Vtilde=V;
    % terminal: QH and exponential discounter coincide
    Policyalt(:,:,N_j)=Policy(:,:,N_j);
    PolicyL2flagalt(1,:,N_j)=PolicyL2flag(1,:,N_j);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=reshape(vfoptions.V_Jplus1,[N_a,1]);
    EVinterp=interp1(a_grid,EV,aprime_grid);

    Vtilde=zeros(N_a,N_j,'gpuArray');

    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);

    %% V (beta)
    entireRHS_ii=ReturnMatrix_ii+beta*shiftdim(EV,-1);
    [~,maxindex1]=max(entireRHS_ii,[],2);
    midpoints_jj(:,1,level1ii)=maxindex1;
    maxgap_V=max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap_V(ii)>0
            loweredge=min(maxindex1(:,1,ii),n_a-maxgap_V(ii));
            aprimeindexes=loweredge+(0:1:maxgap_V(ii));
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), ReturnFnParamsVec,3);
            entireRHS_ii=ReturnMatrix_ii_dc+beta*EV(reshape(aprimeindexes(:),[N_d,(maxgap_V(ii)+1),1]));
            [~,maxindex]=max(entireRHS_ii,[],2);
            midpoints_jj(:,1,curraindex)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(:,1,ii);
            midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1);
        end
    end
    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn,n_d,d_gridvals,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec,2);
    entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp(aprimeindexes(:)),[N_d*n2long,N_a]);
    [Vtempii,maxindexL2alt]=max(entireRHS_L2,[],1);
    V(:,N_j)=shiftdim(Vtempii,1);
    d_indalt=rem(maxindexL2alt-1,N_d)+1;
    allindalt=d_indalt+N_d*aind;
    Policyalt(1,:,N_j)=d_indalt;
    Policyalt(2,:,N_j)=shiftdim(squeeze(midpoints_jj(allindalt)),-1);
    Policyalt(3,:,N_j)=shiftdim(ceil(maxindexL2alt/N_d),-1);

    L2offsetalt=ceil(maxindexL2alt/N_d);
    linidx_loweralt=d_indalt                + N_d*n2long*aind;
    linidx_upperalt=d_indalt + N_d*(n2long-1) + N_d*n2long*aind;
    isInfLoweralt=(ReturnMatrix_L2(linidx_loweralt)==-Inf);
    isInfUpperalt=(ReturnMatrix_L2(linidx_upperalt)==-Inf);
    inLowerStrictalt=(L2offsetalt>=2)         & (L2offsetalt<=n2short+1);
    inUpperStrictalt=(L2offsetalt>=n2short+3) & (L2offsetalt<=n2long-1);
    PolicyL2flagalt(1,:,N_j)=2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);
    %% Vtilde (beta0*beta)
    entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV,-1);
    [~,maxindex1]=max(entireRHS_ii,[],2);
    midpoints_jj(:,1,level1ii)=maxindex1;
    maxgap=max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), ReturnFnParamsVec,3);
            entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*EV(reshape(aprimeindexes(:),[N_d,(maxgap(ii)+1),1]));
            [~,maxindex]=max(entireRHS_ii,[],2);
            midpoints_jj(:,1,curraindex)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(:,1,ii);
            midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1);
        end
    end
    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn,n_d,d_gridvals,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec,2);
    entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp(aprimeindexes(:)),[N_d*n2long,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
    Vtilde(:,N_j)=shiftdim(Vtempii,1);
    d_ind=rem(maxindexL2-1,N_d)+1;
    allind=d_ind+N_d*aind;
    Policy(1,:,N_j)=d_ind;
    Policy(2,:,N_j)=shiftdim(squeeze(midpoints_jj(allind)),-1);
    Policy(3,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);

    L2offset=ceil(maxindexL2/N_d);
    linidx_lower=d_ind                + N_d*n2long*aind;
    linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*aind;
    isInfLower=(ReturnMatrix_L2(linidx_lower)==-Inf);
    isInfUpper=(ReturnMatrix_L2(linidx_upper)==-Inf);
    inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
    inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
    PolicyL2flag(1,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
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

    EVsource=V(:,jj+1);
    EV=EVsource;
    EVinterp=interp1(a_grid,EV,aprime_grid);

    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);

    %% V (beta)
    entireRHS_ii=ReturnMatrix_ii+beta*shiftdim(EV,-1);
    [~,maxindex1]=max(entireRHS_ii,[],2);
    midpoints_jj(:,1,level1ii)=maxindex1;
    maxgap_V=max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap_V(ii)>0
            loweredge=min(maxindex1(:,1,ii),n_a-maxgap_V(ii));
            aprimeindexes=loweredge+(0:1:maxgap_V(ii));
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), ReturnFnParamsVec,3);
            entireRHS_ii=ReturnMatrix_ii_dc+beta*EV(reshape(aprimeindexes(:),[N_d,(maxgap_V(ii)+1),1]));
            [~,maxindex]=max(entireRHS_ii,[],2);
            midpoints_jj(:,1,curraindex)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(:,1,ii);
            midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1);
        end
    end
    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn,n_d,d_gridvals,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec,2);
    entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp(aprimeindexes(:)),[N_d*n2long,N_a]);
    [Vtempii,maxindexL2alt]=max(entireRHS_L2,[],1);
    V(:,jj)=shiftdim(Vtempii,1);
    d_indalt=rem(maxindexL2alt-1,N_d)+1;
    allindalt=d_indalt+N_d*aind;
    Policyalt(1,:,jj)=d_indalt;
    Policyalt(2,:,jj)=shiftdim(squeeze(midpoints_jj(allindalt)),-1);
    Policyalt(3,:,jj)=shiftdim(ceil(maxindexL2alt/N_d),-1);

    L2offsetalt=ceil(maxindexL2alt/N_d);
    linidx_loweralt=d_indalt                + N_d*n2long*aind;
    linidx_upperalt=d_indalt + N_d*(n2long-1) + N_d*n2long*aind;
    isInfLoweralt=(ReturnMatrix_L2(linidx_loweralt)==-Inf);
    isInfUpperalt=(ReturnMatrix_L2(linidx_upperalt)==-Inf);
    inLowerStrictalt=(L2offsetalt>=2)         & (L2offsetalt<=n2short+1);
    inUpperStrictalt=(L2offsetalt>=n2short+3) & (L2offsetalt<=n2long-1);
    PolicyL2flagalt(1,:,jj)=2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);

    %% Vtilde (beta0*beta)
    entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV,-1);
    [~,maxindex1]=max(entireRHS_ii,[],2);
    midpoints_jj(:,1,level1ii)=maxindex1;
    maxgap=max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), ReturnFnParamsVec,3);
            entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*EV(reshape(aprimeindexes(:),[N_d,(maxgap(ii)+1),1]));
            [~,maxindex]=max(entireRHS_ii,[],2);
            midpoints_jj(:,1,curraindex)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(:,1,ii);
            midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1);
        end
    end
    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn,n_d,d_gridvals,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec,2);
    entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp(aprimeindexes(:)),[N_d*n2long,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
    Vtilde(:,jj)=shiftdim(Vtempii,1);
    d_ind=rem(maxindexL2-1,N_d)+1;
    allind=d_ind+N_d*aind;
    Policy(1,:,jj)=d_ind;
    Policy(2,:,jj)=shiftdim(squeeze(midpoints_jj(allind)),-1);
    Policy(3,:,jj)=shiftdim(ceil(maxindexL2/N_d),-1);

    L2offset=ceil(maxindexL2/N_d);
    linidx_lower=d_ind                + N_d*n2long*aind;
    linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*aind;
    isInfLower=(ReturnMatrix_L2(linidx_lower)==-Inf);
    isInfUpper=(ReturnMatrix_L2(linidx_upper)==-Inf);
    inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
    inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
    PolicyL2flag(1,:,jj)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
end

%% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:)<1+n2short+1);
Policy(2,:,:)=Policy(2,:,:)-adjust;
Policy(3,:,:)=adjust.*Policy(3,:,:)+(1-adjust).*(Policy(3,:,:)-n2short-1);

Policy=[Policy;PolicyL2flag];

adjustalt=(Policyalt(3,:,:)<1+n2short+1);
Policyalt(2,:,:)=Policyalt(2,:,:)-adjustalt;
Policyalt(3,:,:)=adjustalt.*Policyalt(3,:,:)+(1-adjustalt).*(Policyalt(3,:,:)-n2short-1);

Policyalt=[Policyalt;PolicyL2flagalt];

end
