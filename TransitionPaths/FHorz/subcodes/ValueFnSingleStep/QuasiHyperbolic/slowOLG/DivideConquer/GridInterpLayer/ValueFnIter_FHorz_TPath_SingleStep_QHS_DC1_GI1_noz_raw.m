function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_QHS_DC1_GI1_noz_raw(V,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The V input is next period value fn (across all ages), the V output is this period.
% Sophisticated QH: V carries Vunderbar; Policy = QH choice.

N_d=prod(n_d);
N_a=prod(n_a);

Policy=zeros(4,N_a,N_j,'gpuArray'); % [d_ind; midpoint; aprimeL2ind; L2flag]

if vfoptions.lowmemory>0
    error('vfoptions.lowmemory>0 not supported for ValueFnIter_FHorz_TPath_SingleStep_QHS_DC1_GI1_noz_raw')
end

%%
midpoints_jj=zeros(N_d,1,N_a,'gpuArray');

aind=0:1:N_a-1;

level1ii=round(linspace(1,n_a,vfoptions.level1n));

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));


%% j=N_j: terminal age has no continuation in TPath
Vtemp_j=V(:,N_j);

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);

[~,maxindex1]=max(ReturnMatrix_ii,[],2);
midpoints_jj(:,1,level1ii)=maxindex1;

maxgap=max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1);
for ii=1:(vfoptions.level1n-1)
    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
    if maxgap(ii)>0
        loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
        aprimeindexes=loweredge+(0:1:maxgap(ii));
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,3);
        [~,maxindex]=max(ReturnMatrix_ii,[],2);
        midpoints_jj(:,1,curraindex,:)=shiftdim(maxindex+(loweredge-1),1);
    else
        loweredge=maxindex1(:,1,ii);
        midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1);
    end
end

midpoints_jj=max(min(midpoints_jj,n_a-1),2);
aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn,n_d, d_gridvals,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec,2);
[Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
V(:,N_j)=shiftdim(Vtempii,1);
d_ind=rem(maxindexL2-1,N_d)+1;
allind=d_ind+N_d*aind;
Policy(1,:,N_j)=d_ind;
Policy(2,:,N_j)=shiftdim(squeeze(midpoints_jj(allind)),-1);
Policy(3,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);
L2offset = ceil(maxindexL2/N_d);
linidx_lower = d_ind                  + N_d*n2long*aind;
linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind;
isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
Policy(4,:,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    VKronNext_j=Vtemp_j;
    Vtemp_j=V(:,jj);

    EV=VKronNext_j;
    EVinterp=interp1(a_grid,EV,aprime_grid);

    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);

    %% Vhat (beta0*beta) — find QH-optimal Policy
    entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV,-1);
    [~,maxindex1]=max(entireRHS_ii,[],2);
    midpoints_jj(:,1,level1ii)=maxindex1;
    maxgap=max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii),n_a-maxgap(:,1,ii));
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,3);
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
    EVfine=reshape(EVinterp(aprimeindexes(:)),[N_d*n2long,N_a]);
    entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine;
    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
    Vhat_jj=shiftdim(Vtempii,1);
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
    Policy(4,:,jj)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);

    linidx=double(reshape(maxindexL2,[1,N_a]))+N_d*n2long*(0:N_a-1);
    EV_at_policy=reshape(EVfine(linidx),[N_a,1]);
    V(:,jj)=Vhat_jj+(beta-beta0beta)*EV_at_policy;
end

%% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:)<1+n2short+1);
Policy(2,:,:)=Policy(2,:,:)-adjust;
Policy(3,:,:)=adjust.*Policy(3,:,:)+(1-adjust).*(Policy(3,:,:)-n2short-1);

end
