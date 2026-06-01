function [V,Policy,Vhat]=ValueFnIter_FHorz_TPath_SingleStep_QHS_DC1_GI1_nod_noz_raw(V,n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The V input is next period value fn (across all ages), the V output is this period.
% Sophisticated QH: V carries Vunderbar; Policy = QH choice; Vhat is the agent's-perspective (beta0*beta) value.

N_a=prod(n_a);

Policy=zeros(3,N_a,N_j,'gpuArray'); % [midpoint; aprimeL2ind; L2flag]
Vhat=zeros(N_a,N_j,'gpuArray');

if vfoptions.lowmemory>0
    error('vfoptions.lowmemory>0 not supported for ValueFnIter_FHorz_TPath_SingleStep_QHS_DC1_GI1_nod_noz_raw')
end

%%
midpoints_jj=zeros(1,N_a,'gpuArray');

level1ii=round(linspace(1,n_a,vfoptions.level1n));

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));


%% j=N_j: terminal age has no continuation in TPath
Vtemp_j=V(:,N_j);

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);

[~,maxindex]=max(ReturnMatrix_ii,[],1);
midpoints_jj(1,level1ii)=maxindex;

for ii=1:(vfoptions.level1n-1)
    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1))), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec);
    [~,maxindex]=max(ReturnMatrix_ii,[],1);
    midpoints_jj(1,curraindex)=maxindex+midpoints_jj(level1ii(ii))-1;
end

midpoints_jj=max(min(midpoints_jj,n_a-1),2);
aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
[Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
V(:,N_j)=shiftdim(Vtempii,1);
Policy(1,:,N_j)=shiftdim(squeeze(midpoints_jj),-1);
Policy(2,:,N_j)=shiftdim(maxindexL2,-1);
isInfLower    = (ReturnMatrix_ii(1,     :) == -Inf);
isInfUpper    = (ReturnMatrix_ii(n2long,:) == -Inf);
inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
Policy(3,:,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);
Vhat(:,N_j)=V(:,N_j); % terminal: Vhat coincides with V (Vunderbar)


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

    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);

    %% Vhat (beta0*beta) — find QH-optimal Policy
    entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
    [~,maxindex]=max(entireRHS_ii,[],1);
    midpoints_jj(1,level1ii)=maxindex;
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1))), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec);
        entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*EV(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1)));
        [~,maxindex]=max(entireRHS_ii,[],1);
        midpoints_jj(1,curraindex)=maxindex+midpoints_jj(level1ii(ii))-1;
    end
    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
    ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    EVfine=reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
    entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine;
    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
    Vhat_jj=shiftdim(Vtempii,1);
    Vhat(:,jj)=Vhat_jj;
    Policy(1,:,jj)=shiftdim(squeeze(midpoints_jj),-1);
    Policy(2,:,jj)=shiftdim(maxindexL2,-1);
    isInfLower    = (ReturnMatrix_L2(1,     :) == -Inf);
    isInfUpper    = (ReturnMatrix_L2(n2long,:) == -Inf);
    inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
    inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
    Policy(3,:,jj) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);

    linidx=double(reshape(maxindexL2,[1,N_a]))+n2long*(0:N_a-1);
    EV_at_policy=reshape(EVfine(linidx),[N_a,1]);
    V(:,jj)=Vhat_jj+(beta-beta0beta)*EV_at_policy;
end

%% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2,:,:)<1+n2short+1);
Policy(1,:,:)=Policy(1,:,:)-adjust;
Policy(2,:,:)=adjust.*Policy(2,:,:)+(1-adjust).*(Policy(2,:,:)-n2short-1);

end
