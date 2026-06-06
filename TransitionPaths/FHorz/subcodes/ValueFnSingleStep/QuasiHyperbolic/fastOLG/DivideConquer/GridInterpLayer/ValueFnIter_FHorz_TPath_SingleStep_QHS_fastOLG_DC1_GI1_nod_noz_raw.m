function [V, Policy, Vhat]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_DC1_GI1_nod_noz_raw(V,n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% V carries Vunderbar for Sophisticated QH

N_a=prod(n_a);

Policy=zeros(3,N_a,N_j,'gpuArray'); % first dim indexes the optimal choice for aprime (midpoint, L2, L2 flag)
Vhat=zeros(N_a,N_j,'gpuArray');

%%

% Preallocate
midpoints_jj=zeros(1,N_a,N_j,'gpuArray');

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

jind=shiftdim(gpuArray(0:1:N_j-1),-1);


%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_aprime by N_a*N_j (note: N_aprime is just equal to N_a)

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

beta_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);
beta0_J=CreateAgeMatrixFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
beta0beta_J=beta0_J.*beta_J;
betaminusbeta0beta_J=beta_J-beta0beta_J;

if vfoptions.EVpre==0
    EV=zeros(N_a,N_j,'gpuArray');
    EV(:,1:N_j-1)=V(:,2:end);
    EV=reshape(EV,[N_a,1,N_j]);
elseif vfoptions.EVpre==1
    % This is used for 'Matched Expecations Path'
    EV=reshape(V,[N_a,1,N_j]); % input V is of size [N_a,N_j] and we want to use the whole thing
end

% Interpolate EV over aprime_grid
EVinterp=interp1(a_grid,EV,aprime_grid);

DiscountedEV=reshape(beta0beta_J,[1,1,N_j]).*EV;
DiscountedEVinterp=reshape(beta0beta_J,[1,1,N_j]).*EVinterp;
DiffDiscountedEVinterp=reshape(betaminusbeta0beta_J,[1,1,N_j]).*EVinterp;

% n-Monotonicity
ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_noz(ReturnFn, N_j, a_grid, a_grid(level1ii), ReturnFnParamsAgeMatrix,1);

%% Vhat (beta0*beta) -> Policy
entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

[~,maxindex1]=max(entireRHS_ii,[],1);
midpoints_jj(1,level1ii,:)=maxindex1;

maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
for ii=1:(vfoptions.level1n-1)
    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
    if maxgap(ii)>0
        loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
        aprimeindexes=loweredge+(0:1:maxgap(ii))';
        ReturnMatrix_ii_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_noz(ReturnFn,N_j,a_grid(aprimeindexes),a_grid(level1ii(ii)+1:level1ii(ii+1)-1),ReturnFnParamsAgeMatrix,2);
        aprimej=repelem(aprimeindexes,1,level1iidiff(ii),1)+N_a*jind;
        entireRHS_ii=ReturnMatrix_ii_dc+reshape(DiscountedEV(aprimej(:)),[maxgap(ii)+1,level1iidiff(ii),N_j]);
        [~,maxindex]=max(entireRHS_ii,[],1);
        midpoints_jj(1,curraindex,:)=maxindex+(loweredge-1);
    else
        loweredge=maxindex1(1,ii,:);
        midpoints_jj(1,curraindex,:)=repelem(loweredge,1,length(curraindex),1);
    end
end

midpoints_jj=max(min(midpoints_jj,n_a-1),2);
aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
ReturnMatrix_L2=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_noz(ReturnFn,N_j,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsAgeMatrix,2);
aprimej=aprimeindexes+n2aprime*jind;
EVfine=reshape(DiffDiscountedEVinterp(aprimej(:)),[n2long,N_a,N_j]);
entireRHS_L2=ReturnMatrix_L2+reshape(DiscountedEVinterp(aprimej(:)),[n2long,N_a,N_j]);
[Vhat,maxindexL2]=max(entireRHS_L2,[],1);
Policy(1,:,:)=shiftdim(squeeze(midpoints_jj),-1);
Policy(2,:,:)=shiftdim(maxindexL2,-1);

isInfLower    = (ReturnMatrix_L2(1,     :,:) == -Inf);
isInfUpper    = (ReturnMatrix_L2(n2long,:,:) == -Inf);
inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
Policy(3,:,:) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);

% Vunderbar = Vhat + (beta - beta0beta)*EV_at_policy
linidx=reshape(maxindexL2,[1,N_a*N_j])+n2long*(0:N_a*N_j-1);
EV_at_policy=reshape(EVfine(linidx),[N_a,N_j]);
Vhat=shiftdim(Vhat,1);
V=Vhat+EV_at_policy;


%% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2,:,:)<1+n2short+1);
Policy(1,:,:)=Policy(1,:,:)-adjust;
Policy(2,:,:)=adjust.*Policy(2,:,:)+(1-adjust).*(Policy(2,:,:)-n2short-1);



end
