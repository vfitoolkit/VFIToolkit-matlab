function [V,Policy]=ValueFnIter_FHorz_EpsteinZin_DC1_GI1_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7, ezc8)
% Divide-and-conquer plus grid-interpolation-layer version of
% ValueFnIter_FHorz_EpsteinZin_nod_noz_raw.
% Grafts the Epstein-Zin transforms onto ValueFnIter_FHorz_DC1_GI1_nod_noz_raw.
% The interpolation acts on EV=(ezc4*V')^ezc5 (the expectations object, exactly
% where the vNM GI layer interpolates its EV); the certainty-equivalent power
% ^ezc6 and the rest of the transform chain are applied pointwise AFTER the
% interpolation. (Linear interpolation of EV commutes with affine maps, so at
% gamma=1/phi this reproduces the vNM GI layer exactly.) The warm-glow fn is
% evaluated exactly on the fine grid (no interpolation needed); the coarse
% divide-and-conquer passes use its strided subset. The return transform and
% the final ^ezc7 wrap each level's entireRHS before its max (a monotone
% transform, so the divide-and-conquer monotonicity logic is unaffected).
% No shocks, so the certainty-equivalent is just the identity (Epstein-Zin
% without shocks does not make much sense, but is allowed so that models can
% be compared with/without shocks without also having to change preferences).

N_a=prod(n_a);

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(2,N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z
PolicyL2flag=2*ones(1,N_a,N_j,'gpuArray'); % 1=all weight to lower coarse pt, 2=usual linear weights, 3=all weight to upper coarse pt
% When ReturnFn is -Inf on one of the course grid points, we will allow fine index between that and the neighbouring course grid point, but we use L2flag to record this and so later avoid that -Inf point when simulating/iteration

%%
% Preallocate
midpoints_jj=zeros(1,N_a,'gpuArray');

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

% For debugging, uncomment next two lines, with this 'aprime_grid' you
% should get exact same value fn as without interpolation (as it doesn't
% really interpolate, it just repeats points)
% aprime_grid=repelem(a_grid,1+n2short,1);
% aprime_grid=aprime_grid(1:(N_a+(N_a-1)*n2short));

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
if vfoptions.EZoneminusbeta==1
    ezc1=1-DiscountFactorParamsVec; % Just in case it depends on age
elseif vfoptions.EZoneminusbeta==2
    ezc1=1-sj(N_j)*DiscountFactorParamsVec;
end

% If there is a warm-glow at end of the final period, evaluate the warmglowfn
% (evaluated exactly on the fine grid; coarse passes use the strided subset)
if warmglow==1
    WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,N_j);
    WGmatrixfineraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n2aprime, aprime_grid, WGParamsVec);
    WGmatrixfine=WGmatrixfineraw;
    WGmatrixfine(isfinite(WGmatrixfineraw))=(ezc4*WGmatrixfineraw(isfinite(WGmatrixfineraw))).^ezc5(N_j);
    WGmatrixfine(WGmatrixfineraw==0)=0; % otherwise zero to negative power is set to infinity
    if ~isfield(vfoptions,'V_Jplus1')
        becareful=(WGmatrixfine==0);
        WGmatrixfine(isfinite(WGmatrixfine))=ezc3*DiscountFactorParamsVec*(((1-sj(N_j))*WGmatrixfine(isfinite(WGmatrixfine)).^ezc8(N_j)).^ezc6(N_j));
        WGmatrixfine(becareful)=0;
    end
    WGmatrix=WGmatrixfine(1:(n2short+1):end); % coarse-grid subset
    WGmatrix=WGmatrix(:); % column over the coarse aprime grid
else
    WGmatrixfine=zeros(n2aprime,1,'gpuArray');
    WGmatrix=zeros(N_a,1,'gpuArray');
end

if ~isfield(vfoptions,'V_Jplus1')

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);
    % Modify the Return Function appropriately for Epstein-Zin Preferences
    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
    ReturnMatrix_ii(becareful)=(ezc1*ReturnMatrix_ii(becareful).^ezc2(N_j)).^ezc7(N_j);
    ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
    entireRHS_ii=ReturnMatrix_ii+WGmatrix; % warm-glow (zero if not using)

    %Calc the max and it's index
    [~,maxindex]=max(entireRHS_ii,[],1);

    % Just keep the 'midpoint' version of maxindex1 [as GI]
    midpoints_jj(1,level1ii)=maxindex;

    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1))), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec);
        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
        ReturnMatrix_ii(becareful)=(ezc1*ReturnMatrix_ii(becareful).^ezc2(N_j)).^ezc7(N_j);
        ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
        entireRHS_ii=ReturnMatrix_ii+WGmatrix(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1)));
        [~,maxindex]=max(entireRHS_ii,[],1);
        midpoints_jj(1,curraindex)=maxindex+midpoints_jj(level1ii(ii))-1;
    end

    % Turn this into the 'midpoint'
    midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is 1-by-n_a
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
    % aprime possibilities are n2long-by-n_a
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
    ReturnMatrix_ii(becareful)=(ezc1*ReturnMatrix_ii(becareful).^ezc2(N_j)).^ezc7(N_j);
    ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
    entireRHS_ii=ReturnMatrix_ii+reshape(WGmatrixfine(aprimeindexes),[n2long,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V(:,N_j)=shiftdim(Vtempii,1);
    Policy(1,:,N_j)=shiftdim(squeeze(midpoints_jj),-1); % midpoint
    Policy(2,:,N_j)=shiftdim(maxindexL2,-1); % aprimeL2ind
    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
    isInfLower    = (ReturnMatrix_ii(1,     :) == -Inf);
    isInfUpper    = (ReturnMatrix_ii(n2long,:) == -Inf);
    inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
    inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
    PolicyL2flag(1,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,1]);

    % Part of Epstein-Zin is before taking expectation
    temp=V_Jplus1;
    temp(isfinite(V_Jplus1))=(ezc4*V_Jplus1(isfinite(V_Jplus1))).^ezc5(N_j);
    temp(V_Jplus1==0)=0;

    % No shocks, so no expectation to take (the certainty-equivalent is just the identity)
    EV=temp;

    % Interpolate EV over aprime_grid (BEFORE the certainty-equivalent power ^ezc6)
    EVinterp=interp1(a_grid,EV,aprime_grid'); % (column, to match WGmatrixfine)

    % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over aprime
    temp4=EV;
    temp4interp=EVinterp;
    if warmglow==1
        becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
        temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrix(becareful).^ezc8(N_j)).^ezc6(N_j);
        temp4((EV==0)&(WGmatrix==0))=0; % Is actually zero
        becareful=logical(isfinite(temp4interp).*isfinite(WGmatrixfine)); % both are finite
        temp4interp(becareful)=(sj(N_j)*temp4interp(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrixfine(becareful).^ezc8(N_j)).^ezc6(N_j);
        temp4interp((EVinterp==0)&(WGmatrixfine==0))=0; % Is actually zero
    else % not using warmglow
        temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4)).^ezc8(N_j)).^ezc6(N_j);
        temp4(EV==0)=0;
        temp4interp(isfinite(temp4interp))=(sj(N_j)*temp4interp(isfinite(temp4interp)).^ezc8(N_j)).^ezc6(N_j);
        temp4interp(EVinterp==0)=0;
    end

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);
    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
    temp2_ii=ReturnMatrix_ii;
    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
    temp2_ii(ReturnMatrix_ii==0)=-Inf;
    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*temp4;
    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
    entireRHS_ii(entireRHS_ii==0)=-Inf;
    %Calc the max and it's index
    [~,maxindex]=max(entireRHS_ii,[],1);

    % Just keep the 'midpoint' version of maxindex1 [as GI]
    midpoints_jj(1,level1ii)=maxindex;

    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1))), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec);
        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
        temp2_ii=ReturnMatrix_ii;
        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
        temp2_ii(ReturnMatrix_ii==0)=-Inf;
        entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*temp4(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1)));
        temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
        entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
        entireRHS_ii(entireRHS_ii==0)=-Inf;
        [~,maxindex]=max(entireRHS_ii,[],1);
        midpoints_jj(1,curraindex)=maxindex+midpoints_jj(level1ii(ii))-1;
    end

    % Turn this into the 'midpoint'
    midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is 1-by-n_a
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
    % aprime possibilities are n2long-by-n_a
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
    temp2_ii=ReturnMatrix_ii;
    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
    temp2_ii(ReturnMatrix_ii==0)=-Inf;
    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4interp(aprimeindexes(:)),[n2long,N_a]);
    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
    entireRHS_ii(entireRHS_ii==0)=-Inf;
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V(:,N_j)=shiftdim(Vtempii,1);
    Policy(1,:,N_j)=shiftdim(squeeze(midpoints_jj),-1); % midpoint
    Policy(2,:,N_j)=shiftdim(maxindexL2,-1); % aprimeL2ind
    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
    isInfLower    = (ReturnMatrix_ii(1,     :) == -Inf);
    isInfUpper    = (ReturnMatrix_ii(n2long,:) == -Inf);
    inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
    inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
    PolicyL2flag(1,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
end


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
    end

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    if vfoptions.EZoneminusbeta==1
        ezc1=1-DiscountFactorParamsVec; % Just in case it depends on age
    elseif vfoptions.EZoneminusbeta==2
        ezc1=1-sj(jj)*DiscountFactorParamsVec;
    end

    % If there is a warm-glow, evaluate the warmglowfn (exactly on the fine grid; coarse passes use the strided subset)
    if warmglow==1
        WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
        WGmatrixfineraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n2aprime, aprime_grid, WGParamsVec);
        WGmatrixfine=WGmatrixfineraw;
        WGmatrixfine(isfinite(WGmatrixfineraw))=(ezc4*WGmatrixfineraw(isfinite(WGmatrixfineraw))).^ezc5(jj);
        WGmatrixfine(WGmatrixfineraw==0)=0; % otherwise zero to negative power is set to infinity
        WGmatrix=WGmatrixfine(1:(n2short+1):end); % coarse-grid subset
        WGmatrix=WGmatrix(:); % column over the coarse aprime grid
    end

    EVpre=V(:,jj+1);
    % Part of Epstein-Zin is before taking expectation
    temp=EVpre;
    temp(isfinite(EVpre))=(ezc4*EVpre(isfinite(EVpre))).^ezc5(jj);
    temp(EVpre==0)=0;

    % No shocks, so no expectation to take (the certainty-equivalent is just the identity)
    EV=temp;

    % Interpolate EV over aprime_grid (BEFORE the certainty-equivalent power ^ezc6)
    EVinterp=interp1(a_grid,EV,aprime_grid'); % (column, to match WGmatrixfine)

    % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over aprime
    temp4=EV;
    temp4interp=EVinterp;
    if warmglow==1
        becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
        temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrix(becareful).^ezc8(jj)).^ezc6(jj);
        temp4((EV==0)&(WGmatrix==0))=0; % Is actually zero
        becareful=logical(isfinite(temp4interp).*isfinite(WGmatrixfine)); % both are finite
        temp4interp(becareful)=(sj(jj)*temp4interp(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrixfine(becareful).^ezc8(jj)).^ezc6(jj);
        temp4interp((EVinterp==0)&(WGmatrixfine==0))=0; % Is actually zero
    else % not using warmglow
        temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
        temp4(EV==0)=0;
        temp4interp(isfinite(temp4interp))=(sj(jj)*temp4interp(isfinite(temp4interp)).^ezc8(jj)).^ezc6(jj);
        temp4interp(EVinterp==0)=0;
    end

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);
    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
    temp2_ii=ReturnMatrix_ii;
    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
    temp2_ii(ReturnMatrix_ii==0)=-Inf;
    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*temp4;
    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
    entireRHS_ii(entireRHS_ii==0)=-Inf;
    %Calc the max and it's index
    [~,maxindex]=max(entireRHS_ii,[],1);

    % Just keep the 'midpoint' version of maxindex1 [as GI]
    midpoints_jj(1,level1ii)=maxindex;

    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1))), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec);
        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
        temp2_ii=ReturnMatrix_ii;
        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
        temp2_ii(ReturnMatrix_ii==0)=-Inf;
        entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*temp4(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1)));
        temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
        entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
        entireRHS_ii(entireRHS_ii==0)=-Inf;
        [~,maxindex]=max(entireRHS_ii,[],1);
        midpoints_jj(1,curraindex)=maxindex+midpoints_jj(level1ii(ii))-1;
    end

    % Turn this into the 'midpoint'
    midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is 1-by-n_a
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
    % aprime possibilities are n2long-by-n_a
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
    temp2_ii=ReturnMatrix_ii;
    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
    temp2_ii(ReturnMatrix_ii==0)=-Inf;
    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4interp(aprimeindexes(:)),[n2long,N_a]);
    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
    entireRHS_ii(entireRHS_ii==0)=-Inf;
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V(:,jj)=shiftdim(Vtempii,1);
    Policy(1,:,jj)=shiftdim(squeeze(midpoints_jj),-1); % midpoint
    Policy(2,:,jj)=shiftdim(maxindexL2,-1); % aprimeL2ind
    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
    isInfLower    = (ReturnMatrix_ii(1,     :) == -Inf);
    isInfUpper    = (ReturnMatrix_ii(n2long,:) == -Inf);
    inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
    inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
    PolicyL2flag(1,:,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
end



% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(1,:,:)=Policy(1,:,:)-adjust; % lower grid point
Policy(2,:,:)=adjust.*Policy(2,:,:)+(1-adjust).*(Policy(2,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=[Policy; PolicyL2flag];


end
