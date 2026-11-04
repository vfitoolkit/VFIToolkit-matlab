function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_LS1_GI1_nod_raw(V,n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Epstein-Zin with local search AND the grid interpolation layer, finite horizon, fastOLG (all ages at
% once), with z, no d. The V input is next period value fn (across all ages), the V output is this period.
% Two sources:
%  - the window, the two layers, the clamp, the L2 flag and the Policy re-spelling are EXACTLY those of
%    fastOLG/LocalSearch/GridInterpLayer/ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LS1_GI1_nod_raw (see
%    that raw): layer 1 is a coarse max over the window (whose answer IS the new reference, unclamped),
%    layer 2 is the interpolation layer one coarse cell either side of that argmax CLAMPED to
%    [loweredge+1,loweredge+2n-1];
%  - every right-hand side is formed EXACTLY as EpsteinZin/fastOLG/GridInterpLayer/
%    ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_GI1_nod_raw forms it: the coarse and the fine
%    (interpolated) discounted certainty-equivalent terms are built over the whole aprime grid,
%    age-vectorised, with the terminal age set explicitly to the warm-glow term alone, then gathered on the
%    window (layer 1, coarse) and on the fine points (layer 2); the return is transformed (^ezc2) on the
%    window only.
% Both windowed return matrices are [window,N_a,N_j,N_z], so age is dim 3 throughout and the
% age-dependent constants are laid along dim 3.
% When the window holds the unrestricted coarse argmax (and the clamped midpoint is the unrestricted one),
% V and Policy are bit-identical to the unrestricted EZ GI1 raw.

N_a=prod(n_a);
N_z=prod(n_z);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw.

% ONLY THE vfoptions.EVpre==0 PATH. Local search is reachable only from a transition path, where
% EVpre=0 (EVpre=1 is the Matched Expectations Path, which never sets localsearch).
if vfoptions.EVpre~=0
    error('vfoptions.localsearch requires vfoptions.EVpre=0 (ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_LS1_GI1_nod_raw)')
end

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_a,N_j,N_z]);

z_gridvals_J=shiftdim(z_gridvals_J,-2); % [1,1,N_j,N_z,l_z]

% Grid interpolation
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=2*n2short+3;            % the interpolation layer's search set, exactly as in the standard GI raw
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

jind=shiftdim(gpuArray(0:1:N_j-1),-1);
zind=shiftdim(gpuArray(0:1:N_z-1),-2);

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

%% The age-dependent Epstein-Zin constants, along the age dimension (dim 3)
ezc2_J=reshape(ezc2,[1,1,N_j]);
ezc5_J=reshape(ezc5,[1,1,N_j]);
ezc6_J=reshape(ezc6,[1,1,N_j]);
ezc7_J=reshape(ezc7,[1,1,N_j]);
ezc8_J=reshape(ezc8,[1,1,N_j]);
sj_J=reshape(sj,[1,1,N_j]);
if vfoptions.EZoneminusbeta==1
    ezc1_J=reshape(1-DiscountFactor_J,[1,1,N_j]); % (1-beta)* on this period's return, age by age
elseif vfoptions.EZoneminusbeta==2
    ezc1_J=reshape(1-sj(:).*DiscountFactor_J,[1,1,N_j]); % (1-sj*beta)* on this period's return, age by age
else
    ezc1_J=ezc1;
end

%% The warm-glow of bequests, exactly on the fine grid, (aprime,1,j); the coarse pass uses the strided subset
if warmglow==1
    WGmatrixfineraw=zeros(n2aprime,1,N_j,'gpuArray');
    for jj=1:N_j
        WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
        WGmatrixfineraw(:,1,jj)=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n2aprime, aprime_grid, WGParamsVec);
    end
    WGfinite=isfinite(WGmatrixfineraw);
    WGmatrixfine=ezc4*WGmatrixfineraw;
    WGmatrixfine(~WGfinite)=1; % set aside, the power is for the finite entries
    WGmatrixfine=WGmatrixfine.^ezc5_J;
    WGmatrixfine(~WGfinite)=WGmatrixfineraw(~WGfinite);
    WGmatrixfine(WGmatrixfineraw==0)=0; % otherwise zero to negative power is set to infinity
    WGmatrixfine(WGmatrixfineraw==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf
    WGmatrix=WGmatrixfine(1:(n2short+1):end,:,:); % coarse-grid subset, (aprime,1,j)
    clear WGmatrixfineraw WGfinite
end

%% First, create the big 'next period (of transition path) expected value fn.
pi_z_J=cat(1,pi_z_J,zeros(1,N_z,N_z,'gpuArray')); % append a j=N_j slot of zeros: there is no continuation value in the final period (input pi_z_J has N_j-1 slices, the transitions from periods 1..N_j-1)
Vnext=reshape(V(N_a+1:end,:),[N_a,1,N_j-1,N_z]); % next period's value fn at ages 2..N_j: the continuation of ages 1..N_j-1
% Part of Epstein-Zin is before taking expectation
Vnextfinite=isfinite(Vnext);
temp=ezc4*Vnext;
temp(~Vnextfinite)=1; % set aside, the power is for the finite entries
temp=temp.^ezc5_J(1,1,1:N_j-1);
temp(~Vnextfinite)=Vnext(~Vnextfinite);
temp(Vnext==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf
EVpre=zeros(N_a,1,N_j,N_z,'gpuArray');
EVpre(:,1,1:N_j-1,:)=temp; % I use zeros in j=N_j so that can just use pi_z_J to create expectations
EV=EVpre.*shiftdim(pi_z_J,-2);
EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z)
clear Vnext Vnextfinite temp EVpre

% Interpolate EV over aprime_grid (BEFORE the certainty-equivalent power ^ezc6)
EVinterp=interp1(a_grid,EV,aprime_grid);

%% The certainty equivalent, age by age, on the coarse and on the fine grid
temp4=EV;
temp4interp=EVinterp;
if warmglow==1
    becareful=(~(temp4==-Inf)) & isfinite(WGmatrix); % both are finite
    temp4b=temp4;
    temp4b(~becareful)=1; % set aside
    WGb=WGmatrix;
    WGb(~isfinite(WGb))=1; % set aside
    temp4b=(sj_J.*temp4b.^ezc8_J+(1-sj_J).*WGb.^ezc8_J).^ezc6_J;
    temp4(becareful)=temp4b(becareful);
    temp4((EV==0)&(WGmatrix==0))=0; % Is actually zero
    becareful=(~(temp4interp==-Inf)) & isfinite(WGmatrixfine); % both are finite
    temp4b=temp4interp;
    temp4b(~becareful)=1; % set aside
    WGb=WGmatrixfine;
    WGb(~isfinite(WGb))=1; % set aside
    temp4b=(sj_J.*temp4b.^ezc8_J+(1-sj_J).*WGb.^ezc8_J).^ezc6_J;
    temp4interp(becareful)=temp4b(becareful);
    temp4interp((EVinterp==0)&(WGmatrixfine==0))=0; % Is actually zero
    clear temp4b WGb
else % not using warmglow
    becareful=~(temp4==-Inf);
    temp4b=temp4;
    temp4b(~becareful)=1; % set aside
    temp4b=(sj_J.*temp4b.^ezc8_J).^ezc6_J;
    temp4(becareful)=temp4b(becareful);
    temp4(EV==0)=0;
    becareful=~(temp4interp==-Inf);
    temp4b=temp4interp;
    temp4b(~becareful)=1; % set aside
    temp4b=(sj_J.*temp4b.^ezc8_J).^ezc6_J;
    temp4interp(becareful)=temp4b(becareful);
    temp4interp(EVinterp==0)=0;
    clear temp4b
end
% The terminal age has no continuation: its certainty equivalent is the warm-glow term alone, exactly the
% terminal formula of the slowOLG raw (the formula above, with its zero expectation, can differ from it:
% 0^ezc8 is Inf when ezc8<0, and sj(N_j)*Inf is NaN when sj(N_j)=0)
if warmglow==1
    WGterminal=WGmatrixfine(:,1,N_j);
    becareful=(WGterminal==0);
    WGterminal(isfinite(WGterminal))=((1-sj(N_j))*WGterminal(isfinite(WGterminal)).^ezc8(N_j)).^ezc6(N_j);
    WGterminal(becareful)=0;
    temp4interp(:,1,N_j,:)=WGterminal.*ones(1,1,1,N_z);
    temp4(:,1,N_j,:)=WGterminal(1:(n2short+1):end).*ones(1,1,1,N_z); % coarse-grid subset
else
    temp4interp(:,1,N_j,:)=0;
    temp4(:,1,N_j,:)=0;
end
clear EV EVinterp becareful

% Discounted over the whole grid and THEN gathered, as the unrestricted EZ GI1 raw
DiscountedEV=(ezc3*reshape(DiscountFactor_J,[1,1,N_j])).*temp4;             % [N_a,1,N_j,N_z]
DiscountedEVinterp=(ezc3*reshape(DiscountFactor_J,[1,1,N_j])).*temp4interp; % [n2aprime,1,N_j,N_z]
clear temp4 temp4interp

Policy=zeros(3,N_a,N_j,N_z,'gpuArray'); % lower coarse point, L2, L2flag

%% The window, in COARSE grid points, for every (a,j,z) at once
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_a,N_j,N_z], a coarse index

%% LAYER 1: the coarse max over the window, on the COARSE discounted certainty-equivalent term
coarseindexes=shiftdim(loweredge,-1)+(0:1:2*n)'; % [nC,N_a,N_j,N_z], coarse indices
% Level 2 of the no-d creator is [N_aprime,N_a,N_j,N_z] with N_aprime=size(aprime_grid,1), unfolded
ReturnMatrix1=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn, n_z, N_j, a_grid(coarseindexes), a_grid, z_gridvals_J, ReturnFnParamsAgeMatrix,2);

% Modify the Return Function appropriately for Epstein-Zin Preferences
becareful=isfinite(ReturnMatrix1); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
ReturnMatrixnonfinite=ReturnMatrix1(~becareful);
ReturnMatrix1(~becareful)=1; % set aside
ReturnMatrix1=ReturnMatrix1.^ezc2_J; % age is dim 3 of the window
ReturnMatrix1(~becareful)=ReturnMatrixnonfinite;
clear ReturnMatrixnonfinite

coarsejz=coarseindexes+N_a*jind+N_a*N_j*zind; % linear index into DiscountedEV, [N_a,1,N_j,N_z]
entireRHS1=ezc1_J.*ReturnMatrix1+DiscountedEV(coarsejz); % [nC,N_a,N_j,N_z]
clear ReturnMatrix1

entireRHS1(isnan(entireRHS1))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
temp5=~(entireRHS1==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
entireRHS1(~temp5)=1; % set aside
entireRHS1=entireRHS1.^ezc7_J;
entireRHS1(~temp5)=-Inf;

[~,maxindex1]=max(entireRHS1,[],1);
clear entireRHS1 temp5
% COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
coarseoffset=reshape(maxindex1,[N_a,N_j,N_z])-1; % 0 to 2n, steps above loweredge

% The new reference: THE LAYER 1 ANSWER, taken straight, UNCLAMPED
aprimeReferencePolicyNew=shiftdim(loweredge+coarseoffset,-1); % [1,N_a,N_j,N_z]

%% LAYER 2: the interpolation layer, one coarse cell either side of the clamped midpoint, on the FINE
% (interpolated) discounted certainty-equivalent term
midpoint=loweredge+min(max(coarseoffset,1),2*n-1); % [N_a,N_j,N_z]
aprimeindexes=shiftdim(midpoint+(midpoint-1)*n2short,-1)+(-n2short-1:1:1+n2short)'; % [n2long,N_a,N_j,N_z]
ReturnMatrix2=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn, n_z, N_j, aprime_grid(aprimeindexes), a_grid, z_gridvals_J, ReturnFnParamsAgeMatrix,2);

% Modify the Return Function appropriately for Epstein-Zin Preferences (in place: -Inf stays -Inf, so
% the L2 flag below still reads the raw -Inf pattern)
becareful=isfinite(ReturnMatrix2); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
ReturnMatrixnonfinite=ReturnMatrix2(~becareful);
ReturnMatrix2(~becareful)=1; % set aside
ReturnMatrix2=ReturnMatrix2.^ezc2_J; % age is dim 3 of the window
ReturnMatrix2(~becareful)=ReturnMatrixnonfinite;
clear ReturnMatrixnonfinite becareful

aprimejz=aprimeindexes+n2aprime*jind+n2aprime*N_j*zind; % linear index into DiscountedEVinterp
entireRHS2=ezc1_J.*ReturnMatrix2+reshape(DiscountedEVinterp(aprimejz(:)),[n2long,N_a,N_j,N_z]);
entireRHS2(isnan(entireRHS2))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
temp5=~(entireRHS2==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
entireRHS2(~temp5)=1; % set aside
entireRHS2=entireRHS2.^ezc7_J;
entireRHS2(~temp5)=-Inf;
clear temp5

[Vtemp,maxindexL2]=max(entireRHS2,[],1); % maxindexL2 is [1,N_a,N_j,N_z]
V=reshape(Vtemp,[N_a*N_j,N_z]);

%% L2 flag, to later avoid putting weight on an aprime where the ReturnFn is -Inf
% (1=all to lower, 2=usual, 3=all to upper.) The standard GI raw's rule; computed from maxindexL2
% BEFORE the re-spelling below.
isInfLower=(ReturnMatrix2(1,     :,:,:)==-Inf);
isInfUpper=(ReturnMatrix2(n2long,:,:,:)==-Inf);
inLowerStrict=(maxindexL2>=2)         & (maxindexL2<=n2short+1);
inUpperStrict=(maxindexL2>=n2short+3) & (maxindexL2<=n2long-1);
Policy(3,:,:,:)=2+(inLowerStrict & isInfLower)-(inUpperStrict & isInfUpper);

%% Policy, re-spelled from (midpoint, offset within n2long) to (lower coarse point, L2)
% The standard GI raw's lines. L2 ends up in [1,n2short+2].
adjust=(maxindexL2<1+n2short+1);
Policy(1,:,:,:)=shiftdim(midpoint,-1)-adjust;
Policy(2,:,:,:)=adjust.*maxindexL2+(1-adjust).*(maxindexL2-n2short-1);

% NOTE ON THE ENCODING, inherited from the standard GI raw. The window's top fine point can come out
% as either (loweredge+2n, 1) or (loweredge+2n-1, n2short+2). Anything comparing these raws must use
% the DECODED fine point (1+n2short)*(P1-1)+L2 and never the raw channels.

end
