function [V, Policy, aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_GI1_nod_noz_raw(V,n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames,vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Epstein-Zin, one period of the FHorz transition path, fastOLG (every age at once), with the grid
% interpolation layer. The V input is next period value fn, the V output is this period.
% The Epstein-Zin graft of ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI1_nod_noz_raw. It does, vectorised
% over the age dimension, exactly what the slowOLG raw ValueFnIter_FHorz_TPath_SingleStep_EZ_GI1_nod_noz_raw
% does age by age. See ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_nod_noz_raw for the graft. Without
% shocks there is no expectation: the transformed continuation (ezc4*V')^ezc5 is interpolated, BEFORE the
% certainty-equivalent power ^ezc6, which is then applied pointwise on both the coarse and the fine grid;
% the warm-glow is evaluated exactly on the fine grid, and the coarse pass uses its strided subset.
% The age dimension is dim 3 of both layers' return matrices, [aprime,a,j], so the age-dependent
% constants are laid along dim 3 throughout.
% THE THIRD OUTPUT is for vfoptions.localsearch, on a SEED sweep (vfoptions.localsearchseed, set by
% Step1): the COARSE aprime argmax per (a,j), [1,N_a,N_j], which is this raw's own midpoint. See
% slowOLG/GridInterpLayer/ValueFnIter_FHorz_TPath_SingleStep_GI1_nod_raw. Only built when asked for.

N_a=prod(n_a);

Policy=zeros(3,N_a,N_j,'gpuArray'); % first dim indexes the optimal choice for aprime (layer 1, layer 2, and L2 flag)

%%

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

jind=shiftdim(gpuArray(0:1:N_j-1),-1);

%% First, create the big 'next period (of transition path) expected value fn.

% VfastOLG will be N_d*N_aprime by N_a*N_j (note: N_aprime is just equal to N_a)

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

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

Vnext=reshape(V(:,2:end),[N_a,1,N_j-1]); % next period's value fn at ages 2..N_j: the continuation of ages 1..N_j-1
% Part of Epstein-Zin is before taking expectation (without shocks there is no expectation)
Vnextfinite=isfinite(Vnext);
temp=ezc4*Vnext;
temp(~Vnextfinite)=1; % set aside, the power is for the finite entries
temp=temp.^ezc5_J(1,1,1:N_j-1);
temp(~Vnextfinite)=Vnext(~Vnextfinite);
temp(Vnext==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf
EV=zeros(N_a,1,N_j,'gpuArray');
EV(:,1,1:N_j-1)=temp;
clear Vnext Vnextfinite temp
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
    temp4interp(:,1,N_j)=WGterminal;
    temp4(:,1,N_j)=WGterminal(1:(n2short+1):end); % coarse-grid subset
else
    temp4interp(:,1,N_j)=0;
    temp4(:,1,N_j)=0;
end
clear EV EVinterp becareful

ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_noz(ReturnFn, N_j, a_grid, a_grid, ReturnFnParamsAgeMatrix,2);

% Modify the Return Function appropriately for Epstein-Zin Preferences
becareful=isfinite(ReturnMatrix); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
ReturnMatrixnonfinite=ReturnMatrix(~becareful);
ReturnMatrix(~becareful)=1; % set aside
ReturnMatrix=ReturnMatrix.^ezc2_J; % in-place: ReturnMatrix now holds what the slowOLG raw calls temp2
ReturnMatrix(~becareful)=ReturnMatrixnonfinite;
clear ReturnMatrixnonfinite

entireRHS=ezc1_J.*ReturnMatrix+(ezc3*reshape(DiscountFactor_J,[1,1,N_j])).*temp4; % [N_aprime,N_a,N_j]
clear ReturnMatrix

entireRHS(isnan(entireRHS))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
temp5=~(entireRHS==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
entireRHS(~temp5)=1; % set aside
entireRHS=entireRHS.^ezc7_J;
entireRHS(~temp5)=-Inf;

%Calc the max and it's index
[~,maxindex]=max(entireRHS,[],1);
clear entireRHS temp5

% Turn this into the 'midpoint'
midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
if nargout>2
    aprimeReferencePolicyNew=reshape(midpoint,[1,N_a,N_j]); % seed reference, see top
end
% midpoint is 1-by-n_a-by-N_j
aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
% aprime possibilities are n2long-by-n_a-by-N_j
ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_noz(ReturnFn,N_j,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsAgeMatrix,2);

% Modify the Return Function appropriately for Epstein-Zin Preferences (in place: -Inf stays -Inf, so
% the L2 flag below still reads the raw -Inf pattern)
becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
ReturnMatrixnonfinite=ReturnMatrix_ii(~becareful);
ReturnMatrix_ii(~becareful)=1; % set aside
ReturnMatrix_ii=ReturnMatrix_ii.^ezc2_J;
ReturnMatrix_ii(~becareful)=ReturnMatrixnonfinite;
clear ReturnMatrixnonfinite

aprimej=aprimeindexes+n2aprime*jind;
entireRHS_ii=ezc1_J.*ReturnMatrix_ii+(ezc3*reshape(DiscountFactor_J,[1,1,N_j])).*reshape(temp4interp(aprimej(:)),[n2long,N_a,N_j]);
entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
entireRHS_ii(~temp5)=1; % set aside
entireRHS_ii=entireRHS_ii.^ezc7_J;
entireRHS_ii(~temp5)=-Inf;
[Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
V=shiftdim(Vtempii,1);
Policy(1,:,:)=shiftdim(squeeze(midpoint),-1); % midpoint
Policy(2,:,:)=shiftdim(maxindexL2,-1); % aprimeL2ind

% L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
isInfLower    = (ReturnMatrix_ii(1,     :,:) == -Inf);
isInfUpper    = (ReturnMatrix_ii(n2long,:,:) == -Inf);
inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
Policy(3,:,:) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);

%% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(1,:,:)=Policy(1,:,:)-adjust; % lower grid point
Policy(2,:,:)=adjust.*Policy(2,:,:)+(1-adjust).*(Policy(2,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)



end
