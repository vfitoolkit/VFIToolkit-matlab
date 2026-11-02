function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_GI1_noz_e_raw(V,n_d,n_a,n_e,N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Epstein-Zin, one period of the FHorz transition path, fastOLG (every age at once), with the grid
% interpolation layer. The V input is next period value fn, the V output is this period.
% The Epstein-Zin graft of ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI1_noz_e_raw. It does, vectorised
% over the age dimension, exactly what the slowOLG raw ValueFnIter_FHorz_TPath_SingleStep_EZ_GI1_noz_e_raw
% does age by age. See ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_noz_e_raw for the graft. The transform
% (ezc4*V')^ezc5 comes before the expectation over e'; next period's V has (a,j) stacked in its rows, so
% the age exponent is laid along the rows there. As in the slowOLG raw, the interpolation acts on
% EV=E[(ezc4*V')^ezc5], BEFORE the certainty-equivalent power ^ezc6, which is then applied pointwise on
% both the coarse and the fine grid; the warm-glow is evaluated exactly on the fine grid, and the coarse
% pass uses its strided subset.
% The age dimension is dim 4 of the first layer's return matrix, [d,aprime,a,j,e], and dim 3 of the
% second layer's, [d & aprime,a,j,e], so each has its own shaping of the age-dependent constants.
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,e), rather than standard (a,e,j)
% V is (a,j)-by-e

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

Policy=zeros(4,N_a,N_j,N_e,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z (d, midpoint, aprimeL2ind, L2flag)
% THE THIRD OUTPUT is for vfoptions.localsearch, on a SEED sweep (vfoptions.localsearchseed, set by
% Step1): the COARSE aprime argmax per (d,a,j,e), [N_d,N_a,N_j,N_e] (the order of Policy), which is this
% raw's own midpoint. It is the unrestricted version of exactly what the restricted GI raw's first layer
% produces, so the reference means one thing whichever sweep built it. midpoint is clamped to
% [2,N_a-1], which is harmless for a window centre: loweredge slides. Only built when asked for, as with
% a d it is N_d times Policy.
if nargout>2
    aprimeReferencePolicyNew=zeros(N_d,N_a,N_j,N_e,'gpuArray');
end

e_gridvals_J=shiftdim(e_gridvals_J,-3); % [1,1,1,N_j,N_e,l_e]

%% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

jind=shiftdim(gpuArray(0:1:N_j-1),-2);
aBind=gpuArray(0:1:N_a-1);
jBind=shiftdim(gpuArray(0:1:N_j-1),-1);
eBind=shiftdim(gpuArray(0:1:N_e-1),-2);

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_e (note: N_aprime is just equal to N_a)

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

%% The age-dependent Epstein-Zin constants, along the age dimension: dim 3 (the expectation, the second
% layer), dim 4 (the first layer)
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
ezc1_J4=shiftdim(ezc1_J,-1); % [1,1,1,N_j] (a scalar is unchanged)
ezc2_J4=shiftdim(ezc2_J,-1);
ezc7_J4=shiftdim(ezc7_J,-1);

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

Vnext=V(N_a+1:end,:); % next period's value fn at ages 2..N_j, (a,j)-by-e: the continuation of ages 1..N_j-1
% Part of Epstein-Zin is before taking expectation
ezc5_rows=repelem(ezc5(1:N_j-1),N_a,1); % the age exponent, laid along the (a,j) rows
Vnextfinite=isfinite(Vnext);
temp=ezc4*Vnext;
temp(~Vnextfinite)=1; % set aside, the power is for the finite entries
temp=temp.^ezc5_rows;
temp(~Vnextfinite)=Vnext(~Vnextfinite);
temp(Vnext==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf
% pi_e_J is (a,j)-by-e
EV=[sum(temp.*pi_e_J(N_a+1:end,:),2); zeros(N_a,1,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_e_J to create expectations
EV=reshape(EV,[N_a,1,N_j]); % (aprime,1,j), 2nd dim will be autofilled with a
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

DiscountedEV=(ezc3*reshape(DiscountFactor_J,[1,1,N_j])).*temp4;
DiscountedEV=repelem(shiftdim(DiscountedEV,-1),N_d,1,1); % [d,aprime,1,j]

DiscountedEVinterp=(ezc3*reshape(DiscountFactor_J,[1,1,N_j])).*temp4interp;
DiscountedEVinterp=repelem(shiftdim(DiscountedEVinterp,-1),N_d,1,1); % [d,aprime,1,j]
clear temp4 temp4interp


if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, n_e, N_j, d_gridvals, a_grid, a_grid, e_gridvals_J, ReturnFnParamsAgeMatrix,1);
    % fastOLG: ReturnMatrix is [d,aprime,a,j,z]

    % Modify the Return Function appropriately for Epstein-Zin Preferences
    becareful=isfinite(ReturnMatrix); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
    ReturnMatrixnonfinite=ReturnMatrix(~becareful);
    ReturnMatrix(~becareful)=1; % set aside
    ReturnMatrix=ReturnMatrix.^ezc2_J4; % in-place: ReturnMatrix now holds what the slowOLG raw calls temp2
    ReturnMatrix(~becareful)=ReturnMatrixnonfinite;
    clear ReturnMatrixnonfinite

    entireRHS=ezc1_J4.*ReturnMatrix+DiscountedEV; %  [d,aprime,a,j,z]
    clear ReturnMatrix

    entireRHS(isnan(entireRHS))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
    temp5=~(entireRHS==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
    entireRHS(~temp5)=1; % set aside
    entireRHS=entireRHS.^ezc7_J4;
    entireRHS(~temp5)=-Inf;

    % First, we want aprime conditional on (d,1,a,j)
    [~,maxindex1]=max(entireRHS,[],2);
    clear entireRHS temp5

    % Turn this into the 'midpoint'
    midpoint=max(min(maxindex1,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    if nargout>2
        aprimeReferencePolicyNew=reshape(midpoint,[N_d,N_a,N_j,N_e]); % seed reference, see top
    end
    % midpoint is n_d-by-1-by-n_a-by-N_j-by-n_e
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
    % aprime possibilities are n_d-by-n2long-by-n_a-by-N_j-by-n_e
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn,n_d,n_e,N_j,d_gridvals,aprime_grid(aprimeindexes),a_grid, e_gridvals_J, ReturnFnParamsAgeMatrix,2);
    % [d & aprime,a,j,e]

    % Modify the Return Function appropriately for Epstein-Zin Preferences (in place: -Inf stays -Inf, so
    % the L2 flag below still reads the raw -Inf pattern)
    becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
    ReturnMatrixnonfinite=ReturnMatrix_ii(~becareful);
    ReturnMatrix_ii(~becareful)=1; % set aside
    ReturnMatrix_ii=ReturnMatrix_ii.^ezc2_J;
    ReturnMatrix_ii(~becareful)=ReturnMatrixnonfinite;
    clear ReturnMatrixnonfinite

    daprimej=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*n2aprime*jind;
    entireRHS_ii=ezc1_J.*ReturnMatrix_ii+reshape(DiscountedEVinterp(daprimej(:)),[N_d*n2long,N_a,N_j,N_e]);
    entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
    temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
    entireRHS_ii(~temp5)=1; % set aside
    entireRHS_ii=entireRHS_ii.^ezc7_J;
    entireRHS_ii(~temp5)=-Inf;
    [V,maxindexL2]=max(entireRHS_ii,[],1);
    V=reshape(V,[N_a*N_j,N_e]);
    d_ind=rem(maxindexL2-1,N_d)+1;
    allind=d_ind+N_d*aBind+N_d*N_a*jBind+N_d*N_a*N_j*eBind; % midpoint is n_d-by-1-by-n_a-by-N_j-by-n_e
    Policy(1,:,:,:)=d_ind; % d
    Policy(2,:,:,:)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
    Policy(3,:,:,:)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
    L2offset=ceil(maxindexL2/N_d);
    linidx_lower=d_ind                  +N_d*n2long*aBind+N_d*n2long*N_a*jBind+N_d*n2long*N_a*N_j*eBind;
    linidx_upper=d_ind+N_d*(n2long-1)   +N_d*n2long*aBind+N_d*n2long*N_a*jBind+N_d*n2long*N_a*N_j*eBind;
    isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
    isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
    inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
    inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
    Policy(4,:,:,:)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);

elseif vfoptions.lowmemory==1

    special_n_e=ones(1,length(n_e));
    V=zeros(N_a*N_j,N_e,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,1,:,e_c,:); % z_gridvals_J has shape (j,prod(n_z),l_z) for fastOLG

        ReturnMatrix_e=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, special_n_e, N_j, d_gridvals, a_grid, a_grid, e_vals, ReturnFnParamsAgeMatrix,1);
        % fastOLG: ReturnMatrix is [d,aprime,a,j]

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=isfinite(ReturnMatrix_e); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
        ReturnMatrixnonfinite=ReturnMatrix_e(~becareful);
        ReturnMatrix_e(~becareful)=1; % set aside
        ReturnMatrix_e=ReturnMatrix_e.^ezc2_J4;
        ReturnMatrix_e(~becareful)=ReturnMatrixnonfinite;

        entireRHS_e=ezc1_J4.*ReturnMatrix_e+DiscountedEV; %(d,aprime)-by-(a,j)

        entireRHS_e(isnan(entireRHS_e))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
        temp5=~(entireRHS_e==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
        entireRHS_e(~temp5)=1; % set aside
        entireRHS_e=entireRHS_e.^ezc7_J4;
        entireRHS_e(~temp5)=-Inf;

        % First, we want aprime conditional on (d,1,a,j)
        [~,maxindex1]=max(entireRHS_e,[],2);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex1,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        if nargout>2
            aprimeReferencePolicyNew(:,:,:,e_c)=reshape(midpoint,[N_d,N_a,N_j]); % seed reference, see top
        end
        % midpoint is n_d-by-1-by-n_a-by-N_j
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-N_j
        ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn,n_d,special_n_e,N_j,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_vals,ReturnFnParamsAgeMatrix,2);

        % Modify the Return Function appropriately for Epstein-Zin Preferences (in place: -Inf stays -Inf, so
        % the L2 flag below still reads the raw -Inf pattern)
        becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
        ReturnMatrixnonfinite=ReturnMatrix_ii(~becareful);
        ReturnMatrix_ii(~becareful)=1; % set aside
        ReturnMatrix_ii=ReturnMatrix_ii.^ezc2_J;
        ReturnMatrix_ii(~becareful)=ReturnMatrixnonfinite;

        daprimej=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*n2aprime*jind;
        entireRHS_ii=ezc1_J.*ReturnMatrix_ii+reshape(DiscountedEVinterp(daprimej(:)),[N_d*n2long,N_a,N_j]);
        entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
        temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
        entireRHS_ii(~temp5)=1; % set aside
        entireRHS_ii=entireRHS_ii.^ezc7_J;
        entireRHS_ii(~temp5)=-Inf;
        [Vtemp,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,e_c)=reshape(Vtemp,[N_a*N_j,1]);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aBind+N_d*N_a*jBind; % midpoint is n_d-by-1-by-n_a-by-N_j
        Policy(1,:,:,e_c)=d_ind; % d
        Policy(2,:,:,e_c)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
        Policy(3,:,:,e_c)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        L2offset=ceil(maxindexL2/N_d);
        linidx_lower=d_ind                  +N_d*n2long*aBind+N_d*n2long*N_a*jBind;
        linidx_upper=d_ind+N_d*(n2long-1)   +N_d*n2long*aBind+N_d*n2long*N_a*jBind;
        isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
        inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
        Policy(4,:,:,e_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);
    end
end



%% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:,:)=Policy(2,:,:,:)-adjust; % lower grid point
Policy(3,:,:,:)=adjust.*Policy(3,:,:,:)+(1-adjust).*(Policy(3,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

%% fastOLG with e, so need to output to take certain shapes
% V=reshape(V,[N_a*N_j,N_e]);
% Policy=reshape(Policy,[size(Policy,1),N_a,N_j,N_e]);



end
