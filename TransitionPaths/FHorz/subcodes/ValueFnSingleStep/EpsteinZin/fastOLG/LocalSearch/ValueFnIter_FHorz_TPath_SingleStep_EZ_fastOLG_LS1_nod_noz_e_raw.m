function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_LS1_nod_noz_e_raw(V,n_a,n_e,N_j, a_grid,e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Epstein-Zin with local search, finite horizon, fastOLG, no d, no z, with e. The V input is next period
% value fn, (a,j)-by-e, the V output is this period.
% The window and outputs are EXACTLY those of ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LS1_nod_noz_e_raw
% (see that raw), and the right-hand side on the window is EXACTLY that of
% ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_nod_noz_e_raw: the transform (ezc4*V')^ezc5 comes before the
% expectation over e', the discounted certainty-equivalent term is built over the whole aprime grid,
% age-vectorised, with the terminal age set explicitly to the warm-glow term alone, then gathered on the
% window; the return is transformed on the window only. So when the window holds the unrestricted argmax, V
% and Policy are bit-identical to the unrestricted EZ raw.
%
% THE AGE DIMENSION of the windowed return [2n+1,N_a,N_j,N_e] is dim 3, as in the unrestricted raw, so the
% ezcX_J are used on the window as they are.

N_a=prod(n_a);
N_e=prod(n_e);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw, because a
% window fits in memory whatever the full sweep needs.

n=vfoptions.nlocalsearch;
nC=2*n+1; % the window

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_a,N_j,N_e]);

e_gridvals_J=shiftdim(e_gridvals_J,-2);

jind=shiftdim(gpuArray(0:1:N_j-1),-1);

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

%% The warm-glow of bequests, (aprime,1,j)
if warmglow==1
    WGmatrixraw=zeros(N_a,1,N_j,'gpuArray');
    for jj=1:N_j
        WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
        WGmatrixraw(:,1,jj)=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a, a_grid, WGParamsVec);
    end
    WGfinite=isfinite(WGmatrixraw);
    WGmatrix=ezc4*WGmatrixraw;
    WGmatrix(~WGfinite)=1; % set aside, the power is for the finite entries
    WGmatrix=WGmatrix.^ezc5_J;
    WGmatrix(~WGfinite)=WGmatrixraw(~WGfinite);
    WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
    WGmatrix(WGmatrixraw==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf
end

%% First, create the big 'next period (of transition path) expected value fn.
Vnext=V(N_a+1:end,:); % next period's value fn at ages 2..N_j, (a,j)-by-e: the continuation of ages 1..N_j-1
% Part of Epstein-Zin is before taking expectation
ezc5_rows=repelem(ezc5(1:N_j-1),N_a,1); % the age exponent, laid along the (a,j) rows
Vnextfinite=isfinite(Vnext);
temp=ezc4*Vnext;
temp(~Vnextfinite)=1; % set aside, the power is for the finite entries
temp=temp.^ezc5_rows;
temp(~Vnextfinite)=Vnext(~Vnextfinite);
temp(Vnext==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf
% Expectation over e
EV=[sum(temp.*pi_e_J(N_a+1:end,:),2); zeros(N_a,1,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_e_J to create expectations
EV=reshape(EV,[N_a,1,N_j]);
clear Vnext Vnextfinite temp

%% The certainty equivalent, age by age
temp4=EV;
if warmglow==1
    becareful=(~(temp4==-Inf)) & isfinite(WGmatrix); % both are finite
    temp4b=temp4;
    temp4b(~becareful)=1; % set aside
    WGb=WGmatrix;
    WGb(~isfinite(WGb))=1; % set aside
    temp4b=(sj_J.*temp4b.^ezc8_J+(1-sj_J).*WGb.^ezc8_J).^ezc6_J;
    temp4(becareful)=temp4b(becareful);
    temp4((EV==0)&(WGmatrix==0))=0; % Is actually zero
    clear temp4b WGb
else % not using warmglow
    becareful=~(temp4==-Inf);
    temp4b=temp4;
    temp4b(~becareful)=1; % set aside
    temp4b=(sj_J.*temp4b.^ezc8_J).^ezc6_J;
    temp4(becareful)=temp4b(becareful);
    temp4(EV==0)=0;
    clear temp4b
end
% The terminal age has no continuation: its certainty equivalent is the warm-glow term alone, exactly the
% terminal formula of the slowOLG raw (0^ezc8 is Inf when ezc8<0, and sj(N_j)*Inf is NaN when sj(N_j)=0)
if warmglow==1
    WGterminal=WGmatrix(:,1,N_j);
    becareful=(WGterminal==0);
    WGterminal(isfinite(WGterminal))=((1-sj(N_j))*WGterminal(isfinite(WGterminal)).^ezc8(N_j)).^ezc6(N_j);
    WGterminal(becareful)=0;
    temp4(:,1,N_j)=WGterminal;
else
    temp4(:,1,N_j)=0;
end
discountedEV=(ezc3*reshape(DiscountFactor_J,[1,1,N_j])).*temp4; % [N_aprime,1,N_j]
clear EV temp4 becareful

%% The window, for every (a,j,e) at once
% Requires N_a>=2n+1, which is checked where nlocalsearch is validated.
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_a,N_j,N_e]
aprimeindexes=shiftdim(loweredge,-1)+(0:1:2*n)';     % [nC,N_a,N_j,N_e]

%% Return matrix on the window only, transformed for Epstein-Zin
% Level 2 takes aprime as [nC,N_a,N_j,N_e] and returns [nC,N_a,N_j,N_e]
ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn,n_e,N_j,a_grid(aprimeindexes),a_grid,e_gridvals_J,ReturnFnParamsAgeMatrix,2);
becareful=isfinite(ReturnMatrix); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
ReturnMatrixnonfinite=ReturnMatrix(~becareful);
ReturnMatrix(~becareful)=1; % set aside
ReturnMatrix=ReturnMatrix.^ezc2_J; % age is dim 3 of the window too
ReturnMatrix(~becareful)=ReturnMatrixnonfinite;
clear ReturnMatrixnonfinite

aprimej=aprimeindexes+N_a*jind; % linear index into discountedEV, which is [N_a,1,N_j]
entireRHS=ezc1_J.*ReturnMatrix+reshape(discountedEV(aprimej(:)),[nC,N_a,N_j,N_e]); % [nC,N_a,N_j,N_e]
clear ReturnMatrix

entireRHS(isnan(entireRHS))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
temp5=~(entireRHS==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
entireRHS(~temp5)=1; % set aside
entireRHS=entireRHS.^ezc7_J;
entireRHS(~temp5)=-Inf;

[Vtemp,maxindex]=max(entireRHS,[],1);
V=reshape(Vtemp,[N_a*N_j,N_e]);

% COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
coarseoffset=reshape(maxindex,[N_a,N_j,N_e])-1; % 0 to 2n, steps above loweredge
aprimeindexstar=loweredge+coarseoffset;

%% Output shape for policy, the same as the unrestricted raw
Policy=reshape(aprimeindexstar,[1,N_a,N_j,N_e]);
% With no d the new reference is just the chosen aprime. Returned all the same, so that a caller
% never has to branch on whether the model has a d.
aprimeReferencePolicyNew=Policy;

end
