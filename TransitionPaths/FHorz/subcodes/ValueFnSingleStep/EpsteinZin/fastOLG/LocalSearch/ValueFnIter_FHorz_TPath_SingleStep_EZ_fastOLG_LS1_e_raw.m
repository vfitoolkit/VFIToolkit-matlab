function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_LS1_e_raw(V,n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Epstein-Zin with local search, finite horizon, fastOLG, with z and e, with a decision variable. The V input
% is next period value fn (across all ages), the V output is this period.
% The window, the per-d handling (d searched in FULL, reference CONDITIONAL ON d, two-stage max) and the
% outputs are EXACTLY those of ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LS1_e_raw (see that raw), and the
% right-hand side on the window is EXACTLY that of ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_e_raw: the
% transform (ezc4*V')^ezc5 comes before BOTH expectations (over e', then over z'), the discounted
% certainty-equivalent term is built over the whole aprime grid, age-vectorised, with the terminal age set
% explicitly to the warm-glow term alone, then gathered on the window (the same values the unrestricted raw
% repelems over d); the return is transformed on the window only. So when the window holds the unrestricted
% argmax, V is bit-identical to the unrestricted EZ raw, and so is Policy up to the tie-breaking across d
% noted in the vNM LS raw (two names for one value).
%
% THE AGE DIMENSION of the windowed return [N_d,2n+1,N_a,N_j,N_z,N_e] is dim 4, not dim 3 as in the
% unrestricted raw, so the constants that touch the windowed arrays (ezc1, ezc2, ezc7) are shifted one
% dimension to the right there (ezcX_W). The values are unchanged.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
pi_z_J=cat(1,pi_z_J,zeros(1,N_z,N_z,'gpuArray')); % append a j=N_j slot of zeros: there is no continuation value in the final period (input pi_z_J has N_j-1 slices, the transitions from periods 1..N_j-1)
N_e=prod(n_e);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw.

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_d,N_a,N_j,N_z,N_e]);

z_gridvals_J=shiftdim(z_gridvals_J,-3); % [1,1,1,N_j,N_z,l_z]
e_gridvals_J=reshape(e_gridvals_J,[1,1,1,N_j,1,N_e,length(n_e)]); % [1,1,1,N_j,1,N_e,l_e]

jind=shiftdim(gpuArray(0:1:N_j-1),-2);
zind=shiftdim(gpuArray(0:1:N_z-1),-3);

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
% On the window the age is dim 4: [N_d,2n+1,N_a,N_j,N_z,N_e]. shiftdim(.,-1) moves [1,1,N_j] to
% [1,1,1,N_j] and leaves a scalar ezc1 a scalar.
ezc1_W=shiftdim(ezc1_J,-1);
ezc2_W=shiftdim(ezc2_J,-1);
ezc7_W=shiftdim(ezc7_J,-1);

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
Vnext=V(N_a+1:end,:,:); % next period's value fn at ages 2..N_j, (a,j)-by-z-by-e: the continuation of ages 1..N_j-1
% Part of Epstein-Zin is before taking expectation
ezc5_rows=repelem(ezc5(1:N_j-1),N_a,1); % the age exponent, laid along the (a,j) rows
Vnextfinite=isfinite(Vnext);
temp=ezc4*Vnext;
temp(~Vnextfinite)=1; % set aside, the power is for the finite entries
temp=temp.^ezc5_rows;
temp(~Vnextfinite)=Vnext(~Vnextfinite);
temp(Vnext==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf
% Expectation over e
EVpre=[sum(temp.*pi_e_J(N_a+1:end,:,:),3); zeros(N_a,N_z,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_z_J to create expectations
EVpre=reshape(EVpre,[N_a,1,N_j,N_z]);
EV=EVpre.*shiftdim(pi_z_J,-2);
EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z)
clear Vnext Vnextfinite temp EVpre

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
    temp4(:,1,N_j,:)=WGterminal.*ones(1,1,1,N_z);
else
    temp4(:,1,N_j,:)=0;
end
% The same values as the unrestricted raw's repelem(...,N_d,1,1,1), which only copies them across d
DiscountedEV=(ezc3*reshape(DiscountFactor_J,[1,1,N_j])).*temp4; % [aprime,1,j,z]
clear EV temp4 becareful

%% The window, one per (d,a,j,z,e)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n);                % [N_d,N_a,N_j,N_z,N_e]
aprimeindexes=reshape(loweredge,[N_d,1,N_a,N_j,N_z,N_e])+(0:1:2*n);   % [N_d,2n+1,N_a,N_j,N_z,N_e], d fastest

%% Return matrix on the window only, transformed for Epstein-Zin
% Level 3 returns [N_d,N_aprime,N_a,N_j,N_z,N_e] without folding d and aprime together (aprime not shifted)
ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, N_j, d_gridvals, a_grid(aprimeindexes), a_grid, z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,3);
becareful=isfinite(ReturnMatrix); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
ReturnMatrixnonfinite=ReturnMatrix(~becareful);
ReturnMatrix(~becareful)=1; % set aside
ReturnMatrix=ReturnMatrix.^ezc2_W; % age is dim 4 of the window
ReturnMatrix(~becareful)=ReturnMatrixnonfinite;
clear ReturnMatrixnonfinite

aprimejz=aprimeindexes+N_a*jind+N_a*N_j*zind; % linear index into DiscountedEV, which is [N_a,1,N_j,N_z]; e autofills
entireRHS=ezc1_W.*ReturnMatrix+DiscountedEV(aprimejz); % [N_d,2n+1,N_a,N_j,N_z,N_e]
clear ReturnMatrix

entireRHS(isnan(entireRHS))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
temp5=~(entireRHS==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
entireRHS(~temp5)=1; % set aside
entireRHS=entireRHS.^ezc7_W;
entireRHS(~temp5)=-Inf;

%% Two maxima: over aprime per d (the new reference), then over d
[Vd,offset1]=max(entireRHS,[],2);                                  % [N_d,1,N_a,N_j,N_z,N_e]
% COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
coarseoffset=reshape(offset1,[N_d,N_a,N_j,N_z,N_e])-1;             % 0 to 2n, steps above loweredge, per d
aprimeReferencePolicyNew=loweredge+coarseoffset;                   % [N_d,N_a,N_j,N_z,N_e]

[Vtemp,dstar]=max(Vd,[],1);                                        % [1,1,N_a,N_j,N_z,N_e]
V=reshape(Vtemp,[N_a*N_j,N_z,N_e]);
dstar=reshape(dstar,[1,N_a,N_j,N_z,N_e]);

% aprime at the chosen d. The lookup must use dstar and not the state index. Reshaped because with
% N_d=1 the source can be a vector, and a vector indexed by a vector takes the SOURCE's orientation.
aprimestar=reshape(aprimeReferencePolicyNew(dstar+N_d*reshape(gpuArray(0:1:N_a*N_j*N_z*N_e-1),[1,N_a,N_j,N_z,N_e])),[1,N_a,N_j,N_z,N_e]); % [1,N_a,N_j,N_z,N_e]

%% Output shape for policy, the same as the unrestricted raw: [1,N_a,N_j,N_z,N_e]
% The folded (d,aprime) index, d fastest, as the unrestricted raw's repelem(DiscountedEV,N_d,...) layout
Policy=dstar+N_d*(aprimestar-1);

end
