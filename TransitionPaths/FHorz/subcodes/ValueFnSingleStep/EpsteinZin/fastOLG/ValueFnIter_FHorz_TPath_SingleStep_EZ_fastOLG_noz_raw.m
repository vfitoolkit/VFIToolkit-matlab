function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_noz_raw(V,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Epstein-Zin, one period of the FHorz transition path, fastOLG (every age at once). The V input is next
% period value fn, the V output is this period.
% The Epstein-Zin graft of ValueFnIter_FHorz_TPath_SingleStep_fastOLG_noz_raw. It does, vectorised over the
% age dimension, exactly what the slowOLG raw ValueFnIter_FHorz_TPath_SingleStep_EZ_noz_raw does age by age.
% See ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_raw for the graft. Without shocks there is no
% expectation (the certainty equivalent of a sure thing is that thing): the transformed continuation
% goes straight into the certainty-equivalent formula.
% fastOLG just means parallelize over "age" (j)
% V is (a,j)

N_d=prod(n_d);
N_a=prod(n_a);

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

%% First, create the big 'next period (of transition path) value fn.
% VfastOLG will be N_d*N_aprime by N_a*N_j (note: N_aprime is just equal to N_a)
Vnext=reshape(V(:,2:end),[N_a,1,N_j-1]); % next period's value fn at ages 2..N_j: the continuation of ages 1..N_j-1
% Part of Epstein-Zin is before taking expectation
Vnextfinite=isfinite(Vnext);
temp=ezc4*Vnext;
temp(~Vnextfinite)=1; % set aside, the power is for the finite entries
temp=temp.^ezc5_J(1,1,1:N_j-1);
temp(~Vnextfinite)=Vnext(~Vnextfinite);
temp(Vnext==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf
EV=zeros(N_a,1,N_j,'gpuArray');
EV(:,1,1:N_j-1)=temp; % Leave N_j as zeros (no continuation in the final period)
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
% terminal formula of the slowOLG raw (the formula above, with its zero expectation, can differ from it:
% 0^ezc8 is Inf when ezc8<0, and sj(N_j)*Inf is NaN when sj(N_j)=0)
if warmglow==1
    WGterminal=WGmatrix(:,1,N_j);
    becareful=(WGterminal==0);
    WGterminal(isfinite(WGterminal))=((1-sj(N_j))*WGterminal(isfinite(WGterminal)).^ezc8(N_j)).^ezc6(N_j);
    WGterminal(becareful)=0;
    temp4(:,1,N_j)=WGterminal;
else
    temp4(:,1,N_j)=0;
end
DiscountedEV=repelem((ezc3*reshape(DiscountFactor_J,[1,1,N_j])).*temp4,N_d,1,1); % [N_d*N_aprime,1,N_j]
clear EV temp4 becareful

ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_noz(ReturnFn, n_d, n_a, N_j, d_gridvals, a_grid, a_grid, ReturnFnParamsAgeMatrix);
% fastOLG: ReturnMatrix is [d & aprime,a,j]

% Modify the Return Function appropriately for Epstein-Zin Preferences
becareful=isfinite(ReturnMatrix); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
ReturnMatrixnonfinite=ReturnMatrix(~becareful);
ReturnMatrix(~becareful)=1; % set aside
ReturnMatrix=ReturnMatrix.^ezc2_J; % in-place: ReturnMatrix now holds what the slowOLG raw calls temp2
ReturnMatrix(~becareful)=ReturnMatrixnonfinite;
clear ReturnMatrixnonfinite

entireRHS=ezc1_J.*ReturnMatrix+DiscountedEV; %(d,aprime)-by-(a,j)
clear ReturnMatrix

entireRHS(isnan(entireRHS))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
temp5=~(entireRHS==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
entireRHS(~temp5)=1; % set aside
entireRHS=entireRHS.^ezc7_J;
entireRHS(~temp5)=-Inf;

% Calc the max and it's index
[V,Policy]=max(entireRHS,[],1);

V=squeeze(V); % V is over (a,j)
Policy=squeeze(Policy); % Policy is over (a,j)

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point

end
