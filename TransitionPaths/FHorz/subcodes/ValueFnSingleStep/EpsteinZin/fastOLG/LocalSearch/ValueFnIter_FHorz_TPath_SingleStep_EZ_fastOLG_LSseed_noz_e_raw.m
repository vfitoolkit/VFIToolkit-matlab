function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_LSseed_noz_e_raw(V,n_d,n_a,n_e,N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Epstein-Zin SEED SWEEP, finite horizon, fastOLG (all ages at once), with a d, no z, with e, and without
% the grid interpolation layer. An UNRESTRICTED sweep that also returns the aprime argmax FOR EACH d,
% [N_d,N_a,N_j,N_e], (a,j,e) being the order of Policy, which is what vfoptions.localsearch needs as its
% reference.
% The sweep (per-d max, max over d, folded Policy, per-d reference, lowmemory branches, shapes) is EXACTLY
% that of ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LSseed_noz_e_raw (see that raw), and the right-hand side
% is EXACTLY that of ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_noz_e_raw: the transform (ezc4*V')^ezc5
% before the expectation over e', the certainty equivalent with the terminal age set to the warm-glow term
% alone, the return ^ezc2 with set-asides, ezc1*F^ezc2+ezc3*beta*CE, NaN to -Inf, ^ezc7 with set-asides.
% So V is bit-identical to the unrestricted EZ raw; Policy differs only in tie-breaking (lowest d, then that
% d's lowest aprime, against linear order).
%
% AGE DIMENSION: as the vNM seed raw, this one keeps the right-hand side FOLDED, [d & aprime,a,j,e], and only
% reshapes it to [N_d,N_aprime,...] for the two maxima. So every Epstein-Zin constant stays in dim 3, exactly
% as the unrestricted EZ raw lays it, and the whole right-hand side is that raw's, line for line.

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

e_gridvals_J=shiftdim(e_gridvals_J,-3); % [1,1,1,N_j,N_e,l_e]

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
% fastOLG will be N_d*N_aprime by N_a*N_j*N_e (note: N_aprime is just equal to N_a)
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
discountedEV=repelem((ezc3*reshape(DiscountFactor_J,[1,1,N_j])).*temp4,N_d,1,1); % [d & aprime, 1, j]
clear EV temp4 becareful

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc(ReturnFn, n_d, n_a, n_e, N_j, d_gridvals, a_grid, a_grid, e_gridvals_J, ReturnFnParamsAgeMatrix);
    % fastOLG: ReturnMatrix is [d*aprime,a,j,e], d fastest

    % Modify the Return Function appropriately for Epstein-Zin Preferences
    becareful=isfinite(ReturnMatrix); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
    ReturnMatrixnonfinite=ReturnMatrix(~becareful);
    ReturnMatrix(~becareful)=1; % set aside
    ReturnMatrix=ReturnMatrix.^ezc2_J; % in-place: ReturnMatrix now holds what the slowOLG raw calls temp2
    ReturnMatrix(~becareful)=ReturnMatrixnonfinite;
    clear ReturnMatrixnonfinite

    entirediscountedEV=ezc1_J.*ReturnMatrix+discountedEV; %(d,aprime)-by-(a,j,e)
    clear ReturnMatrix

    entirediscountedEV(isnan(entirediscountedEV))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
    temp5=~(entirediscountedEV==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
    entirediscountedEV(~temp5)=1; % set aside
    entirediscountedEV=entirediscountedEV.^ezc7_J;
    entirediscountedEV(~temp5)=-Inf;

    % Two maxima. The first is over aprime FOR EACH d and is the whole point of this raw.
    [Vd,offset1]=max(reshape(entirediscountedEV,[N_d,N_a,N_a,N_j,N_e]),[],2); % [N_d,1,N_a,N_j,N_e]
    % No window here, so the argmax is already a grid index in [1,N_a]
    aprimeReferencePolicyNew=reshape(offset1,[N_d,N_a,N_j,N_e]);

    [Vtemp,dstar]=max(reshape(Vd,[N_d,N_a,N_j,N_e]),[],1); % [1,N_a,N_j,N_e]
    V=reshape(Vtemp,[N_a*N_j,N_e]);
    dstar=reshape(dstar,[N_a,N_j,N_e]);

    % aprime at the chosen d. The lookup must use dstar and not the state index. Reshaped because with
    % N_d=1 the source can be a vector, and a vector indexed by a vector takes the SOURCE's orientation.
    aprimestar=reshape(aprimeReferencePolicyNew(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_j-1)+N_d*N_a*N_j*shiftdim(0:1:N_e-1,-1)),[N_a,N_j,N_e]); % [N_a,N_j,N_e]

    % The folded (d,aprime) index, d fastest, the same form as the unrestricted raw's Policy. The unrestricted
    % raw squeezes its [1,N_a,N_j,N_e]; squeeze of this reshape gives the same array.
    Policy=squeeze(reshape(dstar+N_d*(aprimestar-1),[1,N_a,N_j,N_e]));

elseif vfoptions.lowmemory==1

    special_n_e=ones(1,length(n_e));

    Policy=zeros(N_a*N_j,N_e,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z
    aprimeReferencePolicyNew=zeros(N_d,N_a,N_j,N_e,'gpuArray');

    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,1,:,e_c,:); % e_gridvals_J has shape (j,prod(n_e),l_e) for fastOLG with no z

        ReturnMatrix_e=CreateReturnFnMatrix_fastOLG_Disc(ReturnFn, n_d, n_a, special_n_e, N_j, d_gridvals, a_grid, a_grid, e_vals, ReturnFnParamsAgeMatrix);
        % fastOLG: ReturnMatrix is [d*aprime,a,j], d fastest

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=isfinite(ReturnMatrix_e); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
        ReturnMatrixnonfinite=ReturnMatrix_e(~becareful);
        ReturnMatrix_e(~becareful)=1; % set aside
        ReturnMatrix_e=ReturnMatrix_e.^ezc2_J;
        ReturnMatrix_e(~becareful)=ReturnMatrixnonfinite;

        entirediscountedEV_e=ezc1_J.*ReturnMatrix_e+discountedEV; %(d,aprime)-by-(a,j)

        entirediscountedEV_e(isnan(entirediscountedEV_e))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
        temp5=~(entirediscountedEV_e==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
        entirediscountedEV_e(~temp5)=1; % set aside
        entirediscountedEV_e=entirediscountedEV_e.^ezc7_J;
        entirediscountedEV_e(~temp5)=-Inf;

        % Two maxima, as above
        [Vd,offset1]=max(reshape(entirediscountedEV_e,[N_d,N_a,N_a,N_j]),[],2); % [N_d,1,N_a,N_j]
        aprimeRefNew_e=reshape(offset1,[N_d,N_a,N_j]);

        [Vtemp,dstar]=max(reshape(Vd,[N_d,N_a,N_j]),[],1); % [1,N_a,N_j]
        V(:,e_c)=reshape(Vtemp,[N_a*N_j,1]);
        dstar=reshape(dstar,[N_a,N_j]);

        % aprime at the chosen d. The lookup must use dstar and not the state index. Reshaped because with
        % N_d=1 the source can be a vector, and a vector indexed by a vector takes the SOURCE's orientation.
        aprimestar=reshape(aprimeRefNew_e(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_j-1)),[N_a,N_j]); % [N_a,N_j]

        Policy(:,e_c)=reshape(dstar+N_d*(aprimestar-1),[N_a*N_j,1]);
        aprimeReferencePolicyNew(:,:,:,e_c)=aprimeRefNew_e;
    end
end

% Output shape for policy, the same as the unrestricted raw
Policy=shiftdim(Policy,-1); % so first dim is just one point


end
