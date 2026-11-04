function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_LSseed_e_raw(V,n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Epstein-Zin SEED SWEEP, finite horizon, fastOLG (all ages at once), with z and e, with a d and without
% the grid interpolation layer. An UNRESTRICTED sweep that also returns the aprime argmax FOR EACH d,
% [N_d,N_a,N_j,N_z,N_e], which is what vfoptions.localsearch needs as its reference.
% The sweep (unfolded d and aprime, per-d max, max over d, folded Policy, per-d reference, lowmemory
% branches, shapes) is EXACTLY that of ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LSseed_e_raw (see that
% raw), and the right-hand side is EXACTLY that of ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_e_raw: the
% transform (ezc4*V')^ezc5 before BOTH expectations (over e', then over z'), the certainty equivalent with
% the terminal age set to the warm-glow term alone, the return ^ezc2 with set-asides, ezc1*F^ezc2+ezc3*beta*CE,
% NaN to -Inf, ^ezc7 with set-asides. So V is bit-identical to the unrestricted EZ raw; Policy differs only
% in tie-breaking (lowest d, then that d's lowest aprime, against linear order).
%
% AGE DIMENSION: the certainty-equivalent part works on (aprime,1,j,z), age in dim 3, as the unrestricted
% raw. The return is UNFOLDED to (d,aprime,a,j,z,e), age in dim 4, so the constants that meet it (ezc1,
% ezc2, ezc7) are laid along dim 4, and DiscountedEV is repelem'd over d and then unfolded to
% (d,aprime,1,j,z), as in the vNM seed raw.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
pi_z_J=cat(1,pi_z_J,zeros(1,N_z,N_z,'gpuArray')); % append a j=N_j slot of zeros: there is no continuation value in the final period (input pi_z_J has N_j-1 slices, the transitions from periods 1..N_j-1)
N_e=prod(n_e);

z_gridvals_J=shiftdim(z_gridvals_J,-3);
e_gridvals_J=reshape(e_gridvals_J,[1,1,1,N_j,1,N_e,length(n_e)]);

Policy=zeros(1,N_a,N_j,N_z,N_e,'gpuArray');
aprimeReferencePolicyNew=zeros(N_d,N_a,N_j,N_z,N_e,'gpuArray');

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

%% The age-dependent Epstein-Zin constants
% Those of the certainty equivalent along dim 3, as (aprime,1,j,z); those that meet the unfolded return
% (d,aprime,a,j,z,e) along dim 4
ezc2_J=reshape(ezc2,[1,1,1,N_j]); % dim 4: the unfolded return
ezc5_J=reshape(ezc5,[1,1,N_j]);
ezc6_J=reshape(ezc6,[1,1,N_j]);
ezc7_J=reshape(ezc7,[1,1,1,N_j]); % dim 4: the unfolded return
ezc8_J=reshape(ezc8,[1,1,N_j]);
sj_J=reshape(sj,[1,1,N_j]);
if vfoptions.EZoneminusbeta==1
    ezc1_J=reshape(1-DiscountFactor_J,[1,1,1,N_j]); % (1-beta)* on this period's return, age by age; dim 4: the unfolded return
elseif vfoptions.EZoneminusbeta==2
    ezc1_J=reshape(1-sj(:).*DiscountFactor_J,[1,1,1,N_j]); % (1-sj*beta)* on this period's return, age by age; dim 4: the unfolded return
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
EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a
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
% terminal formula of the slowOLG raw (the formula above, with its zero expectation, can differ from it:
% 0^ezc8 is Inf when ezc8<0, and sj(N_j)*Inf is NaN when sj(N_j)=0)
if warmglow==1
    WGterminal=WGmatrix(:,1,N_j);
    becareful=(WGterminal==0);
    WGterminal(isfinite(WGterminal))=((1-sj(N_j))*WGterminal(isfinite(WGterminal)).^ezc8(N_j)).^ezc6(N_j);
    WGterminal(becareful)=0;
    temp4(:,1,N_j,:)=WGterminal.*ones(1,1,1,N_z);
else
    temp4(:,1,N_j,:)=0;
end
DiscountedEV=repelem((ezc3*reshape(DiscountFactor_J,[1,1,N_j])).*temp4,N_d,1,1,1); % [N_d*N_aprime,1,N_j,N_z]
clear EV temp4 becareful
% Unfolded, d fastest, as the per-d max needs it. The repelem copies each value across d, so this is
% the same numbers in the same places as the unrestricted raw adds.
DiscountedEV=reshape(DiscountedEV,[N_d,N_a,1,N_j,N_z]);

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_e(ReturnFn, n_d, n_a, n_z, n_e, N_j, d_gridvals, a_grid, a_grid, z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix);
    % fastOLG: ReturnMatrix is [d*aprime,a,j,z,e], d fastest
    ReturnMatrix=reshape(ReturnMatrix,[N_d,N_a,N_a,N_j,N_z,N_e]); % (d,aprime,a,j,z,e)

    % Modify the Return Function appropriately for Epstein-Zin Preferences
    becareful=isfinite(ReturnMatrix); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
    ReturnMatrixnonfinite=ReturnMatrix(~becareful);
    ReturnMatrix(~becareful)=1; % set aside
    ReturnMatrix=ReturnMatrix.^ezc2_J; % age is dim 4 of the unfolded return
    ReturnMatrix(~becareful)=ReturnMatrixnonfinite;
    clear ReturnMatrixnonfinite

    entireRHS=ezc1_J.*ReturnMatrix+DiscountedEV; % (d,aprime,a,j,z,e)
    clear ReturnMatrix

    entireRHS(isnan(entireRHS))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
    temp5=~(entireRHS==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
    entireRHS(~temp5)=1; % set aside
    entireRHS=entireRHS.^ezc7_J; % age is dim 4 of the unfolded return
    entireRHS(~temp5)=-Inf;

    % Two maxima. The first is over aprime FOR EACH d and is the whole point of this raw.
    [Vd,aprimeRefNew]=max(entireRHS,[],2);           % [N_d,1,N_a,N_j,N_z,N_e]
    % No window here, so the argmax is already a grid index in [1,N_a]
    aprimeReferencePolicyNew=reshape(aprimeRefNew,[N_d,N_a,N_j,N_z,N_e]);

    [Vtemp,dstar]=max(Vd,[],1);                      % [1,1,N_a,N_j,N_z,N_e]
    V=reshape(Vtemp,[N_a*N_j,N_z,N_e]);
    dstar=reshape(dstar,[1,N_a,N_j,N_z,N_e]);

    % aprime at the chosen d. The lookup must use dstar and not the state index. Reshaped because with
    % N_d=1 the source can be a vector, and a vector indexed by a vector takes the SOURCE's orientation.
    aprimestar=reshape(aprimeReferencePolicyNew(dstar+N_d*reshape(gpuArray(0:1:N_a*N_j*N_z*N_e-1),[1,N_a,N_j,N_z,N_e])),[1,N_a,N_j,N_z,N_e]);

    % The folded (d,aprime) index, d fastest, the same form as the unrestricted raw's Policy
    Policy=dstar+N_d*(aprimestar-1);

elseif vfoptions.lowmemory==1

    special_n_e=ones(1,length(n_e));
    V=zeros(N_a*N_j,N_z,N_e,'gpuArray');

    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,1,:,1,e_c,:); % e_gridvals_J has shape (1,1,1,j,1,prod(n_e),l_e) for fastOLG with d

        ReturnMatrix_e=CreateReturnFnMatrix_fastOLG_Disc_e(ReturnFn, n_d, n_a, n_z, special_n_e, N_j, d_gridvals, a_grid, a_grid, z_gridvals_J, e_vals, ReturnFnParamsAgeMatrix);
        % fastOLG: ReturnMatrix is [d*aprime,a,j,z], d fastest
        ReturnMatrix_e=reshape(ReturnMatrix_e,[N_d,N_a,N_a,N_j,N_z]); % (d,aprime,a,j,z)

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=isfinite(ReturnMatrix_e); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
        ReturnMatrixnonfinite=ReturnMatrix_e(~becareful);
        ReturnMatrix_e(~becareful)=1; % set aside
        ReturnMatrix_e=ReturnMatrix_e.^ezc2_J; % age is dim 4 of the unfolded return
        ReturnMatrix_e(~becareful)=ReturnMatrixnonfinite;

        entireRHS_e=ezc1_J.*ReturnMatrix_e+DiscountedEV; % (d,aprime,a,j,z)

        entireRHS_e(isnan(entireRHS_e))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
        temp5=~(entireRHS_e==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
        entireRHS_e(~temp5)=1; % set aside
        entireRHS_e=entireRHS_e.^ezc7_J; % age is dim 4 of the unfolded return
        entireRHS_e(~temp5)=-Inf;

        [Vd,aprimeRefNew]=max(entireRHS_e,[],2);     % [N_d,1,N_a,N_j,N_z]
        aprimeRefNew=reshape(aprimeRefNew,[N_d,N_a,N_j,N_z]);
        aprimeReferencePolicyNew(:,:,:,:,e_c)=aprimeRefNew;

        [Vtemp,dstar]=max(Vd,[],1);                  % [1,1,N_a,N_j,N_z]
        V(:,:,e_c)=reshape(Vtemp,[N_a*N_j,N_z]);
        dstar=reshape(dstar,[N_a,N_j,N_z]);

        aprimestar=reshape(aprimeRefNew(dstar+N_d*reshape(gpuArray(0:1:N_a*N_j*N_z-1),[N_a,N_j,N_z])),[N_a,N_j,N_z]); % [N_a,N_j,N_z]
        Policy(1,:,:,:,e_c)=shiftdim(dstar+N_d*(aprimestar-1),-1);
    end

elseif vfoptions.lowmemory==2

    special_n_e=ones(1,length(n_e));
    special_n_z=ones(1,length(n_z));
    V=zeros(N_a*N_j,N_z,N_e,'gpuArray');

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,1,:,z_c,:); % z_gridvals_J has shape (1,1,1,j,prod(n_z),l_z) for fastOLG with d
        DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);
        for e_c=1:N_e
            e_vals=e_gridvals_J(1,1,1,:,1,e_c,:); % e_gridvals_J has shape (1,1,1,j,1,prod(n_e),l_e) for fastOLG with d

            ReturnMatrix_ze=CreateReturnFnMatrix_fastOLG_Disc_e(ReturnFn, n_d, n_a, special_n_z, special_n_e, N_j, d_gridvals, a_grid, a_grid, z_vals, e_vals, ReturnFnParamsAgeMatrix);
            % fastOLG: ReturnMatrix is [d*aprime,a,j], d fastest
            ReturnMatrix_ze=reshape(ReturnMatrix_ze,[N_d,N_a,N_a,N_j]); % (d,aprime,a,j)

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=isfinite(ReturnMatrix_ze); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
            ReturnMatrixnonfinite=ReturnMatrix_ze(~becareful);
            ReturnMatrix_ze(~becareful)=1; % set aside
            ReturnMatrix_ze=ReturnMatrix_ze.^ezc2_J; % age is dim 4 of the unfolded return
            ReturnMatrix_ze(~becareful)=ReturnMatrixnonfinite;

            entireRHS_ze=ezc1_J.*ReturnMatrix_ze+DiscountedEV_z; % (d,aprime,a,j)

            entireRHS_ze(isnan(entireRHS_ze))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
            temp5=~(entireRHS_ze==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
            entireRHS_ze(~temp5)=1; % set aside
            entireRHS_ze=entireRHS_ze.^ezc7_J; % age is dim 4 of the unfolded return
            entireRHS_ze(~temp5)=-Inf;

            [Vd,aprimeRefNew]=max(entireRHS_ze,[],2); % [N_d,1,N_a,N_j]
            aprimeRefNew=reshape(aprimeRefNew,[N_d,N_a,N_j]);
            aprimeReferencePolicyNew(:,:,:,z_c,e_c)=aprimeRefNew;

            [Vtemp,dstar]=max(Vd,[],1);              % [1,1,N_a,N_j]
            V(:,z_c,e_c)=reshape(Vtemp,[N_a*N_j,1]);
            dstar=reshape(dstar,[N_a,N_j]);

            aprimestar=reshape(aprimeRefNew(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_j-1)),[N_a,N_j]); % [N_a,N_j]
            Policy(1,:,:,z_c,e_c)=shiftdim(dstar+N_d*(aprimestar-1),-1);
        end
    end

else
    error('vfoptions.lowmemory>2 not supported for ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_LSseed_e_raw (nor by the unrestricted EZ fastOLG_e raw)')
end

end
