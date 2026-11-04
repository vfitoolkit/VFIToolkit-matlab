function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_LSseed_raw(V,n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Epstein-Zin SEED SWEEP, finite horizon, fastOLG (all ages at once), with z, with a d and without the
% grid interpolation layer. An UNRESTRICTED sweep that also returns the aprime argmax FOR EACH d,
% [N_d,N_a,N_j,N_z], which is what vfoptions.localsearch needs as its reference.
% The sweep (unfolded d and aprime, per-d max, max over d, folded Policy, per-d reference, shapes) is EXACTLY
% that of ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LSseed_raw (see that raw), and the right-hand side is
% EXACTLY that of ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_raw: the discounted certainty-equivalent term
% built over the whole aprime grid, age-vectorised, with the terminal age set to the warm-glow term alone,
% the return ^ezc2 with set-asides, ezc1*F^ezc2+ezc3*beta*CE, NaN to -Inf, ^ezc7 with set-asides. Every
% operation is elementwise on the same numbers, so V is bit-identical to the unrestricted EZ raw; Policy
% differs only in tie-breaking (lowest d, then that d's lowest aprime, against linear order).
%
% AGE DIMENSION: the certainty-equivalent part works on (aprime,1,j,z), age in dim 3, as the unrestricted
% raw. The return is UNFOLDED to (d,aprime,a,j,z), age in dim 4, so the constants that meet it (ezc1, ezc2,
% ezc7) are laid along dim 4, and DiscountedEV (aprime,1,j,z) is broadcast as shiftdim(DiscountedEV,-1),
% which is (1,aprime,1,j,z).

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

% ONLY THE vfoptions.EVpre==0 PATH. Local search is reachable only from a transition path, where
% EVpre=0 (EVpre=1 is the Matched Expectations Path, which never sets localsearch).
if vfoptions.EVpre~=0
    error('vfoptions.localsearch requires vfoptions.EVpre=0 (ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_LSseed_raw)')
end
if vfoptions.lowmemory>1
    error('vfoptions.lowmemory>1 not supported for ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_LSseed_raw')
end

z_gridvals_J=shiftdim(z_gridvals_J,-3); % [1,1,1,N_j,N_z,l_z]

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

%% The age-dependent Epstein-Zin constants
% Those of the certainty equivalent along dim 3, as (aprime,1,j,z); those that meet the unfolded return
% (d,aprime,a,j,z) along dim 4
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
% EV depends on aprime, which is dimension 2 of the unfolded return matrix, so it broadcasts as
% shiftdim(DiscountedEV,-1). Same products as the unrestricted EZ raw's repelem(...,N_d,1,1,1).
DiscountedEV=(ezc3*reshape(DiscountFactor_J,[1,1,N_j])).*temp4; % [N_a,1,N_j,N_z]
clear EV temp4 becareful

if vfoptions.lowmemory==0

    % The unrestricted raw's creator, unfolded by reshape: [d,aprime,a,j,z]
    ReturnMatrix=reshape(CreateReturnFnMatrix_fastOLG_Disc(ReturnFn, n_d, n_a, n_z, N_j, d_gridvals, a_grid, a_grid, z_gridvals_J, ReturnFnParamsAgeMatrix),[N_d,N_a,N_a,N_j,N_z]);

    % Modify the Return Function appropriately for Epstein-Zin Preferences
    becareful=isfinite(ReturnMatrix); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
    ReturnMatrixnonfinite=ReturnMatrix(~becareful);
    ReturnMatrix(~becareful)=1; % set aside
    ReturnMatrix=ReturnMatrix.^ezc2_J; % age is dim 4 of the unfolded return
    ReturnMatrix(~becareful)=ReturnMatrixnonfinite;
    clear ReturnMatrixnonfinite

    entireRHS=ezc1_J.*ReturnMatrix+shiftdim(DiscountedEV,-1); % [d,aprime,a,j,z]
    clear ReturnMatrix

    entireRHS(isnan(entireRHS))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
    temp5=~(entireRHS==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
    entireRHS(~temp5)=1; % set aside
    entireRHS=entireRHS.^ezc7_J; % age is dim 4 of the unfolded return
    entireRHS(~temp5)=-Inf;

    % Two maxima. The first is over aprime FOR EACH d and is the whole point of this raw.
    [Vd,offset1]=max(entireRHS,[],2);                           % [N_d,1,N_a,N_j,N_z]
    % No window here, so the argmax is already a grid index in [1,N_a]
    aprimeReferencePolicyNew=reshape(offset1,[N_d,N_a,N_j,N_z]);

    [Vtemp,dstar]=max(reshape(Vd,[N_d,N_a,N_j,N_z]),[],1);      % [1,N_a,N_j,N_z]
    V=reshape(Vtemp,[N_a*N_j,N_z]);
    dstar=reshape(dstar,[N_a,N_j,N_z]);

    % aprime at the chosen d. The lookup must use dstar and not the state index. Reshaped because with
    % N_d=1 the source can be a vector, and a vector indexed by a vector takes the SOURCE's orientation.
    aprimestar=reshape(aprimeReferencePolicyNew(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_j-1)+N_d*N_a*N_j*shiftdim((0:1:N_z-1),-1)),[N_a,N_j,N_z]); % [N_a,N_j,N_z]

    % The folded (d,aprime) index, d fastest, the same form as the unrestricted raw's Policy
    Policy=dstar+N_d*(aprimestar-1); % [N_a,N_j,N_z]

elseif vfoptions.lowmemory==1

    Policy=zeros(N_a,N_j,N_z,'gpuArray');
    aprimeReferencePolicyNew=zeros(N_d,N_a,N_j,N_z,'gpuArray');

    special_n_z=ones(1,length(n_z));

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,1,:,z_c,:); % z_gridvals_J has shape (1,1,1,N_j,N_z,l_z) for fastOLG
        DiscountedEV_z=DiscountedEV(:,:,:,z_c);

        % The unrestricted raw's creator, unfolded by reshape: [d,aprime,a,j]
        ReturnMatrix_z=reshape(CreateReturnFnMatrix_fastOLG_Disc(ReturnFn, n_d, n_a, special_n_z, N_j, d_gridvals, a_grid, a_grid, z_vals, ReturnFnParamsAgeMatrix),[N_d,N_a,N_a,N_j]);

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=isfinite(ReturnMatrix_z); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
        ReturnMatrixnonfinite=ReturnMatrix_z(~becareful);
        ReturnMatrix_z(~becareful)=1; % set aside
        ReturnMatrix_z=ReturnMatrix_z.^ezc2_J; % age is dim 4 of the unfolded return
        ReturnMatrix_z(~becareful)=ReturnMatrixnonfinite;

        entireRHS_z=ezc1_J.*ReturnMatrix_z+shiftdim(DiscountedEV_z,-1); % [d,aprime,a,j]

        entireRHS_z(isnan(entireRHS_z))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
        temp5=~(entireRHS_z==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
        entireRHS_z(~temp5)=1; % set aside
        entireRHS_z=entireRHS_z.^ezc7_J; % age is dim 4 of the unfolded return
        entireRHS_z(~temp5)=-Inf;

        [Vd,offset1]=max(entireRHS_z,[],2);                         % [N_d,1,N_a,N_j]
        aprimeRefNew_z=reshape(offset1,[N_d,N_a,N_j]);

        [Vtemp,dstar]=max(reshape(Vd,[N_d,N_a,N_j]),[],1);          % [1,N_a,N_j]
        V(:,z_c)=reshape(Vtemp,[N_a*N_j,1]);
        dstar=reshape(dstar,[N_a,N_j]);

        % aprime at the chosen d. The lookup must use dstar and not the state index. Reshaped because with
        % N_d=1 the source can be a vector, and a vector indexed by a vector takes the SOURCE's orientation.
        aprimestar=reshape(aprimeRefNew_z(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_j-1)),[N_a,N_j]); % [N_a,N_j]

        Policy(:,:,z_c)=dstar+N_d*(aprimestar-1);
        aprimeReferencePolicyNew(:,:,:,z_c)=aprimeRefNew_z;
    end
end

%% Output shape for policy, the same as the unrestricted raw
Policy=shiftdim(Policy,-1); % [1,N_a,N_j,N_z]

end
