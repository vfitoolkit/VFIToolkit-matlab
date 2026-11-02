function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_raw(V,n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Epstein-Zin, one period of the FHorz transition path, fastOLG (every age at once). The V input is next
% period value fn, the V output is this period.
% The Epstein-Zin graft of ValueFnIter_FHorz_TPath_SingleStep_fastOLG_raw. It does, vectorised over the age
% dimension, exactly what the slowOLG raw ValueFnIter_FHorz_TPath_SingleStep_EZ_raw does age by age:
%   (ezc4*V')^ezc5 on next period's value fn, before the expectation over z'
%   the certainty equivalent (sj*EV^ezc8+(1-sj)*WG^ezc8)^ezc6, with the warm-glow of bequests WG
%   F^ezc2 on the return, then ezc1*F^ezc2+ezc3*beta*CE, then ^ezc7 before the max
% with the same masks (a zero return or continuation is a value; -Inf is kept out of every power).
% The terminal age has no continuation (its slot of the expectation is zero), and its certainty
% equivalent is overwritten with the warm-glow term alone, the terminal formula of the slowOLG raw.
% The age-dependent constants (ezc2, ezc5, ezc6, ezc7, ezc8, sj, and ezc1 under EZoneminusbeta) are laid
% along the age dimension (dim 3) and broadcast. A broadcast power cannot be applied to a masked subset,
% so the entries the slowOLG raw leaves out of a power are set aside first (to 1) and put back after.
% fastOLG is done as (a,j,z), rather than standard (a,z,j)
% V is (a,j)-by-z
% Policy is (a,j,z)
% pi_z_J is (j,z',z) for fastOLG; it arrives with N_j-1 slices (the transitions from periods 1..N_j-1) and a j=N_j slot of zeros is appended below
% z_gridvals_J is (j,N_z,l_z) for fastOLG

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

z_gridvals_J=shiftdim(z_gridvals_J,-3); % [1,1,1,N_j,N_z,l_z]

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
% fastOLG will be N_d*N_aprime by N_a*N_j*N_z (note: N_aprime is just equal to N_a)
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
DiscountedEV=repelem((ezc3*reshape(DiscountFactor_J,[1,1,N_j])).*temp4,N_d,1,1,1); % [d & aprime,1,j,z]
clear EV temp4 becareful

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc(ReturnFn, n_d, n_a, n_z, N_j, d_gridvals, a_grid, a_grid, z_gridvals_J, ReturnFnParamsAgeMatrix);
    % fastOLG: ReturnMatrix is [d & aprime,a,j,z]

    % Modify the Return Function appropriately for Epstein-Zin Preferences
    becareful=isfinite(ReturnMatrix); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
    ReturnMatrixnonfinite=ReturnMatrix(~becareful);
    ReturnMatrix(~becareful)=1; % set aside
    ReturnMatrix=ReturnMatrix.^ezc2_J; % in-place: ReturnMatrix now holds what the slowOLG raw calls temp2
    ReturnMatrix(~becareful)=ReturnMatrixnonfinite;
    clear ReturnMatrixnonfinite

    entireRHS=ezc1_J.*ReturnMatrix+DiscountedEV; %  [d & aprime,a,j,z]
    clear ReturnMatrix

    entireRHS(isnan(entireRHS))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
    temp5=~(entireRHS==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
    entireRHS(~temp5)=1; % set aside
    entireRHS=entireRHS.^ezc7_J;
    entireRHS(~temp5)=-Inf;

    %Calc the max and it's index
    [V,Policy]=max(entireRHS,[],1);

    V=reshape(V,[N_a*N_j,N_z]);
    Policy=squeeze(Policy);

elseif vfoptions.lowmemory==1

    Policy=zeros(N_a,N_j,N_z,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

    special_n_z=ones(1,length(n_z));

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,1,:,z_c,:); % z_gridvals_J has shape (j,prod(n_z),l_z) for fastOLG
        DiscountedEV_z=DiscountedEV(:,:,:,z_c);

        ReturnMatrix_z=CreateReturnFnMatrix_fastOLG_Disc(ReturnFn, n_d, n_a, special_n_z, N_j, d_gridvals, a_grid, a_grid, z_vals, ReturnFnParamsAgeMatrix);
        % fastOLG: ReturnMatrix is [d,aprime,a,j]

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=isfinite(ReturnMatrix_z); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
        ReturnMatrixnonfinite=ReturnMatrix_z(~becareful);
        ReturnMatrix_z(~becareful)=1; % set aside
        ReturnMatrix_z=ReturnMatrix_z.^ezc2_J;
        ReturnMatrix_z(~becareful)=ReturnMatrixnonfinite;

        entireRHS_z=ezc1_J.*ReturnMatrix_z+DiscountedEV_z; %(d,aprime)-by-(a,j)

        entireRHS_z(isnan(entireRHS_z))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
        temp5=~(entireRHS_z==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
        entireRHS_z(~temp5)=1; % set aside
        entireRHS_z=entireRHS_z.^ezc7_J;
        entireRHS_z(~temp5)=-Inf;

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS_z,[],1);
        V(:,z_c)=reshape(Vtemp,[N_a*N_j,1]);
        Policy(:,:,z_c)=maxindex;
    end
end

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point


end
