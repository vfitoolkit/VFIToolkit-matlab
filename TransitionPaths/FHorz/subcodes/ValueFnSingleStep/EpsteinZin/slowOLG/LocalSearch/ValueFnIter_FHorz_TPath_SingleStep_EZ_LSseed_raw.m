function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_EZ_LSseed_raw(V,n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Epstein-Zin SEED SWEEP, finite horizon, slowOLG, with a d and without the grid interpolation layer. An
% UNRESTRICTED sweep that also returns the aprime argmax FOR EACH d, [N_d,N_a,N_z,N_j], which is what
% vfoptions.localsearch needs as its reference.
% The search and the outputs are EXACTLY those of ValueFnIter_FHorz_TPath_SingleStep_LSseed_raw (unfolded
% (d,aprime,a,z) return, per-d max over aprime, then max over d, Policy folded; see it for why this
% exists), and the right-hand side is EXACTLY that of ValueFnIter_FHorz_TPath_SingleStep_EZ_raw: the
% continuation is transformed (ezc4,ezc5), its expectation taken, and it is aggregated with survival and
% the warm-glow (ezc6,ezc8) into temp4, which depends on (aprime,z) only, so it is computed once over the
% aprime grid and broadcast over d (the EZ raw repeats it N_d times with repelem/kron, which copies values,
% so every element of the RHS is the same floating-point operation). So V is bit-identical to the EZ raw.
% TIE-BREAKING IS THE ONE DIFFERENCE in Policy (lowest d, then that d's lowest aprime, against linear
% order), as in the vNM LSseed raw.
% The terminal age is, as in the EZ raw, the return composed with the warm-glow alone (no continuation),
% or with zero when there is no warm-glow.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

% No lowmemory branch. On the slowOLG layout TransitionPath_Case1_FHorz refuses lowmemory>0 outright,
% and this raw is reachable only from a TransitionPath_Case1_FHorz solve.
if vfoptions.lowmemory>0
    error('vfoptions.lowmemory>0 not supported for ValueFnIter_FHorz_TPath_SingleStep_EZ_LSseed_raw')
end

Policy=zeros(N_a,N_z,N_j,'gpuArray');
aprimeReferencePolicyNew=zeros(N_d,N_a,N_z,N_j,'gpuArray');

%% Iterate backwards through j, starting from j=N_j
% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,:,N_j);

for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    if vfoptions.EZoneminusbeta==1
        ezc1=1-DiscountFactorParamsVec; % Just in case it depends on age
    elseif vfoptions.EZoneminusbeta==2
        ezc1=1-sj(jj)*DiscountFactorParamsVec;
    end

    % If there is a warm-glow, evaluate the warmglowfn, (aprime,1)
    if warmglow==1
        WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
        WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a, a_grid, WGParamsVec);
        WGmatrix=WGmatrixraw;
        WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5(jj);
        WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
        WGmatrix(WGmatrixraw==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf
    end

    if jj==N_j
        % The terminal age has no continuation: the warm-glow enters INSIDE the ^ezc7 root, as the terminal
        % condition of the recursion (the main-loop composition with the EV term absent), as in the EZ raw
        if warmglow==1
            becareful=(WGmatrix==0);
            WGmatrix(isfinite(WGmatrix))=((1-sj(N_j))*WGmatrix(isfinite(WGmatrix)).^ezc8(N_j)).^ezc6(N_j);
            WGmatrix(becareful)=0;
            temp4=WGmatrix; % [N_a,1], broadcast over (d,a,z) below
        else
            temp4=0; % the EZ raw adds ezc3*beta*0 at the terminal age
        end
    else
        VKronNext_j=Vtemp_j; % Has been presaved before it was replaced
        Vtemp_j=V(:,:,jj); % Grab this before it is replaced/updated
        EVpre=VKronNext_j;
        % Part of Epstein-Zin is before taking expectation
        temp=EVpre;
        temp(isfinite(EVpre))=(ezc4*EVpre(isfinite(EVpre))).^ezc5(jj);
        temp(EVpre==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf
        temp(~isfinite(EVpre))=ezc4*EVpre(~isfinite(EVpre)); % ezc4 is 1 or -1, When it is -1 need to flip the sign of the -Inf, so that ezc3 below (which will also be -1) flips it back

        %Calc the expectation term (except beta)
        EV=temp.*shiftdim(pi_z_J(:,:,jj)',-1);
        EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV=sum(EV,2); % sum over z', [N_a,1,N_z]

        temp4=EV;
        if warmglow==1
            WGmatrix=WGmatrix.*ones(1,1,N_z);
            becareful=logical((~(temp4==-Inf)).*isfinite(WGmatrix)); % both are finite
            temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrix(becareful).^ezc8(jj)).^ezc6(jj);
            temp4((EV==0)&(WGmatrix==0))=0; % Is actually zero
        else % not using warmglow
            temp4(~(temp4==-Inf))=(sj(jj)*temp4(~(temp4==-Inf)).^ezc8(jj)).^ezc6(jj);
            temp4(EV==0)=0;
        end
    end

    % Level 1 keeps d and aprime in separate dimensions, (d,aprime,a,z), which is what the per-d max
    % below needs. temp4 depends on (aprime,z), so it broadcasts as shiftdim(temp4,-1).
    ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

    % Modify the Return Function appropriately for Epstein-Zin Preferences
    becareful=isfinite(ReturnMatrix); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
    temp2=ReturnMatrix;
    temp2(becareful)=ReturnMatrix(becareful).^ezc2(jj);

    entireRHS=ezc1*temp2+ezc3*DiscountFactorParamsVec*shiftdim(temp4,-1);

    entireRHS(isnan(entireRHS))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
    temp5=~(entireRHS==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
    entireRHS(temp5)=entireRHS(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity

    % Two maxima. The first is over aprime FOR EACH d and is the whole point of this raw.
    [Vd,offset1]=max(entireRHS,[],2);                   % [N_d,1,N_a,N_z]
    % No window here, so the argmax is already a grid index in [1,N_a]
    aprimeRefNew_j=reshape(offset1,[N_d,N_a,N_z]);

    [Vtemp,dstar]=max(reshape(Vd,[N_d,N_a,N_z]),[],1);  % [1,N_a,N_z]
    V(:,:,jj)=reshape(Vtemp,[N_a,N_z]);
    dstar=reshape(dstar,[N_a,N_z]);

    % aprime at the chosen d. The lookup must use dstar and not the state index. Reshaped because with
    % N_d=1 the source can be a vector, and a vector indexed by a vector takes the SOURCE's orientation.
    aprimestar=reshape(aprimeRefNew_j(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_z-1)),[N_a,N_z]); % [N_a,N_z]

    % The folded (d,aprime) index, d fastest, the same form as the folded raw's Policy
    Policy(:,:,jj)=dstar+N_d*(aprimestar-1);
    aprimeReferencePolicyNew(:,:,:,jj)=aprimeRefNew_j;
end

%% Output shape for policy, the same as the folded raw
Policy=shiftdim(Policy,-1);

end
