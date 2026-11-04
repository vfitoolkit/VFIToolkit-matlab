function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_EZ_LS1_nod_noz_e_raw(V,n_a,n_e,N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Epstein-Zin with local search, finite horizon, slowOLG, no d, no z, with e. The V input is next period value fn
% (across all ages), the V output is this period.
% The window is EXACTLY that of ValueFnIter_FHorz_TPath_SingleStep_LS1_nod_noz_e_raw (aprime restricted to
% 2*nlocalsearch+1 grid points around aprimeReferencePolicy, sliding at the grid ends, coarseoffset 0-BASED; see
% that raw), and the right-hand side on the window is EXACTLY that of
% ValueFnIter_FHorz_TPath_SingleStep_EZ_nod_noz_e_raw: the continuation is transformed (ezc4/ezc5) on next period V
% for every e, THEN the expectation over e is taken (per age, inside the loop, not up front on the untransformed V
% as in the vNM LS raw), aggregated with survival and warm-glow over the whole aprime grid, and gathered on the
% window; the return is transformed on the window only. So when the window holds the unrestricted argmax, V and
% Policy are bit-identical to the unrestricted EZ raw. The continuation is [N_a,1], shared by all e.
% The terminal age is, as in the EZ raw, the return composed with the warm-glow alone (no continuation).
%
% THE REFERENCE is [1,N_a,N_e,N_j], one window per (a,e,j).

N_a=prod(n_a);
N_e=prod(n_e);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw.

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_a,N_e,N_j]);

Policy=zeros(N_a,N_e,N_j,'gpuArray');
aprimeReferencePolicyNew=zeros(1,N_a,N_e,N_j,'gpuArray');

pi_e_J=shiftdim(pi_e_J,-1); % Move to second dimension (normally -2, but no z so -1)

%% The window, for every age at once
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_a,N_e,N_j]

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
            temp4=WGmatrix; % [N_a,1], the same shape as the continuation at other ages
        else
            temp4=zeros(N_a,1,'gpuArray');
        end
    else
        VKronNext_j=Vtemp_j; % Has been presaved before it was replaced
        Vtemp_j=V(:,:,jj); % Grab this before it is replaced/updated
        EVpre=VKronNext_j;
        % Part of Epstein-Zin is before taking expectation
        temp=EVpre;
        temp(isfinite(EVpre))=(ezc4*EVpre(isfinite(EVpre))).^ezc5(jj);
        temp(EVpre==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf

        % Take expectation over e
        temp=sum(temp.*pi_e_J(1,:,jj+1),2); % [N_a,1]

        temp4=temp;
        if warmglow==1
            becareful=logical((~(temp4==-Inf)).*isfinite(WGmatrix)); % both are finite
            temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrix(becareful).^ezc8(jj)).^ezc6(jj);
            temp4((temp==0)&(WGmatrix==0))=0; % Is actually zero
        else % not using warmglow
            temp4(~(temp4==-Inf))=(sj(jj)*temp4(~(temp4==-Inf)).^ezc8(jj)).^ezc6(jj);
            temp4(temp==0)=0;
        end
    end

    loweredge_j=loweredge(:,:,jj);                       % [N_a,N_e]
    aprimeindexes=shiftdim(loweredge_j,-1)+(0:1:2*n)';   % [2n+1,N_a,N_e]

    %% Return matrix on the window only, transformed for Epstein-Zin
    % No z, so e is passed in the place of z, as the unrestricted raw does
    ReturnMatrix=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_e, a_grid(aprimeindexes), a_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
    becareful=isfinite(ReturnMatrix); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
    temp2=ReturnMatrix;
    temp2(becareful)=ReturnMatrix(becareful).^ezc2(jj);

    % The continuation does not depend on e, so the aprime index is the linear index into temp4, [N_a,1]
    entireRHS=ezc1*temp2+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimeindexes(:)),[2*n+1,N_a,N_e]);

    entireRHS(isnan(entireRHS))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
    temp5=~(entireRHS==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
    entireRHS(temp5)=entireRHS(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity

    [Vtemp,maxindex]=max(entireRHS,[],1);
    V(:,:,jj)=reshape(Vtemp,[N_a,N_e]);

    % COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
    coarseoffset=reshape(maxindex,[N_a,N_e])-1; % 0 to 2n, steps above loweredge
    aprimeindexstar=loweredge_j+coarseoffset;

    Policy(:,:,jj)=aprimeindexstar;
    % With no d the new reference is just the chosen aprime
    aprimeReferencePolicyNew(1,:,:,jj)=shiftdim(aprimeindexstar,-1);
end

%% Output shape for policy, the same as the unrestricted raw
Policy=shiftdim(Policy,-1);

end
