function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_EZ_LS1_noz_raw(V,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Epstein-Zin with local search, finite horizon, slowOLG, with a decision variable, no z. The V input is next
% period value fn (across all ages), the V output is this period.
% The window is EXACTLY that of ValueFnIter_FHorz_TPath_SingleStep_LS1_noz_raw (aprime restricted to
% 2*nlocalsearch+1 grid points around aprimeReferencePolicy, sliding at the grid ends, coarseoffset 0-BASED; d
% searched in FULL, the reference CONDITIONAL ON d, max over aprime per d (the new reference) then over d; see
% that raw), and the right-hand side on the window is EXACTLY that of ValueFnIter_FHorz_TPath_SingleStep_EZ_noz_raw:
% the continuation is transformed over the whole aprime grid (no shocks, so no expectation: the
% certainty-equivalent is the identity), then gathered on the window, and the return is transformed on the window
% only. So when the window holds the unrestricted argmax, V is bit-identical to the unrestricted EZ raw, and so is
% Policy up to tie-breaking across d (lowest d among the per-d maxima, as the vNM LS1 raw).
% The EZ raw's temp4 is kron(EV,ones(N_d,1)), elementwise ops on copies of [N_a,1]; here it is kept as [N_a,1]
% and gathered, which gives the same values.
% The terminal age is, as in the EZ raw, the return composed with the warm-glow alone (no continuation).
%
% THE REFERENCE is [N_d,N_a,N_j].
% Policy comes out in the SAME Kron form as the unrestricted raw, the folded (d,aprime) index with d fastest.

N_d=prod(n_d);
N_a=prod(n_a);

% lowmemory is not read here (the unrestricted noz raw has no lowmemory branch anyway).

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_d,N_a,N_j]);

Policy=zeros(N_a,N_j,'gpuArray');
aprimeReferencePolicyNew=zeros(N_d,N_a,N_j,'gpuArray');

%% The window, one per (d,a,j)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_d,N_a,N_j]

%% Iterate backwards through j, starting from j=N_j
% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,N_j);

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
        Vtemp_j=V(:,jj); % Grab this before it is replaced/updated
        EVpre=VKronNext_j;
        % Part of Epstein-Zin is before taking expectation
        temp=EVpre;
        temp(isfinite(EVpre))=(ezc4*EVpre(isfinite(EVpre))).^ezc5(jj);
        temp(EVpre==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf

        % No shocks, so no expectation to take (the certainty-equivalent is just the identity)
        EV=temp; % [N_a,1]

        temp4=EV;
        if warmglow==1
            becareful=logical((~(temp4==-Inf)).*isfinite(WGmatrix)); % both are finite
            temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrix(becareful).^ezc8(jj)).^ezc6(jj);
            temp4((EV==0)&(WGmatrix==0))=0; % Is actually zero
        else % not using warmglow
            temp4(~(temp4==-Inf))=(sj(jj)*temp4(~(temp4==-Inf)).^ezc8(jj)).^ezc6(jj);
            temp4(EV==0)=0;
        end
    end

    loweredge_j=loweredge(:,:,jj);                                  % [N_d,N_a]
    aprimeindexes=reshape(loweredge_j,[N_d,1,N_a])+(0:1:2*n);       % [N_d,2n+1,N_a], d fastest

    %% Return matrix on the window only, transformed for Epstein-Zin
    % Level 3 returns [N_d,N_aprime,N_a] without folding d and aprime together
    ReturnMatrix=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid, ReturnFnParamsVec,3);
    becareful=isfinite(ReturnMatrix); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
    temp2=ReturnMatrix;
    temp2(becareful)=ReturnMatrix(becareful).^ezc2(jj);

    entireRHS=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4(aprimeindexes);

    entireRHS(isnan(entireRHS))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
    temp5=~(entireRHS==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
    entireRHS(temp5)=entireRHS(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity

    %% Two maxima: over aprime per d (the new reference), then over d
    [Vd,offset1]=max(entireRHS,[],2);                               % [N_d,1,N_a]
    % COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
    coarseoffset=reshape(offset1,[N_d,N_a])-1;                      % 0 to 2n, steps above loweredge, per d
    aprimeRefNew_j=loweredge_j+coarseoffset;

    [Vtemp,dstar]=max(reshape(Vd,[N_d,N_a]),[],1);                  % [1,N_a]
    V(:,jj)=reshape(Vtemp,[N_a,1]);
    dstar=reshape(dstar,[N_a,1]);

    % aprime at the chosen d. The lookup must use dstar and not the state index. Reshaped because with
    % N_d=1 aprimeRefNew_j is a row vector and a vector indexed by a vector takes the SOURCE's orientation.
    aprimestar=reshape(aprimeRefNew_j(dstar+N_d*(0:1:N_a-1)'),[N_a,1]); % [N_a,1]

    % The folded (d,aprime) index, d fastest, as the unrestricted raw's kron(V,ones(N_d,1)) layout
    Policy(:,jj)=dstar+N_d*(aprimestar-1);
    aprimeReferencePolicyNew(:,:,jj)=aprimeRefNew_j;
end

%% Output shape for policy, the same as the unrestricted raw
Policy=shiftdim(Policy,-1);

end
