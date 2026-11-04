function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_EZ_LS1_GI1_nod_noz_raw(V,n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Epstein-Zin with local search and the grid interpolation layer, finite horizon, slowOLG, no d, no z.
% The V input is next period value fn (across all ages), the V output is this period.
% Two sources:
%   The windows, layers, clamp, L2 flag and Policy re-spelling are EXACTLY those of
%   ValueFnIter_FHorz_TPath_SingleStep_LS1_GI1_nod_noz_raw: a coarse max over the window (whose answer IS
%   the new reference, UNCLAMPED), then the interpolation layer one coarse cell either side of that argmax
%   CLAMPED to [loweredge+1,loweredge+2n-1]. See that raw.
%   The right-hand sides are EXACTLY those of ValueFnIter_FHorz_TPath_SingleStep_EZ_GI1_nod_noz_raw:
%   without z there is no expectation, the continuation is next period's V transformed (ezc4,ezc5), over
%   the WHOLE coarse aprime grid, interpolated to the fine grid, and both aggregated with survival/warm-glow
%   (ezc6,ezc8) into temp4 (coarse) and temp4interp (fine; warm-glow evaluated on the fine grid, the
%   coarse pass using its strided subset). Layer 1 gathers temp4 on the coarse window, layer 2 gathers
%   temp4interp on the fine points; the return is created on those points only and ^ezc2 on its finite
%   entries; entireRHS=ezc1*temp2+ezc3*beta*temp4; NaN->-Inf; ^ezc7 on non -Inf.
% So when the coarse window holds the unrestricted coarse argmax, V and Policy are bit-identical to the
% EZ GI1 raw (up to the LS raw's midpoint clamp, the local-search design).
% The terminal age is, as in the EZ GI1 raw, the return composed with the warm-glow alone
% ((1-sj)*WG^ezc8)^ezc6 on the fine grid, its strided subset on the coarse grid), or zeros.
%
% THE REFERENCE is a COARSE aprime index, [1,N_a,N_j].
% Policy comes out in the SAME form as the unrestricted GI raw: (lower coarse point, L2, L2flag),
% [3,N_a,N_j], so the dispatcher's UnKron applies unchanged.

N_a=prod(n_a);

% lowmemory is not read here (the unrestricted noz GI raw has no lowmemory branch anyway).

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_a,N_j]);

% Grid interpolation
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=2*n2short+3;            % the interpolation layer's search set, exactly as in the standard GI raw
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);
nC=2*n+1; % the coarse window

Policy=zeros(3,N_a,N_j,'gpuArray'); % lower coarse point, L2, L2flag
aprimeReferencePolicyNew=zeros(1,N_a,N_j,'gpuArray');

%% The window, in COARSE grid points, for every age at once
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_a,N_j], a coarse index

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

    % If there is a warm-glow, evaluate the warmglowfn (exactly on the fine grid; coarse pass uses the strided subset)
    if warmglow==1
        WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
        WGmatrixfineraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n2aprime, aprime_grid, WGParamsVec);
        WGmatrixfine=WGmatrixfineraw;
        WGmatrixfine(isfinite(WGmatrixfineraw))=(ezc4*WGmatrixfineraw(isfinite(WGmatrixfineraw))).^ezc5(jj);
        WGmatrixfine(WGmatrixfineraw==0)=0; % otherwise zero to negative power is set to infinity
        WGmatrixfine(WGmatrixfineraw==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf
        if jj==N_j
            % The warm-glow enters INSIDE the ^ezc7 root at the terminal age: build the
            % temp4-analogue of the main loop (with no EV term), as the EZ GI1 raw does
            becareful=(WGmatrixfine==0);
            WGmatrixfine(isfinite(WGmatrixfine))=((1-sj(N_j))*WGmatrixfine(isfinite(WGmatrixfine)).^ezc8(N_j)).^ezc6(N_j);
            WGmatrixfine(becareful)=0;
        end
        WGmatrix=WGmatrixfine(1:(n2short+1):end); % coarse-grid subset
        WGmatrix=WGmatrix(:); % column over the coarse aprime grid
    end

    if jj==N_j
        % terminal age has no continuation in TPath, only the warm-glow (or nothing)
        if warmglow==1
            temp4=WGmatrix;             % [N_a,1]
            temp4interp=WGmatrixfine;   % [n2aprime,1]
        else
            temp4=zeros(N_a,1,'gpuArray');
            temp4interp=zeros(n2aprime,1,'gpuArray');
        end
    else
        VKronNext_j=Vtemp_j; % Has been presaved before it was replaced
        Vtemp_j=V(:,jj); % Grab this before it is replaced/updated
        EVpre=VKronNext_j;
        % Part of Epstein-Zin is before taking expectation
        temp=EVpre;
        temp(isfinite(EVpre))=(ezc4*EVpre(isfinite(EVpre))).^ezc5(jj);
        temp(EVpre==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf

        % No z, so no expectation
        EV=temp;

        % Interpolate EV over aprime_grid (BEFORE the certainty-equivalent power ^ezc6)
        EVinterp=interp1(a_grid,EV,aprime_grid'); % (column, to match WGmatrixfine)

        % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over aprime
        temp4=EV;
        temp4interp=EVinterp;
        if warmglow==1
            becareful=logical((~(temp4==-Inf)).*isfinite(WGmatrix)); % both are finite
            temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrix(becareful).^ezc8(jj)).^ezc6(jj);
            temp4((EV==0)&(WGmatrix==0))=0; % Is actually zero
            becareful=logical((~(temp4interp==-Inf)).*isfinite(WGmatrixfine)); % both are finite
            temp4interp(becareful)=(sj(jj)*temp4interp(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrixfine(becareful).^ezc8(jj)).^ezc6(jj);
            temp4interp((EVinterp==0)&(WGmatrixfine==0))=0; % Is actually zero
        else % not using warmglow
            temp4(~(temp4==-Inf))=(sj(jj)*temp4(~(temp4==-Inf)).^ezc8(jj)).^ezc6(jj);
            temp4(EV==0)=0;
            temp4interp(~(temp4interp==-Inf))=(sj(jj)*temp4interp(~(temp4interp==-Inf)).^ezc8(jj)).^ezc6(jj);
            temp4interp(EVinterp==0)=0;
        end
    end

    loweredge_j=loweredge(:,jj); % [N_a,1]

    %% LAYER 1: the coarse max over the window, on the COARSE temp4
    coarseindexes=loweredge_j'+(0:1:2*n)'; % [nC,N_a], coarse indices
    ReturnMatrix1=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn,a_grid(coarseindexes),a_grid,ReturnFnParamsVec);
    becareful=isfinite(ReturnMatrix1); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
    temp2=ReturnMatrix1;
    temp2(becareful)=ReturnMatrix1(becareful).^ezc2(jj);
    entireRHS1=ezc1*temp2+ezc3*DiscountFactorParamsVec*reshape(temp4(coarseindexes(:)),[nC,N_a]);
    entireRHS1(isnan(entireRHS1))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
    temp5=~(entireRHS1==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
    entireRHS1(temp5)=entireRHS1(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
    [~,maxindex1]=max(entireRHS1,[],1);
    % COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
    coarseoffset=reshape(maxindex1,[N_a,1])-1; % 0 to 2n, steps above loweredge

    % The new reference: THE LAYER 1 ANSWER, taken straight, UNCLAMPED
    aprimeReferencePolicyNew(1,:,jj)=(loweredge_j+coarseoffset)';

    %% LAYER 2: the interpolation layer, one coarse cell either side of the clamped midpoint, on temp4interp
    midpoint=loweredge_j'+min(max(coarseoffset',1),2*n-1); % [1,N_a]
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % [n2long,N_a]
    ReturnMatrix2=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    becareful=isfinite(ReturnMatrix2); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
    temp2=ReturnMatrix2; % ReturnMatrix2 itself stays raw, for the L2 flag
    temp2(becareful)=ReturnMatrix2(becareful).^ezc2(jj);
    entireRHS2=ezc1*temp2+ezc3*DiscountFactorParamsVec*reshape(temp4interp(aprimeindexes(:)),[n2long,N_a]);
    entireRHS2(isnan(entireRHS2))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
    temp5=~(entireRHS2==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
    entireRHS2(temp5)=entireRHS2(temp5).^ezc7(jj);
    [Vtemp,maxindexL2]=max(entireRHS2,[],1); % [1,N_a]
    V(:,jj)=reshape(Vtemp,[N_a,1]);

    %% L2 flag, to later avoid putting weight on an aprime where the ReturnFn is -Inf
    % (1=all to lower, 2=usual, 3=all to upper.) The standard GI raw's rule, on the RAW return; computed
    % from maxindexL2 BEFORE the re-spelling below.
    isInfLower    = (ReturnMatrix2(1,     :) == -Inf);
    isInfUpper    = (ReturnMatrix2(n2long,:) == -Inf);
    inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
    inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
    Policy(3,:,jj)=2+(inLowerStrict & isInfLower)-(inUpperStrict & isInfUpper);

    %% Policy, re-spelled from (midpoint, offset within n2long) to (lower coarse point, L2)
    % The standard GI raw's lines. L2 ends up in [1,n2short+2].
    adjust=(maxindexL2<1+n2short+1);
    Policy(1,:,jj)=midpoint-adjust;
    Policy(2,:,jj)=adjust.*maxindexL2+(1-adjust).*(maxindexL2-n2short-1);
end

% NOTE ON THE ENCODING, inherited from the standard GI raw: the window's top fine point can come out as
% either (loweredge+2n, 1) or (loweredge+2n-1, n2short+2). Anything comparing these raws must use the
% DECODED fine point (1+n2short)*(P1-1)+L2 and never the raw channels.

end
