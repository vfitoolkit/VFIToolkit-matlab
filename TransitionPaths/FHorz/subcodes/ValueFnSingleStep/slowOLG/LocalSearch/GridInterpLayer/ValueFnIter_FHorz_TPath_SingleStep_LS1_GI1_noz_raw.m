function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_LS1_GI1_noz_raw(V,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search with a decision variable AND the grid interpolation layer, finite horizon, slowOLG, no z.
% The V input is next period value fn (across all ages), the V output is this period.
% EXACTLY ValueFnIter_FHorz_TPath_SingleStep_LS1_GI1_raw with the z dimension removed; see it (and the
% InfHorz raw it points to). TWO LAYERS, BOTH CONDITIONAL ON d, then the max over d. Without z there is
% no expectation: EV is next period's V, exactly as in ValueFnIter_FHorz_TPath_SingleStep_GI1_noz_raw.
%
% THE REFERENCE is a COARSE aprime index per (d,a,j), [N_d,N_a,N_j].
% j=N_j uses EV=0 and EVinterp=0 with a zero discount factor, so R+0 is R exactly.
%
% Policy comes out in the SAME form as the unrestricted GI raw: (d, lower coarse point, L2, L2flag),
% [4,N_a,N_j], so the dispatcher's UnKron applies unchanged.

N_d=prod(n_d);
N_a=prod(n_a);

% lowmemory is not read here (the unrestricted noz GI raw has no lowmemory branch anyway).

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_d,N_a,N_j]);

% Grid interpolation
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=2*n2short+3;            % the interpolation layer's search set, exactly as in the standard GI raw
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);
nC=2*n+1; % the coarse window

Policy=zeros(4,N_a,N_j,'gpuArray'); % d, lower coarse point, L2, L2flag
aprimeReferencePolicyNew=zeros(N_d,N_a,N_j,'gpuArray');

%% The window, in COARSE grid points, one per (d,a,j)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_d,N_a,N_j], a coarse index

%% Iterate backwards through j, starting from j=N_j
% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,N_j);

for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);

    if jj==N_j
        % terminal age has no continuation in TPath
        DiscountFactorParamsVec=0;
        EV=zeros(N_a,1,'gpuArray');
        EVinterp=zeros(1,n2aprime,'gpuArray');
    else
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
        DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

        EV=Vtemp_j; % Has been presaved before it was replaced
        Vtemp_j=V(:,jj); % Grab this before it is replaced/updated

        % Interpolate EV over aprime_grid
        EVinterp=interp1(a_grid,EV,aprime_grid);
    end

    loweredge_j=loweredge(:,:,jj); % [N_d,N_a]

    %% LAYER 1: the coarse max over the window, per d, on the COARSE EV
    coarseindexes=reshape(loweredge_j,[N_d,1,N_a])+(0:1:2*n); % [N_d,nC,N_a], coarse indices
    % Level 3 returns [N_d,N_aprime,N_a] without folding d and aprime together
    ReturnMatrix1=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid(coarseindexes), a_grid, ReturnFnParamsVec,3);
    entireRHS1=ReturnMatrix1+DiscountFactorParamsVec*reshape(EV(coarseindexes(:)),[N_d,nC,N_a]);
    [~,offset1]=max(entireRHS1,[],2);                      % [N_d,1,N_a]
    % COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
    coarseoffset=reshape(offset1,[N_d,N_a])-1;             % 0 to 2n, steps above loweredge, per d

    % The new reference: THE LAYER 1 ANSWER per d, taken straight, UNCLAMPED
    aprimeReferencePolicyNew(:,:,jj)=loweredge_j+coarseoffset;

    %% LAYER 2: the interpolation layer, one coarse cell either side of the clamped midpoint, per d
    midpoint=loweredge_j+min(max(coarseoffset,1),2*n-1);   % [N_d,N_a]
    aprimeindexes=reshape(midpoint+(midpoint-1)*n2short,[N_d,1,N_a])+(-n2short-1:1:1+n2short); % [N_d,n2long,N_a]
    ReturnMatrix2=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, aprime_grid(aprimeindexes), a_grid, ReturnFnParamsVec,3);
    % Discounting applied AFTER the gather, the same order the GI raw uses
    entireRHS2=ReturnMatrix2+DiscountFactorParamsVec*reshape(EVinterp(aprimeindexes(:)),[N_d,n2long,N_a]);
    [Vd,offsetL2]=max(entireRHS2,[],2);                    % [N_d,1,N_a]
    L2d=reshape(offsetL2,[N_d,N_a]);                       % 1 to n2long, per d

    %% Then the max over d
    [Vtemp,dstar]=max(reshape(Vd,[N_d,N_a]),[],1);         % [1,N_a]
    V(:,jj)=reshape(Vtemp,[N_a,1]);
    dstar=reshape(dstar,[N_a,1]);

    % Everything below is read AT THE CHOSEN d. Reshaped because with N_d=1 L2d and midpoint are row
    % vectors and a vector indexed by a vector takes the SOURCE's orientation.
    lin=dstar+N_d*(0:1:N_a-1)';                            % [N_a,1], into an [N_d,N_a] array
    L2star=reshape(L2d(lin),[N_a,1]);
    midstar=reshape(midpoint(lin),[N_a,1]);

    %% L2 flag, for the cell the chosen d landed in
    % The standard GI raw's rule, with the two cell endpoints at window positions 1 and n2long in the
    % chosen d's row of ReturnMatrix2. Computed from L2star BEFORE the re-spelling below.
    base=dstar+N_d*n2long*(0:1:N_a-1)';                    % window position 1, at the chosen d
    linidx_lower=base;
    linidx_upper=base+N_d*(n2long-1);
    isInfLower=reshape(ReturnMatrix2(linidx_lower)==-Inf,[N_a,1]);
    isInfUpper=reshape(ReturnMatrix2(linidx_upper)==-Inf,[N_a,1]);
    inLowerStrict=(L2star>=2)         & (L2star<=n2short+1);
    inUpperStrict=(L2star>=n2short+3) & (L2star<=n2long-1);
    Policy(4,:,jj)=(2+(inLowerStrict & isInfLower)-(inUpperStrict & isInfUpper))';

    %% Policy, re-spelled from (midpoint, offset within n2long) to (lower coarse point, L2)
    % The standard GI raw's lines. L2 ends up in [1,n2short+2].
    adjust=(L2star<1+n2short+1);
    Policy(1,:,jj)=dstar';
    Policy(2,:,jj)=(midstar-adjust)';
    Policy(3,:,jj)=(adjust.*L2star+(1-adjust).*(L2star-n2short-1))';
end

% NOTE ON THE ENCODING, inherited from the standard GI raw: the window's top fine point can come out as
% either (loweredge+2n, 1) or (loweredge+2n-1, n2short+2). Anything comparing these raws must use the
% DECODED fine point (1+n2short)*(P1-1)+L2 and never the raw channels.

end
