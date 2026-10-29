function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_LS1_GI1_nod_noz_raw(V,n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search with the grid interpolation layer, finite horizon, slowOLG, no d, no z. The V input is
% next period value fn (across all ages), the V output is this period.
% EXACTLY ValueFnIter_FHorz_TPath_SingleStep_LS1_GI1_nod_raw with the z dimension removed; see it (and
% the InfHorz raw it points to) for the TWO LAYERS and the clamp. Without z there is no expectation: EV
% is next period's V, exactly as in ValueFnIter_FHorz_TPath_SingleStep_GI1_nod_noz_raw.
%
% THE REFERENCE is a COARSE aprime index, [1,N_a,N_j]. Layer 1's answer IS the new reference, the
% UNCLAMPED coarse argmax.
% j=N_j uses EV=0 and EVinterp=0 with a zero discount factor, so R+0 is R exactly.
%
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

    loweredge_j=loweredge(:,jj); % [N_a,1]

    %% LAYER 1: the coarse max over the window, on the COARSE EV
    coarseindexes=loweredge_j'+(0:1:2*n)'; % [nC,N_a], coarse indices
    ReturnMatrix1=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn,a_grid(coarseindexes),a_grid,ReturnFnParamsVec);
    entireRHS1=ReturnMatrix1+DiscountFactorParamsVec*reshape(EV(coarseindexes(:)),[nC,N_a]);
    [~,maxindex1]=max(entireRHS1,[],1);
    % COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
    coarseoffset=reshape(maxindex1,[N_a,1])-1; % 0 to 2n, steps above loweredge

    % The new reference: THE LAYER 1 ANSWER, taken straight, UNCLAMPED
    aprimeReferencePolicyNew(1,:,jj)=(loweredge_j+coarseoffset)';

    %% LAYER 2: the interpolation layer, one coarse cell either side of the clamped midpoint
    midpoint=loweredge_j'+min(max(coarseoffset',1),2*n-1); % [1,N_a]
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % [n2long,N_a]
    ReturnMatrix2=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    % Discounting applied AFTER the gather, the same order the GI raw uses
    entireRHS2=ReturnMatrix2+DiscountFactorParamsVec*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
    [Vtemp,maxindexL2]=max(entireRHS2,[],1); % [1,N_a]
    V(:,jj)=reshape(Vtemp,[N_a,1]);

    %% L2 flag, to later avoid putting weight on an aprime where the ReturnFn is -Inf
    % (1=all to lower, 2=usual, 3=all to upper.) The standard GI raw's rule; computed from maxindexL2
    % BEFORE the re-spelling below.
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
