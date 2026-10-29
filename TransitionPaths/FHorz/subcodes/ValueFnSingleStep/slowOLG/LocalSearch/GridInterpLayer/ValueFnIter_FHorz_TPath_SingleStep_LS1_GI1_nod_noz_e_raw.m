function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_LS1_GI1_nod_noz_e_raw(V,n_a,n_e,N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search with the grid interpolation layer, finite horizon, slowOLG, no d, no z, with e. The V
% input is next period value fn (across all ages), the V output is this period.
% EXACTLY ValueFnIter_FHorz_TPath_SingleStep_LS1_GI1_nod_raw with e in the place of z: a coarse max over
% the window (whose answer IS the new reference, unclamped), then the interpolation layer one coarse cell
% either side of that argmax clamped to [loweredge+1,loweredge+2n-1]. See that raw for the reasoning.
%
% THE REFERENCE is a COARSE aprime index, [1,N_a,N_e,N_j].
% e IS START-OF-PERIOD: EV is the expectation of next period V over next period e, [N_a,1], shared by
% all e. Built the way the unrestricted GI raw ValueFnIter_FHorz_TPath_SingleStep_GI1_nod_noz_e_raw
% builds it, BUT with the age-jj+1 column of pi_e_J (Reading A, as every non-GI and DC raw of this
% variant). [That GI raw reads column jj, so with an age-dependent pi_e the two differ; with an
% age-constant pi_e they are bit-identical at the cap.]
% j=N_j uses EV=0 and EVinterp=0 with a zero discount factor, so R+0 is R exactly.
%
% Policy comes out in the SAME form as the unrestricted GI raw: (lower coarse point, L2, L2flag), so
% the dispatcher's UnKron applies unchanged.

N_a=prod(n_a);
N_e=prod(n_e);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw.

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_a,N_e,N_j]);

% Grid interpolation
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=2*n2short+3;            % the interpolation layer's search set, exactly as in the standard GI raw
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);
nC=2*n+1; % the coarse window

Policy=zeros(3,N_a,N_e,N_j,'gpuArray'); % lower coarse point, L2, L2flag
aprimeReferencePolicyNew=zeros(1,N_a,N_e,N_j,'gpuArray');

aind=0:1:N_a-1;
eind=shiftdim(0:1:N_e-1,-1);

pi_e_J=shiftdim(pi_e_J,-1); % Move to second dimension (normally -2, but no z so -1)

%% The window, in COARSE grid points, for every age at once
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_a,N_e,N_j], a coarse index

%% Iterate backwards through j, starting from j=N_j
% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,:,N_j);

for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);

    if jj==N_j
        % terminal age has no continuation in TPath
        DiscountFactorParamsVec=0;
        EV=zeros(N_a,1,'gpuArray');
        EVinterp=zeros(n2aprime,1,'gpuArray');
    else
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
        DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

        VKronNext_j=Vtemp_j; % Has been presaved before it was replaced
        Vtemp_j=V(:,:,jj); % Grab this before it is replaced/updated

        EV=sum(VKronNext_j.*pi_e_J(1,:,jj+1),2); % [N_a,1], weight V at age jj+1 by pi_e_J(:,jj+1), the distribution of the e realized in period jj+1

        % Interpolate EV over aprime_grid
        EVinterp=interp1(a_grid,EV,aprime_grid);
    end

    loweredge_j=loweredge(:,:,jj); % [N_a,N_e]

    %% LAYER 1: the coarse max over the window, on the COARSE EV
    coarseindexes=shiftdim(loweredge_j,-1)+(0:1:2*n)'; % [nC,N_a,N_e], coarse indices
    % No z, so e is passed in the place of z, as the unrestricted GI raw does
    ReturnMatrix1=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,n_e,a_grid(coarseindexes),a_grid,e_gridvals_J(:,:,jj),ReturnFnParamsVec,2);
    % EV does not depend on e, so the coarse index is the linear index into EV, [N_a,1]
    entireRHS1=ReturnMatrix1+DiscountFactorParamsVec*reshape(EV(coarseindexes(:)),[nC,N_a,N_e]);
    [~,maxindex1]=max(entireRHS1,[],1);
    % COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
    coarseoffset=reshape(maxindex1,[N_a,N_e])-1; % 0 to 2n, steps above loweredge

    % The new reference: THE LAYER 1 ANSWER, taken straight, UNCLAMPED
    aprimeReferencePolicyNew(1,:,:,jj)=shiftdim(loweredge_j+coarseoffset,-1);

    %% LAYER 2: the interpolation layer, one coarse cell either side of the clamped midpoint
    midpoint=loweredge_j+min(max(coarseoffset,1),2*n-1); % [N_a,N_e]
    aprimeindexes=shiftdim(midpoint+(midpoint-1)*n2short,-1)+(-n2short-1:1:1+n2short)'; % [n2long,N_a,N_e]
    ReturnMatrix2=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,n_e,aprime_grid(aprimeindexes),a_grid,e_gridvals_J(:,:,jj),ReturnFnParamsVec,2);
    % Discounting applied AFTER the gather, the same order the GI raw uses
    entireRHS2=ReturnMatrix2+DiscountFactorParamsVec*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a,N_e]);
    [Vtemp,maxindexL2]=max(entireRHS2,[],1);
    V(:,:,jj)=reshape(Vtemp,[N_a,N_e]);
    % maxindexL2 is deliberately LEFT as [1,N_a,N_e], the shape the flag's linear indexes are built in

    %% L2 flag, to later avoid putting weight on an aprime where the ReturnFn is -Inf
    % (1=all to lower, 2=usual, 3=all to upper.) The standard GI raw's rule; computed from maxindexL2
    % BEFORE the re-spelling below.
    linidx_lower=1      +n2long*aind+n2long*N_a*eind;
    linidx_upper=n2long +n2long*aind+n2long*N_a*eind;
    isInfLower=(ReturnMatrix2(linidx_lower)==-Inf);
    isInfUpper=(ReturnMatrix2(linidx_upper)==-Inf);
    inLowerStrict=(maxindexL2>=2)         & (maxindexL2<=n2short+1);
    inUpperStrict=(maxindexL2>=n2short+3) & (maxindexL2<=n2long-1);
    Policy(3,:,:,jj)=2+(inLowerStrict & isInfLower)-(inUpperStrict & isInfUpper);

    %% Policy, re-spelled from (midpoint, offset within n2long) to (lower coarse point, L2)
    % The standard GI raw's lines. L2 ends up in [1,n2short+2].
    adjust=(maxindexL2<1+n2short+1);
    Policy(1,:,:,jj)=shiftdim(midpoint,-1)-adjust;
    Policy(2,:,:,jj)=adjust.*maxindexL2+(1-adjust).*(maxindexL2-n2short-1);
end

% NOTE ON THE ENCODING, inherited from the standard GI raw: the window's top fine point can come out as
% either (loweredge+2n, 1) or (loweredge+2n-1, n2short+2). Anything comparing these raws must use the
% DECODED fine point (1+n2short)*(P1-1)+L2 and never the raw channels.

end
