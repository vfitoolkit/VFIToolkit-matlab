function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_LS1_GI1_e_raw(V,n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search with a decision variable AND the grid interpolation layer, finite horizon, slowOLG, with
% z and e. The V input is next period value fn (across all ages), the V output is this period.
% EXACTLY ValueFnIter_FHorz_TPath_SingleStep_LS1_GI1_raw (see it, and the InfHorz original, for the
% reasoning) with e added: TWO LAYERS, BOTH CONDITIONAL ON d (coarse max over the window per d, whose
% answer IS the new reference, unclamped; then the interpolation layer per d around that argmax clamped
% to [loweredge+1,loweredge+2n-1]), and only then the max over d.
%
% THE REFERENCE is a COARSE aprime index per (d,a,z,e,j), [N_d,N_a,N_z,N_e,N_j].
%
% EV is built EXACTLY as ValueFnIter_FHorz_TPath_SingleStep_GI1_e_raw builds it: the Vtemp_j trick,
% with the expectation over e taken per age (NOT the up-front Vnext of the non-GI e raws).
% j=N_j uses EV=0 and EVinterp=0 with a zero discount factor, so R+0 is R exactly.
%
% Policy comes out in the SAME form as the unrestricted GI raw: (d, lower coarse point, L2, L2flag).

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw.

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_d,N_a,N_z,N_e,N_j]);

% Grid interpolation
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=2*n2short+3;            % the interpolation layer's search set, exactly as in the standard GI raw
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

Policy=zeros(4,N_a,N_z,N_e,N_j,'gpuArray'); % d, lower coarse point, L2, L2flag
aprimeReferencePolicyNew=zeros(N_d,N_a,N_z,N_e,N_j,'gpuArray');

pi_e_J=shiftdim(pi_e_J,-2); % Move to third dimension

%% The window, in COARSE grid points, one per (d,a,z,e,j)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_d,N_a,N_z,N_e,N_j], a coarse index

%% Iterate backwards through j, starting from j=N_j
% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,:,:,N_j);

for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);

    if jj==N_j
        % terminal age has no continuation in TPath
        DiscountFactorParamsVec=0;
        EV=zeros(N_a,1,N_z,'gpuArray');
        EVinterp=zeros(n2aprime,1,N_z,'gpuArray');
    else
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
        DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

        VKronNext_j=Vtemp_j; % Has been presaved before it was replaced
        Vtemp_j=V(:,:,:,jj); % Grab this before it is replaced/updated

        EV=sum(VKronNext_j.*pi_e_J(1,1,:,jj+1),3);

        EV=EV.*shiftdim(pi_z_J(:,:,jj)',-1);
        EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV=sum(EV,2); % [N_a,1,N_z]

        % Interpolate EV over aprime_grid
        EVinterp=interp1(a_grid,EV,aprime_grid);
    end

    loweredge_j=loweredge(:,:,:,:,jj); % [N_d,N_a,N_z,N_e]

    %% LAYER 1: the coarse max over the window, per d, on the COARSE EV
    coarseindexes=reshape(loweredge_j,[N_d,1,N_a,N_z,N_e])+(0:1:2*n); % [N_d,nC,N_a,N_z,N_e], coarse indices
    % Level 6 takes the aprime array as given (d in dim 1, aprime in dim 2) and returns
    % [N_d,N_aprime,N_a,N_z,N_e] without folding d and aprime together. (Level 3 would shiftdim it.)
    ReturnMatrix1=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(coarseindexes), a_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,6);
    coarsez=coarseindexes+N_a*shiftdim((0:1:N_z-1),-2); % EV does not depend on e
    entireRHS1=ReturnMatrix1+DiscountFactorParamsVec*EV(coarsez);
    [~,offset1]=max(entireRHS1,[],2);                      % [N_d,1,N_a,N_z,N_e]
    % COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
    coarseoffset=reshape(offset1,[N_d,N_a,N_z,N_e])-1;     % 0 to 2n, steps above loweredge, per d

    % The new reference: THE LAYER 1 ANSWER per d, taken straight, UNCLAMPED
    aprimeReferencePolicyNew(:,:,:,:,jj)=loweredge_j+coarseoffset;

    %% LAYER 2: the interpolation layer, one coarse cell either side of the clamped midpoint, per d
    midpoint=loweredge_j+min(max(coarseoffset,1),2*n-1);   % [N_d,N_a,N_z,N_e]
    aprimeindexes=reshape(midpoint+(midpoint-1)*n2short,[N_d,1,N_a,N_z,N_e])+(-n2short-1:1:1+n2short); % [N_d,n2long,N_a,N_z,N_e]
    ReturnMatrix2=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, d_gridvals, aprime_grid(aprimeindexes), a_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,6);
    aprimez=aprimeindexes+n2aprime*shiftdim((0:1:N_z-1),-2);
    % Discounting applied AFTER the gather, the same order the GI raw uses
    entireRHS2=ReturnMatrix2+DiscountFactorParamsVec*EVinterp(aprimez);
    [Vd,offsetL2]=max(entireRHS2,[],2);                    % [N_d,1,N_a,N_z,N_e]
    L2d=reshape(offsetL2,[N_d,N_a,N_z,N_e]);               % 1 to n2long, per d

    %% Then the max over d
    [Vtemp,dstar]=max(reshape(Vd,[N_d,N_a,N_z,N_e]),[],1); % [1,N_a,N_z,N_e]
    V(:,:,:,jj)=reshape(Vtemp,[N_a,N_z,N_e]);
    dstar=reshape(dstar,[N_a,N_z,N_e]);

    % Everything below is read AT THE CHOSEN d
    lin=dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_z-1)+N_d*N_a*N_z*shiftdim((0:1:N_e-1),-1); % [N_a,N_z,N_e], into an [N_d,N_a,N_z,N_e] array
    L2star=L2d(lin);
    midstar=midpoint(lin);

    %% L2 flag, for the cell the chosen d landed in
    % The standard GI raw's rule, with the two cell endpoints at window positions 1 and n2long in the
    % chosen d's row of ReturnMatrix2. Computed from L2star BEFORE the re-spelling below.
    base=dstar+N_d*n2long*(0:1:N_a-1)'+N_d*n2long*N_a*(0:1:N_z-1)+N_d*n2long*N_a*N_z*shiftdim((0:1:N_e-1),-1); % window position 1, at the chosen d
    linidx_lower=base;
    linidx_upper=base+N_d*(n2long-1);
    isInfLower=(ReturnMatrix2(linidx_lower)==-Inf);
    isInfUpper=(ReturnMatrix2(linidx_upper)==-Inf);
    inLowerStrict=(L2star>=2)         & (L2star<=n2short+1);
    inUpperStrict=(L2star>=n2short+3) & (L2star<=n2long-1);
    Policy(4,:,:,:,jj)=shiftdim(2+(inLowerStrict & isInfLower)-(inUpperStrict & isInfUpper),-1);

    %% Policy, re-spelled from (midpoint, offset within n2long) to (lower coarse point, L2)
    % The standard GI raw's lines. L2 ends up in [1,n2short+2].
    adjust=(L2star<1+n2short+1);
    Policy(1,:,:,:,jj)=shiftdim(dstar,-1);
    Policy(2,:,:,:,jj)=shiftdim(midstar-adjust,-1);
    Policy(3,:,:,:,jj)=shiftdim(adjust.*L2star+(1-adjust).*(L2star-n2short-1),-1);
end

% NOTE ON THE ENCODING, inherited from the standard GI raw: the window's top fine point can come out as
% either (loweredge+2n, 1) or (loweredge+2n-1, n2short+2). Compare DECODED fine points only.

end
