function [V, Policy2, aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LS1_GI1_e_raw(Vnext,n_d,n_a,n_z,n_e, d_gridvals, a_grid, z_gridvals, e_gridvals, pi_z, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search with a decision variable AND the grid interpolation layer, with z and iid e: a coarse max over the window
% per d, then the interpolation layer around the winner per d, then the max over d.
% ValueFnIter_InfHorz_TPath_SingleStep_LS1_GI1_raw with an e dimension; see its header for why both layers are
% conditional on d, and the interior clamp. The reference is a COARSE index per (d,a,z,e), [N_d,N_a,N_z,N_e].
% Vnext is [N_a,N_z,N_e]; pi_z is [N_z,N_zprime]; pi_e is [N_e,1], the distribution of next period e.
%
% Policy2 has the d variables unpacked into their own rows, [l_d+3,N_a,N_z,N_e] (d..., lower coarse point, L2,
% L2flag), as ValueFnIter_InfHorz_TPath_SingleStep_GI1_e_raw returns it.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw, because a
% window fits in memory whatever the full sweep needs; the seed sweeps are the ones that honour it.

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_d,N_a,N_z,N_e]);

% Grid interpolation
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=2*n2short+3;            % the interpolation layer's search set, exactly as in the standard GI raw
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

Policy=zeros(4,N_a,N_z,N_e,'gpuArray'); % d (joint index), lower coarse point, L2, L2flag

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

%% Expected value (first e', then z'), and its interpolation onto the fine grid, as in the GI e raw
EV=Vnext.*shiftdim(pi_e,-2); % [N_a,N_z,N_e], aprime-by-zprime-by-eprime
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
EV=sum(EV,3); % sum over e', [N_a,N_z]
EV=EV.*shiftdim(pi_z',-1); % [N_a,N_zprime,N_z]
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=sum(EV,2); % [N_a,1,N_z]

EVinterp=interp1(a_grid,EV,aprime_grid); % [n2aprime,1,N_z]

%% The window, in COARSE grid points, one per (d,a,z,e)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_d,N_a,N_z,N_e], a coarse index

%% LAYER 1: the coarse max over the window, per d
coarseindexes=reshape(loweredge,[N_d,1,N_a,N_z,N_e])+(0:1:2*n); % [N_d,nC,N_a,N_z,N_e], coarse indices
% Level 6 of the e command is what the no-e command calls Level 3 (see ValueFnIter_InfHorz_TPath_SingleStep_LS1_e_raw)
ReturnMatrix1=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(coarseindexes), a_grid, z_gridvals, e_gridvals, ReturnFnParamsVec,6);
coarsez=coarseindexes+N_a*shiftdim((0:1:N_z-1),-2); % EV does not depend on e, so no e offset
entireRHS1=ReturnMatrix1+DiscountFactorParamsVec*EV(coarsez);
[~,offset1]=max(entireRHS1,[],2);                          % [N_d,1,N_a,N_z,N_e]
coarseoffset=reshape(offset1,[N_d,N_a,N_z,N_e])-1;         % 0 to 2n, steps above loweredge, per d

%% The new reference: the layer 1 answer per d, unclamped
aprimeReferencePolicyNew=loweredge+coarseoffset;

%% LAYER 2: the interpolation layer, one coarse cell either side of the clamped midpoint, per d
midpoint=loweredge+min(max(coarseoffset,1),2*n-1);         % [N_d,N_a,N_z,N_e]
aprimeindexes=reshape(midpoint+(midpoint-1)*n2short,[N_d,1,N_a,N_z,N_e])+(-n2short-1:1:1+n2short); % [N_d,n2long,N_a,N_z,N_e]
ReturnMatrix2=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, d_gridvals, aprime_grid(aprimeindexes), a_grid, z_gridvals, e_gridvals, ReturnFnParamsVec,6);
aprimez=aprimeindexes+n2aprime*shiftdim((0:1:N_z-1),-2); % index into EVinterp [n2aprime,1,N_z]
% Discounting applied AFTER the gather, the same order the GI raws use
entireRHS2=ReturnMatrix2+DiscountFactorParamsVec*EVinterp(aprimez);
[Vd,offsetL2]=max(entireRHS2,[],2);                        % [N_d,1,N_a,N_z,N_e]
L2d=reshape(offsetL2,[N_d,N_a,N_z,N_e]);                   % 1 to n2long, per d

%% Then the max over d
[V,dstar]=max(reshape(Vd,[N_d,N_a,N_z,N_e]),[],1);         % [1,N_a,N_z,N_e]
V=reshape(V,[N_a,N_z,N_e]);
dstar=reshape(dstar,[N_a,N_z,N_e]);

% Everything below is read AT THE CHOSEN d, which is the thing to get wrong here
lin=dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_z-1)+N_d*N_a*N_z*shiftdim((0:1:N_e-1),-1); % [N_a,N_z,N_e], into an [N_d,N_a,N_z,N_e] array
L2star=L2d(lin);
midstar=midpoint(lin);

%% L2 flag, for the cell the chosen d landed in (the standard GI raw's rule)
base=dstar+N_d*n2long*(0:1:N_a-1)'+N_d*n2long*N_a*(0:1:N_z-1)+N_d*n2long*N_a*N_z*shiftdim((0:1:N_e-1),-1); % window position 1, at the chosen d
linidx_lower=base;
linidx_upper=base+N_d*(n2long-1);
isInfLower=(ReturnMatrix2(linidx_lower)==-Inf);
isInfUpper=(ReturnMatrix2(linidx_upper)==-Inf);
inLowerStrict=(L2star>=2)         & (L2star<=n2short+1);
inUpperStrict=(L2star>=n2short+3) & (L2star<=n2long-1);
Policy(4,:,:,:)=shiftdim(2+(inLowerStrict & isInfLower)-(inUpperStrict & isInfUpper),-1);

%% Policy, re-spelled from (midpoint, offset within n2long) to (lower coarse point, L2), as in the standard GI raw
adjust=(L2star<1+n2short+1);
Policy(1,:,:,:)=shiftdim(dstar,-1);
Policy(2,:,:,:)=shiftdim(midstar-adjust,-1);
Policy(3,:,:,:)=shiftdim(adjust.*L2star+(1-adjust).*(L2star-n2short-1),-1);

%% Policy in transition paths: the d variables unpacked into their own rows, as in the GI e raw
l_d=length(n_d);
Policy2=zeros(l_d+3,N_a,N_z,N_e,'gpuArray'); % +3 = lower coarse point, L2, L2flag
Policy2(1,:,:,:)=rem(Policy(1,:,:,:)-1,n_d(1))+1;
if l_d>1
    if l_d>2
        for ii=2:l_d-1
            Policy2(ii,:,:,:)=rem(ceil(Policy(1,:,:,:)/prod(n_d(1:ii-1)))-1,n_d(ii))+1;
        end
    end
    Policy2(l_d,:,:,:)=ceil(Policy(1,:,:,:)/prod(n_d(1:l_d-1)));
end
Policy2(l_d+1:end,:,:,:)=Policy(2:4,:,:,:);

end
