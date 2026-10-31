function [V, Policy2, aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LS1_GI1_noz_raw(Vnext,n_d,n_a, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search with a decision variable AND the grid interpolation layer, no z, no e: a coarse max over the window per
% d, then the interpolation layer around the winner per d, then the max over d.
% ValueFnIter_InfHorz_TPath_SingleStep_LS1_GI1_raw without the z dimension; see its header for why both layers are
% conditional on d, and the interior clamp. The reference is a COARSE index per (d,a), [N_d,N_a].
% Vnext is [N_a,1].
%
% Policy2 has the d variables unpacked into their own rows, [l_d+3,N_a] (d..., lower coarse point, L2, L2flag), as
% ValueFnIter_InfHorz_TPath_SingleStep_GI1_noz_raw returns it.

N_d=prod(n_d);
N_a=prod(n_a);

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_d,N_a]);

% Grid interpolation
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=2*n2short+3;            % the interpolation layer's search set, exactly as in the standard GI raw
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));

Policy=zeros(4,N_a,'gpuArray'); % d (joint index), lower coarse point, L2, L2flag

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

EV=Vnext; % [N_a,1], no exogenous state so nothing to take expectations over
EVinterp=interp1(a_grid,EV,aprime_grid);

%% The window, in COARSE grid points, one per (d,a)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_d,N_a], a coarse index

%% LAYER 1: the coarse max over the window, per d
coarseindexes=reshape(loweredge,[N_d,1,N_a])+(0:1:2*n); % [N_d,nC,N_a], coarse indices
ReturnMatrix1=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid(coarseindexes), a_grid, ReturnFnParamsVec,3); % [N_d,nC,N_a]
entireRHS1=ReturnMatrix1+DiscountFactorParamsVec*EV(coarseindexes);
[~,offset1]=max(entireRHS1,[],2);                    % [N_d,1,N_a]
coarseoffset=reshape(offset1,[N_d,N_a])-1;           % 0 to 2n, steps above loweredge, per d

%% The new reference: the layer 1 answer per d, unclamped
aprimeReferencePolicyNew=loweredge+coarseoffset;

%% LAYER 2: the interpolation layer, one coarse cell either side of the clamped midpoint, per d
midpoint=loweredge+min(max(coarseoffset,1),2*n-1);   % [N_d,N_a]
aprimeindexes=reshape(midpoint+(midpoint-1)*n2short,[N_d,1,N_a])+(-n2short-1:1:1+n2short); % [N_d,n2long,N_a]
ReturnMatrix2=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, aprime_grid(aprimeindexes), a_grid, ReturnFnParamsVec,3); % [N_d,n2long,N_a]
entireRHS2=ReturnMatrix2+DiscountFactorParamsVec*reshape(EVinterp(aprimeindexes(:)),[N_d,n2long,N_a]);
[Vd,offsetL2]=max(entireRHS2,[],2);                  % [N_d,1,N_a]
L2d=reshape(offsetL2,[N_d,N_a]);                     % 1 to n2long, per d

%% Then the max over d
[V,dstar]=max(reshape(Vd,[N_d,N_a]),[],1);           % [1,N_a]
V=reshape(V,[N_a,1]);

% Everything below is read AT THE CHOSEN d, which is the thing to get wrong here
lin=dstar+N_d*(0:1:N_a-1); % [1,N_a], into an [N_d,N_a] array
L2star=L2d(lin);
midstar=midpoint(lin);

%% L2 flag, for the cell the chosen d landed in (the standard GI raw's rule)
base=dstar+N_d*n2long*(0:1:N_a-1); % window position 1, at the chosen d
linidx_lower=base;
linidx_upper=base+N_d*(n2long-1);
isInfLower=(ReturnMatrix2(linidx_lower)==-Inf);
isInfUpper=(ReturnMatrix2(linidx_upper)==-Inf);
inLowerStrict=(L2star>=2)         & (L2star<=n2short+1);
inUpperStrict=(L2star>=n2short+3) & (L2star<=n2long-1);
Policy(4,:)=2+(inLowerStrict & isInfLower)-(inUpperStrict & isInfUpper);

%% Policy, re-spelled from (midpoint, offset within n2long) to (lower coarse point, L2), as in the standard GI raw
adjust=(L2star<1+n2short+1);
Policy(1,:)=dstar;
Policy(2,:)=midstar-adjust;
Policy(3,:)=adjust.*L2star+(1-adjust).*(L2star-n2short-1);

%% Policy in transition paths: the d variables unpacked into their own rows, as in the GI noz raw
l_d=length(n_d);
Policy2=zeros(l_d+3,N_a,'gpuArray'); % +3 = lower coarse point, L2, L2flag
Policy2(1,:)=rem(Policy(1,:)-1,n_d(1))+1;
if l_d>1
    if l_d>2
        for ii=2:l_d-1
            Policy2(ii,:)=rem(ceil(Policy(1,:)/prod(n_d(1:ii-1)))-1,n_d(ii))+1;
        end
    end
    Policy2(l_d,:)=ceil(Policy(1,:)/prod(n_d(1:l_d-1)));
end
Policy2(l_d+1:end,:)=Policy(2:4,:);

end
