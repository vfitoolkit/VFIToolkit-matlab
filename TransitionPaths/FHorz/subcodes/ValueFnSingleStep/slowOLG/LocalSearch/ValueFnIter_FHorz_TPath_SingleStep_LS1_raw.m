function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_LS1_raw(V,n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search, finite horizon, slowOLG, with a decision variable. The V input is next period value fn
% (across all ages), the V output is this period.
% Each age is EXACTLY the InfHorz restricted step, ValueFnIter_InfHorz_TPath_SingleStep_LS1_raw:
% aprime is restricted to a window of 2*nlocalsearch+1 grid points around aprimeReferencePolicy, d is
% searched in FULL, the reference is CONDITIONAL ON d, and the max is taken in two stages (over aprime
% per d, which is the new reference, then over d). See that raw for the reasoning.
%
% THE REFERENCE GAINS AN AGE DIMENSION: [N_d,N_a,N_z,N_j].
% j=N_j uses EV=0, see the no-d raw.
%
% Policy comes out in the SAME Kron form as the unrestricted raw, the folded (d,aprime) index with d
% fastest, so the dispatcher's UnKron applies unchanged.
% Tie-breaking differs from the folded raw (lowest d among the per-d maxima, rather than linear order);
% two names for one value, V is unaffected.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw.

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_d,N_a,N_z,N_j]);

Policy=zeros(N_a,N_z,N_j,'gpuArray');
aprimeReferencePolicyNew=zeros(N_d,N_a,N_z,N_j,'gpuArray');

%% The window, one per (d,a,z,j)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_d,N_a,N_z,N_j]

%% Iterate backwards through j, starting from j=N_j
% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,:,N_j);

for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);

    if jj==N_j
        % terminal age has no continuation in TPath
        DiscountedEV=zeros(N_a,1,N_z,'gpuArray');
    else
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
        DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

        VKronNext_j=Vtemp_j; % Has been presaved before it was replaced
        Vtemp_j=V(:,:,jj); % Grab this before it is replaced/updated

        EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
        EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV=sum(EV,2); % [N_a,1,N_z]
        DiscountedEV=DiscountFactorParamsVec*EV;
    end

    loweredge_j=loweredge(:,:,:,jj);                                % [N_d,N_a,N_z]
    aprimeindexes=reshape(loweredge_j,[N_d,1,N_a,N_z])+(0:1:2*n);   % [N_d,2n+1,N_a,N_z], d fastest

    %% Return matrix on the window only
    % Level 3 returns [N_d,N_aprime,N_a,N_z] without folding d and aprime together
    ReturnMatrix=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, a_grid(aprimeindexes), a_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec,3);

    aprimez=aprimeindexes+N_a*shiftdim((0:1:N_z-1),-2); % linear index into DiscountedEV, which is [N_a,1,N_z]
    entireRHS=ReturnMatrix+DiscountedEV(aprimez);

    %% Two maxima: over aprime per d (the new reference), then over d
    [Vd,offset1]=max(entireRHS,[],2);                               % [N_d,1,N_a,N_z]
    % COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
    coarseoffset=reshape(offset1,[N_d,N_a,N_z])-1;                  % 0 to 2n, steps above loweredge, per d
    aprimeRefNew_j=loweredge_j+coarseoffset;

    [Vtemp,dstar]=max(reshape(Vd,[N_d,N_a,N_z]),[],1);              % [1,N_a,N_z]
    V(:,:,jj)=reshape(Vtemp,[N_a,N_z]);
    dstar=reshape(dstar,[N_a,N_z]);

    % aprime at the chosen d. The lookup must use dstar and not the state index.
    aprimestar=aprimeRefNew_j(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_z-1)); % [N_a,N_z]

    % The folded (d,aprime) index, d fastest, as the unrestricted raw's repelem(EV,N_d,1,1) layout
    Policy(:,:,jj)=dstar+N_d*(aprimestar-1);
    aprimeReferencePolicyNew(:,:,:,jj)=aprimeRefNew_j;
end

%% Output shape for policy, the same as the unrestricted raw
Policy=shiftdim(Policy,-1);

end
