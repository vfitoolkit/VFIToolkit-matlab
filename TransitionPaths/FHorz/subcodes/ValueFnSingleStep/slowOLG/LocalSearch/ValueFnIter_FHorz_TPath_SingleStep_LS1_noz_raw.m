function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_LS1_noz_raw(V,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search, finite horizon, slowOLG, with a decision variable, no z. The V input is next period
% value fn (across all ages), the V output is this period.
% EXACTLY ValueFnIter_FHorz_TPath_SingleStep_LS1_raw with the z dimension removed; see it (and the
% InfHorz raw it points to) for the reasoning. d is searched in FULL, the reference is CONDITIONAL ON d,
% and the max is taken in two stages (over aprime per d, which is the new reference, then over d).
% Without z there is no expectation: EV is next period's V.
%
% THE REFERENCE is [N_d,N_a,N_j].
% j=N_j uses EV=0, see the no-d raw.
%
% Policy comes out in the SAME Kron form as ValueFnIter_FHorz_TPath_SingleStep_noz_raw, the folded
% (d,aprime) index with d fastest, so the dispatcher's UnKron applies unchanged.
% Tie-breaking differs from the folded raw (lowest d among the per-d maxima, rather than linear order);
% two names for one value, V is unaffected.

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

    if jj==N_j
        % terminal age has no continuation in TPath
        DiscountedEV=zeros(N_a,1,'gpuArray');
    else
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
        DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

        VKronNext_j=Vtemp_j; % Has been presaved before it was replaced
        Vtemp_j=V(:,jj); % Grab this before it is replaced/updated

        DiscountedEV=DiscountFactorParamsVec*VKronNext_j; % [N_a,1], no z so no expectation
    end

    loweredge_j=loweredge(:,:,jj);                                  % [N_d,N_a]
    aprimeindexes=reshape(loweredge_j,[N_d,1,N_a])+(0:1:2*n);       % [N_d,2n+1,N_a], d fastest

    %% Return matrix on the window only
    % Level 3 returns [N_d,N_aprime,N_a] without folding d and aprime together
    ReturnMatrix=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid, ReturnFnParamsVec,3);

    entireRHS=ReturnMatrix+DiscountedEV(aprimeindexes);

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
