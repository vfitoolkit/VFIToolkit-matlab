function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_LS1_noz_e_raw(V,n_d,n_a,n_e,N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search, finite horizon, slowOLG, with a decision variable, no z, with e. The V input is next
% period value fn (across all ages), the V output is this period.
% EXACTLY ValueFnIter_FHorz_TPath_SingleStep_LS1_raw with e in the place of z: aprime restricted to a
% window, d searched in FULL, the reference CONDITIONAL ON d, max over aprime per d (the new reference)
% then over d. See that raw for the reasoning.
%
% THE REFERENCE is [N_d,N_a,N_e,N_j].
% e IS START-OF-PERIOD: EV is the expectation of next period V over next period e, [N_a,1], computed
% exactly as the unrestricted raw ValueFnIter_FHorz_TPath_SingleStep_noz_e_raw does.
% j=N_j uses EV=0, see the no-d raw.
%
% Policy comes out in the SAME Kron form as the unrestricted raw, the folded (d,aprime) index with d
% fastest, so the dispatcher's UnKron applies unchanged.
% Tie-breaking differs from the folded raw (lowest d among the per-d maxima, rather than linear order);
% two names for one value, V is unaffected.

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw.

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_d,N_a,N_e,N_j]);

Policy=zeros(N_a,N_e,N_j,'gpuArray');
aprimeReferencePolicyNew=zeros(N_d,N_a,N_e,N_j,'gpuArray');

Vnext=sum(V.*shiftdim(pi_e_J,-1),2); % Take expectations over e: Vnext(...,jj+1) is read for current age jj, so weight V at age jj+1 by pi_e_J(:,jj+1), the distribution of the e realized in period jj+1; first column is never read

%% The window, one per (d,a,e,j)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_d,N_a,N_e,N_j]

%% Iterate backwards through j, starting from j=N_j
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

        EV=Vnext(:,1,jj+1); % [N_a,1]
        DiscountedEV=DiscountFactorParamsVec*EV;
    end

    loweredge_j=loweredge(:,:,:,jj);                                % [N_d,N_a,N_e]
    aprimeindexes=reshape(loweredge_j,[N_d,1,N_a,N_e])+(0:1:2*n);   % [N_d,2n+1,N_a,N_e], d fastest

    %% Return matrix on the window only
    % Level 3 returns [N_d,N_aprime,N_a,N_e] without folding d and aprime together. No z, so e is passed
    % in the place of z, as the unrestricted and DC raws of this variant do.
    ReturnMatrix=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_e, d_gridvals, a_grid(aprimeindexes), a_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec,3);

    % EV does not depend on e, so the aprime index is the linear index into DiscountedEV, [N_a,1]
    entireRHS=ReturnMatrix+reshape(DiscountedEV(aprimeindexes(:)),[N_d,2*n+1,N_a,N_e]);

    %% Two maxima: over aprime per d (the new reference), then over d
    [Vd,offset1]=max(entireRHS,[],2);                               % [N_d,1,N_a,N_e]
    % COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
    coarseoffset=reshape(offset1,[N_d,N_a,N_e])-1;                  % 0 to 2n, steps above loweredge, per d
    aprimeRefNew_j=loweredge_j+coarseoffset;

    [Vtemp,dstar]=max(reshape(Vd,[N_d,N_a,N_e]),[],1);              % [1,N_a,N_e]
    V(:,:,jj)=reshape(Vtemp,[N_a,N_e]);
    dstar=reshape(dstar,[N_a,N_e]);

    % aprime at the chosen d. The lookup must use dstar and not the state index.
    aprimestar=aprimeRefNew_j(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_e-1)); % [N_a,N_e]

    % The folded (d,aprime) index, d fastest, as the unrestricted raw's repelem(EV,N_d,1,1) layout
    Policy(:,:,jj)=dstar+N_d*(aprimestar-1);
    aprimeReferencePolicyNew(:,:,:,jj)=aprimeRefNew_j;
end

%% Output shape for policy, the same as the unrestricted raw
Policy=shiftdim(Policy,-1);

end
