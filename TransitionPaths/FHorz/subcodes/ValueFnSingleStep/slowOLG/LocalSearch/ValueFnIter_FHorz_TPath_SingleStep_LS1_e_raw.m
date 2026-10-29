function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_LS1_e_raw(V,n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search, finite horizon, slowOLG, with z and e, with a decision variable. The V input is next
% period value fn (across all ages), the V output is this period.
% EXACTLY ValueFnIter_FHorz_TPath_SingleStep_LS1_raw (see it, and the InfHorz original, for the
% reasoning) with e added: d is searched in FULL, the reference is CONDITIONAL ON d, and the max is
% taken in two stages (over aprime per d, which is the new reference, then over d).
%
% THE REFERENCE is [N_d,N_a,N_z,N_e,N_j].
%
% EV is built EXACTLY as ValueFnIter_FHorz_TPath_SingleStep_e_raw builds it: the expectation over e is
% taken UP FRONT for every age (Vnext), then z expectations per age. j=N_j uses EV=0, R+0 is R.
%
% Policy comes out in the SAME Kron form as the unrestricted raw, the folded (d,aprime) index with d
% fastest. Tie-breaking differs from the folded raw (lowest d among the per-d maxima); V is unaffected.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw.

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_d,N_a,N_z,N_e,N_j]);

Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray');
aprimeReferencePolicyNew=zeros(N_d,N_a,N_z,N_e,N_j,'gpuArray');

Vnext=sum(V.*shiftdim(pi_e_J,-2),3); % Take expectations over e: Vnext(...,jj+1) is read for current age jj, so weight V at age jj+1 by pi_e_J(:,jj+1), the distribution of the e realized in period jj+1; first column is never read

%% The window, one per (d,a,z,e,j)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_d,N_a,N_z,N_e,N_j]

%% Iterate backwards through j, starting from j=N_j
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

        VKronNext_j=Vnext(:,:,1,jj+1);

        EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
        EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV=sum(EV,2); % [N_a,1,N_z]
        DiscountedEV=DiscountFactorParamsVec*EV;
    end

    loweredge_j=loweredge(:,:,:,:,jj);                                   % [N_d,N_a,N_z,N_e]
    aprimeindexes=reshape(loweredge_j,[N_d,1,N_a,N_z,N_e])+(0:1:2*n);    % [N_d,2n+1,N_a,N_z,N_e], d fastest

    %% Return matrix on the window only
    % Level 6 takes the aprime array as given (d in dim 1, aprime in dim 2) and returns
    % [N_d,N_aprime,N_a,N_z,N_e] without folding d and aprime together. (Level 3 would shiftdim it.)
    ReturnMatrix=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(aprimeindexes), a_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,6);

    aprimez=aprimeindexes+N_a*shiftdim((0:1:N_z-1),-2); % linear index into DiscountedEV, which is [N_a,1,N_z]; EV does not depend on e
    entireRHS=ReturnMatrix+DiscountedEV(aprimez);

    %% Two maxima: over aprime per d (the new reference), then over d
    [Vd,offset1]=max(entireRHS,[],2);                               % [N_d,1,N_a,N_z,N_e]
    % COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
    coarseoffset=reshape(offset1,[N_d,N_a,N_z,N_e])-1;              % 0 to 2n, steps above loweredge, per d
    aprimeRefNew_j=loweredge_j+coarseoffset;

    [Vtemp,dstar]=max(reshape(Vd,[N_d,N_a,N_z,N_e]),[],1);          % [1,N_a,N_z,N_e]
    V(:,:,:,jj)=reshape(Vtemp,[N_a,N_z,N_e]);
    dstar=reshape(dstar,[N_a,N_z,N_e]);

    % aprime at the chosen d. The lookup must use dstar and not the state index.
    aprimestar=aprimeRefNew_j(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_z-1)+N_d*N_a*N_z*shiftdim((0:1:N_e-1),-1)); % [N_a,N_z,N_e]

    % The folded (d,aprime) index, d fastest, as the unrestricted raw's repelem(EV,N_d,1,1) layout
    Policy(:,:,:,jj)=dstar+N_d*(aprimestar-1);
    aprimeReferencePolicyNew(:,:,:,:,jj)=aprimeRefNew_j;
end

%% Output shape for policy, the same as the unrestricted raw
Policy=shiftdim(Policy,-1);

end
