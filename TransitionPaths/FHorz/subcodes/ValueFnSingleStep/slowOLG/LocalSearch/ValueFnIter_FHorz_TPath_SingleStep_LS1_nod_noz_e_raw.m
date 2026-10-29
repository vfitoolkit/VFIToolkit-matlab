function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_LS1_nod_noz_e_raw(V,n_a,n_e,N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search, finite horizon, slowOLG, no d, no z, with e. The V input is next period value fn
% (across all ages), the V output is this period.
% EXACTLY ValueFnIter_FHorz_TPath_SingleStep_LS1_nod_raw with e in the place of z; see it (and the
% InfHorz raw it points to) for the reasoning. Only what e changes is noted here.
%
% THE REFERENCE is [1,N_a,N_e,N_j], one window per (a,e,j).
%
% e IS START-OF-PERIOD: the continuation is the expectation of next period V over next period e, which
% does not depend on this period e. It is computed for every age at once, exactly as the unrestricted
% raw ValueFnIter_FHorz_TPath_SingleStep_nod_noz_e_raw does, so EV is [N_a,1] and is shared by all e.
% j=N_j uses EV=0; R+0 is R exactly, so at the cap this is bit-identical to the unrestricted raw.

N_a=prod(n_a);
N_e=prod(n_e);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw.

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_a,N_e,N_j]);

Policy=zeros(N_a,N_e,N_j,'gpuArray');
aprimeReferencePolicyNew=zeros(1,N_a,N_e,N_j,'gpuArray');

Vnext=sum(V.*shiftdim(pi_e_J,-1),2); % Take expectations over e: Vnext(...,jj+1) is read for current age jj, so weight V at age jj+1 by pi_e_J(:,jj+1), the distribution of the e realized in period jj+1; first column is never read

%% The window, for every age at once
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_a,N_e,N_j]

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

    loweredge_j=loweredge(:,:,jj);                       % [N_a,N_e]
    aprimeindexes=shiftdim(loweredge_j,-1)+(0:1:2*n)';   % [2n+1,N_a,N_e]

    %% Return matrix on the window only
    % No z, so e is passed in the place of z, as the unrestricted raw does
    ReturnMatrix=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_e, a_grid(aprimeindexes), a_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);

    % EV does not depend on e, so the aprime index is the linear index into DiscountedEV, [N_a,1]
    entireRHS=ReturnMatrix+reshape(DiscountedEV(aprimeindexes(:)),[2*n+1,N_a,N_e]);

    [Vtemp,maxindex]=max(entireRHS,[],1);
    V(:,:,jj)=reshape(Vtemp,[N_a,N_e]);

    % COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
    coarseoffset=reshape(maxindex,[N_a,N_e])-1; % 0 to 2n, steps above loweredge
    aprimeindexstar=loweredge_j+coarseoffset;

    Policy(:,:,jj)=aprimeindexstar;
    aprimeReferencePolicyNew(1,:,:,jj)=shiftdim(aprimeindexstar,-1);
end

%% Output shape for policy, the same as the unrestricted raw
Policy=shiftdim(Policy,-1);

end
