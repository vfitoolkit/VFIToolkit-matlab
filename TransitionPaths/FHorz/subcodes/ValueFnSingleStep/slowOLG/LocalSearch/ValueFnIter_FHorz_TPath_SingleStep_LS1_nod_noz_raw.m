function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_LS1_nod_noz_raw(V,n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search, finite horizon, slowOLG, no d, no z. The V input is next period value fn (across all
% ages), the V output is this period.
% EXACTLY ValueFnIter_FHorz_TPath_SingleStep_LS1_nod_raw with the z dimension removed; see it (and the
% InfHorz raw it points to) for the reasoning. Without z there is no expectation: EV is next period's V.
%
% THE REFERENCE is [1,N_a,N_j], one window per (a,j).
% j=N_j uses EV=0, and R+0 is R exactly, so at the cap this is bit-identical to
% ValueFnIter_FHorz_TPath_SingleStep_nod_noz_raw, which uses the return matrix alone at j=N_j.

N_a=prod(n_a);

% lowmemory is not read here (the unrestricted noz raw has no lowmemory branch anyway).

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_a,N_j]);

Policy=zeros(N_a,N_j,'gpuArray');
aprimeReferencePolicyNew=zeros(1,N_a,N_j,'gpuArray');

%% The window, for every age at once
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_a,N_j]

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

    loweredge_j=loweredge(:,jj);                    % [N_a,1]
    aprimeindexes=loweredge_j'+(0:1:2*n)';          % [2n+1,N_a]

    %% Return matrix on the window only
    ReturnMatrix=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(aprimeindexes), a_grid, ReturnFnParamsVec); % [2n+1,N_a]

    entireRHS=ReturnMatrix+DiscountedEV(aprimeindexes);

    [Vtemp,maxindex]=max(entireRHS,[],1);
    V(:,jj)=reshape(Vtemp,[N_a,1]);

    % COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
    coarseoffset=reshape(maxindex,[N_a,1])-1; % 0 to 2n, steps above loweredge
    aprimeindexstar=loweredge_j+coarseoffset;

    Policy(:,jj)=aprimeindexstar;
    % With no d the new reference is just the chosen aprime
    aprimeReferencePolicyNew(1,:,jj)=aprimeindexstar';
end

%% Output shape for policy, the same as the unrestricted raw
Policy=shiftdim(Policy,-1);

end
