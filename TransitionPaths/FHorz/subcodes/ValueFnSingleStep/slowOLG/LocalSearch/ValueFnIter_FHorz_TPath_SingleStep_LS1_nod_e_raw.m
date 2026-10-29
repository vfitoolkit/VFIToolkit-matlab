function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_LS1_nod_e_raw(V,n_a,n_z,n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search, finite horizon, slowOLG, with z and e, no d. The V input is next period value fn
% (across all ages), the V output is this period.
% EXACTLY ValueFnIter_FHorz_TPath_SingleStep_LS1_nod_raw (see it, and the InfHorz original, for the
% reasoning) with e added: e is one more state the window is indexed by.
%
% THE REFERENCE is [1,N_a,N_z,N_e,N_j], one window per (a,z,e,j).
%
% EV is built EXACTLY as ValueFnIter_FHorz_TPath_SingleStep_nod_e_raw builds it: the expectation over e
% is taken UP FRONT for every age (Vnext), then z expectations per age. j=N_j uses EV=0, R+0 is R.

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw.

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_a,N_z,N_e,N_j]);

Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray');
aprimeReferencePolicyNew=zeros(1,N_a,N_z,N_e,N_j,'gpuArray');

Vnext=sum(V.*shiftdim(pi_e_J,-2),3); % Take expectations over e: Vnext(...,jj+1) is read for current age jj, so weight V at age jj+1 by pi_e_J(:,jj+1), the distribution of the e realized in period jj+1; first column is never read

%% The window, for every age at once
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_a,N_z,N_e,N_j]

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

    loweredge_j=loweredge(:,:,:,jj);                     % [N_a,N_z,N_e]
    aprimeindexes=shiftdim(loweredge_j,-1)+(0:1:2*n)';   % [2n+1,N_a,N_z,N_e]

    %% Return matrix on the window only
    ReturnMatrix=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, n_z, n_e, a_grid(aprimeindexes), a_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);

    aprimez=aprimeindexes+N_a*shiftdim((0:1:N_z-1),-1); % linear index into DiscountedEV, which is [N_a,1,N_z]; EV does not depend on e
    entireRHS=ReturnMatrix+DiscountedEV(aprimez);

    [Vtemp,maxindex]=max(entireRHS,[],1);
    V(:,:,:,jj)=reshape(Vtemp,[N_a,N_z,N_e]);

    % COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
    coarseoffset=reshape(maxindex,[N_a,N_z,N_e])-1; % 0 to 2n, steps above loweredge
    aprimeindexstar=loweredge_j+coarseoffset;

    Policy(:,:,:,jj)=aprimeindexstar;
    aprimeReferencePolicyNew(1,:,:,:,jj)=shiftdim(aprimeindexstar,-1);
end

%% Output shape for policy, the same as the unrestricted raw
Policy=shiftdim(Policy,-1);

end
