function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_LS1_nod_raw(V,n_a,n_z,N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search, finite horizon, slowOLG, no d. The V input is next period value fn (across all ages),
% the V output is this period.
% Each age is EXACTLY the InfHorz restricted step, ValueFnIter_InfHorz_TPath_SingleStep_LS1_nod_raw:
% aprime is restricted to a window of 2*nlocalsearch+1 grid points around aprimeReferencePolicy, the
% window SLIDES at the grid ends, and coarseoffset is 0-BASED. See that raw for the reasoning; only
% what the age dimension changes is noted here.
%
% THE REFERENCE GAINS AN AGE DIMENSION: [1,N_a,N_z,N_j], one window per (a,z,j). It is the previous GE
% iteration's answer at the same (j,t), never this sweep's answer at a neighbouring age: only the
% iteration across GE sweeps is certified by a verification sweep.
%
% THE TERMINAL AGE IS NOT A SPECIAL CASE. j=N_j has no continuation value, but aprime still enters the
% return, so the window applies there too. EV is set to zero at j=N_j, and R+0 is R exactly, so at the
% cap this is bit-identical to the unrestricted raw, which uses the return matrix alone at j=N_j.

N_a=prod(n_a);
N_z=prod(n_z);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw, because a
% window fits in memory whatever the full sweep needs.

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_a,N_z,N_j]);

Policy=zeros(N_a,N_z,N_j,'gpuArray');
aprimeReferencePolicyNew=zeros(1,N_a,N_z,N_j,'gpuArray');

%% The window, for every age at once
% Same line as every other localsearch raw, with an age dimension. Requires N_a>=2n+1, which is
% checked where nlocalsearch is validated.
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_a,N_z,N_j]

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

    loweredge_j=loweredge(:,:,jj);                       % [N_a,N_z]
    aprimeindexes=shiftdim(loweredge_j,-1)+(0:1:2*n)';   % [2n+1,N_a,N_z]

    %% Return matrix on the window only
    ReturnMatrix=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_z, a_grid(aprimeindexes), a_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec,2);

    aprimez=aprimeindexes+N_a*shiftdim((0:1:N_z-1),-1); % linear index into DiscountedEV, which is [N_a,1,N_z]
    entireRHS=ReturnMatrix+DiscountedEV(aprimez);

    [Vtemp,maxindex]=max(entireRHS,[],1);
    V(:,:,jj)=reshape(Vtemp,[N_a,N_z]);

    % COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
    coarseoffset=reshape(maxindex,[N_a,N_z])-1; % 0 to 2n, steps above loweredge
    aprimeindexstar=loweredge_j+coarseoffset;

    Policy(:,:,jj)=aprimeindexstar;
    % With no d the new reference is just the chosen aprime. Returned all the same, so that a caller
    % never has to branch on whether the model has a d.
    aprimeReferencePolicyNew(1,:,:,jj)=shiftdim(aprimeindexstar,-1);
end

%% Output shape for policy, the same as the unrestricted raw
Policy=shiftdim(Policy,-1);

end
