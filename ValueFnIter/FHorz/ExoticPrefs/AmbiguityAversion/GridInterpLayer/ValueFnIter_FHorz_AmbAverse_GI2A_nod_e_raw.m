function [V,Policy]=ValueFnIter_FHorz_AmbAverse_GI2A_nod_e_raw(n_ambiguity, n_a,n_z,n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, ambiguity_pi_z_J, ambiguity_pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Ambiguity aversion: multiple priors over the shock process; the continuation EV is the worst case over the priors.

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(4,N_a,N_z,N_e,N_j,'gpuArray'); % first dim is (a1prime midpoint,a2prime,a1prime L2)
Policy(4,:,:,:,:)=2; % 1=all weight to lower coarse a1, 2=usual linear weights, 3=all weight to upper coarse a1
% When ReturnFn is -Inf on one of the course grid points, we will allow fine index between that and the neighbouring course grid point, but we use L2flag to record this and so later avoid that -Inf point when simulating/iteration

%%
n_a1=n_a(1);
n_a2=n_a(2:end);
N_a1=n_a1;
N_a2=n_a2;
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);
% aprime_grid=[a1prime_grid; a2_grid];

% precompute
a2ind=gpuArray(0:1:N_a2-1); % already includes -1
zind=shiftdim(gpuArray(0:1:N_z-1),-1); % already includes -1
eind=shiftdim(gpuArray(0:1:N_e-1),-2); % already includes -1
zBind=shiftdim(gpuArray(0:1:N_z-1),-3); % already includes -1

a12ind=repmat(gpuArray(0:1:N_a1-1),1,N_a2)+N_a1*repelem(gpuArray(0:1:N_a2-1),1,N_a1);

ambiguity_pi_e_J=shiftdim(ambiguity_pi_e_J,-2); % Move to third dimension

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A_nod_e(ReturnFn, n_z,n_e, a1_grid, a2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

    % Calc the max and it's index: a1prime(a2prime,a1,a2,z,e)
    [~,maxindex]=max(ReturnMatrix,[],1);

    % Turn this into the 'midpoint'
    midpoint=max(min(maxindex,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is 1-by-n_a2-by-n_a1-by-n_a2-by-n_z-by-n_e
    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
    % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z-by-n_e
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod_e(ReturnFn,n_z,n_e,a1prime_grid(a1primeindexes),a2_grid,a1_grid,a2_grid,z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
    maxindexL2a1=rem(maxindexL2-1,n2long)+1;
    maxindexL2a2=ceil(maxindexL2/n2long);

    % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
    linidx_lower  = 1                  + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind + n2long*N_a2*N_a*N_z*eind;
    linidx_upper  = n2long*maxindexL2a2                          + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind + n2long*N_a2*N_a*N_z*eind;
    isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
    isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
    inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
    inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
    Policy(4,:,:,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    V(:,:,:,N_j)=shiftdim(Vtempii,1);
    Policy(1,:,:,:,N_j)=midpoint(maxindexL2a2+N_a2*a12ind+N_a2*N_a*zind+N_a2*N_a*N_z*eind); % a1prime midpoint
    Policy(2,:,:,:,N_j)=maxindexL2a2; % a2prime
    Policy(3,:,:,:,N_j)=maxindexL2a1; % a1primeL2ind
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpree=reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]); % iid-e expectation of V is taken first
    ambEVe=zeros(N_a,N_z,n_ambiguity(N_j),'gpuArray'); % (aprime,z,prior)
    for amb_c=1:n_ambiguity(N_j) % evaluate the iid-e expectation under each of the multiple priors
        EVe=EVpree.*ambiguity_pi_e_J(1,1,:,N_j+1,amb_c);
        EVe(isnan(EVe))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EVe=sum(EVe,3); % sum over e', leaving a singular third dimension
        ambEVe(:,:,amb_c)=EVe;
    end
    EVpre=min(ambEVe,[],3); % take the worst-case over the priors (iid e); the z expectation is next

    ambEV=zeros(N_a,1,N_z,n_ambiguity(N_j),'gpuArray'); % (aprime,1,z,prior)
    for amb_c=1:n_ambiguity(N_j) % evaluate the expectation under each of the multiple priors
        EV=EVpre.*shiftdim(ambiguity_pi_z_J(:,:,N_j,amb_c)',-1);
        EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        ambEV(:,:,:,amb_c)=EV;
    end
    EV=min(ambEV,[],4); % take the worst-case over the priors
    % From here, can just use EV as normal

    EV=reshape(EV,[N_a1,N_a2,1,1,N_z]);
    % Interpolate EV over aprime_grid
    EVinterp=min(interp1(a1_grid,reshape(ambEV,[N_a1,N_a2,1,1,N_z,size(ambEV,4)]),a1prime_grid),[],6); % interpolate each prior's EV over a1prime_grid (conditional on the prior), then take the worst case over the priors

    ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A_nod_e(ReturnFn, n_z,n_e, a1_grid, a2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV;

    %Calc the max and it's index
    [~,maxindex]=max(entireRHS,[],1);

    % Turn this into the 'midpoint'
    midpoint=max(min(maxindex,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is 1-by-n_a2-by-n_a1-by-n_a2-by-n_z-by-n_e
    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
    % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z-by-n_e
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod_e(ReturnFn,n_z,n_e,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid,z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
    aprimez=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimez),[n2long*N_a2,N_a,N_z,N_e]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    maxindexL2a1=rem(maxindexL2-1,n2long)+1;
    maxindexL2a2=ceil(maxindexL2/n2long);

    % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
    linidx_lower  = 1                  + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind + n2long*N_a2*N_a*N_z*eind;
    linidx_upper  = n2long*maxindexL2a2                          + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind + n2long*N_a2*N_a*N_z*eind;
    isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
    isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
    inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
    inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
    Policy(4,:,:,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    V(:,:,:,N_j)=shiftdim(Vtempii,1);
    Policy(1,:,:,:,N_j)=midpoint(maxindexL2a2+N_a2*a12ind+N_a2*N_a*zind+N_a2*N_a*N_z*eind); % a1prime midpoint
    Policy(2,:,:,:,N_j)=maxindexL2a2; % a2prime
    Policy(3,:,:,:,N_j)=maxindexL2a1; % a1primeL2ind
end


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
    end


    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpree=V(:,:,:,jj+1); % iid-e expectation of V is taken first
    ambEVe=zeros(N_a,N_z,n_ambiguity(jj),'gpuArray'); % (aprime,z,prior)
    for amb_c=1:n_ambiguity(jj) % evaluate the iid-e expectation under each of the multiple priors
        EVe=EVpree.*ambiguity_pi_e_J(1,1,:,jj+1,amb_c);
        EVe(isnan(EVe))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EVe=sum(EVe,3); % sum over e', leaving a singular third dimension
        ambEVe(:,:,amb_c)=EVe;
    end
    EVpre=min(ambEVe,[],3); % take the worst-case over the priors (iid e); the z expectation is next

    ambEV=zeros(N_a,1,N_z,n_ambiguity(jj),'gpuArray'); % (aprime,1,z,prior)
    for amb_c=1:n_ambiguity(jj) % evaluate the expectation under each of the multiple priors
        EV=EVpre.*shiftdim(ambiguity_pi_z_J(:,:,jj,amb_c)',-1);
        EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        ambEV(:,:,:,amb_c)=EV;
    end
    EV=min(ambEV,[],4); % take the worst-case over the priors
    % From here, can just use EV as normal

    EV=reshape(EV,[N_a1,N_a2,1,1,N_z]);
    % Interpolate EV over aprime_grid
    EVinterp=min(interp1(a1_grid,reshape(ambEV,[N_a1,N_a2,1,1,N_z,size(ambEV,4)]),a1prime_grid),[],6); % interpolate each prior's EV over a1prime_grid (conditional on the prior), then take the worst case over the priors

    ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A_nod_e(ReturnFn, n_z,n_e, a1_grid, a2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,jj),e_gridvals_J(:,:,jj),  ReturnFnParamsVec,1);
    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV;

    %Calc the max and it's index: a1prime(a2prime,a1,a2)
    [~,maxindex]=max(entireRHS,[],1);

    % Turn this into the 'midpoint'
    midpoint=max(min(maxindex,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is 1-by-n_a2-by-n_a1-by-n_a2-by-n_z-by-n_e
    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
    % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z-by-n_e
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod_e(ReturnFn,n_z,n_e,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid,z_gridvals_J(:,:,jj),e_gridvals_J(:,:,jj),  ReturnFnParamsVec,2);
    aprimez=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimez),[n2long*N_a2,N_a,N_z,N_e]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    maxindexL2a1=rem(maxindexL2-1,n2long)+1;
    maxindexL2a2=ceil(maxindexL2/n2long);

    % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
    linidx_lower  = 1                  + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind + n2long*N_a2*N_a*N_z*eind;
    linidx_upper  = n2long*maxindexL2a2                          + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind + n2long*N_a2*N_a*N_z*eind;
    isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
    isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
    inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
    inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
    Policy(4,:,:,:,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    V(:,:,:,jj)=shiftdim(Vtempii,1);
    Policy(1,:,:,:,jj)=midpoint(maxindexL2a2+N_a2*a12ind+N_a2*N_a*zind+N_a2*N_a*N_z*eind); % a1prime midpoint
    Policy(2,:,:,:,jj)=maxindexL2a2; % a2prime
    Policy(3,:,:,:,jj)=maxindexL2a1; % a1primeL2ind
end


%% Currently Policy(1,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(1,:,:,:,:)=Policy(1,:,:,:,:)-adjust; % lower grid point
Policy(3,:,:,:,:)=Policy(3,:,:,:,:)-(n2short+1)*(~adjust); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

% Policy=Policy(1,:,:,:,:)+N_a1*(Policy(2,:,:,:,:)-1)+N_a1*N_a2*(Policy(3,:,:,:,:)-1)+N_a1*N_a2*(n2short+2)*(Policy(4,:,:,:,:)-1);



end