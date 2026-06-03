function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_GI2A_nod_raw(V,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The V input is next period value fn (across all ages), the V output is this period.

N_a=prod(n_a);
N_z=prod(n_z);

Policy=zeros(3,N_a,N_z,N_j,'gpuArray'); % first dim is (a1prime midpoint,a2prime,a1prime L2)
PolicyL2flag=2*ones(1,N_a,N_z,N_j,'gpuArray'); % 1=all weight to lower coarse a1, 2=usual linear weights, 3=all weight to upper coarse a1
% When ReturnFn is -Inf on one of the course grid points, we will allow fine index between that and the neighbouring course grid point, but we use L2flag to record this and so later avoid that -Inf point when simulating/iteration

%%
if vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
elseif vfoptions.lowmemory>=2
    error('vfoptions.lowmemory>=2 not supported')
end

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
zBind=shiftdim(gpuArray(0:1:N_z-1),-3); % already includes -1

a12ind=repmat(gpuArray(0:1:N_a1-1),1,N_a2)+N_a1*repelem(gpuArray(0:1:N_a2-1),1,N_a1);


%% j=N_j: terminal age has no continuation in TPath
% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,:,N_j);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if vfoptions.lowmemory==0
    ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, n_z, a1_grid, a2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

    % Calc the max and it's index: a1prime(a2prime,a1,a2,z)
    [~,maxindex]=max(ReturnMatrix,[],1);

    % Turn this into the 'midpoint'
    midpoint=max(min(maxindex,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is 1-by-n_a2-by-n_a1-by-n_a2-by-n_z
    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
    % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn,n_z,a1prime_grid(a1primeindexes),a2_grid,a1_grid,a2_grid,z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
    maxindexL2a1=rem(maxindexL2-1,n2long)+1;
    maxindexL2a2=ceil(maxindexL2/n2long);

    % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
    linidx_lower  = 1                  + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
    linidx_upper  = n2long*maxindexL2a2                          + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
    isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
    isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
    inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
    inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
    PolicyL2flag(1,:,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    V(:,:,N_j)=shiftdim(Vtempii,1);
    Policy(1,:,:,N_j)=midpoint(maxindexL2a2+N_a2*a12ind+N_a2*N_a*zind); % a1prime midpoint
    Policy(2,:,:,N_j)=maxindexL2a2; % a2prime
    Policy(3,:,:,N_j)=maxindexL2a1; % a1primeL2ind
elseif vfoptions.lowmemory==1
    for z_c=1:N_z
        z_val=z_gridvals_J(z_c,:,N_j);
        ReturnMatrix_z=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, special_n_z, a1_grid, a2_grid, a1_grid, a2_grid, z_val, ReturnFnParamsVec,1);

        [~,maxindex]=max(ReturnMatrix_z,[],1);

        midpoint=max(min(maxindex,n_a1-1),2);
        % midpoint is 1-by-n_a2-by-n_a1-by-n_a2
        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
        % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2
        ReturnMatrix_ii_z=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn,special_n_z,a1prime_grid(a1primeindexes),a2_grid,a1_grid,a2_grid,z_val, ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii_z,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);

        linidx_lower  = 1                  + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
        linidx_upper  = n2long*maxindexL2a2                          + n2long*N_a2*a12ind;
        isInfLower    = (ReturnMatrix_ii_z(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii_z(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        PolicyL2flag(1,:,z_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        V(:,z_c,N_j)=shiftdim(Vtempii,1);
        Policy(1,:,z_c,N_j)=midpoint(maxindexL2a2+N_a2*a12ind); % a1prime midpoint
        Policy(2,:,z_c,N_j)=maxindexL2a2; % a2prime
        Policy(3,:,z_c,N_j)=maxindexL2a1; % a1primeL2ind
    end
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

    EV=Vtemp_j; % Has been presaved before it was replaced
    Vtemp_j=V(:,:,jj); % Grab this before it is replaced/updated

    EV=EV.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    EV=reshape(EV,[N_a1,N_a2,1,1,N_z]);
    % Interpolate EV over aprime_grid
    EVinterp=interp1(a1_grid,EV,a1prime_grid);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, n_z, a1_grid, a2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec,1);
        entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV;

        %Calc the max and it's index: a1prime(a2prime,a1,a2)
        [~,maxindex]=max(entireRHS,[],1);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is 1-by-n_a2-by-n_a1-by-n_a2-by-n_z
        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
        % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn,n_z,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid,z_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
        aprimez=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimez),[n2long*N_a2,N_a,N_z]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);

        % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
        linidx_lower  = 1                  + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
        linidx_upper  = n2long*maxindexL2a2                          + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        PolicyL2flag(1,:,:,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        V(:,:,jj)=shiftdim(Vtempii,1);
        Policy(1,:,:,jj)=midpoint(maxindexL2a2+N_a2*a12ind+N_a2*N_a*zind); % a1prime midpoint
        Policy(2,:,:,jj)=maxindexL2a2; % a2prime
        Policy(3,:,:,jj)=maxindexL2a1; % a1primeL2ind
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            EV_z=EV(:,:,:,:,z_c);
            EVinterp_z=EVinterp(:,:,:,:,z_c);

            ReturnMatrix_z=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, special_n_z, a1_grid, a2_grid, a1_grid, a2_grid, z_val, ReturnFnParamsVec,1);
            entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*EV_z;

            [~,maxindex]=max(entireRHS_z,[],1);

            midpoint=max(min(maxindex,n_a1-1),2);
            % midpoint is 1-by-n_a2-by-n_a1-by-n_a2
            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
            % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2
            ReturnMatrix_ii_z=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn,special_n_z,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid,z_val, ReturnFnParamsVec,2);
            aprimez=a1primeindexes+N_a1fine*a2ind;
            entireRHS_ii_z=ReturnMatrix_ii_z+DiscountFactorParamsVec*reshape(EVinterp_z(aprimez),[n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);

            linidx_lower  = 1                  + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            linidx_upper  = n2long*maxindexL2a2                          + n2long*N_a2*a12ind;
            isInfLower    = (ReturnMatrix_ii_z(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii_z(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            PolicyL2flag(1,:,z_c,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            V(:,z_c,jj)=shiftdim(Vtempii,1);
            Policy(1,:,z_c,jj)=midpoint(maxindexL2a2+N_a2*a12ind); % a1prime midpoint
            Policy(2,:,z_c,jj)=maxindexL2a2; % a2prime
            Policy(3,:,z_c,jj)=maxindexL2a1; % a1primeL2ind
        end
    end
end


%% Currently Policy(1,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(1,:,:,:)=Policy(1,:,:,:)-adjust; % lower grid point
Policy(3,:,:,:)=adjust.*Policy(3,:,:,:)+(1-adjust).*(Policy(3,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=[Policy;PolicyL2flag];


end
