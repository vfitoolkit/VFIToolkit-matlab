function [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_GI2A_e_raw(V,n_d,n_a,n_z,n_e, N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The V input is next period value fn (across all ages), the V output is this period.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

Policy=zeros(4,N_a,N_z,N_e,N_j,'gpuArray'); % first dim is (d,a1prime midpoint,a2prime,a1prime L2)
PolicyL2flag=2*ones(1,N_a,N_z,N_e,N_j,'gpuArray'); % 1=all weight to lower coarse a1, 2=usual linear weights, 3=all weight to upper coarse a1
% When ReturnFn is -Inf on one of the course grid points, we will allow fine index between that and the neighbouring course grid point, but we use L2flag to record this and so later avoid that -Inf point when simulating/iteration

%%
if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_z=ones(1,length(n_z));
end
if vfoptions.lowmemory>=3
    error('vfoptions.lowmemory>=3 not supported')
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
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-1); % already includes -1
zind=shiftdim(gpuArray(0:1:N_z-1),-1); % already includes -1
eind=shiftdim(gpuArray(0:1:N_e-1),-2); % already includes -1
zBind=shiftdim(gpuArray(0:1:N_z-1),-4); % already includes -1

a12ind=repmat(gpuArray(0:1:N_a1-1),1,N_a2)+N_a1*repelem(gpuArray(0:1:N_a2-1),1,N_a1);

pi_e_J=shiftdim(pi_e_J,-2); % Move to third dimension

%% j=N_j: terminal age has no continuation in TPath
% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,:,:,N_j);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if vfoptions.lowmemory==0
    ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,n_z,n_e,d_gridvals,a1_grid, a2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,N_j),e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

    % Calc the max and it's index: a1prime(d,1,a2prime,a1,a2,z,e)
    [~,maxindex]=max(ReturnMatrix,[],2);

    % Turn this into the 'midpoint'
    midpoint=max(min(maxindex,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2-by-n_z-by-n_e
    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
    % aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z-by-n_e
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,n_z,n_e,d_gridvals,a1prime_grid(a1primeindexes),a2_grid,a1_grid,a2_grid, z_gridvals_J(:,:,N_j),e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
    maxindexL2d=rem(maxindexL2-1,N_d)+1;
    maxindexL2a=ceil(maxindexL2/N_d);
    maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
    maxindexL2a2=ceil(maxindexL2a/n2long);

    % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d, a2prime)
    linidx_lower  = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind + N_d*n2long*N_a2*N_a*N_z*eind;
    linidx_upper  = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind + N_d*n2long*N_a2*N_a*N_z*eind;
    isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
    isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
    inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
    inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
    PolicyL2flag(1,:,:,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    V(:,:,:,N_j)=shiftdim(Vtempii,1);
    Policy(1,:,:,:,N_j)=maxindexL2d; % d
    Policy(2,:,:,:,N_j)=midpoint(maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind+N_d*N_a2*N_a*zind+N_d*N_a2*N_a*N_z*eind); % a1prime midpoint
    Policy(3,:,:,:,N_j)=maxindexL2a2; % a2prime
    Policy(4,:,:,:,N_j)=maxindexL2a1; % a1primeL2ind
elseif vfoptions.lowmemory==1
    for e_c=1:N_e
        e_val=e_gridvals_J(e_c,:,N_j);
        ReturnMatrix_e=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,n_z,special_n_e,d_gridvals,a1_grid, a2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,N_j),e_val, ReturnFnParamsVec,1,0);

        [~,maxindex]=max(ReturnMatrix_e,[],2);

        midpoint=max(min(maxindex,n_a1-1),2);
        % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2-by-n_z
        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        % aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z
        ReturnMatrix_ii_e=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,n_z,special_n_e,d_gridvals,a1prime_grid(a1primeindexes),a2_grid,a1_grid,a2_grid, z_gridvals_J(:,:,N_j),e_val, ReturnFnParamsVec,2,0);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii_e,[],1);
        maxindexL2d=rem(maxindexL2-1,N_d)+1;
        maxindexL2a=ceil(maxindexL2/N_d);
        maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2a/n2long);

        linidx_lower  = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind;
        linidx_upper  = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind;
        isInfLower    = (ReturnMatrix_ii_e(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii_e(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        PolicyL2flag(1,:,:,e_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        V(:,:,e_c,N_j)=shiftdim(Vtempii,1);
        Policy(1,:,:,e_c,N_j)=maxindexL2d; % d
        Policy(2,:,:,e_c,N_j)=midpoint(maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind+N_d*N_a2*N_a*zind); % a1prime midpoint
        Policy(3,:,:,e_c,N_j)=maxindexL2a2; % a2prime
        Policy(4,:,:,e_c,N_j)=maxindexL2a1; % a1primeL2ind
    end
elseif vfoptions.lowmemory==2
    for e_c=1:N_e
        e_val=e_gridvals_J(e_c,:,N_j);
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_ze=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,special_n_z,special_n_e,d_gridvals,a1_grid, a2_grid, a1_grid, a2_grid, z_val,e_val, ReturnFnParamsVec,1,0);

            [~,maxindex]=max(ReturnMatrix_ze,[],2);

            midpoint=max(min(maxindex,n_a1-1),2);
            % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2
            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            % aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2
            ReturnMatrix_ii_ze=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,special_n_z,special_n_e,d_gridvals,a1prime_grid(a1primeindexes),a2_grid,a1_grid,a2_grid, z_val,e_val, ReturnFnParamsVec,2,0);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_ze,[],1);
            maxindexL2d=rem(maxindexL2-1,N_d)+1;
            maxindexL2a=ceil(maxindexL2/N_d);
            maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2a/n2long);

            linidx_lower  = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind;
            linidx_upper  = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind;
            isInfLower    = (ReturnMatrix_ii_ze(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii_ze(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            PolicyL2flag(1,:,z_c,e_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            V(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(1,:,z_c,e_c,N_j)=maxindexL2d; % d
            Policy(2,:,z_c,e_c,N_j)=midpoint(maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind); % a1prime midpoint
            Policy(3,:,z_c,e_c,N_j)=maxindexL2a2; % a2prime
            Policy(4,:,z_c,e_c,N_j)=maxindexL2a1; % a1primeL2ind
        end
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

    VKronNext_j=Vtemp_j; % Has been presaved before it was replaced
    Vtemp_j=V(:,:,:,jj); % Grab this before it is replaced/updated

    EV=sum(VKronNext_j.*pi_e_J(1,1,:,jj),3);

    EV=EV.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    EV=reshape(EV,[N_a1,N_a2,1,1,N_z]);
    % Interpolate EV over aprime_grid
    EVinterp=interp1(a1_grid,EV,a1prime_grid);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,n_z,n_e,d_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,jj),e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);
        entireRHS=ReturnMatrix+DiscountFactorParamsVec*shiftdim(EV,-1);

        % Calc the max and it's index: a1prime(d,1,a2prime,a1,a2,z,e)
        [~,maxindex]=max(entireRHS,[],2);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2-by-n_z-by-n_e
        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,n_z,n_e,d_gridvals,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,jj),e_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0);
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EVinterp(aprime),[N_d*n2long*N_a2,N_a,N_z,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        maxindexL2d=rem(maxindexL2-1,N_d)+1;
        maxindexL2a=ceil(maxindexL2/N_d);
        maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2a/n2long);

        % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d, a2prime)
        linidx_lower  = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind + N_d*n2long*N_a2*N_a*N_z*eind;
        linidx_upper  = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind + N_d*n2long*N_a2*N_a*N_z*eind;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        PolicyL2flag(1,:,:,:,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        V(:,:,:,jj)=shiftdim(Vtempii,1);
        Policy(1,:,:,:,jj)=maxindexL2d; % d
        Policy(2,:,:,:,jj)=midpoint(maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind+N_d*N_a2*N_a*zind+N_d*N_a2*N_a*N_z*eind); % a1prime midpoint
        Policy(3,:,:,:,jj)=maxindexL2a2; % a2prime
        Policy(4,:,:,:,jj)=maxindexL2a1; % a1primeL2ind
    elseif vfoptions.lowmemory==1
        % EV depends only on z (e was integrated out); reuse across e_c
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,n_z,special_n_e,d_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,jj),e_val, ReturnFnParamsVec,1,0);
            entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*shiftdim(EV,-1);

            [~,maxindex]=max(entireRHS_e,[],2);

            midpoint=max(min(maxindex,n_a1-1),2);
            % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2-by-n_z
            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            % aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z
            ReturnMatrix_ii_e=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,n_z,special_n_e,d_gridvals,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,jj),e_val, ReturnFnParamsVec,2,0);
            aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
            entireRHS_ii_e=ReturnMatrix_ii_e+DiscountFactorParamsVec*reshape(EVinterp(aprime),[N_d*n2long*N_a2,N_a,N_z]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_e,[],1);
            maxindexL2d=rem(maxindexL2-1,N_d)+1;
            maxindexL2a=ceil(maxindexL2/N_d);
            maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2a/n2long);

            linidx_lower  = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind;
            linidx_upper  = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind;
            isInfLower    = (ReturnMatrix_ii_e(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii_e(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            PolicyL2flag(1,:,:,e_c,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            V(:,:,e_c,jj)=shiftdim(Vtempii,1);
            Policy(1,:,:,e_c,jj)=maxindexL2d; % d
            Policy(2,:,:,e_c,jj)=midpoint(maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind+N_d*N_a2*N_a*zind); % a1prime midpoint
            Policy(3,:,:,e_c,jj)=maxindexL2a2; % a2prime
            Policy(4,:,:,e_c,jj)=maxindexL2a1; % a1primeL2ind
        end
    elseif vfoptions.lowmemory==2
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            for z_c=1:N_z
                z_val=z_gridvals_J(z_c,:,jj);
                EV_z=EV(:,:,:,:,z_c);
                EVinterp_z=EVinterp(:,:,:,:,z_c);

                ReturnMatrix_ze=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,special_n_z,special_n_e,d_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, z_val,e_val, ReturnFnParamsVec,1,0);
                entireRHS_ze=ReturnMatrix_ze+DiscountFactorParamsVec*shiftdim(EV_z,-1);

                [~,maxindex]=max(entireRHS_ze,[],2);

                midpoint=max(min(maxindex,n_a1-1),2);
                % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2
                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                % aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,special_n_z,special_n_e,d_gridvals,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid, z_val,e_val, ReturnFnParamsVec,2,0);
                aprime=a1primeindexes+N_a1fine*a2ind;
                entireRHS_ii_ze=ReturnMatrix_ii_ze+DiscountFactorParamsVec*reshape(EVinterp_z(aprime),[N_d*n2long*N_a2,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_ze,[],1);
                maxindexL2d=rem(maxindexL2-1,N_d)+1;
                maxindexL2a=ceil(maxindexL2/N_d);
                maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
                maxindexL2a2=ceil(maxindexL2a/n2long);

                linidx_lower  = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind;
                linidx_upper  = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind;
                isInfLower    = (ReturnMatrix_ii_ze(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii_ze(linidx_upper) == -Inf);
                inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                PolicyL2flag(1,:,z_c,e_c,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                V(:,z_c,e_c,jj)=shiftdim(Vtempii,1);
                Policy(1,:,z_c,e_c,jj)=maxindexL2d; % d
                Policy(2,:,z_c,e_c,jj)=midpoint(maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind); % a1prime midpoint
                Policy(3,:,z_c,e_c,jj)=maxindexL2a2; % a2prime
                Policy(4,:,z_c,e_c,jj)=maxindexL2a1; % a1primeL2ind
            end
        end
    end
end


%% Currently Policy(2,:) is the midpoint, and Policy(4,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(4,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(4,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:,:,:)=Policy(2,:,:,:,:)-adjust; % lower grid point
Policy(4,:,:,:,:)=adjust.*Policy(4,:,:,:,:)+(1-adjust).*(Policy(4,:,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=[Policy;PolicyL2flag];


end
