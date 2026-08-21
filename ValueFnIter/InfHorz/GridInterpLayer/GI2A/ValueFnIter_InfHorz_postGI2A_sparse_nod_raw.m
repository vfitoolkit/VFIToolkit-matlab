function [VKron, Policy]=ValueFnIter_InfHorz_postGI2A_sparse_nod_raw(VKron, n_a, n_z, a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParams, vfoptions)
% Sparse-matrix (iterated) Howards, with the grid interpolation layer (postGI), two endogenous
% states, no decision variable.
%
% Structure is the multi-grid postGI one: first solve on the coarse a_grid, then build the
% +-vfoptions.maxaprimediff window ONCE (on the first endogenous state only) and solve on the fine
% (interpolated) grid within it. Howards is done by building the sparse transition matrix T_E and
% iterating with it (vfoptions.howards times).
%
% Note: the non-sparse ValueFnIter_InfHorz_postGI2A_nod_raw has no Howards at all in its second
% (fine grid) stage; the indexed version was too slow to be worth using because it has to
% re-interpolate EV inside every Howards step. Building T_E bakes the interpolation weights in once,
% so each Howards step is a single sparse matrix multiply and the cost disappears.
%
% The grid interpolation layer is on the FIRST endogenous state only. The joint index over a is
% a1+N_a1*(a2-1), so a1 varies fastest, which means the two points that an interpolated a1prime
% spreads mass over are adjacent in the joint index (stride 1), exactly as in the one endogenous
% state case.

N_a=prod(n_a);
N_z=prod(n_z);

N_a1=n_a(1);
N_a2=prod(n_a(2:end));
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);
a_gridvals=CreateGridvals(n_a,a_grid,1);

ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn,n_a, n_a, n_z, a_gridvals, a_gridvals, z_gridvals, ReturnFnParams);

pi_z_alt=shiftdim(pi_z',-1);

addindexforaz=gpuArray(N_a*(0:1:N_a-1)'+N_a*N_a*(0:1:N_z-1));

% Setup specific to the sparse-matrix Howards
N_a_times_zind=N_a*gpuArray(0:1:N_z-1); % already contains -1
azind1=repmat(gpuArray(1:1:N_a*N_z)',1,N_z); % (a-z,zprime)
pi_z_big1=gpuArray(repelem(pi_z,N_a,1)); % (a-z,zprime)

%%
tempcounter=1;
currdist=Inf;

%% First, just consider a_grid for next period
while currdist>(vfoptions.multigridswitch*vfoptions.tolerance) && tempcounter<=vfoptions.maxiter
    VKronold=VKron;

    % Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=VKronold.*pi_z_alt;
    EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; % aprime by a by z

    %Calc the max and it's index
    [VKron,Policy]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1); % a by z

    VKrondist=VKron(:)-VKronold(:);
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    % Use Howards Improvement, iterating with the sparse transition matrix (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
        tempmaxindex=shiftdim(Policy,1)+addindexforaz; % aprime index, add the index for a and z
        Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a*N_z,1]); % keep return function of optimal policy for using in Howards

        % On the coarse grid Policy is already the joint aprime index, so this is the same as the one endogenous state case
        T_E=sparse(azind1,Policy(:)+N_a_times_zind,pi_z_big1,N_a*N_z,N_a*N_z);

        VKron=reshape(VKron,[N_a*N_z,1]);
        for h_c=1:vfoptions.howards
            VKron=Ftemp+DiscountFactorParamsVec*(T_E*VKron); % T_E already contains pi_z, so T_E*V is the expected continuation
        end
        VKron=reshape(VKron,[N_a,N_z]);
    end

    tempcounter=tempcounter+1;

end
Policy=reshape(Policy,[1,N_a,N_z]); % Howards can mess with the size
Policy_a1=rem(Policy-1,N_a1)+1;

%% Now that we have solved on the rough grid, we resolve on the fine grid
% Based on solving a bunch of value fns with and without grid interpolation, the 'lower grid index'
% with grid interpolation is always within a point or two of the solution on the rough grid. So here
% we only consider +-vfoptions.maxaprimediff to set up the fine/interpolated a1prime_grid

% First, create an a1prime_grid that is just the +-vfoptions.maxaprimediff
n_a1primediff=1+2*vfoptions.maxaprimediff;
N_a1primediff=prod(n_a1primediff);
a1primeshifter=min(max(Policy_a1,1+vfoptions.maxaprimediff),N_a1-vfoptions.maxaprimediff);
a1primeindex=(-vfoptions.maxaprimediff:1:vfoptions.maxaprimediff)' +a1primeshifter; % size n_aprime-by-n_a
a1prime_grid=a1_grid(a1primeindex);
% Second, interpolate this
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n_a1prime=n_a1primediff+(n_a1primediff-1)*vfoptions.ngridinterp;
N_a1prime=prod(n_a1prime);
a1prime_grid=interp1((1:1:N_a1primediff)',a1prime_grid,linspace(1,N_a1primediff,N_a1primediff+(N_a1primediff-1)*vfoptions.ngridinterp)');
% Note: a1prime_grid is N_a1prime-by-N_a-by-N_z

a1prime_grid=reshape(a1prime_grid,[N_a1prime,1,N_a1,N_a2,N_z]);

EVinterpindex1=(1:1:N_a1primediff)';
EVinterpindex2=linspace(1,N_a1primediff,N_a1primediff+(N_a1primediff-1)*vfoptions.ngridinterp)';

N_aprime=N_a1prime*N_a2;
N_aprimediff=N_a1primediff*N_a2;
aprimeindex=repmat(a1primeindex,N_a2,1,1)+N_a1*repelem((0:1:N_a2-1)',N_a1primediff,1,1);

ReturnMatrixfine=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, n_z, a1prime_grid, a2_grid, a1_grid, a2_grid, z_gridvals, ReturnFnParams, 2);

% For Howards we need
addindexforazfine=gpuArray(N_aprime*(0:1:N_a-1)'+N_aprime*N_a*(0:1:N_z-1));

pi_z_alt2=shiftdim(pi_z,-2);

% Setup specific to the sparse-matrix Howards
azind2=repmat(gpuArray(1:1:N_a*N_z)',2,N_z); % (a-z-2,zprime)
pi_z_big2=gpuArray(repmat(pi_z_big1,2,1)); % (a-z-2,zprime)

%% Now switch to considering the fine/interpolated aprime_grid
currdist=1; % force going into the next while loop at least one iteration
tempcounter=1; % reset tempcounter
while currdist>vfoptions.tolerance && tempcounter<=vfoptions.maxiter
    VKronold=VKron;

    % Switch VKron into being over vfoptions.maxaprimediff
    EVpre=reshape(VKron(aprimeindex,:),[N_aprimediff,N_a,N_z,N_z]); % last dimension is zprime

    % Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=EVpre.*pi_z_alt2;
    EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=squeeze(sum(EV,4)); % sum over z', leaving a singular second dimension
    % EV is now [N_aprimediff,N_a,N_z]
    % Interpolate EV over aprime_grid

    EVinterp=reshape(interp1(EVinterpindex1,reshape(EV,[N_a1primediff,N_a2,N_a,N_z]),EVinterpindex2),[N_aprime,N_a,N_z]);

    entireRHS=ReturnMatrixfine+DiscountFactorParamsVec*EVinterp; % aprime by a by z

    % Calc the max and it's index
    [VKron,Policy]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1); % a by z

    VKrondist=VKron(:)-VKronold(:);
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    % Use Howards Improvement, iterating with the sparse transition matrix (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
        tempmaxindex=shiftdim(Policy,1)+addindexforazfine; % aprime index, add the index for a and z
        Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a*N_z,1]); % keep return function of optimal policy for using in Howards

        % Split the fine index into its a1prime (interpolated) and a2prime (exact) parts
        Policy_fine1=rem(Policy(:)-1,N_a1prime)+1; % a1prime, in the post-GI (fine, within-window) index
        Policy_fine2=ceil(Policy(:)/N_a1prime); % a2prime index, on the coarse a2 grid
        Policy_L1a=ceil((Policy_fine1-1)/(n2short+1))-1;
        Policy_lowerind=max(Policy_L1a-vfoptions.maxaprimediff+a1primeshifter(:),1); % Policy_L1a is the index within the +-maxaprimediff window, so +a1primeshifter converts it to the index on the full a1_grid
        Policy_lowerprob=1- ((Policy_fine1-max(Policy_L1a,0)*(n2short+1))-1)/(n2short+1);
        % Joint index over a is a1+N_a1*(a2-1), so the upper interpolation point is the next one along
        indp = Policy_lowerind+N_a1*(Policy_fine2-1)+N_a_times_zind; % with all tomorrows z (a-z,zprime)

        T_E=sparse(azind2,[indp;indp+1],[Policy_lowerprob;1-Policy_lowerprob].*pi_z_big2,N_a*N_z,N_a*N_z);

        VKron=reshape(VKron,[N_a*N_z,1]);
        for h_c=1:vfoptions.howards
            VKron=Ftemp+DiscountFactorParamsVec*(T_E*VKron); % T_E already contains pi_z, so T_E*V is the expected continuation
        end
        VKron=reshape(VKron,[N_a,N_z]);
    end

    tempcounter=tempcounter+1;
end

%% Do another post-GI layer
% Note: is just a copy-paste of the previous post-GI layer code
% Only difference is that before we start there are a couple of lines of code to convert the policy
% back into being about the nearest rough grid index
while vfoptions.postGIrepeat>0
    vfoptions.postGIrepeat=vfoptions.postGIrepeat-1;

    % First, we switch the policy to be the nearest point on the rough grid
    Policy=reshape(Policy,[1,N_a,N_z]); % Howards can mess with the size
    Policy_a1=rem(Policy-1,N_a1prime)+1;
    Policy_a1=ceil((Policy_a1-1)/(n2short+1))-vfoptions.maxaprimediff+a1primeshifter;
    % ceil((Policy_a1-1)/(n2short+1))-vfoptions.maxaprimediff ranges -vfoptions.maxaprimediff:1:vfoptions.maxaprimediff


    %% Now that we have solved on the rough grid, we resolve on the fine grid
    % Based on solving a bunch of value fns with and without grid interpolation, the 'lower grid index'
    % with grid interpolation is always within a point or two of the solution on the rough grid. So here
    % we only consider +-vfoptions.maxaprimediff to set up the fine/interpolated a1prime_grid

    % First, create an a1prime_grid that is just the +-vfoptions.maxaprimediff
    n_a1primediff=1+2*vfoptions.maxaprimediff;
    N_a1primediff=prod(n_a1primediff);
    a1primeshifter=min(max(Policy_a1,1+vfoptions.maxaprimediff),N_a1-vfoptions.maxaprimediff);
    a1primeindex=(-vfoptions.maxaprimediff:1:vfoptions.maxaprimediff)' +a1primeshifter; % size n_aprime-by-n_a
    a1prime_grid=a1_grid(a1primeindex);
    % Second, interpolate this
    n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
    n_a1prime=n_a1primediff+(n_a1primediff-1)*vfoptions.ngridinterp;
    N_a1prime=prod(n_a1prime);
    a1prime_grid=interp1((1:1:N_a1primediff)',a1prime_grid,linspace(1,N_a1primediff,N_a1primediff+(N_a1primediff-1)*vfoptions.ngridinterp)');
    % Note: a1prime_grid is N_a1prime-by-N_a-by-N_z

    a1prime_grid=reshape(a1prime_grid,[N_a1prime,1,N_a1,N_a2,N_z]);

    EVinterpindex1=(1:1:N_a1primediff)';
    EVinterpindex2=linspace(1,N_a1primediff,N_a1primediff+(N_a1primediff-1)*vfoptions.ngridinterp)';

    N_aprime=N_a1prime*N_a2;
    N_aprimediff=N_a1primediff*N_a2;
    aprimeindex=repmat(a1primeindex,N_a2,1,1)+N_a1*repelem((0:1:N_a2-1)',N_a1primediff,1,1);

    ReturnMatrixfine=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, n_z, a1prime_grid, a2_grid, a1_grid, a2_grid, z_gridvals, ReturnFnParams, 2);

    % For Howards we need
    addindexforazfine=gpuArray(N_aprime*(0:1:N_a-1)'+N_aprime*N_a*(0:1:N_z-1));

    pi_z_alt2=shiftdim(pi_z,-2);

    % Setup specific to the sparse-matrix Howards
    azind2=repmat(gpuArray(1:1:N_a*N_z)',2,N_z); % (a-z-2,zprime)
    pi_z_big2=gpuArray(repmat(pi_z_big1,2,1)); % (a-z-2,zprime)

    %% Now switch to considering the fine/interpolated aprime_grid
    currdist=1; % force going into the next while loop at least one iteration
    tempcounter=1; % reset tempcounter
    while currdist>vfoptions.tolerance && tempcounter<=vfoptions.maxiter
        VKronold=VKron;

        % Switch VKron into being over vfoptions.maxaprimediff
        EVpre=reshape(VKron(aprimeindex,:),[N_aprimediff,N_a,N_z,N_z]); % last dimension is zprime

        % Calc the condl expectation term (except beta), which depends on z but not on control variables
        EV=EVpre.*pi_z_alt2;
        EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV=squeeze(sum(EV,4)); % sum over z', leaving a singular second dimension
        % EV is now [N_aprimediff,N_a,N_z]
        % Interpolate EV over aprime_grid

        EVinterp=reshape(interp1(EVinterpindex1,reshape(EV,[N_a1primediff,N_a2,N_a,N_z]),EVinterpindex2),[N_aprime,N_a,N_z]);

        entireRHS=ReturnMatrixfine+DiscountFactorParamsVec*EVinterp; % aprime by a by z

        % Calc the max and it's index
        [VKron,Policy]=max(entireRHS,[],1);
        VKron=shiftdim(VKron,1); % a by z

        VKrondist=VKron(:)-VKronold(:);
        VKrondist(isnan(VKrondist))=0;
        currdist=max(abs(VKrondist));

        % Use Howards Improvement, iterating with the sparse transition matrix (except for first few and last few iterations, as it is not a good idea there)
        if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
            tempmaxindex=shiftdim(Policy,1)+addindexforazfine; % aprime index, add the index for a and z
            Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a*N_z,1]); % keep return function of optimal policy for using in Howards

            % Split the fine index into its a1prime (interpolated) and a2prime (exact) parts
            Policy_fine1=rem(Policy(:)-1,N_a1prime)+1; % a1prime, in the post-GI (fine, within-window) index
            Policy_fine2=ceil(Policy(:)/N_a1prime); % a2prime index, on the coarse a2 grid
            Policy_L1a=ceil((Policy_fine1-1)/(n2short+1))-1;
            Policy_lowerind=max(Policy_L1a-vfoptions.maxaprimediff+a1primeshifter(:),1); % Policy_L1a is the index within the +-maxaprimediff window, so +a1primeshifter converts it to the index on the full a1_grid
            Policy_lowerprob=1- ((Policy_fine1-max(Policy_L1a,0)*(n2short+1))-1)/(n2short+1);
            % Joint index over a is a1+N_a1*(a2-1), so the upper interpolation point is the next one along
            indp = Policy_lowerind+N_a1*(Policy_fine2-1)+N_a_times_zind; % with all tomorrows z (a-z,zprime)

            T_E=sparse(azind2,[indp;indp+1],[Policy_lowerprob;1-Policy_lowerprob].*pi_z_big2,N_a*N_z,N_a*N_z);

            VKron=reshape(VKron,[N_a*N_z,1]);
            for h_c=1:vfoptions.howards
                VKron=Ftemp+DiscountFactorParamsVec*(T_E*VKron); % T_E already contains pi_z, so T_E*V is the expected continuation
            end
            VKron=reshape(VKron,[N_a,N_z]);
        end

        tempcounter=tempcounter+1;
    end
end


%% Switch policy to lower grid index and L2 index (is currently index on fine grid)
fineindex=reshape(Policy,[1,N_a,N_z]);
Policy=zeros(4,N_a,N_z,'gpuArray'); % +1 channel for PolicyL2flag
fineindexvec1=rem(fineindex-1,N_a1prime)+1; % a1prime, but in post-GI index
fineindexvec2=ceil(fineindex/N_a1prime); % a2prime index

fineindexvec1=reshape(fineindexvec1,[N_a*N_z,1]);
L1a=ceil((fineindexvec1-1)/(n2short+1))-1; % this ranges -vfoptions.maxaprimediff:1:vfoptions.maxaprimediff
L1=max(L1a-vfoptions.maxaprimediff+1+a1primeshifter(:)-1,1); % lower grid point index (on the full grid), so this ranges 0 to n_a-1
L1intermediate=max(L1a,0)+1; % lower grid point index (on the small grid, in form so we can get L2)
L2=fineindexvec1-(L1intermediate-1)*(n2short+1); % L2 index

Policy(1,:,:)=reshape(L1,[1,N_a,N_z]);
Policy(2,:,:)=fineindexvec2;
Policy(3,:,:)=reshape(L2,[1,N_a,N_z]);

% L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
% Computed once, post-convergence. Flag checks a1-coarse endpoints at chosen (L1 segment, a2prime).
% ReturnMatrixfine is [N_aprime,N_a,N_z] with N_aprime=N_a1prime*N_a2 and aprime packed as a1prime_fine + N_a1prime*(a2prime-1).
fineindexvec2_flat = reshape(fineindexvec2,[N_a*N_z,1]);
aprime_lower = (L1intermediate-1)*(n2short+1) + 1 + N_a1prime*(fineindexvec2_flat-1);
aprime_upper = L1intermediate*(n2short+1)     + 1 + N_a1prime*(fineindexvec2_flat-1);
linidx_lower = aprime_lower + reshape(addindexforazfine,[N_a*N_z,1]);
linidx_upper = aprime_upper + reshape(addindexforazfine,[N_a*N_z,1]);
isInfLower = (ReturnMatrixfine(linidx_lower) == -Inf);
isInfUpper = (ReturnMatrixfine(linidx_upper) == -Inf);
inInterior = (L2 >= 2) & (L2 <= n2short+1);
Policy(4,:,:) = reshape(2 + (inInterior & isInfLower) - (inInterior & isInfUpper), [1,N_a,N_z]);

% Note: unlike Howards-greedy (which solves the linear system), iterating with the sparse T_E is
% fine when V contains values of -Inf, so there is no need to warn about that here.
if currdist > vfoptions.tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, vfoptions.tolerance)
end

end
