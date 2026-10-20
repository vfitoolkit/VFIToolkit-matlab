function [VKron, Policy]=ValueFnIter_InfHorz_postGI_nod_e_raw(VKron, n_a, n_z, n_e, a_grid, z_gridvals, e_gridvals, pi_z, pi_e, DiscountFactorParamsVec, ReturnFn, ReturnFnParams, vfoptions)
% postGI, no decision variable, with z and iid e
% Take a multigrid approach, using just a_grid for aprime until near convergence, then switch to use the fine
% (interpolated) grid for aprime, but only within +-vfoptions.maxaprimediff of what was optimal on the rough grid.
% VKron is a-by-z-by-e, pi_e is a column vector of probabilities
% e is iid, so the expectation of next period value is EV(aprime,z): it does not depend on e. This is computed first,
% and only then evaluated at the aprime (and interpolated aprime) points.

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_a, n_a, n_z, n_e, a_grid, a_grid, z_gridvals, e_gridvals, ReturnFnParams); % aprime by a by z by e

pi_z_alt=shiftdim(pi_z',-1); % 1-by-zprime-by-z
pi_e_alt=shiftdim(pi_e,-2); % 1-by-1-by-e

addindexforaze=gpuArray(N_a*(0:1:N_a-1)'+N_a*N_a*(0:1:N_z-1)+N_a*N_a*N_z*shiftdim((0:1:N_e-1),-1));
addindexforz=gpuArray(repmat(N_a*(0:1:N_z-1),N_a,1,N_e)); % a-by-z-by-e, for indexing EV (which is aprime-by-z)

%%
tempcounter=1;
currdist=Inf;

%% First, just consider a_grid for next period
while currdist>(vfoptions.multigridswitch*vfoptions.tolerance) && tempcounter<=vfoptions.maxiter
    VKronold=VKron;

    % Calc the condl expectation term (except beta), which depends on z but not on control variables
    % First take the expectation over e', which is iid
    EV=VKronold.*pi_e_alt;
    EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
    EV=sum(EV,3); % sum over e', aprime-by-zprime
    % Then the expectation over z' conditional on z
    EV=EV.*pi_z_alt;
    EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension, aprime-by-1-by-z

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; % aprime by a by z by e (EV autoexpands over a and e)

    % Calc the max and it's index
    [VKron,Policy]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1); % a by z by e

    VKrondist=VKron(:)-VKronold(:);
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
        tempmaxindex=shiftdim(Policy,1)+addindexforaze; % aprime index, add the index for a, z and e
        Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a,N_z,N_e]); % keep return function of optimal policy for using in Howards
        Policyz=shiftdim(Policy,1)+addindexforz; % aprime index, add the index for z (this shape is just convenient for Howards)
        for Howards_counter=1:vfoptions.howards
            EVKrontemp=VKron.*pi_e_alt;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=sum(EVKrontemp,3); % sum over e', aprime-by-zprime
            EVKrontemp=EVKrontemp.*pi_z_alt;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]); % sum over z', aprime-by-z
            VKron=Ftemp+DiscountFactorParamsVec*EVKrontemp(Policyz);
        end
    end

    tempcounter=tempcounter+1;

end
Policy=reshape(Policy,[1,N_a,N_z,N_e]); % Howards can mess with the size

%% Now that we have solved on the rough grid, we resolve on the fine grid
% Based on solving a bunch of value fns with and without grid
% interpolation, the 'lower grid index' with grid interpolation is always
% within a point or two of the solution on the rough grid. So here we only
% consider +-vfoptions.maxaprimediff to set up the fine/interpolated aprime_grid

% First, create an aprime_grid that is just the +-vfoptions.maxaprimediff
% Note: this code is for models with a single endogenous state
n_aprimediff=1+2*vfoptions.maxaprimediff;
N_aprimediff=prod(n_aprimediff);
aprimeshifter=min(max(Policy,1+vfoptions.maxaprimediff),N_a-vfoptions.maxaprimediff);
aprimeindex=(-vfoptions.maxaprimediff:1:vfoptions.maxaprimediff)' +aprimeshifter; % size n_aprimediff-by-n_a-by-n_z-by-n_e
aprime_grid=a_grid(aprimeindex);
% Second, interpolate this
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n_aprime=n_aprimediff+(n_aprimediff-1)*vfoptions.ngridinterp;
N_aprime=prod(n_aprime);
aprime_grid=interp1((1:1:N_aprimediff)',aprime_grid,linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp)');
% Note: aprime_grid is N_aprime-by-N_a-by-N_z-by-N_e

ReturnMatrixfine=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, n_z, n_e, aprime_grid, a_grid, z_gridvals, e_gridvals, ReturnFnParams,2); % aprime by a by z by e

EVinterpindex1=(1:1:N_aprimediff)';
EVinterpindex2=linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp)';

% EV is aprime-by-z, so to evaluate it on the window add the index for z
aprimeindexz=aprimeindex+N_a*shiftdim((0:1:N_z-1),-1); % n_aprimediff-by-n_a-by-n_z-by-n_e

% For Howards we need
addindexforazefine=gpuArray(N_aprime*(0:1:N_a-1)'+N_aprime*N_a*(0:1:N_z-1)+N_aprime*N_a*N_z*shiftdim((0:1:N_e-1),-1));
zindexforhowards=reshape(addindexforz,[N_a*N_z*N_e,1]); % index for z, for indexing EV (which is aprime-by-z)


%% Now switch to considering the fine/interpolated aprime_grid
tempcounter=1; % reset the counter
currdist=1; % force going into the next while loop at least one iteration
while currdist>vfoptions.tolerance && tempcounter<=vfoptions.maxiter
    VKronold=VKron;

    % Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=VKron.*pi_e_alt;
    EV(isnan(EV))=0;
    EV=sum(EV,3); % sum over e', aprime-by-zprime
    EV=EV.*pi_z_alt;
    EV(isnan(EV))=0;
    EV=reshape(sum(EV,2),[N_a,N_z]); % sum over z', aprime-by-z
    % Switch EV into being over vfoptions.maxaprimediff
    EV=EV(aprimeindexz); % [N_aprimediff,N_a,N_z,N_e]
    % Interpolate EV over aprime_grid
    EVinterp=interp1(EVinterpindex1,EV,EVinterpindex2);

    entireRHS=ReturnMatrixfine+DiscountFactorParamsVec*EVinterp; % aprime by a by z by e

    % Calc the max and it's index
    [VKron,Policy]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1); % a by z by e

    VKrondist=VKron(:)-VKronold(:);
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
        tempmaxindex=shiftdim(Policy,1)+addindexforazefine; % aprime index, add the index for a, z and e, size is [N_a,N_z,N_e]
        Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a*N_z*N_e,1]); % keep return function of optimal policy for using in Howards
        % Resolve the interpolation to a (lower grid index, weight) pair once, here
        Policy_L1a=ceil((Policy(:)-1)/(n2short+1))-1;
        % max(Policy_L1a,0) so the lower index and the weight agree at the window's bottom node
        Policy_lowerind=max(Policy_L1a,0)-vfoptions.maxaprimediff+aprimeshifter(:); % index on the full a_grid
        Policy_lowerprob=1- ((Policy(:)-max(Policy_L1a,0)*(n2short+1))-1)/(n2short+1);
        Policy_lowerindz=Policy_lowerind+zindexforhowards; % add the index for z, for indexing EV
        for Howards_counter=1:vfoptions.howards
            EVKrontemp=VKron.*pi_e_alt;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=sum(EVKrontemp,3); % sum over e', aprime-by-zprime
            EVKrontemp=EVKrontemp.*pi_z_alt;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]); % sum over z', aprime-by-z
            EVlower=Policy_lowerprob.*EVKrontemp(Policy_lowerindz);
            EVlower(Policy_lowerprob==0)=0; % zero weight times -Inf gives NaN, this replaces them with zeros
            EVupper=(1-Policy_lowerprob).*EVKrontemp(Policy_lowerindz+1);
            EVupper(Policy_lowerprob==1)=0; % zero weight times -Inf gives NaN, this replaces them with zeros
            VKron=reshape(Ftemp+DiscountFactorParamsVec*(EVlower+EVupper),[N_a,N_z,N_e]);
        end
    end

    tempcounter=tempcounter+1;
end


%% Do another post-GI layer
% Note: is just a copy-paste of the previous post-GI layer code
% Only difference that before we start there are two lines of code to convert Policy back into being about the nearest rough grid index
while vfoptions.postGIrepeat>0
    vfoptions.postGIrepeat=vfoptions.postGIrepeat-1;

    % First, we switch Policy to be the nearest point on the rough grid
    Policy=reshape(Policy,[1,N_a,N_z,N_e]); % Howards can mess with the size
    Policy=ceil((Policy-1)/(n2short+1))-vfoptions.maxaprimediff+aprimeshifter;
    % ceil((Policy-1)/(n2short+1))-vfoptions.maxaprimediff ranges -vfoptions.maxaprimediff:1:vfoptions.maxaprimediff

    % First, create an aprime_grid that is just the +-vfoptions.maxaprimediff
    % Note: this code is for models with a single endogenous state
    n_aprimediff=1+2*vfoptions.maxaprimediff;
    N_aprimediff=prod(n_aprimediff);
    aprimeshifter=min(max(Policy,1+vfoptions.maxaprimediff),N_a-vfoptions.maxaprimediff);
    aprimeindex=(-vfoptions.maxaprimediff:1:vfoptions.maxaprimediff)' +aprimeshifter; % size n_aprimediff-by-n_a-by-n_z-by-n_e
    aprime_grid=a_grid(aprimeindex);
    % Second, interpolate this
    n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
    n_aprime=n_aprimediff+(n_aprimediff-1)*vfoptions.ngridinterp;
    N_aprime=prod(n_aprime);
    aprime_grid=interp1((1:1:N_aprimediff)',aprime_grid,linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp)');
    % Note: aprime_grid is N_aprime-by-N_a-by-N_z-by-N_e

    ReturnMatrixfine=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, n_z, n_e, aprime_grid, a_grid, z_gridvals, e_gridvals, ReturnFnParams,2); % aprime by a by z by e

    EVinterpindex1=(1:1:N_aprimediff)';
    EVinterpindex2=linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp)';

    % EV is aprime-by-z, so to evaluate it on the window add the index for z
    aprimeindexz=aprimeindex+N_a*shiftdim((0:1:N_z-1),-1); % n_aprimediff-by-n_a-by-n_z-by-n_e

    % For Howards we need
    addindexforazefine=gpuArray(N_aprime*(0:1:N_a-1)'+N_aprime*N_a*(0:1:N_z-1)+N_aprime*N_a*N_z*shiftdim((0:1:N_e-1),-1));
    zindexforhowards=reshape(addindexforz,[N_a*N_z*N_e,1]); % index for z, for indexing EV (which is aprime-by-z)


    %% Now switch to considering the fine/interpolated aprime_grid
    tempcounter=1; % reset the counter
    currdist=1; % force going into the next while loop at least one iteration
    while currdist>vfoptions.tolerance && tempcounter<=vfoptions.maxiter
        VKronold=VKron;

        % Calc the condl expectation term (except beta), which depends on z but not on control variables
        EV=VKron.*pi_e_alt;
        EV(isnan(EV))=0;
        EV=sum(EV,3); % sum over e', aprime-by-zprime
        EV=EV.*pi_z_alt;
        EV(isnan(EV))=0;
        EV=reshape(sum(EV,2),[N_a,N_z]); % sum over z', aprime-by-z
        % Switch EV into being over vfoptions.maxaprimediff
        EV=EV(aprimeindexz); % [N_aprimediff,N_a,N_z,N_e]
        % Interpolate EV over aprime_grid
        EVinterp=interp1(EVinterpindex1,EV,EVinterpindex2);

        entireRHS=ReturnMatrixfine+DiscountFactorParamsVec*EVinterp; % aprime by a by z by e

        % Calc the max and it's index
        [VKron,Policy]=max(entireRHS,[],1);
        VKron=shiftdim(VKron,1); % a by z by e

        VKrondist=VKron(:)-VKronold(:);
        VKrondist(isnan(VKrondist))=0;
        currdist=max(abs(VKrondist));

        % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
        if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
            tempmaxindex=shiftdim(Policy,1)+addindexforazefine; % aprime index, add the index for a, z and e, size is [N_a,N_z,N_e]
            Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a*N_z*N_e,1]); % keep return function of optimal policy for using in Howards
            % Resolve the interpolation to a (lower grid index, weight) pair once, here
            Policy_L1a=ceil((Policy(:)-1)/(n2short+1))-1;
            % max(Policy_L1a,0) so the lower index and the weight agree at the window's bottom node
            Policy_lowerind=max(Policy_L1a,0)-vfoptions.maxaprimediff+aprimeshifter(:); % index on the full a_grid
            Policy_lowerprob=1- ((Policy(:)-max(Policy_L1a,0)*(n2short+1))-1)/(n2short+1);
            Policy_lowerindz=Policy_lowerind+zindexforhowards; % add the index for z, for indexing EV
            for Howards_counter=1:vfoptions.howards
                EVKrontemp=VKron.*pi_e_alt;
                EVKrontemp(isnan(EVKrontemp))=0;
                EVKrontemp=sum(EVKrontemp,3); % sum over e', aprime-by-zprime
                EVKrontemp=EVKrontemp.*pi_z_alt;
                EVKrontemp(isnan(EVKrontemp))=0;
                EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]); % sum over z', aprime-by-z
                EVlower=Policy_lowerprob.*EVKrontemp(Policy_lowerindz);
                EVlower(Policy_lowerprob==0)=0; % zero weight times -Inf gives NaN, this replaces them with zeros
                EVupper=(1-Policy_lowerprob).*EVKrontemp(Policy_lowerindz+1);
                EVupper(Policy_lowerprob==1)=0; % zero weight times -Inf gives NaN, this replaces them with zeros
                VKron=reshape(Ftemp+DiscountFactorParamsVec*(EVlower+EVupper),[N_a,N_z,N_e]);
            end
        end

        tempcounter=tempcounter+1;
    end
end


%% Switch policy to lower grid index and L2 index (is currently index on fine grid)
% Separate Policy into L1 and L2
fineindex=reshape(Policy,[N_a*N_z*N_e,1]);
L1a=ceil((fineindex-1)/(n2short+1))-1;  % this ranges -1:0:2*vfoptions.maxaprimediff-1
% the max(L1a,0) matters only at the window's bottom node, where L1a is -1; L2 below is built off the same thing, so L1 has to be too
L1=max(L1a,0)-vfoptions.maxaprimediff+aprimeshifter(:); % lower grid point index on the full grid; the window bounds put this in 1:N_a-1, so no clamp is needed
L1intermediate=max(L1a,0)+1; % lower grid point index (on the small grid, in form so we can get L2)
L2=fineindex-(L1intermediate-1)*(n2short+1); % L2 index

Policy=zeros(3,N_a,N_z,N_e,'gpuArray'); % +1 channel for PolicyL2flag
Policy(1,:,:,:)=reshape(L1,[1,N_a,N_z,N_e]);
Policy(2,:,:,:)=reshape(L2,[1,N_a,N_z,N_e]);

% L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
% Computed once, post-convergence, using final ReturnMatrixfine and final aprimeshifter
fineindex_lower = (L1intermediate-1)*(n2short+1) + 1;          % fine pos of lower coarse for chosen L1 segment
fineindex_upper = L1intermediate*(n2short+1) + 1;              % fine pos of upper coarse
linidx_lower = reshape(fineindex_lower,[N_a,N_z,N_e]) + addindexforazefine;
linidx_upper = reshape(fineindex_upper,[N_a,N_z,N_e]) + addindexforazefine;
isInfLower = (ReturnMatrixfine(linidx_lower(:)) == -Inf);
isInfUpper = (ReturnMatrixfine(linidx_upper(:)) == -Inf);
inInterior = (L2 >= 2) & (L2 <= n2short+1);
Policy(3,:,:,:) = reshape(2 + (inInterior & isInfLower) - (inInterior & isInfUpper), [1,N_a,N_z,N_e]);

if currdist > vfoptions.tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, vfoptions.tolerance)
end

end
