function [VKron, Policy]=ValueFnIter_InfHorz_RelativeVFI_postGI_nod_noz_raw(VKron, n_a, a_grid, DiscountFactorParamsVec, ReturnFn, ReturnFnParams, vfoptions)
% Relative VFI (Bray, 2019): identical iterations to ValueFnIter_InfHorz_postGI_nod_noz_raw, but every loop stops on the
% span over a of V-Vold within each exogenous state (tolerances apply to that span), and V is recovered in levels after
% each fine-grid loop.
% preGI: create the whole ReturnMatrix based on aprime
% Then take a multigrid approach, using just a_grid for aprime until near
% convergence, then switch to use the fine grid for aprime.

N_a=prod(n_a);

ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_noz(ReturnFn, n_a, n_a, a_grid, a_grid, ReturnFnParams);

addindexfora=gpuArray(N_a*(0:1:N_a-1)');

%%
tempcounter=1;
currdist=Inf;

%% First, just consider a_grid for next period
while currdist>(vfoptions.multigridswitch*vfoptions.tolerance) && tempcounter<=vfoptions.maxiter
    VKronold=VKron;

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*VKronold; % aprime by a

    % Calc the max and it's index
    [VKron,Policy]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1); % a by 1

    VKrondist=reshape(VKron,[N_a,numel(VKron)/N_a])-reshape(VKronold,[N_a,numel(VKronold)/N_a]); % a-by-(exogenous states)
    currdist=max(max(VKrondist,[],1)-min(VKrondist,[],1),[],'all'); % endogenous VFI: the span over a, within each exogenous state (max and min omit the NaN from -Inf minus -Inf)

    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
        tempmaxindex=shiftdim(Policy,1)+addindexfora; % aprime index, add the index for a
        Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a,1]); % keep return function of optimal policy for using in Howards
        Policy=Policy(:); % a by 1 (this shape is just convenient for Howards)
        for Howards_counter=1:vfoptions.howards
            EVKrontemp=VKron(Policy,:);
            VKron=Ftemp+DiscountFactorParamsVec*EVKrontemp; % interpolate EV
        end
    end

    tempcounter=tempcounter+1;

end
Policy=reshape(Policy,[1,N_a]); % Howards can mess with the size

%% Now that we have solved on the rough grid, we resolve on the fine grid
% Based on solving a bunch of value fns with and without grid
% interpolation, the 'lower grid index' with grid interpolation is always
% within a point or two of the solution on the rough grid. So here we only
% consider +-vfoptions.maxaprimediff to set up the fine/interpolated aprime_grid

% Current optimal aprime is Policy_a
% So create an aprime_grid that is just an interpolation within +-vfoptions.maxaprimediff

% First, create an aprime_grid that is just the +-vfoptions.maxaprimediff
% Note: this code is for models with a single endogenous state
n_aprimediff=1+2*vfoptions.maxaprimediff;
N_aprimediff=prod(n_aprimediff);
aprimeshifter=min(max(Policy,1+vfoptions.maxaprimediff),N_a-vfoptions.maxaprimediff);
aprimeindex=(-vfoptions.maxaprimediff:1:vfoptions.maxaprimediff)' +aprimeshifter; % size n_aprime-by-n_a
aprime_grid=a_grid(aprimeindex);
% Second, interpolate this
% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n_aprime=n_aprimediff+(n_aprimediff-1)*vfoptions.ngridinterp;
N_aprime=prod(n_aprime);
aprime_grid=interp1((1:1:N_aprimediff)',aprime_grid,linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp)');
% Note: aprime_grid is N_aprime-by-N_a-by-N_z

ReturnMatrixfine=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, aprime_grid, a_grid, ReturnFnParams);

EVinterpindex1=(1:1:N_aprimediff)';
EVinterpindex2=linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp)';

% For Howards we need
addindexforafine=gpuArray(N_aprime*(0:1:N_a-1)');

%% Now switch to considering the fine/interpolated aprime_grid
tempcounter=1; % reset the counter
currdist=1; % force going into the next while loop at least one iteration
while currdist>vfoptions.tolerance && tempcounter<=vfoptions.maxiter
    VKronold=VKron;

    % Switch VKron into being over vfoptions.maxaprimediff
    EV=reshape(VKron(aprimeindex,:),[N_aprimediff,N_a]); % last dimension is zprime

    % Interpolate EV over aprime_grid
    EVinterp=interp1(EVinterpindex1,EV,EVinterpindex2);

    entireRHS=ReturnMatrixfine+DiscountFactorParamsVec*EVinterp; % aprime by a

    % Calc the max and it's index
    [VKron,Policy]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1); % a by 1

    VKrondist=reshape(VKron,[N_a,numel(VKron)/N_a])-reshape(VKronold,[N_a,numel(VKronold)/N_a]); % a-by-(exogenous states)
    currdist=max(max(VKrondist,[],1)-min(VKrondist,[],1),[],'all'); % endogenous VFI: the span over a, within each exogenous state (max and min omit the NaN from -Inf minus -Inf)

    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
        tempmaxindex=shiftdim(Policy,1)+addindexforafine; % aprime index, add the index for a, size is [N_a,1]
        Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a,1]); % keep return function of optimal policy for using in Howards
        % Resolve the interpolation to a (lower grid index, weight) pair once, here, instead of
        % re-interpolating the whole of EV inside every Howards step. The old version built an
        % [N_aprime,...] array each step and then kept one row per state, which at n_a=500, n_z=75
        % came to 13.5GB and ran the GPU out of memory.
        Policy_L1a=ceil((Policy(:)-1)/(n2short+1))-1;
        % as above, max(Policy_L1a,0) so the lower index and the weight agree at the window's bottom node
        Policy_lowerind=max(Policy_L1a,0)-vfoptions.maxaprimediff+aprimeshifter(:); % index on the full a_grid
        Policy_lowerprob=1- ((Policy(:)-max(Policy_L1a,0)*(n2short+1))-1)/(n2short+1);
        lowerweightzero=(Policy_lowerprob==0); % the chosen point is a coarse node, coded as all the weight on the upper node
        upperweightzero=(Policy_lowerprob==1); % the chosen point is the window's bottom node
        for Howards_counter=1:vfoptions.howards
            EVlower=Policy_lowerprob.*VKron(Policy_lowerind); % [N_a,1]
            EVlower(lowerweightzero)=0; % zero weight times -Inf gives NaN, a zero weight contributes nothing
            EVupper=(1-Policy_lowerprob).*VKron(Policy_lowerind+1);
            EVupper(upperweightzero)=0;
            EVKrontemp=EVlower+EVupper;
            VKron=Ftemp+DiscountFactorParamsVec*EVKrontemp;
        end
    end

    tempcounter=tempcounter+1;
end
% Endogenous VFI: recover V in levels. Once converged, VKron=T(VKronold) (the Howards step is skipped in the last
% iteration) and VKron-VKronold is a function of the exogenous states only, up to the tolerance; call it dexo (the
% mean over a of the finite entries). VKronold+c is the fixed point when c=dexo+beta*E[c'], and V=T(VKronold+c)=VKron+beta*E[c'].
% (The interpolation weights sum to one, so adding a function of the exogenous states to V adds it to the interpolated V too.)
VKrondist=reshape(VKron,[N_a,numel(VKron)/N_a])-reshape(VKronold,[N_a,numel(VKronold)/N_a]);
finitedist=isfinite(VKrondist);
VKrondist(~finitedist)=0;
dexo=sum(VKrondist,1)./max(sum(finitedist,1),1); % 1-by-(exogenous states), zero where there is no finite entry
% No exogenous states, so c=dexo/(1-beta) and E[c']=c (this is relative VFI)
VKron=VKron+DiscountFactorParamsVec*dexo/(1-DiscountFactorParamsVec);




%% Do another post-GI layer
% Note: is just a copy-paste of the previous post-GI layer code
% Only difference that before we start there are two lines of code to
% convert Policy_a back into being about the nearest rough grid index
while vfoptions.postGIrepeat>0
    vfoptions.postGIrepeat=vfoptions.postGIrepeat-1;

    % Current optimal aprime is Policy
    % So create an aprime_grid that is just an interpolation within +-vfoptions.maxaprimediff

    % First, we switch Policy_a to be the nearest point on the rough grid
    Policy=reshape(Policy,[1,N_a]); % Howards can mess with the size
    Policy=ceil((Policy-1)/(n2short+1))-vfoptions.maxaprimediff+aprimeshifter;
    % ceil((Policy-1)/(n2short+1))-vfoptions.maxaprimediff ranges -vfoptions.maxaprimediff:1:vfoptions.maxaprimediff

    % First, create an aprime_grid that is just the +-vfoptions.maxaprimediff
    % Note: this code is for models with a single endogenous state
    n_aprimediff=1+2*vfoptions.maxaprimediff;
    N_aprimediff=prod(n_aprimediff);
    aprimeshifter=min(max(Policy,1+vfoptions.maxaprimediff),N_a-vfoptions.maxaprimediff);
    aprimeindex=(-vfoptions.maxaprimediff:1:vfoptions.maxaprimediff)' +aprimeshifter; % size n_aprime-by-n_a
    aprime_grid=a_grid(aprimeindex);
    % Second, interpolate this
    % Grid interpolation
    % vfoptions.ngridinterp=9;
    n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
    n_aprime=n_aprimediff+(n_aprimediff-1)*vfoptions.ngridinterp;
    N_aprime=prod(n_aprime);
    aprime_grid=interp1((1:1:N_aprimediff)',aprime_grid,linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp)');
    % Note: aprime_grid is N_aprime-by-N_a-by-N_z

    ReturnMatrixfine=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, aprime_grid, a_grid, ReturnFnParams);

    EVinterpindex1=(1:1:N_aprimediff)';
    EVinterpindex2=linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp)';

    % For Howards we need
    addindexforafine=gpuArray(N_aprime*(0:1:N_a-1)');


    %% Now switch to considering the fine/interpolated aprime_grid
    tempcounter=1; % reset the counter
    currdist=1; % force going into the next while loop at least one iteration
    while currdist>vfoptions.tolerance && tempcounter<=vfoptions.maxiter
        VKronold=VKron;

        % Switch VKron into being over vfoptions.maxaprimediff
        EV=reshape(VKron(aprimeindex),[N_aprimediff,N_a]); % last dimension is zprime
        % Interpolate EV over aprime_grid
        EVinterp=interp1(EVinterpindex1,EV,EVinterpindex2);

        entireRHS=ReturnMatrixfine+DiscountFactorParamsVec*EVinterp; % aprime by a

        % Calc the max and it's index
        [VKron,Policy]=max(entireRHS,[],1);
        VKron=shiftdim(VKron,1); % a by 1

        VKrondist=reshape(VKron,[N_a,numel(VKron)/N_a])-reshape(VKronold,[N_a,numel(VKronold)/N_a]); % a-by-(exogenous states)
        currdist=max(max(VKrondist,[],1)-min(VKrondist,[],1),[],'all'); % endogenous VFI: the span over a, within each exogenous state (max and min omit the NaN from -Inf minus -Inf)

        % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
        if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
            tempmaxindex=shiftdim(Policy,1)+addindexforafine; % aprime index, add the index for a, size is [N_a,1]
            Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a,1]); % keep return function of optimal policy for using in Howards
            % Resolve the interpolation to a (lower grid index, weight) pair once, here, instead of
            % re-interpolating the whole of EV inside every Howards step. The old version built an
            % [N_aprime,...] array each step and then kept one row per state, which at n_a=500, n_z=75
            % came to 13.5GB and ran the GPU out of memory.
            Policy_L1a=ceil((Policy(:)-1)/(n2short+1))-1;
            Policy_lowerind=max(Policy_L1a,0)-vfoptions.maxaprimediff+aprimeshifter(:); % index on the full a_grid
            Policy_lowerprob=1- ((Policy(:)-max(Policy_L1a,0)*(n2short+1))-1)/(n2short+1);
            lowerweightzero=(Policy_lowerprob==0); % the chosen point is a coarse node, coded as all the weight on the upper node
            upperweightzero=(Policy_lowerprob==1); % the chosen point is the window's bottom node
            for Howards_counter=1:vfoptions.howards
                EVlower=Policy_lowerprob.*VKron(Policy_lowerind); % [N_a,1]
                EVlower(lowerweightzero)=0; % zero weight times -Inf gives NaN, a zero weight contributes nothing
                EVupper=(1-Policy_lowerprob).*VKron(Policy_lowerind+1);
                EVupper(upperweightzero)=0;
                EVKrontemp=EVlower+EVupper;
                VKron=Ftemp+DiscountFactorParamsVec*EVKrontemp;
            end
        end

        tempcounter=tempcounter+1;
    end
    % Endogenous VFI: recover V in levels. Once converged, VKron=T(VKronold) (the Howards step is skipped in the last
    % iteration) and VKron-VKronold is a function of the exogenous states only, up to the tolerance; call it dexo (the
    % mean over a of the finite entries). VKronold+c is the fixed point when c=dexo+beta*E[c'], and V=T(VKronold+c)=VKron+beta*E[c'].
    % (The interpolation weights sum to one, so adding a function of the exogenous states to V adds it to the interpolated V too.)
    VKrondist=reshape(VKron,[N_a,numel(VKron)/N_a])-reshape(VKronold,[N_a,numel(VKronold)/N_a]);
    finitedist=isfinite(VKrondist);
    VKrondist(~finitedist)=0;
    dexo=sum(VKrondist,1)./max(sum(finitedist,1),1); % 1-by-(exogenous states), zero where there is no finite entry
    % No exogenous states, so c=dexo/(1-beta) and E[c']=c (this is relative VFI)
    VKron=VKron+DiscountFactorParamsVec*dexo/(1-DiscountFactorParamsVec);

end


%% Switch policy to lower grid index and L2 index (is currently index on fine grid)
% Separate Policy into L1 and L2
fineindex=reshape(Policy,[N_a,1]);
L1a=ceil((fineindex-1)/(n2short+1))-1;  % this ranges -1:0:2*vfoptions.maxaprimediff-1
% (L1a-vfoptions.maxaprimediff+1) ranges -vfoptions.maxaprimediff:1:vfoptions.maxaprimediff
% the max(L1a,0) matters only at the window's bottom node, where L1a is -1; L2 below is built off the same thing, so L1 has to be too
L1=max(L1a,0)-vfoptions.maxaprimediff+aprimeshifter(:); % lower grid point index on the full grid; the window bounds put this in 1:N_a-1, so no clamp is needed
L1intermediate=max(L1a,0)+1; % lower grid point index (on the small grid, in form so we can get L2)
L2=fineindex-(L1intermediate-1)*(n2short+1); % L2 index

Policy=zeros(3,N_a,'gpuArray'); % +1 channel for PolicyL2flag
Policy(1,:)=reshape(L1,[1,N_a]);
Policy(2,:)=reshape(L2,[1,N_a]);

% L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
% Computed once, post-convergence, using final ReturnMatrixfine and final aprimeshifter
fineindex_lower = (L1intermediate-1)*(n2short+1) + 1;          % fine pos of lower coarse for chosen L1 segment
fineindex_upper = L1intermediate*(n2short+1) + 1;              % fine pos of upper coarse
linidx_lower = reshape(fineindex_lower,[N_a,1]) + addindexforafine;
linidx_upper = reshape(fineindex_upper,[N_a,1]) + addindexforafine;
isInfLower = (ReturnMatrixfine(linidx_lower(:)) == -Inf);
isInfUpper = (ReturnMatrixfine(linidx_upper(:)) == -Inf);
inInterior = (L2 >= 2) & (L2 <= n2short+1);
Policy(3,:,:) = reshape(2 + (inInterior & isInfLower) - (inInterior & isInfUpper), [1,N_a,1]);

if currdist > vfoptions.tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, vfoptions.tolerance)
end

end
