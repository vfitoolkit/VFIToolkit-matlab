function [VKron,Policy]=ValueFnIter_InfHorz_Refine_postGI_noz_raw(VKron,n_d,n_a,d_gridvals,a_grid,ReturnFn,DiscountFactorParamsVec,ReturnFnParams,vfoptions)
% When using refinement, lowmemory is implemented in the first stage (return fn) but not the second (the actual iteration).
% Refine, so there is at least one d variable

N_d=prod(n_d);
N_a=prod(n_a);

n_da=[n_d,n_a];
da_gridvals=[repmat(d_gridvals,N_a,1),repelem(a_grid,N_d,1)]; % only one aprime

%% Refinement: calculate ReturnMatrix and 'remove' the d dimension
% Since the return function is independent of time creating it once and then using it every iteration is good for
% speed, but it does use a lot of memory. Hence the lowmemory option is here.
ReturnMatrixraw=CreateReturnFnMatrix_Case2_Disc_noz(ReturnFn,n_da, n_a, da_gridvals, a_grid, ReturnFnParams);
ReturnMatrixraw=reshape(ReturnMatrixraw,[N_d,N_a,N_a]);

% For refinement, now we solve for d*(aprime,a) that maximizes the ReturnFn
[ReturnMatrix,~]=max(ReturnMatrixraw,[],1);
ReturnMatrix=shiftdim(ReturnMatrix,1); % 'refined' return matrix
% The rest, except putting d back into Policy at the end, is all just copy-paste from ValueFnIter_InfHorz_postGI_nod_noz_raw()

addindexfora=gpuArray(N_a*(0:1:N_a-1)');

%%
tempcounter=1;
currdist=Inf;

%% First, just consider a_grid for next period
while currdist>(vfoptions.multigridswitch*vfoptions.tolerance) && tempcounter<=vfoptions.maxiter
    VKronold=VKron;

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*VKronold; % aprime by a

    % Calc the max and it's index
    [VKron,Policy_a]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1); % a by 1

    VKrondist=VKron-VKronold;
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
        tempmaxindex=shiftdim(Policy_a,1)+addindexfora; % aprime index, add the index for a
        Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a,1]); % keep return function of optimal policy for using in Howards
        Policy_a=Policy_a(:); % a by 1 (this shape is just convenient for Howards)
        for Howards_counter=1:vfoptions.howards
            EVKrontemp=VKron(Policy_a,:);
            VKron=Ftemp+DiscountFactorParamsVec*EVKrontemp; % interpolate EV
        end
    end

    tempcounter=tempcounter+1;

end
Policy_a=reshape(Policy_a,[1,N_a]); % Howards can mess with the size

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
aprimeshifter=min(max(Policy_a,1+vfoptions.maxaprimediff),N_a-vfoptions.maxaprimediff);
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

ReturnMatrixfine=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, aprime_grid, a_grid, ReturnFnParams,1);
% ReturnMatrixfineraw=reshape(ReturnMatrixfineraw,[N_d,N_aprime,N_a]);

% For refinement, now we solve for d*(aprime,a) that maximizes the ReturnFn
[ReturnMatrixfine,dstar]=max(ReturnMatrixfine,[],1);
ReturnMatrixfine=shiftdim(ReturnMatrixfine,1);

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
    [VKron,Policy_a]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1); % a by 1

    VKrondist=VKron(:)-VKronold(:);
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
        tempmaxindex=shiftdim(Policy_a,1)+addindexforafine; % aprime index, add the index for a, size is [N_a,1]
        Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a,1]); % keep return function of optimal policy for using in Howards
        tempmaxindex2=Policy_a(:)+N_aprime*(0:1:N_a-1)'; % size is [N_a,1], contains the (aprime,a) index; (this shape is just convenient for Howards)
        for Howards_counter=1:vfoptions.howards
            EVpre=reshape(VKron(aprimeindex,:),[N_aprimediff,N_a]); % last dimension is zprime
            EVKrontemp=interp1(EVinterpindex1,EVpre,EVinterpindex2); % interpolate V as Policy points to the interpolated indexes
            EVKrontemp=reshape(EVKrontemp,[N_aprime*N_a,1]);  % last dimension is zprime
            EVKrontemp=EVKrontemp(tempmaxindex2,1);
            VKron=Ftemp+DiscountFactorParamsVec*EVKrontemp;
        end
    end

    tempcounter=tempcounter+1;
end




%% Do another post-GI layer
% Note: is just a copy-paste of the previous post-GI layer code
% Only difference that before we start there are two lines of code to
% convert Policy_a back into being about the nearest rough grid index
while vfoptions.postGIrepeat>0
    vfoptions.postGIrepeat=vfoptions.postGIrepeat-1;

    % Current optimal aprime is Policy
    % So create an aprime_grid that is just an interpolation within +-vfoptions.maxaprimediff

    % First, we switch Policy_a to be the nearest point on the rough grid
    Policy_a=reshape(Policy_a,[1,N_a]); % Howards can mess with the size
    Policy_a=ceil((Policy_a-1)/(n2short+1))-vfoptions.maxaprimediff+aprimeshifter;
    % ceil((Policy-1)/(n2short+1))-vfoptions.maxaprimediff ranges -vfoptions.maxaprimediff:1:vfoptions.maxaprimediff

    % First, create an aprime_grid that is just the +-vfoptions.maxaprimediff
    % Note: this code is for models with a single endogenous state
    n_aprimediff=1+2*vfoptions.maxaprimediff;
    N_aprimediff=prod(n_aprimediff);
    aprimeshifter=min(max(Policy_a,1+vfoptions.maxaprimediff),N_a-vfoptions.maxaprimediff);
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

    ReturnMatrixfine=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, aprime_grid, a_grid, ReturnFnParams,1);
    % ReturnMatrixfineraw=reshape(ReturnMatrixfineraw,[N_d,N_aprime,N_a]);

    % For refinement, now we solve for d*(aprime,a) that maximizes the ReturnFn
    [ReturnMatrixfine,dstar]=max(ReturnMatrixfine,[],1);
    ReturnMatrixfine=shiftdim(ReturnMatrixfine,1);

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
        [VKron,Policy_a]=max(entireRHS,[],1);
        VKron=shiftdim(VKron,1); % a by 1

        VKrondist=VKron(:)-VKronold(:);
        VKrondist(isnan(VKrondist))=0;
        currdist=max(abs(VKrondist));

        % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
        if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
            tempmaxindex=shiftdim(Policy_a,1)+addindexforafine; % aprime index, add the index for a, size is [N_a,1]
            Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a,1]); % keep return function of optimal policy for using in Howards
            tempmaxindex2=Policy_a(:)+N_aprime*(0:1:N_a-1)'; % size is [N_a,1], contains the (aprime,a) index; (this shape is just convenient for Howards)
            for Howards_counter=1:vfoptions.howards
                EVpre=reshape(VKron(aprimeindex),[N_aprimediff,N_a]); % last dimension is zprime
                EVKrontemp=interp1(EVinterpindex1,EVpre,EVinterpindex2); % interpolate V as Policy points to the interpolated indexes
                EVKrontemp=EVKrontemp(tempmaxindex2,1);
                VKron=Ftemp+DiscountFactorParamsVec*EVKrontemp;
            end
        end

        tempcounter=tempcounter+1;
    end

end

%% For refinement, add d back into Policy
Policy=zeros(4,N_a,'gpuArray'); % +1 channel for PolicyL2flag
temppolicyindex=reshape(Policy_a,[1,N_a])+N_aprime*(0:1:N_a-1);

Policy(1,:,:)=reshape(dstar(temppolicyindex),[N_a,1]); % note: dstar is defined on the fine grid

%% Switch policy to lower grid index and L2 index (is currently index on fine grid)
% Separate Policy into L1 and L2
fineindex=reshape(Policy_a,[N_a,1]);
L1a=ceil((fineindex-1)/(n2short+1))-1;  % this ranges -1:0:2*vfoptions.maxaprimediff-1
% (L1a-vfoptions.maxaprimediff+1) ranges -vfoptions.maxaprimediff:1:vfoptions.maxaprimediff
L1=max(L1a-vfoptions.maxaprimediff+1+aprimeshifter(:)-1,1); % lower grid point index (on the full grid), so this ranges 0 to n_a-1
L1intermediate=max(L1a,0)+1; % lower grid point index (on the small grid, in form so we can get L2)
L2=fineindex-(L1intermediate-1)*(n2short+1); % L2 index

Policy(2,:)=reshape(L1,[1,N_a]);
Policy(3,:)=reshape(L2,[1,N_a]);

% L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
% Computed once, post-convergence, using final ReturnMatrixfine and final aprimeshifter
fineindex_lower = (L1intermediate-1)*(n2short+1) + 1;          % fine pos of lower coarse for chosen L1 segment
fineindex_upper = L1intermediate*(n2short+1) + 1;              % fine pos of upper coarse
linidx_lower = reshape(fineindex_lower,[N_a,1]) + addindexforafine;
linidx_upper = reshape(fineindex_upper,[N_a,1]) + addindexforafine;
isInfLower = (ReturnMatrixfine(linidx_lower(:)) == -Inf);
isInfUpper = (ReturnMatrixfine(linidx_upper(:)) == -Inf);
inInterior = (L2 >= 2) & (L2 <= n2short+1);
Policy(4,:,:) = reshape(2 + (inInterior & isInfLower) - (inInterior & isInfUpper), [1,N_a,1]);

if currdist > vfoptions.tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, vfoptions.tolerance)
end



end