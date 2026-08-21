function [VKron,Policy]=ValueFnIter_InfHorz_Refine_postGI2A_sparse_raw(VKron,n_d,n_a,n_z,d_gridvals,a_grid,z_gridvals,pi_z,ReturnFn,DiscountFactorParamsVec,ReturnFnParams,vfoptions)
% Sparse-matrix (iterated) Howards, with the grid interpolation layer (postGI), two endogenous states.
% When using refinement, lowmemory is implemented in the first stage (return fn) but not the second (the actual iteration).
% Refine, so there is at least one d variable
%
% Structure is the multi-grid postGI one: first solve on the coarse a_grid, then build the
% +-vfoptions.maxaprimediff window ONCE (on the first endogenous state only) and solve on the fine
% (interpolated) grid within it. Howards is done by building the sparse transition matrix T_E and
% iterating with it (vfoptions.howards times).
%
% Note: the non-sparse ValueFnIter_InfHorz_Refine_postGI2A_raw has no Howards at all in its second
% (fine grid) stage; the indexed version was too slow to be worth using because it has to
% re-interpolate EV inside every Howards step. Building T_E bakes the interpolation weights in once,
% so each Howards step is a single sparse matrix multiply and the cost disappears.
%
% The grid interpolation layer is on the FIRST endogenous state only. The joint index over a is
% a1+N_a1*(a2-1), so a1 varies fastest, which means the two points that an interpolated a1prime
% spreads mass over are adjacent in the joint index (stride 1), exactly as in the one endogenous
% state case.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

N_a1=n_a(1);
N_a2=prod(n_a(2:end));
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);
a_gridvals=CreateGridvals(n_a,a_grid,1);

n_da=[n_d,n_a];
da_gridvals=[repmat(d_gridvals,N_a,1),repelem(a_gridvals,N_d,1)];

%% CreateReturnFnMatrix_Disc_CPU creates a matrix of dimension (d and aprime)-by-a-by-z.
% Since the return function is independent of time creating it once and
% then using it every iteration is good for speed, but it does use a
% lot of memory. Hence the lowmemory option is here.
if vfoptions.lowmemory==0
    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn,n_da, n_a, n_z, da_gridvals, a_gridvals, z_gridvals, ReturnFnParams);
    ReturnMatrix=reshape(ReturnMatrix,[N_d,N_a,N_a,N_z]);

    % For refinement, now we solve for d*(aprime,a,z) that maximizes the ReturnFn
    [ReturnMatrix,~]=max(ReturnMatrix,[],1);
    ReturnMatrix=shiftdim(ReturnMatrix,1);

elseif vfoptions.lowmemory==1 % loop over z
    %% Refinement: calculate ReturnMatrix and 'remove' the d dimension
    ReturnMatrix=zeros(N_a,N_a,N_z,'gpuArray'); % 'refined' return matrix
    l_z=length(n_z);
    special_n_z=ones(1,l_z);
    for z_c=1:N_z
        zvals=z_gridvals(z_c,:);
        ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn,n_da, n_a, special_n_z, da_gridvals, a_gridvals, zvals, ReturnFnParams);
        ReturnMatrix_z=reshape(ReturnMatrix_z,[N_d,N_a,N_a]);
        [ReturnMatrix_z,~]=max(ReturnMatrix_z,[],1); % the coarse-stage dstar is not kept (only the fine-stage one is used, for Policy)
        ReturnMatrix(:,:,z_c)=shiftdim(ReturnMatrix_z,1);
    end
end

pi_z_alt=shiftdim(pi_z',-1);

addindexforaz=gpuArray(N_a*(0:1:N_a-1)'+N_a*N_a*(0:1:N_z-1));

% Setup specific to the sparse-matrix Howards
N_a_times_zind=N_a*gpuArray(0:1:N_z-1); % already contains -1
azind1=repmat(gpuArray(1:1:N_a*N_z)',1,N_z); % (a-z,zprime)
pi_z_big1=gpuArray(repelem(pi_z,N_a,1)); % (a-z,zprime)

%%
tempcounter=1;
currdist=1;

%% First, just consider a_grid for next period
while currdist>(vfoptions.multigridswitch*vfoptions.tolerance) && tempcounter<=vfoptions.maxiter
    VKronold=VKron;

    % Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=VKronold.*pi_z_alt;
    EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; % aprime by a by z

    %Calc the max and it's index
    [VKron,Policy_a]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1); % a by z

    VKrondist=VKron(:)-VKronold(:);
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    % Use Howards Improvement, iterating with the sparse transition matrix (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
        tempmaxindex=shiftdim(Policy_a,1)+addindexforaz; % aprime index, add the index for a and z
        Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a*N_z,1]); % keep return function of optimal policy for using in Howards

        % On the coarse grid Policy_a is already the joint aprime index, so this is the same as the one endogenous state case
        T_E=sparse(azind1,Policy_a(:)+N_a_times_zind,pi_z_big1,N_a*N_z,N_a*N_z);

        VKron=reshape(VKron,[N_a*N_z,1]);
        for h_c=1:vfoptions.howards
            VKron=Ftemp+DiscountFactorParamsVec*(T_E*VKron); % T_E already contains pi_z, so T_E*V is the expected continuation
        end
        VKron=reshape(VKron,[N_a,N_z]);
    end

    tempcounter=tempcounter+1;
end

Policy_a=reshape(Policy_a,[1,N_a,N_z]); % Howards can mess with the size
Policy_a1=rem(Policy_a-1,N_a1)+1;

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

if vfoptions.lowmemory==0
    ReturnMatrixfine=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, n_d, n_z, d_gridvals, a1prime_grid,a2_grid, a1_grid,a2_grid, z_gridvals, ReturnFnParams,1,0);
    ReturnMatrixfine=reshape(ReturnMatrixfine,[N_d,N_aprime,N_a,N_z]);

    % For refinement, now we solve for d*(aprime,a,z) that maximizes the ReturnFn
    [ReturnMatrixfine,dstar]=max(ReturnMatrixfine,[],1);
    ReturnMatrixfine=shiftdim(ReturnMatrixfine,1);

elseif vfoptions.lowmemory==1 % loop over z
    % Refinement: calculate ReturnMatrix and 'remove' the d dimension
    % Note: dstar comes out [N_aprime,N_a,N_z] here and [1,N_aprime,N_a,N_z] above, but it is only
    % ever read as dstar(temppolicyindex), and the linear ordering (aprime,a,z) is the same either way
    ReturnMatrixfine=zeros(N_aprime,N_a,N_z,'gpuArray'); % 'refined' return matrix
    dstar=zeros(N_aprime,N_a,N_z,'gpuArray');
    l_z=length(n_z);
    special_n_z=ones(1,l_z);
    for z_c=1:N_z
        zvals=z_gridvals(z_c,:);
        ReturnMatrixfine_z=CreateReturnFnMatrix_Disc_DC2A(ReturnFn,n_d, special_n_z, d_gridvals, a1prime_grid(:,1,:,:,z_c),a2_grid, a1_grid,a2_grid, zvals, ReturnFnParams,1,0);
        ReturnMatrixfine_z=reshape(ReturnMatrixfine_z,[N_d,N_aprime,N_a]);
        [ReturnMatrixfine_z,dstar_z]=max(ReturnMatrixfine_z,[],1); % solve for dstar
        ReturnMatrixfine(:,:,z_c)=shiftdim(ReturnMatrixfine_z,1);
        dstar(:,:,z_c)=shiftdim(dstar_z,1);
    end
end

pi_z_alt2=shiftdim(pi_z,-2);

% For Howards we need
addindexforazfine=gpuArray(N_aprime*(0:1:N_a-1)'+N_aprime*N_a*(0:1:N_z-1));

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
    [VKron,Policy_a]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1); % a by z

    VKrondist=VKron(:)-VKronold(:);
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    % Use Howards Improvement, iterating with the sparse transition matrix (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
        tempmaxindex=shiftdim(Policy_a,1)+addindexforazfine; % aprime index, add the index for a and z
        Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a*N_z,1]); % keep return function of optimal policy for using in Howards

        % Split the fine index into its a1prime (interpolated) and a2prime (exact) parts
        Policy_fine1=rem(Policy_a(:)-1,N_a1prime)+1; % a1prime, in the post-GI (fine, within-window) index
        Policy_fine2=ceil(Policy_a(:)/N_a1prime); % a2prime index, on the coarse a2 grid
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
    Policy_a=reshape(Policy_a,[1,N_a,N_z]); % Howards can mess with the size
    Policy_a1=rem(Policy_a-1,N_a1prime)+1;
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

    if vfoptions.lowmemory==0
        ReturnMatrixfine=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, n_d, n_z, d_gridvals, a1prime_grid,a2_grid, a1_grid,a2_grid, z_gridvals, ReturnFnParams,1,0);
        ReturnMatrixfine=reshape(ReturnMatrixfine,[N_d,N_aprime,N_a,N_z]);

        % For refinement, now we solve for d*(aprime,a,z) that maximizes the ReturnFn
        [ReturnMatrixfine,dstar]=max(ReturnMatrixfine,[],1);
        ReturnMatrixfine=shiftdim(ReturnMatrixfine,1);

    elseif vfoptions.lowmemory==1 % loop over z
        % Refinement: calculate ReturnMatrix and 'remove' the d dimension
        % Note: dstar comes out [N_aprime,N_a,N_z] here and [1,N_aprime,N_a,N_z] above, but it is only
        % ever read as dstar(temppolicyindex), and the linear ordering (aprime,a,z) is the same either way
        ReturnMatrixfine=zeros(N_aprime,N_a,N_z,'gpuArray'); % 'refined' return matrix
        dstar=zeros(N_aprime,N_a,N_z,'gpuArray');
        l_z=length(n_z);
        special_n_z=ones(1,l_z);
        for z_c=1:N_z
            zvals=z_gridvals(z_c,:);
            ReturnMatrixfine_z=CreateReturnFnMatrix_Disc_DC2A(ReturnFn,n_d, special_n_z, d_gridvals, a1prime_grid(:,1,:,:,z_c),a2_grid, a1_grid,a2_grid, zvals, ReturnFnParams,1,0);
            ReturnMatrixfine_z=reshape(ReturnMatrixfine_z,[N_d,N_aprime,N_a]);
            [ReturnMatrixfine_z,dstar_z]=max(ReturnMatrixfine_z,[],1); % solve for dstar
            ReturnMatrixfine(:,:,z_c)=shiftdim(ReturnMatrixfine_z,1);
            dstar(:,:,z_c)=shiftdim(dstar_z,1);
        end
    end

    pi_z_alt2=shiftdim(pi_z,-2);

    % For Howards we need
    addindexforazfine=gpuArray(N_aprime*(0:1:N_a-1)'+N_aprime*N_a*(0:1:N_z-1));

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
        [VKron,Policy_a]=max(entireRHS,[],1);
        VKron=shiftdim(VKron,1); % a by z

        VKrondist=VKron(:)-VKronold(:);
        VKrondist(isnan(VKrondist))=0;
        currdist=max(abs(VKrondist));

        % Use Howards Improvement, iterating with the sparse transition matrix (except for first few and last few iterations, as it is not a good idea there)
        if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
            tempmaxindex=shiftdim(Policy_a,1)+addindexforazfine; % aprime index, add the index for a and z
            Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a*N_z,1]); % keep return function of optimal policy for using in Howards

            % Split the fine index into its a1prime (interpolated) and a2prime (exact) parts
            Policy_fine1=rem(Policy_a(:)-1,N_a1prime)+1; % a1prime, in the post-GI (fine, within-window) index
            Policy_fine2=ceil(Policy_a(:)/N_a1prime); % a2prime index, on the coarse a2 grid
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
fineindex=reshape(Policy_a,[1,N_a,N_z]);
Policy=zeros(5,N_a,N_z,'gpuArray'); % +1 channel for PolicyL2flag
fineindexvec1=rem(fineindex-1,N_a1prime)+1; % a1prime, but in post-GI index
fineindexvec2=ceil(fineindex/N_a1prime); % a2prime index

fineindexvec1=reshape(fineindexvec1,[N_a*N_z,1]);
L1a=ceil((fineindexvec1-1)/(n2short+1))-1; % this ranges -1:0:2*vfoptions.maxaprimediff-1
L1=max(L1a-vfoptions.maxaprimediff+1+a1primeshifter(:)-1,1); % lower grid point index (on the full grid), so this ranges 0 to n_a-1
L1intermediate=max(L1a,0)+1; % lower grid point index (on the small grid, in form so we can get L2)
L2=fineindexvec1-(L1intermediate-1)*(n2short+1); % L2 index

Policy(2,:,:)=reshape(L1,[1,N_a,N_z]);
Policy(3,:,:)=fineindexvec2;
Policy(4,:,:)=reshape(L2,[1,N_a,N_z]);

%% For refinement, add d back into Policy
temppolicyindex=fineindex(:)+N_aprime*(0:1:N_a*N_z-1)';
Policy(1,:,:)=reshape(dstar(temppolicyindex),[N_a,N_z]); % note: dstar is defined on the fine grid

% L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
% Computed once, post-convergence. ReturnMatrixfine is [N_aprime,N_a,N_z] (after d-argmax) with aprime = a1prime_fine + N_a1prime*(a2prime-1).
fineindexvec2_flat = reshape(fineindexvec2,[N_a*N_z,1]);
aprime_lower = (L1intermediate-1)*(n2short+1) + 1 + N_a1prime*(fineindexvec2_flat-1);
aprime_upper = L1intermediate*(n2short+1)     + 1 + N_a1prime*(fineindexvec2_flat-1);
linidx_lower = aprime_lower + reshape(addindexforazfine,[N_a*N_z,1]);
linidx_upper = aprime_upper + reshape(addindexforazfine,[N_a*N_z,1]);
isInfLower = (ReturnMatrixfine(linidx_lower) == -Inf);
isInfUpper = (ReturnMatrixfine(linidx_upper) == -Inf);
inInterior = (L2 >= 2) & (L2 <= n2short+1);
Policy(5,:,:) = reshape(2 + (inInterior & isInfLower) - (inInterior & isInfUpper), [1,N_a,N_z]);

% Note: unlike Howards-greedy (which solves the linear system), iterating with the sparse T_E is
% fine when V contains values of -Inf, so there is no need to warn about that here.
if currdist > vfoptions.tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, vfoptions.tolerance)
end

end
