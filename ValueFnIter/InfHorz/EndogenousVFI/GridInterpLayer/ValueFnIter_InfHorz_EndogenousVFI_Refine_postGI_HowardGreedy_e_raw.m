function [VKron,Policy]=ValueFnIter_InfHorz_EndogenousVFI_Refine_postGI_HowardGreedy_e_raw(VKron,n_d,n_a,n_z,n_e,d_gridvals,a_grid,z_gridvals,e_gridvals,pi_z,pi_e,ReturnFn,DiscountFactorParamsVec,ReturnFnParams,vfoptions)
% Endogenous VFI (Bray, 2019): identical iterations to ValueFnIter_InfHorz_Refine_postGI_HowardGreedy_e_raw, but every loop stops on the
% span over a of V-Vold within each exogenous state (tolerances apply to that span), and V is recovered in levels after
% each fine-grid loop.
% postGI, with z and iid e
% When using refinement, lowmemory is implemented in the first stage (return fn) but not the second (the actual iteration).
% Refine, so there is at least one d variable
% VKron is a-by-z-by-e, pi_e is a column vector of probabilities
% Greedy Howards (a.k.a. Policy Function Iteration): Howards is done by solving the linear system for V given the
% policy, using the sparse transition matrix from (a,z,e) to (aprime,zprime,eprime), which has probabilities
% pi_z(z,zprime)*pi_e(eprime) (and the two grid interpolation weights on the fine grid).
% e is iid, so the expectation of next period value is EV(aprime,z): it does not depend on e. This is computed first,
% and only then evaluated at the aprime (and interpolated aprime) points.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

n_da=[n_d,n_a];
da_gridvals=[repmat(d_gridvals,N_a,1),repelem(a_grid,N_d,1)]; % only one aprime

%% Refinement: calculate ReturnMatrix and 'remove' the d dimension
% Since the return function is independent of time creating it once and then using it every iteration is good for
% speed, but it does use a lot of memory. Hence the lowmemory option is here.
if vfoptions.lowmemory==0
    ReturnMatrixraw=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,n_da, n_a, n_z, n_e, da_gridvals, a_grid, z_gridvals, e_gridvals, ReturnFnParams);
    ReturnMatrixraw=reshape(ReturnMatrixraw,[N_d,N_a,N_a,N_z,N_e]);

    % For refinement, now we solve for d*(aprime,a,z,e) that maximizes the ReturnFn
    [ReturnMatrix,~]=max(ReturnMatrixraw,[],1);
    ReturnMatrix=shiftdim(ReturnMatrix,1); % 'refined' return matrix
    clear ReturnMatrixraw

elseif vfoptions.lowmemory==1 % loop over e
    ReturnMatrix=zeros(N_a,N_a,N_z,N_e,'gpuArray'); % 'refined' return matrix
    l_e=length(n_e);
    special_n_e=ones(1,l_e);
    for e_c=1:N_e
        evals=e_gridvals(e_c,:);
        ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,n_da, n_a, n_z, special_n_e, da_gridvals, a_grid, z_gridvals, evals, ReturnFnParams);
        ReturnMatrix_e=reshape(ReturnMatrix_e,[N_d,N_a,N_a,N_z]);
        ReturnMatrix_e=max(ReturnMatrix_e,[],1); % solve for dstar
        ReturnMatrix(:,:,:,e_c)=shiftdim(ReturnMatrix_e,1);
    end
end
% The rest, except putting d back into Policy at the end, is all just copy-paste from ValueFnIter_InfHorz_postGI_HowardGreedy_nod_e_raw()

pi_z_alt=shiftdim(pi_z',-1); % 1-by-zprime-by-z
pi_e_alt=shiftdim(pi_e,-2); % 1-by-1-by-e

addindexforaze=gpuArray(N_a*(0:1:N_a-1)'+N_a*N_a*(0:1:N_z-1)+N_a*N_a*N_z*shiftdim((0:1:N_e-1),-1));
addindexforz=gpuArray(repmat(N_a*(0:1:N_z-1),N_a,1,N_e)); % a-by-z-by-e, for indexing EV (which is aprime-by-z)

% Setup specific to greedy Howards
% Each row (a,z,e) goes to the columns (aprime,zprime,eprime) with probability pi_z(z,zprime)*pi_e(eprime)
spI=gpuArray.speye(N_a*N_z*N_e);
greedyHind1=gpuArray(repelem((1:1:N_a*N_z*N_e)',1,N_z*N_e));
greedyHind2=gpuArray(reshape(N_a*(0:1:N_z-1)'+N_a*N_z*(0:1:N_e-1),[1,N_z*N_e])); % zprime fastest, then eprime
greedyHpi=gpuArray(repmat(repelem(kron(pi_e',pi_z),N_a,1),N_e,1)); % kron(pi_e',pi_z) is z-by-(zprime,eprime)

%% First, just consider a_grid for next period
tempcounter=1;
currdist=1;
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
    [VKron,Policy_a]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1); % a by z by e

    VKrondist=reshape(VKron,[N_a,numel(VKron)/N_a])-reshape(VKronold,[N_a,numel(VKronold)/N_a]); % a-by-(exogenous states)
    currdist=max(max(VKrondist,[],1)-min(VKrondist,[],1),[],'all'); % endogenous VFI: the span over a, within each exogenous state (max and min omit the NaN from -Inf minus -Inf)

    % Use greedy-Howards Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
        tempmaxindex=shiftdim(Policy_a,1)+addindexforaze; % aprime index, add the index for a, z and e
        Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a,N_z,N_e]); % keep return function of optimal policy for using in Howards
        Ftemp=reshape(Ftemp,[N_a*N_z*N_e,1]);
        T_E=sparse(greedyHind1,Policy_a(:)+greedyHind2,greedyHpi,N_a*N_z*N_e,N_a*N_z*N_e);

        % Greedy Howards with -Inf in V: a direct solve is global, so one -Inf in Ftemp gives NaN everywhere, and a finite Ftemp at a
        % state whose policy puts positive weight on a dead state gives a finite V where the truth is -Inf. So solve on the live states
        % only: a state is dead if V is -Inf there or if its policy puts positive weight on a dead state (the closure under T_E);
        % (I-beta*T_E) is block triangular in live-then-dead order, so the live block solves exactly on its own and V is -Inf on the
        % dead set. (CoreInfHorzVFIAlgoTests doPart(11); InfHorzGI_InfNaN_proposal.md, section 8)
        dead=(VKron(:)==-Inf);
        if any(dead)
            newdead=dead;
            while any(newdead)
                newdead=(T_E*double(dead)>0) & ~dead;
                dead=dead | newdead;
            end
            live=~dead;
            N_live=gather(sum(live));
            livemap=cumsum(live);
            [T_Erow,T_Ecol,T_Eval]=find(T_E);
            keep=live(T_Erow) & live(T_Ecol); % a live row has no positive weight on a dead column, so this is just the live rows
            T_E=sparse(livemap(T_Erow(keep)),livemap(T_Ecol(keep)),T_Eval(keep),N_live,N_live);
            VKrontemp=-Inf(numel(Ftemp),1,'gpuArray');
            VKrontemp(live)=(gpuArray.speye(N_live)-DiscountFactorParamsVec*T_E)\Ftemp(live);
            VKron=VKrontemp;
        else
            VKron=(spI-DiscountFactorParamsVec*T_E)\Ftemp;
        end
        VKron=reshape(VKron,[N_a,N_z,N_e]);
    end

    tempcounter=tempcounter+1;
end
Policy_a=reshape(Policy_a,[1,N_a,N_z,N_e]); % Howards can mess with the size

%% Now that we have solved on the rough grid, we re-solve on the fine grid
% Based on solving a bunch of value fns with and without grid
% interpolation, the 'lower grid index' with grid interpolation is always
% within a point or two of the solution on the rough grid. So here we only
% consider +-vfoptions.maxaprimediff to set up the fine/interpolated aprime_grid

% First, create an aprime_grid that is just the +-vfoptions.maxaprimediff
% Note: this code is for models with a single endogenous state
n_aprimediff=1+2*vfoptions.maxaprimediff;
N_aprimediff=prod(n_aprimediff);
aprimeshifter=min(max(Policy_a,1+vfoptions.maxaprimediff),N_a-vfoptions.maxaprimediff);
aprimeindex=(-vfoptions.maxaprimediff:1:vfoptions.maxaprimediff)' +aprimeshifter; % size n_aprimediff-by-n_a-by-n_z-by-n_e
aprime_grid=a_grid(aprimeindex);

% Second, interpolate this
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n_aprime=n_aprimediff+(n_aprimediff-1)*vfoptions.ngridinterp;
N_aprime=prod(n_aprime);
aprime_grid=interp1((1:1:N_aprimediff)',aprime_grid,linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp)');
% Note: aprime_grid is N_aprime-by-N_a-by-N_z-by-N_e

if vfoptions.lowmemory==0
    ReturnMatrixfine=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, d_gridvals, aprime_grid, a_grid, z_gridvals, e_gridvals, ReturnFnParams,1);
    % ReturnMatrixfine is [N_d,N_aprime,N_a,N_z,N_e]

    % For refinement, now we solve for d*(aprime,a,z,e) that maximizes the ReturnFn
    [ReturnMatrixfine,dstar]=max(ReturnMatrixfine,[],1);
    ReturnMatrixfine=shiftdim(ReturnMatrixfine,1);
    dstar=shiftdim(dstar,1);

elseif vfoptions.lowmemory==1 % loop over e
    % Refinement: calculate ReturnMatrix and 'remove' the d dimension
    ReturnMatrixfine=zeros(N_aprime,N_a,N_z,N_e,'gpuArray'); % 'refined' return matrix
    dstar=zeros(N_aprime,N_a,N_z,N_e,'gpuArray');
    l_e=length(n_e);
    special_n_e=ones(1,l_e);
    for e_c=1:N_e
        evals=e_gridvals(e_c,:);
        ReturnMatrixfine_e=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, aprime_grid(:,:,:,e_c), a_grid, z_gridvals, evals, ReturnFnParams,1);
        ReturnMatrixfine_e=reshape(ReturnMatrixfine_e,[N_d,N_aprime,N_a,N_z]);
        [ReturnMatrixfine_e,dstar_e]=max(ReturnMatrixfine_e,[],1); % solve for dstar
        ReturnMatrixfine(:,:,:,e_c)=shiftdim(ReturnMatrixfine_e,1);
        dstar(:,:,:,e_c)=shiftdim(dstar_e,1);
    end
end

%% Now switch to considering the fine/interpolated aprime_grid
EVinterpindex1=gpuArray(1:1:N_aprimediff)';
EVinterpindex2=gpuArray(linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp))';

% EV is aprime-by-z, so to evaluate it on the window add the index for z
aprimeindexz=aprimeindex+N_a*shiftdim((0:1:N_z-1),-1); % n_aprimediff-by-n_a-by-n_z-by-n_e

% For Howards we need
addindexforazefine=gpuArray(N_aprime*(0:1:N_a-1)'+N_aprime*N_a*(0:1:N_z-1)+N_aprime*N_a*N_z*shiftdim((0:1:N_e-1),-1));

% Perform value fn iteration
currdist=1; % force going into the next while loop at least one iteration
tempcounter=1; % reset tempcounter
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

    [VKron,Policy_a]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1); % a by z by e

    VKrondist=reshape(VKron,[N_a,numel(VKron)/N_a])-reshape(VKronold,[N_a,numel(VKronold)/N_a]); % a-by-(exogenous states)
    currdist=max(max(VKrondist,[],1)-min(VKrondist,[],1),[],'all'); % endogenous VFI: the span over a, within each exogenous state (max and min omit the NaN from -Inf minus -Inf)

    % Use greedy-Howards Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
        tempmaxindex=shiftdim(Policy_a,1)+addindexforazefine; % aprime index, add the index for a, z and e, size is [N_a,N_z,N_e]
        Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a*N_z*N_e,1]); % keep return function of optimal policy for using in Howards
        % Resolve the interpolation to a (lower grid index, weight) pair once, here
        Policy_L1a=ceil((Policy_a(:)-1)/(n2short+1))-1;
        % max(Policy_L1a,0) so the lower index and the weight agree at the window's bottom node
        Policy_lowerind=max(Policy_L1a,0)-vfoptions.maxaprimediff+aprimeshifter(:); % index on the full a_grid
        Policy_lowerprob=1- ((Policy_a(:)-max(Policy_L1a,0)*(n2short+1))-1)/(n2short+1);
        T_E=sparse([greedyHind1;greedyHind1],[Policy_lowerind;Policy_lowerind+1]+greedyHind2,[Policy_lowerprob;1-Policy_lowerprob].*[greedyHpi;greedyHpi],N_a*N_z*N_e,N_a*N_z*N_e); % lower and upper grid points either side of the fine aprime

        % Greedy Howards with -Inf in V: a direct solve is global, so one -Inf in Ftemp gives NaN everywhere, and a finite Ftemp at a
        % state whose policy puts positive weight on a dead state gives a finite V where the truth is -Inf. So solve on the live states
        % only: a state is dead if V is -Inf there or if its policy puts positive weight on a dead state (the closure under T_E);
        % (I-beta*T_E) is block triangular in live-then-dead order, so the live block solves exactly on its own and V is -Inf on the
        % dead set. (CoreInfHorzVFIAlgoTests doPart(11); InfHorzGI_InfNaN_proposal.md, section 8)
        dead=(VKron(:)==-Inf);
        if any(dead)
            newdead=dead;
            while any(newdead)
                newdead=(T_E*double(dead)>0) & ~dead;
                dead=dead | newdead;
            end
            live=~dead;
            N_live=gather(sum(live));
            livemap=cumsum(live);
            [T_Erow,T_Ecol,T_Eval]=find(T_E);
            keep=live(T_Erow) & live(T_Ecol); % a live row has no positive weight on a dead column, so this is just the live rows
            T_E=sparse(livemap(T_Erow(keep)),livemap(T_Ecol(keep)),T_Eval(keep),N_live,N_live);
            VKrontemp=-Inf(numel(Ftemp),1,'gpuArray');
            VKrontemp(live)=(gpuArray.speye(N_live)-DiscountFactorParamsVec*T_E)\Ftemp(live);
            VKron=VKrontemp;
        else
            VKron=(spI-DiscountFactorParamsVec*T_E)\Ftemp;
        end
        VKron=reshape(VKron,[N_a,N_z,N_e]);
    end

    tempcounter=tempcounter+1;
end
% Endogenous VFI: recover V in levels. Once converged, VKron=T(VKronold) (the Howards step is skipped in the last
% iteration) and VKron-VKronold is a function of the exogenous states only, up to the tolerance; call it dexo (the
% mean over a of the finite entries). VKronold+c is the fixed point when c=dexo+beta*E[c'], and V=T(VKronold+c)=VKron+beta*E[c'].
% (The interpolation weights sum to one, so adding a function of the exogenous states to V adds it to the interpolated V too.)
szVKron=size(VKron);
VKrondist=reshape(VKron,[N_a,numel(VKron)/N_a])-reshape(VKronold,[N_a,numel(VKronold)/N_a]);
finitedist=isfinite(VKrondist);
VKrondist(~finitedist)=0;
dexo=sum(VKrondist,1)./max(sum(finitedist,1),1); % 1-by-(exogenous states), zero where there is no finite entry
% e is iid, so E[c'|z]=pi_z*cbar where cbar(z)=sum_e pi_e(e)*c(z,e), and cbar solves cbar=dexo*pi_e+beta*pi_z*cbar
cbar=(eye(N_z,'gpuArray')-DiscountFactorParamsVec*pi_z)\(reshape(dexo,[N_z,N_e])*pi_e); % z-by-1
VKron=reshape(reshape(VKron,[N_a,N_z,N_e])+DiscountFactorParamsVec*(pi_z*cbar)',szVKron); % the 1-by-z autoexpands over a and e


%% Do another post-GI layer
% Note: is just a copy-paste of the previous post-GI layer code
% Only difference that before we start there are two lines of code to convert Policy_a back into being about the nearest rough grid index
while vfoptions.postGIrepeat>0
    vfoptions.postGIrepeat=vfoptions.postGIrepeat-1;

    % First, we switch Policy_a to be the nearest point on the rough grid
    Policy_a=reshape(Policy_a,[1,N_a,N_z,N_e]); % Howards can mess with the size
    Policy_a=ceil((Policy_a-1)/(n2short+1))-vfoptions.maxaprimediff+aprimeshifter;
    % ceil((Policy_a-1)/(n2short+1))-vfoptions.maxaprimediff ranges -vfoptions.maxaprimediff:1:vfoptions.maxaprimediff

    % First, create an aprime_grid that is just the +-vfoptions.maxaprimediff
    % Note: this code is for models with a single endogenous state
    n_aprimediff=1+2*vfoptions.maxaprimediff;
    N_aprimediff=prod(n_aprimediff);
    aprimeshifter=min(max(Policy_a,1+vfoptions.maxaprimediff),N_a-vfoptions.maxaprimediff);
    aprimeindex=(-vfoptions.maxaprimediff:1:vfoptions.maxaprimediff)' +aprimeshifter; % size n_aprimediff-by-n_a-by-n_z-by-n_e
    aprime_grid=a_grid(aprimeindex);

    % Second, interpolate this
    n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
    n_aprime=n_aprimediff+(n_aprimediff-1)*vfoptions.ngridinterp;
    N_aprime=prod(n_aprime);
    aprime_grid=interp1((1:1:N_aprimediff)',aprime_grid,linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp)');
    % Note: aprime_grid is N_aprime-by-N_a-by-N_z-by-N_e

    if vfoptions.lowmemory==0
        ReturnMatrixfine=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, d_gridvals, aprime_grid, a_grid, z_gridvals, e_gridvals, ReturnFnParams,1);
        % ReturnMatrixfine is [N_d,N_aprime,N_a,N_z,N_e]

        % For refinement, now we solve for d*(aprime,a,z,e) that maximizes the ReturnFn
        [ReturnMatrixfine,dstar]=max(ReturnMatrixfine,[],1);
        ReturnMatrixfine=shiftdim(ReturnMatrixfine,1);
        dstar=shiftdim(dstar,1);

    elseif vfoptions.lowmemory==1 % loop over e
        % Refinement: calculate ReturnMatrix and 'remove' the d dimension
        ReturnMatrixfine=zeros(N_aprime,N_a,N_z,N_e,'gpuArray'); % 'refined' return matrix
        dstar=zeros(N_aprime,N_a,N_z,N_e,'gpuArray');
        l_e=length(n_e);
        special_n_e=ones(1,l_e);
        for e_c=1:N_e
            evals=e_gridvals(e_c,:);
            ReturnMatrixfine_e=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, aprime_grid(:,:,:,e_c), a_grid, z_gridvals, evals, ReturnFnParams,1);
            ReturnMatrixfine_e=reshape(ReturnMatrixfine_e,[N_d,N_aprime,N_a,N_z]);
            [ReturnMatrixfine_e,dstar_e]=max(ReturnMatrixfine_e,[],1); % solve for dstar
            ReturnMatrixfine(:,:,:,e_c)=shiftdim(ReturnMatrixfine_e,1);
            dstar(:,:,:,e_c)=shiftdim(dstar_e,1);
        end
    end

    %% Now switch to considering the fine/interpolated aprime_grid
    EVinterpindex1=gpuArray(1:1:N_aprimediff)';
    EVinterpindex2=gpuArray(linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp))';

    % EV is aprime-by-z, so to evaluate it on the window add the index for z
    aprimeindexz=aprimeindex+N_a*shiftdim((0:1:N_z-1),-1); % n_aprimediff-by-n_a-by-n_z-by-n_e

    % For Howards we need
    addindexforazefine=gpuArray(N_aprime*(0:1:N_a-1)'+N_aprime*N_a*(0:1:N_z-1)+N_aprime*N_a*N_z*shiftdim((0:1:N_e-1),-1));

    % Perform value fn iteration
    currdist=1; % force going into the next while loop at least one iteration
    tempcounter=1; % reset tempcounter
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

        [VKron,Policy_a]=max(entireRHS,[],1);
        VKron=shiftdim(VKron,1); % a by z by e

        VKrondist=reshape(VKron,[N_a,numel(VKron)/N_a])-reshape(VKronold,[N_a,numel(VKronold)/N_a]); % a-by-(exogenous states)
        currdist=max(max(VKrondist,[],1)-min(VKrondist,[],1),[],'all'); % endogenous VFI: the span over a, within each exogenous state (max and min omit the NaN from -Inf minus -Inf)

        % Use greedy-Howards Improvement (except for first few and last few iterations, as it is not a good idea there)
        if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
            tempmaxindex=shiftdim(Policy_a,1)+addindexforazefine; % aprime index, add the index for a, z and e, size is [N_a,N_z,N_e]
            Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a*N_z*N_e,1]); % keep return function of optimal policy for using in Howards
            % Resolve the interpolation to a (lower grid index, weight) pair once, here
            Policy_L1a=ceil((Policy_a(:)-1)/(n2short+1))-1;
            % max(Policy_L1a,0) so the lower index and the weight agree at the window's bottom node
            Policy_lowerind=max(Policy_L1a,0)-vfoptions.maxaprimediff+aprimeshifter(:); % index on the full a_grid
            Policy_lowerprob=1- ((Policy_a(:)-max(Policy_L1a,0)*(n2short+1))-1)/(n2short+1);
            T_E=sparse([greedyHind1;greedyHind1],[Policy_lowerind;Policy_lowerind+1]+greedyHind2,[Policy_lowerprob;1-Policy_lowerprob].*[greedyHpi;greedyHpi],N_a*N_z*N_e,N_a*N_z*N_e); % lower and upper grid points either side of the fine aprime

            % Greedy Howards with -Inf in V: a direct solve is global, so one -Inf in Ftemp gives NaN everywhere, and a finite Ftemp at a
            % state whose policy puts positive weight on a dead state gives a finite V where the truth is -Inf. So solve on the live states
            % only: a state is dead if V is -Inf there or if its policy puts positive weight on a dead state (the closure under T_E);
            % (I-beta*T_E) is block triangular in live-then-dead order, so the live block solves exactly on its own and V is -Inf on the
            % dead set. (CoreInfHorzVFIAlgoTests doPart(11); InfHorzGI_InfNaN_proposal.md, section 8)
            dead=(VKron(:)==-Inf);
            if any(dead)
                newdead=dead;
                while any(newdead)
                    newdead=(T_E*double(dead)>0) & ~dead;
                    dead=dead | newdead;
                end
                live=~dead;
                N_live=gather(sum(live));
                livemap=cumsum(live);
                [T_Erow,T_Ecol,T_Eval]=find(T_E);
                keep=live(T_Erow) & live(T_Ecol); % a live row has no positive weight on a dead column, so this is just the live rows
                T_E=sparse(livemap(T_Erow(keep)),livemap(T_Ecol(keep)),T_Eval(keep),N_live,N_live);
                VKrontemp=-Inf(numel(Ftemp),1,'gpuArray');
                VKrontemp(live)=(gpuArray.speye(N_live)-DiscountFactorParamsVec*T_E)\Ftemp(live);
                VKron=VKrontemp;
            else
                VKron=(spI-DiscountFactorParamsVec*T_E)\Ftemp;
            end
            VKron=reshape(VKron,[N_a,N_z,N_e]);
        end

        tempcounter=tempcounter+1;
    end
    % Endogenous VFI: recover V in levels. Once converged, VKron=T(VKronold) (the Howards step is skipped in the last
    % iteration) and VKron-VKronold is a function of the exogenous states only, up to the tolerance; call it dexo (the
    % mean over a of the finite entries). VKronold+c is the fixed point when c=dexo+beta*E[c'], and V=T(VKronold+c)=VKron+beta*E[c'].
    % (The interpolation weights sum to one, so adding a function of the exogenous states to V adds it to the interpolated V too.)
    szVKron=size(VKron);
    VKrondist=reshape(VKron,[N_a,numel(VKron)/N_a])-reshape(VKronold,[N_a,numel(VKronold)/N_a]);
    finitedist=isfinite(VKrondist);
    VKrondist(~finitedist)=0;
    dexo=sum(VKrondist,1)./max(sum(finitedist,1),1); % 1-by-(exogenous states), zero where there is no finite entry
    % e is iid, so E[c'|z]=pi_z*cbar where cbar(z)=sum_e pi_e(e)*c(z,e), and cbar solves cbar=dexo*pi_e+beta*pi_z*cbar
    cbar=(eye(N_z,'gpuArray')-DiscountFactorParamsVec*pi_z)\(reshape(dexo,[N_z,N_e])*pi_e); % z-by-1
    VKron=reshape(reshape(VKron,[N_a,N_z,N_e])+DiscountFactorParamsVec*(pi_z*cbar)',szVKron); % the 1-by-z autoexpands over a and e
end


%% For refinement, add d back into Policy
Policy=zeros(4,N_a,N_z,N_e,'gpuArray'); % +1 channel for PolicyL2flag
temppolicyindex=reshape(Policy_a,[1,N_a*N_z*N_e])+N_aprime*(0:1:N_a*N_z*N_e-1);

Policy(1,:,:,:)=reshape(dstar(temppolicyindex),[N_a,N_z,N_e]); % note: dstar is defined on the fine grid

%% Switch policy to lower grid index and L2 index (is currently index on fine grid)
% Separate Policy into L1 and L2
fineindex=reshape(Policy_a,[N_a*N_z*N_e,1]);
L1a=ceil((fineindex-1)/(n2short+1))-1; % this ranges -1:0:2*vfoptions.maxaprimediff-1
% the max(L1a,0) matters only at the window's bottom node, where L1a is -1; L2 below is built off the same thing, so L1 has to be too
L1=max(L1a,0)-vfoptions.maxaprimediff+aprimeshifter(:); % lower grid point index on the full grid; the window bounds put this in 1:N_a-1, so no clamp is needed
L1intermediate=max(L1a,0)+1; % lower grid point index (on the small grid, in form so we can get L2)
L2=fineindex-(L1intermediate-1)*(n2short+1); % L2 index

Policy(2,:,:,:)=reshape(L1,[1,N_a,N_z,N_e]);
Policy(3,:,:,:)=reshape(L2,[1,N_a,N_z,N_e]);

% L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
% Computed once, post-convergence, using final (d-argmaxed) ReturnMatrixfine and final aprimeshifter
fineindex_lower = (L1intermediate-1)*(n2short+1) + 1;
fineindex_upper = L1intermediate*(n2short+1) + 1;
linidx_lower = reshape(fineindex_lower,[N_a,N_z,N_e]) + addindexforazefine;
linidx_upper = reshape(fineindex_upper,[N_a,N_z,N_e]) + addindexforazefine;
isInfLower = (ReturnMatrixfine(linidx_lower(:)) == -Inf);
isInfUpper = (ReturnMatrixfine(linidx_upper(:)) == -Inf);
inInterior = (L2 >= 2) & (L2 <= n2short+1);
Policy(4,:,:,:) = reshape(2 + (inInterior & isInfLower) - (inInterior & isInfUpper), [1,N_a,N_z,N_e]);

if currdist > vfoptions.tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, vfoptions.tolerance)
end

end
