function [VKron,Policy]=ValueFnIter_Refine_postGI_raw(VKron,n_d,n_a,n_z,d_gridvals,a_grid,z_gridvals,pi_z,ReturnFn,DiscountFactorParamsVec,ReturnFnParams,vfoptions)
% When using refinement, lowmemory is implemented in the first stage (return fn) but not the second (the actual iteration).
% Refine, so there is at least one d variable

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

n_da=[n_d,n_a];
da_gridvals=[repmat(d_gridvals,N_a,1),repelem(a_grid,N_d,1)]; % only one aprime

%% CreateReturnFnMatrix_Case1_Disc creates a matrix of dimension (d and aprime)-by-a-by-z.
% Since the return function is independent of time creating it once and
% then using it every iteration is good for speed, but it does use a
% lot of memory.

ReturnMatrixraw=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,n_da, n_a, n_z, da_gridvals, a_grid, z_gridvals, ReturnFnParams);
ReturnMatrixraw=reshape(ReturnMatrixraw,[N_d,N_a,N_a,N_z]);

% For refinement, now we solve for d*(aprime,a,z) that maximizes the ReturnFn
[ReturnMatrix,~]=max(ReturnMatrixraw,[],1);
ReturnMatrix=shiftdim(ReturnMatrix,1);

%% The rest, except putting d back into Policy at the end, is all just copy-paste from ValueFnIter_preGI_nod_raw()
pi_z_alt=shiftdim(pi_z',-1);
pi_z_howards=repelem(pi_z,N_a,1);

addindexforaz=gpuArray(N_a*(0:1:N_a-1)'+N_a*N_a*(0:1:N_z-1));

%% First, just consider a_grid for next period
tempcounter=1;
currdist=1;
while currdist>(vfoptions.multigridswitch*vfoptions.tolerance) && tempcounter<=vfoptions.maxiter
    VKronold=VKron;
    
    % Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=VKronold.*pi_z_alt;
    EV(isnan(EV))=0; % multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; % aprime by a by z

    % Calc the max and it's index
    [VKron,Policy_a]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1); % a by z

    VKrondist=VKron(:)-VKronold(:); 
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    
    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards 
        tempmaxindex=shiftdim(Policy_a,1)+addindexforaz; % aprime index, add the index for a and z
        Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a,N_z]); % keep return function of optimal policy for using in Howards
        Policy_a=Policy_a(:); % a by z (this shape is just convenient for Howards)
        for Howards_counter=1:vfoptions.howards
            EVKrontemp=VKron(Policy_a,:);
            EVKrontemp=EVKrontemp.*pi_z_howards;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=Ftemp+DiscountFactorParamsVec*EVKrontemp; % interpolate EV
        end
    end

    tempcounter=tempcounter+1;
end
Policy_a=reshape(Policy_a,[1,N_a,N_z]); % Howards can mess with the size

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

ReturnMatrixfineraw=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_z, d_gridvals, aprime_grid, a_grid, z_gridvals, ReturnFnParams,1);
% ReturnMatrixfineraw=reshape(ReturnMatrixfineraw,[N_d,N_aprime,N_a,N_z]);

% For refinement, now we solve for d*(aprime,a,z) that maximizes the ReturnFn
[ReturnMatrixfine,dstar]=max(ReturnMatrixfineraw,[],1);
ReturnMatrixfine=shiftdim(ReturnMatrixfine,1);

EVinterpindex1=(1:1:N_aprimediff)';
EVinterpindex2=linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp)';

% For Howards we need
addindexforazfine=gpuArray(N_aprime*(0:1:N_a-1)'+N_aprime*N_a*(0:1:N_z-1));

pi_z_alt2=shiftdim(pi_z,-2);

%% Now switch to considering the fine/interpolated aprime_grid
currdist=1; % force going into the next while loop at least one iteration
while currdist>vfoptions.tolerance && tempcounter<=vfoptions.maxiter
    VKronold=VKron;

    % Switch VKron into being over vfoptions.maxaprimediff
    EVpre=reshape(VKron(aprimeindex,:),[N_aprimediff,N_a,N_z,N_z]); % last dimension is zprime    
    % Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=EVpre.*pi_z_alt2;
    EV(isnan(EV))=0; % multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=squeeze(sum(EV,4)); % sum over z', leaving a singular second dimension
    % EV is now [N_aprimediff,N_a,N_z]
    % Interpolate EV over aprime_grid

    EVinterp=interp1(EVinterpindex1,EV,EVinterpindex2);
    
    entireRHS=ReturnMatrixfine+DiscountFactorParamsVec*EVinterp; % aprime by a by z
    
    % Calc the max and it's index
    [VKron,Policy_a]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1); % a by z

    VKrondist=VKron(:)-VKronold(:); 
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards 
        tempmaxindex=shiftdim(Policy_a,1)+addindexforazfine; % aprime index, add the index for a and z, size is [N_a,N_z]
        Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a,N_z]); % keep return function of optimal policy for using in Howards
        tempmaxindex2=Policy_a(:)+N_aprime*(0:1:N_a*N_z-1)'; % size is [N_a*N_z,1], contains the (aprime,a,z) index; (this shape is just convenient for Howards)
        for Howards_counter=1:vfoptions.howards
            EVpre=reshape(VKron(aprimeindex,:),[N_aprimediff,N_a*N_z,N_z]); % last dimension is zprime
            EVKrontemp=interp1(EVinterpindex1,EVpre,EVinterpindex2); % interpolate V as Policy points to the interpolated indexes
            EVKrontemp=reshape(EVKrontemp,[N_aprime*N_a*N_z,N_z]);  % last dimension is zprime
            EVKrontemp=EVKrontemp(tempmaxindex2,:);
            EVKrontemp=EVKrontemp.*pi_z_howards;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=Ftemp+DiscountFactorParamsVec*EVKrontemp;
        end
    end

    tempcounter=tempcounter+1;
end

%% For refinement, add d back into Policy
Policy=zeros(3,N_a,N_z,'gpuArray');
temppolicyindex=reshape(Policy_a,[1,N_a*N_z])+N_aprime*(0:1:N_a*N_z-1);
Policy(1,:,:)=reshape(dstar(temppolicyindex),[N_a,N_z]); % note: dstar is defined on the fine grid

%% Switch policy to lower grid index and L2 index (is currently index on fine grid)
% Separate Policy into L1 and L2
fineindex=reshape(Policy_a,[N_a*N_z,1]);
L1a=ceil((fineindex-1)/(n2short+1))-1; % this ranges -1:0:2*vfoptions.maxaprimediff-1
% (L1a-vfoptions.maxaprimediff+1) ranges -vfoptions.maxaprimediff:1:vfoptions.maxaprimediff
L1=max(L1a-vfoptions.maxaprimediff+1+aprimeshifter(:)-1,1); % lower grid point index (on the full grid), so this ranges 0 to n_a-1
L1intermediate=max(L1a,0)+1; % lower grid point index (on the small grid, in form so we can get L2)
L2=fineindex-(L1intermediate-1)*(n2short+1); % L2 index

Policy(2,:,:)=reshape(L1,[1,N_a,N_z]);
Policy(3,:,:)=reshape(L2,[1,N_a,N_z]);


end
