function [VKron, Policy]=ValueFnIter_preGI_HowardMix_nod_raw(VKron, n_a, n_z, a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParams, vfoptions)
% preGI: create the whole ReturnMatrix based on aprime
% Then take a multigrid approach, using just a_grid for aprime until near
% convergence, then switch to use the fine grid for aprime.

N_a=prod(n_a);
N_z=prod(n_z);

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)

n_aprime=n_a+(n_a-1)*vfoptions.ngridinterp;
N_aprime=prod(n_aprime);
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*vfoptions.ngridinterp))';
ReturnMatrixfine=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,n_aprime, n_a, n_z, aprime_grid, a_grid, z_gridvals, ReturnFnParams);
ReturnMatrix=ReturnMatrixfine(1:vfoptions.ngridinterp+1:n_aprime,:,:);


pi_z_alt=shiftdim(pi_z',-1);

addindexforaz=gpuArray(N_a*(0:1:N_a-1)'+N_a*N_a*(0:1:N_z-1));
addindexforazfine=gpuArray(N_aprime*(0:1:N_a-1)'+N_aprime*N_a*(0:1:N_z-1));

% Setup specific to Howard iterations
pi_z_howards=repelem(pi_z,N_a,1);

% Setup specific to greedy Howards
spI = gpuArray.speye(N_a*N_z);
N_a_times_zind=N_a*gpuArray(0:1:N_z-1); % already contains -1
azind1=repmat(gpuArray(1:1:N_a*N_z)',1,N_z); % (a-z,zprime)
% azind2=repmat(gpuArray(1:1:N_a*N_z)',2,N_z); % (a-z-2,zprime)
pi_z_big1=gpuArray(repelem(pi_z,N_a,1)); % (a-z,zprime)
% pi_z_big2=gpuArray(repmat(pi_z_big1,2,1)); % (a-z-2,zprime)


%%
tempcounter=1;
currdist=1;

%% First, just consider a_grid for next period, use Howards greedy
while currdist>(vfoptions.multigridswitch*vfoptions.tolerance) && tempcounter<=vfoptions.maxiter
    VKronold=VKron;
    
    % Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=VKronold.*pi_z_alt;
    EV(isnan(EV))=0; % multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; % aprime by a by z

    %Calc the max and it's index
    [VKron,Policy]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1); % a by z

    VKrondist=VKron(:)-VKronold(:); 
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    % Use greedy-Howards Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
        tempmaxindex=shiftdim(Policy,1)+addindexforaz; % aprime index, add the index for a and z
        Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a*N_z,1]); % keep return function of optimal policy for using in Howards

        T_E=sparse(azind1,Policy(:)+N_a_times_zind,pi_z_big1,N_a*N_z,N_a*N_z);

        VKron=(spI-DiscountFactorParamsVec*T_E)\Ftemp;
        VKron=reshape(VKron,[N_a,N_z]);
    end

    tempcounter=tempcounter+1;

end

%% Now switch to considering the fine/interpolated aprime_grid, use Howards iteration
currdist=1; % force going into the next while loop at least one iteration
while currdist>vfoptions.tolerance && tempcounter<=vfoptions.maxiter
    VKronold=VKron;
    
    % Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=VKronold.*pi_z_alt;
    EV(isnan(EV))=0; % multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    % Interpolate EV over aprime_grid
    EVinterp=interp1(a_grid,EV,aprime_grid);

    entireRHS=ReturnMatrixfine+DiscountFactorParamsVec*EVinterp; % aprime by a by z

    %Calc the max and it's index
    [VKron,Policy]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1); % a by z

    VKrondist=VKron(:)-VKronold(:); 
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    
    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards 
        tempmaxindex=shiftdim(Policy,1)+addindexforazfine; % aprime index, add the index for a and z
        Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a,N_z]); % keep return function of optimal policy for using in Howards
        Policy=Policy(:); % a by z (this shape is just convenient for Howards)
        for Howards_counter=1:vfoptions.howards
            EVKrontemp=interp1(a_grid,VKron,aprime_grid); % interpolate V as Policy points to the interpolated indexes
            EVKrontemp=EVKrontemp(Policy,:);
            EVKrontemp=EVKrontemp.*pi_z_howards;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=Ftemp+DiscountFactorParamsVec*EVKrontemp;
        end
    end

    tempcounter=tempcounter+1;

end


%% Switch policy to lower grid index and L2 index (is currently index on fine grid)
fineindex=reshape(Policy,[N_a*N_z,1]);
Policy=zeros(2,N_a,N_z,'gpuArray');
L1a=ceil((fineindex-1)/(n2short+1))-1;
L1=max(L1a,0)+1; % lower grid point index
L2=fineindex-(L1-1)*(n2short+1); % L2 index
Policy(1,:,:)=reshape(L1,[1,N_a,N_z]);
Policy(2,:,:)=reshape(L2,[1,N_a,N_z]);

end
