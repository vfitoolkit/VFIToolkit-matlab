function [VKron, Policy]=ValueFnIter_preGI2B_HowardGreedy_nod_raw(VKron, n_a, n_z, a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParams, vfoptions)
% preGI: create the whole ReturnMatrix based on aprime
% Then take a multigrid approach, using just a_grid for aprime until near
% convergence, then switch to use the fine grid for aprime.

N_a1=n_a(1);
N_a2=prod(n_a(2:end));
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);
a_gridvals=CreateGridvals(n_a,a_grid,1);

N_a=prod(n_a);
N_z=prod(n_z);

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)

N_a1prime=N_a1+(N_a1-1)*vfoptions.ngridinterp;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*vfoptions.ngridinterp))';
n_aprime=[N_a1prime,n_a(2:end)];
N_aprime=prod(n_aprime);
aprime_gridvals=CreateGridvals(n_aprime,[a1prime_grid; a2_grid],1);
ReturnMatrixfine=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,n_aprime, n_a, n_z, aprime_gridvals, a_gridvals, z_gridvals, ReturnFnParams);
originalindex=gpuArray(1:vfoptions.ngridinterp+1:N_a1prime)'+N_a1prime*gpuArray(0:1:N_a2-1);
ReturnMatrix=ReturnMatrixfine(originalindex(:),:,:);

pi_z_alt=shiftdim(pi_z',-1);

addindexforaz=gpuArray(N_a*(0:1:N_a-1)'+N_a*N_a*(0:1:N_z-1));
addindexforazfine=gpuArray(N_aprime*(0:1:N_a-1)'+N_aprime*N_a*(0:1:N_z-1));

% Setup specific to greedy Howards
spI = gpuArray.speye(N_a*N_z);
N_a_times_zind=N_a*gpuArray(0:1:N_z-1); % already contains -1
azind1=repmat(gpuArray(1:1:N_a*N_z)',1,N_z); % (a-z,zprime)
azind2=repmat(gpuArray(1:1:N_a*N_z)',2,N_z); % (a-z-2,zprime)
pi_z_big1=gpuArray(repelem(pi_z,N_a,1)); % (a-z,zprime)
pi_z_big2=gpuArray(repmat(pi_z_big1,2,1)); % (a-z-2,zprime)

%%
tempcounter=1;
currdist=1;

%% First, just consider a_grid for next period
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

%% Now switch to considering the fine/interpolated aprime_grid
currdist=1; % force going into the next while loop at least one iteration
while currdist>vfoptions.tolerance && tempcounter<=vfoptions.maxiter
    VKronold=VKron;
    
    % Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=VKronold.*pi_z_alt;
    EV(isnan(EV))=0; % multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    % Interpolate EV over aprime_grid
    EVinterp=reshape(interp1(a1_grid,reshape(EV,[N_a1,N_a2,N_z]),a1prime_grid),[N_aprime,1,N_z]);

    entireRHS=ReturnMatrixfine+DiscountFactorParamsVec*EVinterp; % aprime by a by z

    %Calc the max and it's index
    [VKron,Policy]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1); % a by z

    VKrondist=VKron(:)-VKronold(:); 
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    
    % Use greedy-Howards Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
        tempmaxindex=shiftdim(Policy,1)+addindexforazfine; % aprime index, add the index for a and z
        Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a*N_z,1]); % keep return function of optimal policy for using in Howards

        % Split Policy into a1 and a2, then switch a1 on fine, to a1 lower grid point index and probability
        Policya1=rem(Policy-1,N_a1prime)+1;
        Policya2=ceil(Policy/N_a1prime);
        Policy_lowerind=max(ceil((Policya1(:)-1)/(n2short+1))-1,0)+1;  % lower grid point index (of first asset)
        Policy_lowerprob=1- ((Policya1(:)-(Policy_lowerind-1)*(n2short+1))-1)/(n2short+1); % Policy-(Policy_lowerind-1)*(n2short+1) is 2nd layer index
        indp = (Policy_lowerind +N_a1*(Policya2(:)-1))+N_a_times_zind; % with all tomorrows z (a-z,zprime)

        T_E=sparse(azind2,[indp;indp+1],[Policy_lowerprob;1-Policy_lowerprob].*pi_z_big2,N_a*N_z,N_a*N_z);

        VKron=(spI-DiscountFactorParamsVec*T_E)\Ftemp;
        VKron=reshape(VKron,[N_a,N_z]);
    end

    tempcounter=tempcounter+1;

end


%% Switch policy to lower grid index and L2 index (is currently index on fine grid)
fineindex=reshape(Policy,[1,N_a,N_z]);
Policy=zeros(2,N_a,N_z,'gpuArray');
fineindexvec1=rem(fineindex-1,N_a1prime)+1;
fineindexvec2=ceil(fineindex/N_a1prime);
L1a=ceil((fineindexvec1-1)/(n2short+1))-1;
L1=max(L1a,0)+1; % lower grid point index
L2=fineindexvec1-(L1-1)*(n2short+1); % L2 index
Policy(1,:,:)=L1;
Policy(2,:,:)=fineindexvec2;
Policy(3,:,:)=L2;

end
