function [VKron,Policy]=ValueFnIter_Refine_preGI2B_raw(VKron,n_d,n_a,n_z,d_gridvals,a_grid,z_gridvals,pi_z,ReturnFn,DiscountFactorParamsVec,ReturnFnParams,vfoptions)
% When using refinement, lowmemory is implemented in the first stage (return fn) but not the second (the actual iteration).
% Refine, so there is at least one d variable

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

N_a1=n_a(1);
N_a2=prod(n_a(2:end));
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);
a_gridvals=CreateGridvals(n_a,a_grid,1);

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)

N_a1prime=N_a1+(N_a1-1)*vfoptions.ngridinterp;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*vfoptions.ngridinterp))';
n_aprime=[N_a1prime,n_a(2:end)];
N_aprime=prod(n_aprime);
aprime_grid=[a1prime_grid; a2_grid];
aprime_gridvals=CreateGridvals(n_aprime,aprime_grid,1);

n_daprime=[n_d,n_aprime];
daprime_gridvals=[repmat(d_gridvals,N_aprime,1),repelem(aprime_gridvals,N_d,1)];

%% CreateReturnFnMatrix_Case1_Disc creates a matrix of dimension (d and aprime)-by-a-by-z.
% Since the return function is independent of time creating it once and
% then using it every iteration is good for speed, but it does use a
% lot of memory.

if vfoptions.lowmemory==0
    ReturnMatrixfine=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,n_daprime, n_a, n_z, daprime_gridvals, a_gridvals, z_gridvals, ReturnFnParams);
    ReturnMatrixfine=reshape(ReturnMatrixfine,[N_d,N_aprime,N_a,N_z]);
    
    % For refinement, now we solve for d*(aprime,a,z) that maximizes the ReturnFn
    [ReturnMatrixfine,dstar]=max(ReturnMatrixfine,[],1);
    ReturnMatrixfine=shiftdim(ReturnMatrixfine,1);

    originalindex=gpuArray(1:vfoptions.ngridinterp+1:N_a1prime)'+N_a1prime*gpuArray(0:1:N_a2-1);
    ReturnMatrix=ReturnMatrixfine(originalindex(:),:,:);

elseif vfoptions.lowmemory==1 % loop over z
    %% Refinement: calculate ReturnMatrix and 'remove' the d dimension
    ReturnMatrixfine=zeros(N_aprime,N_a,N_z,'gpuArray'); % 'refined' return matrix
    dstar=zeros(N_aprime,N_a,N_z,'gpuArray');
    l_z=length(n_z);
    special_n_z=ones(1,l_z);
    for z_c=1:N_z
        zvals=z_gridvals(z_c,:);
        ReturnMatrixfine_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,n_daprime, n_a, special_n_z, daprime_gridvals, a_gridvals, zvals, ReturnFnParams);
        ReturnMatrixfine_z=reshape(ReturnMatrixfine_z,[N_d,N_aprime,N_a]);
        [ReturnMatrixfine_z,dstar_z]=max(ReturnMatrixfine_z,[],1); % solve for dstar
        ReturnMatrixfine(:,:,z_c)=shiftdim(ReturnMatrixfine_z,1);
        dstar(:,:,z_c)=shiftdim(dstar_z,1);
    end

    originalindex=gpuArray(1:vfoptions.ngridinterp+1:N_a1prime)'+N_a1prime*gpuArray(0:1:N_a2-1);
    ReturnMatrix=ReturnMatrixfine(originalindex(:),:,:);
end

%% The rest, except putting d back into Policy at the end, is all just copy-paste from ValueFnIter_preGI_nod_raw()
pi_z_alt=shiftdim(pi_z',-1);
pi_z_howards=repelem(pi_z,N_a,1);

addindexforaz=gpuArray(N_a*(0:1:N_a-1)'+N_a*N_a*(0:1:N_z-1));
addindexforazfine=gpuArray(N_aprime*(0:1:N_a-1)'+N_aprime*N_a*(0:1:N_z-1));

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

    % Calc the max and it's index
    [VKron,Policy_a]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1); % a by z

    VKrondist=VKron(:)-VKronold(:); 
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    
    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards 
        tempmaxindex=shiftdim(Policy_a,1)+addindexforazfine; % aprime index, add the index for a and z
        Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a,N_z]); % keep return function of optimal policy for using in Howards
        Policy_a=Policy_a(:); % a by z (this shape is just convenient for Howards)
        for Howards_counter=1:vfoptions.howards
            EVKrontemp=reshape(interp1(a1_grid,reshape(VKron,[N_a1,N_a2,N_z]),a1prime_grid),[N_aprime,1,N_z]); % interpolate V as Policy points to the interpolated indexes
            EVKrontemp=EVKrontemp(Policy_a,:);
            EVKrontemp=EVKrontemp.*pi_z_howards;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=Ftemp+DiscountFactorParamsVec*EVKrontemp;
        end
    end

    tempcounter=tempcounter+1;

end

%% Switch policy to lower grid index and L2 index (is currently index on fine grid)
fineindex=reshape(Policy_a,[1,N_a,N_z]);
Policy=zeros(4,N_a,N_z,'gpuArray');
fineindexvec1=rem(fineindex-1,N_a1prime)+1;
fineindexvec2=ceil(fineindex/N_a1prime);

L1a=ceil((fineindexvec1-1)/(n2short+1))-1;
L1=max(L1a,0)+1; % lower grid point index
L2=fineindexvec1-(L1-1)*(n2short+1); % L2 index

Policy(2,:,:)=L1;
Policy(3,:,:)=fineindexvec2;
Policy(4,:,:)=L2;

%% For refinement, add d back into Policy
temppolicyindex=fineindex(:)+N_aprime*(0:1:N_a*N_z-1)';
Policy(1,:,:)=reshape(dstar(temppolicyindex),[N_a,N_z]); % note: dstar is defined on the fine grid


end
