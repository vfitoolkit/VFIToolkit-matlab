function [VKron,Policy]=ValueFnIter_Refine_postGI2B_raw(VKron,n_d,n_a,n_z,d_gridvals,a_grid,z_gridvals,pi_z,ReturnFn,DiscountFactorParamsVec,ReturnFnParams,vfoptions)
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

n_da=[n_d,n_a];
da_gridvals=[repmat(d_gridvals,N_a,1),repelem(a_gridvals,N_d,1)];

%% CreateReturnFnMatrix_Case1_Disc creates a matrix of dimension (d and aprime)-by-a-by-z.
% Since the return function is independent of time creating it once and
% then using it every iteration is good for speed, but it does use a
% lot of memory.

if vfoptions.lowmemory==0
    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,n_da, n_a, n_z, da_gridvals, a_gridvals, z_gridvals, ReturnFnParams);
    ReturnMatrix=reshape(ReturnMatrix,[N_d,N_a,N_a,N_z]);
    
    % For refinement, now we solve for d*(aprime,a,z) that maximizes the ReturnFn
    [ReturnMatrix,dstar]=max(ReturnMatrix,[],1);
    ReturnMatrix=shiftdim(ReturnMatrix,1);

elseif vfoptions.lowmemory==1 % loop over z
    %% Refinement: calculate ReturnMatrix and 'remove' the d dimension
    ReturnMatrix=zeros(N_a,N_a,N_z,'gpuArray'); % 'refined' return matrix
    dstar=zeros(N_a,N_a,N_z,'gpuArray');
    l_z=length(n_z);
    special_n_z=ones(1,l_z);
    for z_c=1:N_z
        zvals=z_gridvals(z_c,:);
        ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,n_da, n_a, special_n_z, da_gridvals, a_gridvals, zvals, ReturnFnParams);
        ReturnMatrix_z=reshape(ReturnMatrix_z,[N_d,N_a,N_a]);
        [ReturnMatrix_z,dstar_z]=max(ReturnMatrix_z,[],1); % solve for dstar
        ReturnMatrix(:,:,z_c)=shiftdim(ReturnMatrix_z,1);
        dstar(:,:,z_c)=shiftdim(dstar_z,1);
    end
end

%% The rest, except putting d back into Policy at the end, is all just copy-paste from ValueFnIter_preGI_nod_raw()
pi_z_alt=shiftdim(pi_z',-1);
pi_z_howards=repelem(pi_z,N_a,1);

addindexforaz=gpuArray(N_a*(0:1:N_a-1)'+N_a*N_a*(0:1:N_z-1));

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

Policy_a=reshape(Policy_a,[1,N_a,N_z]); % Howards can mess with the size
Policy_a1=rem(Policy_a-1,N_a1)+1;
% Policy_a2=ceil(Policy/N_a1);


%% Now that we have solved on the rough grid, we resolve on the fine grid
% Based on solving a bunch of value fns with and without grid
% interpolation, the 'lower grid index' with grid interpolation is always
% within a point or two of the solution on the rough grid. So here we only
% consider +-vfoptions.maxaprimediff to set up the fine/interpolated aprime_grid

% Current optimal aprime is Policy_a
% So create an aprime_grid that is just an interpolation within +-vfoptions.maxaprimediff

% First, create an a1prime_grid that is just the +-vfoptions.maxaprimediff
% Note: this code is for models with a single endogenous state
n_a1primediff=1+2*vfoptions.maxaprimediff;
N_a1primediff=prod(n_a1primediff);
a1primeshifter=min(max(Policy_a1,1+vfoptions.maxaprimediff),N_a1-vfoptions.maxaprimediff);
a1primeindex=(-vfoptions.maxaprimediff:1:vfoptions.maxaprimediff)' +a1primeshifter; % size n_aprime-by-n_a
a1prime_grid=a1_grid(a1primeindex);
% Second, interpolate this
% Grid interpolation
% vfoptions.ngridinterp=9;
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
    ReturnMatrixfine=CreateReturnFnMatrix_Case1_Disc_DC2B_Par2(ReturnFn, n_d, n_z, d_gridvals, a1prime_grid,a2_grid, a1_grid,a2_grid, z_gridvals, ReturnFnParams,1);
    ReturnMatrixfine=reshape(ReturnMatrixfine,[N_d,N_aprime,N_a,N_z]);
    
    % For refinement, now we solve for d*(aprime,a,z) that maximizes the ReturnFn
    [ReturnMatrixfine,dstar]=max(ReturnMatrixfine,[],1);
    ReturnMatrixfine=shiftdim(ReturnMatrixfine,1);

elseif vfoptions.lowmemory==1 % loop over z
    % Refinement: calculate ReturnMatrix and 'remove' the d dimension
    ReturnMatrixfine=zeros(N_aprime,N_a,N_z,'gpuArray'); % 'refined' return matrix
    dstar=zeros(N_aprime,N_a,N_z,'gpuArray');
    l_z=length(n_z);
    special_n_z=ones(1,l_z);
    for z_c=1:N_z
        zvals=z_gridvals(z_c,:);
        ReturnMatrixfine_z=CreateReturnFnMatrix_Case1_Disc_DC2B_Par2(ReturnFn,n_d, special_n_z, d_gridvals, a1prime_grid(:,1,:,:,z_c),a2_grid, a1_grid,a2_grid, zvals, ReturnFnParams,1);
        ReturnMatrixfine_z=reshape(ReturnMatrixfine_z,[N_d,N_aprime,N_a]);
        [ReturnMatrixfine_z,dstar_z]=max(ReturnMatrixfine_z,[],1); % solve for dstar
        ReturnMatrixfine(:,:,z_c)=shiftdim(ReturnMatrixfine_z,1);
        dstar(:,:,z_c)=shiftdim(dstar_z,1);
    end
end

pi_z_alt2=shiftdim(pi_z,-2);

% For Howards we need
addindexforazfine=gpuArray(N_aprime*(0:1:N_a-1)'+N_aprime*N_a*(0:1:N_z-1));


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

    EVinterp=reshape(interp1(EVinterpindex1,reshape(EV,[N_a1primediff,N_a2,N_a,N_z]),EVinterpindex2),[N_aprime,N_a,N_z]);

    entireRHS=ReturnMatrixfine+DiscountFactorParamsVec*EVinterp; % aprime by a by z

    % Calc the max and it's index
    [VKron,Policy_a]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1); % a by z

    VKrondist=VKron(:)-VKronold(:); 
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    
    % THIS IMPLENTATION OF HOWARDS ITER IS ACTUALLY SLOWER
    % I THINK IT IS JUST THAT I DONT REALLY UNDERSTAND HOW TO DO HOWARDS WELL WITH THIS Post-GI APPROACH
    % % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    % if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards 
    %     tempmaxindex=shiftdim(Policy,1)+addindexforazfine; % aprime index, add the index for a and z, size is [N_a,N_z]
    %     Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a,N_z]); % keep return function of optimal policy for using in Howards
    %     tempmaxindex2=Policy(:)+N_aprime*(0:1:N_a*N_z-1)'; % size is [N_a*N_z,1], contains the (aprime,a,z) index; (this shape is just convenient for Howards)
    %     for Howards_counter=1:vfoptions.howards
    %         EVpre=reshape(VKron(aprimeindex,:),[N_a1primediff,N_a2,N_a*N_z,N_z]); % last dimension is zprime
    %         EVKrontemp=interp1(EVinterpindex1,EVpre,EVinterpindex2); % interpolate V as Policy points to the interpolated indexes
    %         EVKrontemp=reshape(EVKrontemp,[N_aprime*N_a*N_z,N_z]);  % last dimension is zprime
    %         EVKrontemp=EVKrontemp(tempmaxindex2,:);
    %         EVKrontemp=EVKrontemp.*pi_z_howards;
    %         EVKrontemp(isnan(EVKrontemp))=0;
    %         EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
    %         VKron=Ftemp+DiscountFactorParamsVec*EVKrontemp;
    %     end
    % end

    tempcounter=tempcounter+1;
end


%% Switch policy to lower grid index and L2 index (is currently index on fine grid)
fineindex=reshape(Policy_a,[1,N_a,N_z]);
Policy=zeros(4,N_a,N_z,'gpuArray');
fineindexvec1=rem(fineindex-1,N_a1prime)+1; % a1prime, but in post-GI index
fineindexvec2=ceil(fineindex/N_a1prime); % a2prime index

fineindexvec1=reshape(fineindexvec1,[N_a*N_z,1]);
L1a=ceil((fineindexvec1-1)/(n2short+1))-1; % this ranges -1:0:2*vfoptions.maxaprimediff-1
% (L1a-vfoptions.maxaprimediff+1) ranges -vfoptions.maxaprimediff:1:vfoptions.maxaprimediff
L1=max(L1a-vfoptions.maxaprimediff+1+a1primeshifter(:)-1,1); % lower grid point index (on the full grid), so this ranges 0 to n_a-1
L1intermediate=max(L1a,0)+1; % lower grid point index (on the small grid, in form so we can get L2)
L2=fineindexvec1-(L1intermediate-1)*(n2short+1); % L2 index

Policy(2,:,:)=reshape(L1,[1,N_a,N_z]);
Policy(3,:,:)=fineindexvec2;
Policy(4,:,:)=reshape(L2,[1,N_a,N_z]);

%% For refinement, add d back into Policy
temppolicyindex=fineindex(:)+N_aprime*(0:1:N_a*N_z-1)';
Policy(1,:,:)=reshape(dstar(temppolicyindex),[N_a,N_z]); % note: dstar is defined on the fine grid

end
