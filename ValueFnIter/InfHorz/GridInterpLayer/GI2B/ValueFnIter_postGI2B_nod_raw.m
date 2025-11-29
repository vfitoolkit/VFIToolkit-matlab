function [VKron, Policy]=ValueFnIter_postGI2B_nod_raw(VKron, n_a, n_z, a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParams, vfoptions)
% preGI: create the whole ReturnMatrix based on aprime
% Then take a multigrid approach, using just a_grid for aprime until near
% convergence, then switch to use the fine grid for aprime.
% 2B: two endogenous states, just use grid interpolation layer on the first

N_a=prod(n_a);
N_z=prod(n_z);

N_a1=n_a(1);
N_a2=prod(n_a(2:end));
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);
a_gridvals=CreateGridvals(n_a,a_grid,1);


ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,n_a, n_a, n_z, a_gridvals, a_gridvals, z_gridvals, ReturnFnParams);

pi_z_alt=shiftdim(pi_z',-1);
pi_z_howards=repelem(pi_z,N_a,1);

addindexforaz=gpuArray(N_a*(0:1:N_a-1)'+N_a*N_a*(0:1:N_z-1));

%%
tempcounter=1;
currdist=Inf;


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
    
    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards 
        tempmaxindex=shiftdim(Policy,1)+addindexforaz; % aprime index, add the index for a and z
        Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a,N_z]); % keep return function of optimal policy for using in Howards
        Policy=Policy(:); % a by z (this shape is just convenient for Howards)
        for Howards_counter=1:vfoptions.howards
            EVKrontemp=VKron(Policy,:);
            EVKrontemp=EVKrontemp.*pi_z_howards;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=Ftemp+DiscountFactorParamsVec*EVKrontemp; % interpolate EV
        end
    end

    tempcounter=tempcounter+1;
end

Policy=reshape(Policy,[1,N_a,N_z]); % Howards can mess with the size
Policy_a1=rem(Policy-1,N_a1)+1;
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

ReturnMatrixfine=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, n_z, a1prime_grid, a2_grid, a1_grid, a2_grid, z_gridvals, ReturnFnParams, 2);

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

    EVinterp=reshape(interp1(EVinterpindex1,reshape(EV,[N_a1primediff,N_a2,N_a,N_z]),EVinterpindex2),[N_aprime,N_a,N_z]);

    entireRHS=ReturnMatrixfine+DiscountFactorParamsVec*EVinterp; % aprime by a by z

    % Calc the max and it's index
    [VKron,Policy]=max(entireRHS,[],1);
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
fineindex=reshape(Policy,[1,N_a,N_z]);
Policy=zeros(3,N_a,N_z,'gpuArray');
fineindexvec1=rem(fineindex-1,N_a1prime)+1; % a1prime, but in post-GI index
fineindexvec2=ceil(fineindex/N_a1prime); % a2prime index

fineindexvec1=reshape(fineindexvec1,[N_a*N_z,1]);
L1a=ceil((fineindexvec1-1)/(n2short+1))-1; % this ranges -vfoptions.maxaprimediff:1:vfoptions.maxaprimediff
L1=max(L1a-vfoptions.maxaprimediff+1+a1primeshifter(:)-1,1); % lower grid point index (on the full grid), so this ranges 0 to n_a-1
L1intermediate=max(L1a,0)+1; % lower grid point index (on the small grid, in form so we can get L2)
L2=fineindexvec1-(L1intermediate-1)*(n2short+1); % L2 index

Policy(1,:,:)=reshape(L1,[1,N_a,N_z]);
Policy(2,:,:)=fineindexvec2;
Policy(3,:,:)=reshape(L2,[1,N_a,N_z]);


end
