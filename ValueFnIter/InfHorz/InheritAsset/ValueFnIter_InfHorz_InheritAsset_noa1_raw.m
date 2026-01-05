function [VKron, Policy]=ValueFnIter_InfHorz_InheritAsset_noa1_raw(VKron,n_d1,n_d2,n_a2,n_z, d_gridvals, d2_grid, a2_grid, z_gridvals, pi_z, ReturnFn, aprimeFn, DiscountFactorParamsVec, ReturnFnParamsVec, aprimeFnParamsVec, vfoptions)

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_a2=prod(n_a2);
N_a=N_a2;
N_z=prod(n_z);

% V=zeros(N_a,N_z,'gpuArray');
% Policy=zeros(N_a,N_z,'gpuArray');

%% Refine d1
% CreateReturnFnMatrix_Case1_Disc creates a matrix of dimension (d1 and d2 and aprime)-by-a-by-z.
% Since the return function is independent of time creating it once and then using it every iteration is good for speed, but it does use a lot of memory.

if vfoptions.lowmemory==0
    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, [n_d1,n_d2], n_a2, n_z, d_gridvals, a2_grid, z_gridvals, ReturnFnParamsVec);
    ReturnMatrix=reshape(ReturnMatrix,[N_d1,N_d2,N_a2,N_z]);

    % For refinement, now we solve for d1*(d2,a,z) that maximizes the ReturnFn
    [ReturnMatrix,d1star]=max(ReturnMatrix,[],1);
    ReturnMatrix=shiftdim(ReturnMatrix,1);

elseif vfoptions.lowmemory==1 % loop over z
    %% Refinement: calculate ReturnMatrix and 'remove' the d1 dimension
    ReturnMatrix=zeros(N_d2,N_a,N_z,'gpuArray'); % 'refined' return matrix
    d1star=zeros(N_d2,N_a,N_z,'gpuArray');
    l_z=length(n_z);
    special_n_z=ones(1,l_z,'gpuArray');
    for z_c=1:N_z
        zvals=z_gridvals(z_c,:);
        ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, [n_d1,n_d2], n_a2, special_n_z, d_gridvals, a2_grid, zvals, ReturnFnParamsVec);
        ReturnMatrix_z=reshape(ReturnMatrix_z,[N_d1,N_d2,N_a2,N_z]);

        [ReturnMatrix_z,d1star_z]=max(ReturnMatrix_z,[],1); % solve for d1star
        ReturnMatrix(:,:,z_c)=shiftdim(ReturnMatrix_z,1);
        d1star(:,:,z_c)=shiftdim(d1star_z,1);
    end
end


%% Create aprimeFn Matrix
[a2primeIndex,a2primeProbs]=CreateInheritanceAssetFnMatrix_Case1(aprimeFn, n_d2, n_a2, n_z, n_z, d2_grid, a2_grid, z_gridvals, z_gridvals, aprimeFnParamsVec); % Note, is actually a2prime_grid (but a2_grid is anyway same for all ages)
% Note: aprimeIndex is [N_d2,N_zprime,N_z], and aprimeProbs is also [N_d2,N_zprime,N_z]

addindexforzprime=N_a*gpuArray(0:1:N_z-1);


% pi_z_alt=shiftdim(pi_z',-1);
% pi_z_howards=repelem(pi_z,N_a,1);
% 
% addindexforaz=gpuArray(N_a*(0:1:N_a-1)'+N_a*N_a*(0:1:N_z-1));

%% Do the value fn iteration
%%
tempcounter=1;
currdist=Inf;
while currdist>vfoptions.tolerance && tempcounter<=vfoptions.maxiter
    VKronold=VKron;
    
    Vlower=reshape(VKronold(a2primeIndex+addindexforzprime),[N_d2,N_z,N_z]); % (d2,zprime,z)
    Vupper=reshape(VKronold(a2primeIndex+1+addindexforzprime),[N_d2,N_z,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    a2primeProbs(skipinterp)=0; % effectively skips interpolation
   
    % Switch EV from being in terps of a2prime to being in terms of d2 and a2
    EV=a2primeProbs.*Vlower+(1-a2primeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)
    
    EV=EV.*shiftdim(pi_z',-1);
    EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
    EV=sum(EV,2);
    % EV is over (d2,1,z)

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV;

    % Calc the max and it's index
    [VKron,Policy]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1); % a by z

    VKrondist=VKron(:)-VKronold(:); 
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    
    % % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    % if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards 
    %     tempmaxindex=shiftdim(Policy,1)+addindexforaz; % aprime index, add the index for a and z
    %     Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a,N_z]); % keep return function of optimal policy for using in Howards
    %     Policy=Policy(:); % a by z (this shape is just convenient for Howards)
    % 
    %     for Howards_counter=1:vfoptions.howards
    %         EVKrontemp=VKron(Policy,:);
    %         EVKrontemp=EVKrontemp.*pi_z_howards;
    %         EVKrontemp(isnan(EVKrontemp))=0;
    %         EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
    %         VKron=Ftemp+DiscountFactorParamsVec*EVKrontemp;
    %     end
    % end

    tempcounter=tempcounter+1;
end

Policy=reshape(Policy,[N_a,N_z]);

%% For refinement, add d1 into Policy (which is just d2)
% Policy is currently
temppolicyindex=reshape(Policy,[1,N_a*N_z])+(0:1:N_a*N_z-1)*N_a;
Policy_d1=reshape(d1star(temppolicyindex),[N_a,N_z]);
Policy=Policy_d1+N_d1*(Policy-1);






end