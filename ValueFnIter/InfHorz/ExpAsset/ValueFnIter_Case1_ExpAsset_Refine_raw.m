function [V, Policy]=ValueFnIter_Case1_ExpAsset_Refine_raw(V0,n_d1,n_d2,n_a1,n_a2,n_z, d1_gridvals, d2_grid, a1_gridvals, a2_grid, z_gridvals, pi_z, ReturnFn, aprimeFn, Parameters, DiscountFactorParamsVec, ReturnFnParamsVec, aprimeFnParamNames, vfoptions)

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);

d2_gridvals=CreateGridvals(n_d2,d2_grid,1);
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
n_a1prime=n_a1;
% a1prime_gridvals=a1_gridvals;

%% Refine the ReturnMatrix (removing the d1 dimension)
% Since the return function is independent of time creating it once and
% then using it every iteration is good for speed, but it does use a
% lot of memory.

if vfoptions.lowmemory==0
    ReturnMatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, n_d1,n_d2, n_a1prime, n_a1,n_a2,n_z, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals,z_gridvals, ReturnFnParamsVec,0,1);
    
    % For refinement, now we solve for d*(aprime,a,z) that maximizes the ReturnFn
    [ReturnMatrix,dstar]=max(ReturnMatrix,[],1);
    ReturnMatrix=shiftdim(ReturnMatrix,1);
elseif vfoptions.lowmemory==1 % loop over z
    %% Refinement: calculate ReturnMatrix and 'remove' the d dimension
    ReturnMatrix=zeros(N_d2*N_a1,N_a,N_z,'gpuArray'); % 'refined' return matrix
    dstar=zeros(N_d2*N_a1,N_a,N_z,'gpuArray');
    l_z=length(n_z);
    special_n_z=ones(1,l_z,'gpuArray');
    for z_c=1:N_z
        zvals=z_gridvals(z_c,:);
        ReturnMatrix_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, n_d1,n_d2, n_a1prime, n_a1,n_a2,special_n_z, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals,zvals, ReturnFnParamsVec,0,1);
        [ReturnMatrix_z,dstar_z]=max(ReturnMatrix_z,[],1); % solve for dstar
        ReturnMatrix(:,:,z_c)=shiftdim(ReturnMatrix_z,1);
        dstar(:,:,z_c)=shiftdim(dstar_z,1);
    end
end
% Following is just copy-paste from nod1 case, except right at end when add d1 back in.


%%
V=reshape(V0,[N_a,N_z]);
Policy=zeros(N_a,N_z,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

% for Howards, preallocate
Ftemp=zeros(N_a,N_z,'gpuArray');
% and we need [because of experienceasset, this is very different to usual]
aaa=repelem(pi_z,N_a,1); % pi_z in the form we need to compute expectations in Howards (a1a2z,zprime)

% precompute
Epi_z=shiftdim(pi_z',-2); % pi_z in the form we need it to compute the expectations

% I want the print that tells you distance to have number of decimal points corresponding to vfoptions.tolerance
distvstolstr=['ValueFnIter: after %i iterations the dist is %4.',num2str(-round(log10(vfoptions.tolerance))),'f \n'];

%% Precompute some aspects of experienceasset
aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames);
[a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_Case1(aprimeFn, n_d2, n_a2, d2_grid, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
% Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2]
aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2]
aprimeProbs=repmat(a2primeProbs,N_a1,1,N_z);  % [N_d2*N_a1,N_a2,N_z]

%%
currdist=Inf;
tempcounter=1;
while currdist>vfoptions.tolerance && tempcounter<=vfoptions.maxiter

    Vold=V;
    
    Vlower=reshape(Vold(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_z]);
    Vupper=reshape(Vold(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs2=aprimeProbs; % version that I can modify with skipinterp
    aprimeProbs2(skipinterp)=0; % effectively skips interpolation
    
    % Switch EV from being in terps of a2prime to being in terms of d2 and a2
    EV=aprimeProbs2.*Vlower+(1-aprimeProbs2).*Vupper; % (d2,a1prime,a2,zprime)
    % Already applied the probabilities from interpolating onto grid
    
    %Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=EV.*Epi_z;
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=squeeze(sum(EV,3)); % sum over z', leaving a singular second dimension
    % EV is over (d2 & a1prime,a2,1,z)

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(EV,1,N_a1,1,1);
    
    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);

    V=shiftdim(Vtemp,1);
    Policy=shiftdim(maxindex,1);
    
    %% Finish up
    % Update currdist
    Vdist=V(:)-Vold(:);
    Vdist(isnan(Vdist))=0;
    currdist=max(abs(Vdist));
    
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && vfoptions.maxhowards>0 % Use Howards Policy Fn Iteration Improvement

        Ftemp=reshape(ReturnMatrix(Policy+N_d2*N_a1*(0:1:N_a-1)'+N_d2*N_a1*N_a*(0:1:N_z-1)),[N_a*N_z,1]);
        Policy_d2ind=rem(Policy(:)-1,N_d2)+1;
        Policy_a1primeind=ceil(Policy(:)/N_d2); % size(Policy_a1primeind) is [N_a*N_z,1]

        % a2primeIndex is [N_d2,N_a2]
        temp=Policy_d2ind+N_d2*repmat(repelem((0:1:N_a2-1)',N_a1,1),N_z,1); % (d2,a2) indexes in terms of (a1,a2,z)
        a2primeind=a2primeIndex(temp); % this is the lower grid point index for a2prime in terms of (a1,a2,z)
        % combine a1prime, a2prime, and z
        aprimeind=Policy_a1primeind+N_a1*(a2primeind-1); %+N_a1*N_a2*repelem((0:1:N_z-1)',N_a1*N_a2,1);
        aprimeind=reshape(aprimeind,[N_a*N_z,1]);
        aprimeplus1ind=aprimeind+N_a1; % add one to a2prime index, which means adding N_a1*1
        aprimeProbs_Howards=reshape(a2primeProbs(temp),[N_a*N_z,1]); %  a2primeProbs is [N_d2,N_a2]
        
        for Howards_counter=1:vfoptions.howards
            % Note: Different from outside Howards, as optimal policy depends on z, and we also need to keep Vprime in terms of
            % zprime. So get Vlower and Vupper that depend on (a,z,zprime).

            % Take expectation over a2lower and a2upper
            Vlower=V(aprimeind(:),:); % EVlower in terms of policy (so size is a-by-z-by-zprime)
            Vupper=V(aprimeplus1ind(:),:); % EVupper in terms of policy (so size is a-by-z-by-zprime)
            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(Vlower==Vupper);
            aprimeProbs_Howards2=aprimeProbs_Howards.*ones(1,N_z);
            aprimeProbs_Howards2(skipinterp)=0; % effectively skips interpolation

            % Switch EV from being in terps of a2prime to being in terms of d2 and a2
            EV=aprimeProbs_Howards2.*Vlower+(1-aprimeProbs_Howards2).*Vupper; % (a1prime-by-a2prime,zprime)
            % Already applied the probabilities from interpolating onto grid

            % Calc the condl expectation term (except beta), which depends on z but not on control variables
            EV=aaa.*EV;
            EV(isnan(EV))=0; % multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=squeeze(sum(EV,2)); % sum over z', leaving a singular second dimension

            V=Ftemp+DiscountFactorParamsVec*EV;
            V=reshape(V,[N_a,N_z]);
        end
    end

    if vfoptions.verbose==1
        if rem(tempcounter,10)==0 % Every 10 iterations
            fprintf(distvstolstr, tempcounter,currdist) % use enough decimal points to be able to see countdown of currdist to 0
        end
    end

    tempcounter=tempcounter+1;    
end

%% Refine, so add d1 back into Policy
% Policy is currently
temppolicyindex=reshape(Policy,[1,N_a*N_z])+gpuArray(0:1:N_a*N_z-1)*N_a;
Policy=reshape(dstar(temppolicyindex),[N_a,N_z]) + N_d1*(Policy-1); % Add in d1, but keep in a fully vectorized form

%% For experience asset, just output Policy as is and then use Case2 to UnKron

end

