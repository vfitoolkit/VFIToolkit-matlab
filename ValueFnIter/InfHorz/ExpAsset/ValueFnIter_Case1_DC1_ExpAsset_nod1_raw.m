function [V, Policy]=ValueFnIter_Case1_DC1_ExpAsset_nod1_raw(V0,n_d2,n_a1,n_a2,n_z , d2_grid, a1_grid, a2_grid, z_gridvals, pi_z, ReturnFn, aprimeFn, Parameters, DiscountFactorParamsVec, ReturnFnParamsVec, aprimeFnParamNames, vfoptions)

N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);

%%
d2_grid=gpuArray(d2_grid);
d2_gridvals=CreateGridvals(n_d2,d2_grid,1);
a1_grid=gpuArray(a1_grid);
a2_grid=gpuArray(a2_grid);

% precompute
zind=shiftdim((0:1:N_z-1),-1); % already includes -1

% n-Monotonicity
% vfoptions.level1n=21;
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% Start by setting up ReturnFn for the first-level (as we can reuse this every iteration)
ReturnMatrixLvl1=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d2, n_z, d2_gridvals, a1_grid, a1_grid(level1ii), a2_grid, z_gridvals, ReturnFnParamsVec, 1);

V=reshape(V0,[N_a,N_z]);
Policy=zeros(N_a,N_z,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

% for Howards, preallocate
Ftemp=zeros(N_a,N_z,'gpuArray');
% and we need 
aaa=ones(N_a,1,1,'gpuArray').*shiftdim(pi_z',-1); % (a,z',z)

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
    % EV is over (d2,a1prime,a2,z)
    DiscountedentireEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2,N_z]); % (d2,a1prime,1,a2,zprime)

    %% Level 1
    % We can just reuse ReturnMatrixLvl1
    entireRHS_ii=ReturnMatrixLvl1+DiscountedentireEV;

    % First, we want a1prime conditional on (d,1,a)
    [~,maxindex1]=max(entireRHS_ii,[],2);

    % Now, get and store the full (d,aprime)
    [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d2*N_a1,vfoptions.level1n*N_a2,N_z]),[],1);

    % Store
    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
    V(curraindex,:)=shiftdim(Vtempii,1);
    Policy(curraindex,:)=shiftdim(maxindex2,1);

    % Need to keep Ftemp for Howards policy iteration improvement
    Ftemp(curraindex,:)=ReturnMatrixLvl1(shiftdim(maxindex2,1)+N_d2*N_a1*(0:1:vfoptions.level1n*N_a2-1)'+N_d2*N_a1*vfoptions.level1n*N_a2*(0:1:N_z-1));
    
    % Attempt for improved version
    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d2, n_z, d2_gridvals, a1_grid(a1primeindexes), a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals, ReturnFnParamsVec,2);
            daprime=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_z-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedentireEV(reshape(daprime,[N_d2*(maxgap(ii)+1),N_a2,N_z])),1,level1iidiff(ii),1);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,:)=shiftdim(Vtempii,1);
            % maxindex does not need reworking, as with expasset there is no a2prime
            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            dind=(rem(maxindex-1,N_d2)+1);
            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
            allind=dind+N_d2*a2ind+N_d2*N_a2*zind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_a2
            Policy(curraindex,:)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);

            % Need to keep Ftemp for Howards policy iteration improvement
            Ftemp(curraindex,:)=ReturnMatrix_ii(shiftdim(maxindex,1)+N_d2*(maxgap(ii)+1)*(0:1:level1iidiff(ii)*N_a2-1)'+N_d2*(maxgap(ii)+1)*level1iidiff(ii)*N_a2*(0:1:N_z-1));
        else
            loweredge=maxindex1(:,1,ii,:,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d2, n_z, d2_gridvals, a1_grid(loweredge), a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals, ReturnFnParamsVec,2);
            daprime=(1:1:N_d2)'+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_z-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedentireEV(reshape(daprime,[N_d2*1,N_a2,N_z])),1,level1iidiff(ii),1);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,:)=shiftdim(Vtempii,1);
            % maxindex does not need reworking, as with expasset there is no a2prime
            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            dind=(rem(maxindex-1,N_d2)+1);
            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
            allind=dind+N_d2*a2ind+N_d2*N_a2*zind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_z
            Policy(curraindex,:)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);

            % Need to keep Ftemp for Howards policy iteration improvement
            Ftemp(curraindex,:)=ReturnMatrix_ii(shiftdim(maxindex,1)+N_d2*1*(0:1:level1iidiff(ii)*N_a2-1)'+N_d2*1*level1iidiff(ii)*N_a2*(0:1:N_z-1));
        end
    end
    
    %% Finish up
    % Update currdist
    Vdist=V(:)-Vold(:);
    Vdist(isnan(Vdist))=0;
    currdist=max(abs(Vdist));
    
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards % Use Howards Policy Fn Iteration Improvement
        Policy_d2ind=rem(Policy(:)-1,N_d2)+1;
        Policy_a1primeind=ceil(Policy(:)/N_d2);

        temp=Policy_d2ind+N_d2*repmat(repelem((0:1:N_a2-1)',N_a1,1),N_z,1); % (d2,a2) indexes in terms of (a1,a2,z)

        a2primeind=a2primeIndex(temp); % a2primeIndex is [N_d2,N_a2]
        azprimeind=Policy_a1primeind+N_a1*(a2primeind-1)+N_a1*N_a2*repelem((0:1:N_z-1)',N_a1*N_a2,1);
        azprimeind=reshape(azprimeind,[N_a,N_z]);
        azprimeplus1ind=azprimeind+N_a1; % add one to a2prime index, which means adding N_a1*1
        aprimeProbs_Howards=reshape(a2primeProbs(temp),[N_a,N_z]); %  a2primeProbs is [N_d2,N_a2]
        
        for Howards_counter=1:vfoptions.howards
            Vlower=V(azprimeind); % Vlower in terms of policy (so size is a-by-z)
            Vupper=V(azprimeplus1ind); % Vupper in terms of policy (so size is a-by-z)
            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(Vlower==Vupper);
            aprimeProbs_Howards2=aprimeProbs_Howards;
            aprimeProbs_Howards2(skipinterp)=0; % effectively skips interpolation

            % Switch EV from being in terps of a2prime to being in terms of d2 and a2
            EV=aprimeProbs_Howards2.*Vlower+(1-aprimeProbs_Howards2).*Vupper; % (a1prime-by-a2prime,zprime)
            % Already applied the probabilities from interpolating onto grid
            
            % Calc the condl expectation term (except beta), which depends on z but not on control variables
            EV=EV.*aaa;
            EV(isnan(EV))=0; % multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=squeeze(sum(EV,2)); % sum over z', leaving a singular second dimension
            
            V=Ftemp+DiscountFactorParamsVec*EV;
        end
    end

    if vfoptions.verbose==1
        if rem(tempcounter,10)==0 % Every 10 iterations
            fprintf(distvstolstr, tempcounter,currdist) % use enough decimal points to be able to see countdown of currdist to 0
        end
    end

    tempcounter=tempcounter+1;    
end




end