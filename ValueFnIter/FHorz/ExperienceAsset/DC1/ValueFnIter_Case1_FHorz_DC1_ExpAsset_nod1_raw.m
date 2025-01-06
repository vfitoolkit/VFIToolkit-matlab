function [V,Policy]=ValueFnIter_Case1_FHorz_DC1_ExpAsset_nod1_raw(n_d2,n_a1,n_a2,n_z,N_j, d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

%%
d2_grid=gpuArray(d2_grid);
d2_gridvals=CreateGridvals(n_d2,d2_grid,1);
a1_grid=gpuArray(a1_grid);
a2_grid=gpuArray(a2_grid);

if vfoptions.lowmemory>0
    special_n_z=ones(1,length(n_z));
else
    % precompute
    zind=shiftdim((0:1:N_z-1),-1); % already includes -1
end

% n-Monotonicity
% vfoptions.level1n=21;
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d2, n_z, d2_gridvals, a1_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        % First, we want a1prime conditional on (d,1,a)
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d2*N_a1,vfoptions.level1n*N_a2,N_z]),[],1);

        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V(curraindex,:,N_j)=shiftdim(Vtempii,1);
        Policy(curraindex,:,N_j)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_z
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d2, n_z, d2_gridvals, a1_grid(a1primeindexes), a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d2)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=dind+N_d2*a2ind+N_d2*N_a2*zind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_a2
                Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d2, n_z, d2_gridvals, a1_grid(loweredge), a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d2)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=dind+N_d2*a2ind+N_d2*N_a2*zind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_z
                Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
            end
        end
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            % n-Monotonicity
            ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d2, special_n_z, d2_gridvals, a1_grid, a1_grid(level1ii), a2_grid, z_val, ReturnFnParamsVec,1);

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(ReturnMatrix_ii_z,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_z,[N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
            Policy(curraindex,z_c,N_j)=shiftdim(maxindex2,1);

            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d2, special_n_z, d2_gridvals, a1_grid(a1primeindexes), a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                    Policy(curraindex,z_c,N_j)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d2, special_n_z, d2_gridvals, a1_grid(loweredge), a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                    Policy(curraindex,z_c,N_j)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                end
            end
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    Vnext=reshape(vfoptions.V_Jplus1,[N_a,N_z]);    % First, switch V_Jplus1 into Kron form

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_Case1(aprimeFn, n_d2, n_a2, d2_grid, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,N_z);  % [N_d2*N_a1,N_a2,N_z]

    Vlower=reshape(Vnext(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_z]);
    Vupper=reshape(Vnext(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terps of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,zprime)
    % Already applied the probabilities from interpolating onto grid

    if vfoptions.lowmemory==0
        EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-2);
        EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
        EV=squeeze(sum(EV,3));
        % EV is over (d2,a1prime,a2,z)

        DiscountedentireEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2,N_z]); % (d2,a1prime,1,a2,zprime)

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d2, n_z, d2_gridvals, a1_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        entireRHS_ii=ReturnMatrix_ii+DiscountedentireEV;

        % First, we want a1prime conditional on (d,1,a)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d2*N_a1,vfoptions.level1n*N_a2,N_z]),[],1);

        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V(curraindex,:,N_j)=shiftdim(Vtempii,1);
        Policy(curraindex,:,N_j)=shiftdim(maxindex2,1);

        DiscountedentireEV=repelem(DiscountedentireEV,1,1,N_a1,1,1); % (d,a1prime,a1,a2,z)

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_z
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d2, n_z, d2_gridvals, a1_grid(a1primeindexes), a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                daprime=(1:1:N_d2)'+N_d2*repelem(a1primeindexes-1,1,1,level1iidiff(ii),1,1)+N_d2*N_a1*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a1*N_a2*shiftdim((0:1:N_z-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii+DiscountedentireEV(reshape(daprime,[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_z]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d2)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=dind+N_d2*a2ind+N_d2*N_a2*zind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_a2
                Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d2, n_z, d2_gridvals, a1_grid(loweredge), a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                daprime=(1:1:N_d2)'+N_d2*repelem(loweredge-1,1,1,level1iidiff(ii),1,1)+N_d2*N_a1*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a1*N_a2*shiftdim((0:1:N_z-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii+DiscountedentireEV(reshape(daprime,[N_d2*1,level1iidiff(ii)*N_a2,N_z]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d2)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=dind+N_d2*a2ind+N_d2*N_a2*zind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_z
                Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
            end
        end

    elseif vfoptions.lowmemory==1

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            % Calc the condl expectation term (except beta), which depends on z but not on control variables
            EV_z=EV.*shiftdim(pi_z_J(z_c,:,N_j)',-2);
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,3);

            DiscountedentireEV_z=DiscountFactorParamsVec*reshape(EV_z,[N_d2,N_a1,1,N_a2]); % (d,a1prime,1,a2)

            % n-Monotonicity
            ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d2, special_n_z, d2_gridvals, a1_grid, a1_grid(level1ii), a2_grid, z_val, ReturnFnParamsVec,1);

            entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedentireEV_z;

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(entireRHS_ii_z,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_z,[N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
            Policy(curraindex,z_c,N_j)=shiftdim(maxindex2,1);

            DiscountedentireEV_z=repelem(DiscountedentireEV_z,1,1,N_a1,1); % (d,a1prime,a1,a2)
            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d2, special_n_z, d2_gridvals, a1_grid(a1primeindexes), a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, ReturnFnParamsVec,2);
                    daprime=(1:1:N_d2)'+N_d2*repelem(a1primeindexes-1,1,1,level1iidiff(ii),1)+N_d2*N_a1*N_a1*shiftdim((0:1:N_a2-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedentireEV_z(reshape(daprime,[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2]));
                    [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                    Policy(curraindex,z_c,N_j)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d2, special_n_z, d2_gridvals, a1_grid(loweredge), a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, ReturnFnParamsVec,2);
                    daprime=(1:1:N_d2)'+N_d2*repelem(a1primeindexes-1,1,1,level1iidiff(ii),1)+N_d2*N_a1*N_a1*shiftdim((0:1:N_a2-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedentireEV_z(reshape(daprime,[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2]));
                    [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                    Policy(curraindex,z_c,N_j)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                end
            end
        end
    end
end


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end
    
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_Case1(aprimeFn, n_d2, n_a2, d2_grid, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,N_z);  % [N_d2*N_a1,N_a2,N_z]

    Vlower=reshape(V(aprimeIndex(:),:,jj+1),[N_d2*N_a1,N_a2,N_z]);
    Vupper=reshape(V(aprimeplus1Index(:),:,jj+1),[N_d2*N_a1,N_a2,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terps of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,zprime)
    % Already applied the probabilities from interpolating onto grid

    if vfoptions.lowmemory==0
        % EV is over (d2,a1prime,a2,zprime)
        EV=EV.*shiftdim(pi_z_J(:,:,jj)',-2);
        EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
        EV=squeeze(sum(EV,3));
        % EV is over (d2,a1prime,a2,z)
        DiscountedentireEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2,N_z]); % (d2,a1prime,1,a2,zprime)

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d2, n_z, d2_gridvals, a1_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        entireRHS_ii=ReturnMatrix_ii+DiscountedentireEV;

        % First, we want a1prime conditional on (d,1,a)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d2*N_a1,vfoptions.level1n*N_a2,N_z]),[],1);

        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V(curraindex,:,jj)=shiftdim(Vtempii,1);
        Policy(curraindex,:,jj)=shiftdim(maxindex2,1);

        DiscountedentireEV=repelem(DiscountedentireEV,1,1,N_a1,1,1); % (d,a1prime,a1,a2,z)

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_z
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d2, n_z, d2_gridvals, a1_grid(a1primeindexes), a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                daprime=(1:1:N_d2)'+N_d2*repelem(a1primeindexes-1,1,1,level1iidiff(ii),1,1)+N_d2*N_a1*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a1*N_a2*shiftdim((0:1:N_z-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii+DiscountedentireEV(reshape(daprime,[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_z]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,jj)=shiftdim(Vtempii,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d2)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=dind+N_d2*a2ind+N_d2*N_a2*zind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_a2
                Policy(curraindex,:,jj)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d2, n_z, d2_gridvals, a1_grid(loweredge), a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                daprime=(1:1:N_d2)'+N_d2*repelem(loweredge-1,1,1,level1iidiff(ii),1,1)+N_d2*N_a1*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a1*N_a2*shiftdim((0:1:N_z-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii+DiscountedentireEV(reshape(daprime,[N_d2*1,level1iidiff(ii)*N_a2,N_z]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,jj)=shiftdim(Vtempii,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d2)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=dind+N_d2*a2ind+N_d2*N_a2*zind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_z
                Policy(curraindex,:,jj)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
            end
        end

    elseif vfoptions.lowmemory==1

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            % Calc the condl expectation term (except beta), which depends on z but not on control variables
            EV_z=EV.*shiftdim(pi_z_J(z_c,:,jj)',-2);
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,3);
            
            DiscountedentireEV_z=DiscountFactorParamsVec*reshape(EV_z,[N_d2,N_a1,1,N_a2]); % (d,a1prime,1,a2)

            % n-Monotonicity
            ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d2, special_n_z, d2_gridvals, a1_grid, a1_grid(level1ii), a2_grid, z_val, ReturnFnParamsVec,1);

            entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedentireEV_z;

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(entireRHS_ii_z,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_z,[N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V(curraindex,z_c,jj)=shiftdim(Vtempii,1);
            Policy(curraindex,z_c,jj)=shiftdim(maxindex2,1);

            DiscountedentireEV_z=repelem(DiscountedentireEV_z,1,1,N_a1,1); % (d,a1prime,a1,a2)
            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d2, special_n_z, d2_gridvals, a1_grid(a1primeindexes), a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, ReturnFnParamsVec,2);
                    daprime=(1:1:N_d2)'+N_d2*repelem(a1primeindexes-1,1,1,level1iidiff(ii),1)+N_d2*N_a1*N_a1*shiftdim((0:1:N_a2-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedentireEV_z(reshape(daprime,[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2]));
                    [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                    V(curraindex,z_c,jj)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                    Policy(curraindex,z_c,jj)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d2, special_n_z, d2_gridvals, a1_grid(loweredge), a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, ReturnFnParamsVec,2);
                    daprime=(1:1:N_d2)'+N_d2*repelem(a1primeindexes-1,1,1,level1iidiff(ii),1)+N_d2*N_a1*N_a1*shiftdim((0:1:N_a2-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedentireEV_z(reshape(daprime,[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2]));
                    [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                    V(curraindex,z_c,jj)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                    Policy(curraindex,z_c,jj)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                end
            end
        end
    end
end


%% For experience asset, just output Policy as is and then use Case2 to UnKron


end