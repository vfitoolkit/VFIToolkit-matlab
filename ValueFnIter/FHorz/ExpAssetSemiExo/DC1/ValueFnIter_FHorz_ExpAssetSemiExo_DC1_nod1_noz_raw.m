function [V,Policy3]=ValueFnIter_FHorz_ExpAssetSemiExo_DC1_nod1_noz_raw(n_d2,n_d3,n_a1,n_a2,n_semiz,N_j, d2_grid, d3_grid, a1_grid, a2_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d2 determines experience asset, d3 determines semi-exog state
% a is endogenous state, a2 is experience asset
% semiz is semi-exog state

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);

V=zeros(N_a,N_semiz,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d2,d3,a1prime seperately
Policy3=zeros(3,N_a,N_semiz,N_j,'gpuArray');

%%
d2_grid=gpuArray(d2_grid);
d3_grid=gpuArray(d3_grid);
a1_grid=gpuArray(a1_grid);
a2_grid=gpuArray(a2_grid);

% For the return function we just want (I'm just guessing that as I need them N_j times it will be fractionally faster to put them together now)
n_d=[n_d2,n_d3];
N_d=prod(n_d);
d_grid=[d2_grid;d3_grid];
d_gridvals=CreateGridvals(n_d,d_grid,1);

if vfoptions.lowmemory>0
    special_n_semiz=ones(1,length(n_semiz));
else
    % precompute
    semizind=shiftdim((0:1:N_semiz-1),-1); % already includes -1
end

% Preallocate
V_ford3_jj=zeros(N_a,N_semiz,N_d3,'gpuArray');
Policy_ford3_jj=zeros(N_a,N_semiz,N_d3,'gpuArray');


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
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d, n_semiz, d_gridvals, a1_grid, a1_grid(level1ii), a2_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        % First, we want a1prime conditional on (d,1,a)
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a1,vfoptions.level1n*N_a2,N_semiz]),[],1);

        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V(curraindex,:,N_j)=shiftdim(Vtempii,1);
        dind=rem(maxindex2-1,N_d)+1; % Do I need this shiftdim(), can probably delete all these
        Policy3(1,curraindex,:,N_j)=rem(dind-1,N_d2)+1;
        Policy3(2,curraindex,:,N_j)=ceil(dind/N_d2);
        Policy3(3,curraindex,:,N_j)=ceil(maxindex2/N_d);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_z
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d, n_semiz, d_gridvals, a1_grid(a1primeindexes), a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=dind+N_d*a2ind+N_d*N_a2*semizind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_a2
                % Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
                Policy3(1,curraindex,:,N_j)=rem(dind-1,N_d2)+1;
                Policy3(2,curraindex,:,N_j)=ceil(dind/N_d2);
                Policy3(3,curraindex,:,N_j)=ceil(maxindex/N_d+loweredge(allind)-1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d, n_semiz, d_gridvals, a1_grid(loweredge), a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=dind+N_d*a2ind+N_d*N_a2*semizind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_z
                % Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
                Policy3(1,curraindex,:,N_j)=rem(dind-1,N_d2)+1;
                Policy3(2,curraindex,:,N_j)=ceil(dind/N_d2);
                Policy3(3,curraindex,:,N_j)=ceil(maxindex/N_d+loweredge(allind)-1);
            end
        end
        ReturnMatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, n_d, n_a1,n_a2, n_semiz, d_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec); % [N_d*N_a1,N_a1*N_a2,N_z]
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        d_ind=rem(maxindex-1,N_d)+1; % Do I need this shiftdim(), can probably delete all these
        Policy3(1,:,:,N_j)=rem(d_ind-1,N_d2)+1;
        Policy3(2,:,:,N_j)=ceil(d_ind/N_d2);
        Policy3(3,:,:,N_j)=ceil(maxindex/N_d);

    elseif vfoptions.lowmemory==1

        for z_c=1:N_semiz
            z_val=semiz_gridvals_J(z_c,:,N_j);

            % n-Monotonicity
            ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d, special_n_semiz, d_gridvals, a1_grid, a1_grid(level1ii), a2_grid, z_val, ReturnFnParamsVec,1);

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(ReturnMatrix_ii_z,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_z,[N_d*N_a1,vfoptions.level1n*N_a2]),[],1);

            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
            dind=rem(maxindex2-1,N_d)+1; % Do I need this shiftdim(), can probably delete all these
            Policy3(1,curraindex,z_c,N_j)=rem(dind-1,N_d2)+1;
            Policy3(2,curraindex,z_c,N_j)=ceil(dind/N_d2);
            Policy3(3,curraindex,z_c,N_j)=ceil(maxindex2/N_d);

            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d, special_n_semiz, d_gridvals, a1_grid(a1primeindexes), a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_z,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_a2
                    % Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
                    Policy3(1,curraindex,z_c,N_j)=rem(dind-1,N_d2)+1;
                    Policy3(2,curraindex,z_c,N_j)=ceil(dind/N_d2);
                    Policy3(3,curraindex,z_c,N_j)=ceil(maxindex/N_d+loweredge(allind)-1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d, special_n_semiz, d_gridvals, a1_grid(loweredge), a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_z,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                    Policy3(1,curraindex,z_c,N_j)=rem(dind-1,N_d2)+1;
                    Policy3(2,curraindex,z_c,N_j)=ceil(dind/N_d2);
                    Policy3(3,curraindex,z_c,N_j)=ceil(maxindex/N_d+loweredge(allind)-1);
                end
            end
        end
    end
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_Case1(aprimeFn, n_d2, n_a2, d2_grid, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2]
    if vfoptions.lowmemory>0 || vfoptions.paroverz==0
        aprimeProbs=repmat(a2primeProbs,N_a1,1); % [N_d2*N_a1,N_a2]
    else % lowmemory=0 and paroverz=1
        aprimeProbs=repmat(a2primeProbs,N_a1,1,N_semiz);  % [N_d2*N_a1,N_a2,N_semiz]
    end

    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            % d3_val=d3_grid(d3_c);
            d23_gridvals=[d2_grid,d3_grid(d3_c)*ones(n_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=V_Jplus1.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_semiz]); % (d2,a1prime,a2,z), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_semiz]); % (d2,a1prime,a2,z), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            entireEV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % entireEV is (d2,a1prime, a2,z)

            DiscountedentireEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_semiz]); % (d2,a1prime,1,a2,zprime)

            % n-Monotonicity
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, [n_d2,1], n_semiz, d23_gridvals, a1_grid, a1_grid(level1ii), a2_grid, pi_semiz_d3, ReturnFnParamsVec,1);

            entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedentireEV;

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1,vfoptions.level1n*N_a2,N_semiz]),[],1);

            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V_ford3_jj(curraindex,:,d3_c)=shiftdim(Vtempii,1);
            Policy_ford3_jj(curraindex,:,d3_c)=shiftdim(maxindex2,1);

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
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, [n_d2,1], n_semiz, d23_gridvals, a1_grid(a1primeindexes), a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, pi_semiz_d3, ReturnFnParamsVec,2);
                    daprime=(1:1:N_d2)'+N_d2*repelem(a1primeindexes-1,1,1,level1iidiff(ii),1,1)+N_d2*N_a1*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a1*N_a2*shiftdim((0:1:N_semiz-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii_d3+DiscountedentireEV(reshape(daprime,[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_semiz]));
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_jj(curraindex,:,d3_c)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d2*a2ind+N_d2*N_a2*semizind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_a2
                    Policy_ford3_jj(curraindex,:,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, [n_d2,1], n_semiz, d23_gridvals, a1_grid(loweredge), a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, pi_semiz_d3, ReturnFnParamsVec,2);
                    daprime=(1:1:N_d2)'+N_d2*repelem(loweredge-1,1,1,level1iidiff(ii),1,1)+N_d2*N_a1*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a1*N_a2*shiftdim((0:1:N_semiz-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii_d3+DiscountedentireEV(reshape(daprime,[N_d2*1,level1iidiff(ii)*N_a2,N_semiz]));
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_jj(curraindex,:,d3_c)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d2*a2ind+N_d2*N_a2*semizind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_z
                    Policy_ford3_jj(curraindex,:,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                end
            end
        end
        
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            % d3_val=d3_grid(d3_c);
            d23_gridvals=[d2_grid,d3_grid(d3_c)*ones(n_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);

            for z_c=1:N_bothz
                z_val=semiz_gridvals_J(z_c,:,N_j);

                %Calc the condl expectation term (except beta), which depends on z but not on control variables
                EV_z=V_Jplus1.*shiftdim(pi_semiz_d3(z_c,:)',1);
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);

                % Switch EV_z from being in terms of aprime to being in terms of d and a
                EV1=reshape(EV_z(aprimeIndex),[N_d2*N_a1,N_a2]); % (d2,a1prime,a2), the lower aprime
                EV2=reshape(EV_z(aprimeplus1Index),[N_d2*N_a1,N_a2]); % (d2,a1prime,a2), the upper aprime

                % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
                skipinterp=(EV1==EV2);
                aprimeProbs(skipinterp)=0; % effectively skips interpolation

                % Apply the aprimeProbs
                entireEV_z=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
                % entireEV_z is (d,a1prime, a2)

                DiscountedentireEV_z=DiscountFactorParamsVec*reshape(entireEV_z,[N_d2,N_a1,1,N_a2]); % (d,a1prime,1,a2)

                % n-Monotonicity
                ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, [n_d2,1], special_n_semiz, d23_gridvals, a1_grid, a1_grid(level1ii), a2_grid, z_val, ReturnFnParamsVec,1);

                entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedentireEV_z;

                % First, we want a1prime conditional on (d,1,a)
                [~,maxindex1]=max(entireRHS_ii_z,[],2);

                % Now, get and store the full (d,aprime)
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_z,[N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

                % Store
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V_ford3_jj(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_jj(curraindex,z_c,d3_c)=shiftdim(maxindex2,1);

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
                        ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, [n_d2,1], special_n_semiz, d23_gridvals, a1_grid(a1primeindexes), a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, ReturnFnParamsVec,2);
                        daprime=(1:1:N_d2)'+N_d2*repelem(a1primeindexes-1,1,1,level1iidiff(ii),1)+N_d2*N_a1*N_a1*shiftdim((0:1:N_a2-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedentireEV_z(reshape(daprime,[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2]));
                        [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                        V_ford3_jj(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                        % maxindex does not need reworking, as with expasset there is no a2prime
                        %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                        dind=(rem(maxindex-1,N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                        allind=dind+N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                        Policy_ford3_jj(curraindex,z_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, [n_d2,1], special_n_semiz, d23_gridvals, a1_grid(loweredge), a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, ReturnFnParamsVec,2);
                        daprime=(1:1:N_d2)'+N_d2*repelem(a1primeindexes-1,1,1,level1iidiff(ii),1)+N_d2*N_a1*N_a1*shiftdim((0:1:N_a2-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedentireEV_z(reshape(daprime,[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2]));
                        [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                        V_ford3_jj(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                        % maxindex does not need reworking, as with expasset there is no a2prime
                        %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                        dind=(rem(maxindex-1,N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                        allind=dind+N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                        Policy_ford3_jj(curraindex,z_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                    end
                end
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_jj,[],3); % max over d2
    V(:,:,N_j)=V_jj;
    Policy3(2,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d2a1prime_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
    Policy3(1,:,:,N_j)=rem(d2a1prime_ind-1,N_d2)+1; % d2
    Policy3(3,:,:,N_j)=ceil(d2a1prime_ind/N_d2); % a1prime
    
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
    % Note: aprimeIndex is [N_d2*N_a2,1], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2]
    if vfoptions.lowmemory>0 || vfoptions.paroverz==0
        aprimeProbs=repmat(a2primeProbs,N_a1,1); % [N_d2*N_a1,N_a2]
    else % lowmemory=0 and paroverz=1
        aprimeProbs=repmat(a2primeProbs,N_a1,1,N_semiz);  % [N_d2*N_a1,N_a2,N_semiz]
    end

    VKronNext_j=V(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            % d3_val=d3_grid(d3_c);
            d23_gridvals=[d2_grid,d3_grid(d3_c)*ones(n_d2,1)];

            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=VKronNext_j.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_semiz]); % (d2,a1prime,a2,z), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_semiz]); % (d2,a1prime,a2,z), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            entireEV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % entireEV is (d,a1prime, a2,z)

            DiscountedentireEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]); % (d2,a1prime,1,a2,zprime)

            % n-Monotonicity
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, [n_d2,1], n_semiz, d23_gridvals, a1_grid, a1_grid(level1ii), a2_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

            entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedentireEV;

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1,vfoptions.level1n*N_a2,N_semiz]),[],1);

            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V_ford3_jj(curraindex,:,d3_c)=shiftdim(Vtempii,1);
            Policy_ford3_jj(curraindex,:,d3_c)=shiftdim(maxindex2,1);

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
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, [n_d2,1], n_semiz, d23_gridvals, a1_grid(a1primeindexes), a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                    daprime=(1:1:N_d2)'+N_d2*repelem(a1primeindexes-1,1,1,level1iidiff(ii),1,1)+N_d2*N_a1*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a1*N_a2*shiftdim((0:1:N_semiz-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii_d3+DiscountedentireEV(reshape(daprime,[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_semiz]));
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_jj(curraindex,:,d3_c)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d2*a2ind+N_d2*N_a2*semizind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_a2
                    Policy_ford3_jj(curraindex,:,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, [n_d2,1], n_semiz, d23_gridvals, a1_grid(loweredge), a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                    daprime=(1:1:N_d2)'+N_d2*repelem(loweredge-1,1,1,level1iidiff(ii),1,1)+N_d2*N_a1*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a1*N_a2*shiftdim((0:1:N_semiz-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii_d3+DiscountedentireEV(reshape(daprime,[N_d2*1,level1iidiff(ii)*N_a2,N_semiz]));
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_jj(curraindex,:,d3_c)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d2*a2ind+N_d2*N_a2*semizind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_z
                    Policy_ford3_jj(curraindex,:,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                end
            end

        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            % d3_val=d3_grid(d3_c);
            d23_gridvals=[d2_grid,d3_grid(d3_c)*ones(n_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);

            for z_c=1:N_bothz
                z_val=semiz_gridvals_J(z_c,:,jj);

                %Calc the condl expectation term (except beta), which depends on z but not on control variables
                EV_z=VKronNext_j.*shiftdim(pi_semiz_d3(z_c,:)',1);
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);

                % Switch EV_z from being in terms of aprime to being in terms of d and a
                EV1=reshape(EV_z(aprimeIndex),[N_d2*N_a1,N_a2]); % (d2,a1prime,a2), the lower aprime
                EV2=reshape(EV_z(aprimeplus1Index),[N_d2*N_a1,N_a2]); % (d2,a1prime,a2), the upper aprime

                % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
                skipinterp=(EV1==EV2);
                aprimeProbs(skipinterp)=0; % effectively skips interpolation

                % Apply the aprimeProbs
                entireEV_z=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
                % entireEV_z is (d,a1prime, a2)

                DiscountedentireEV_z=DiscountFactorParamsVec*reshape(entireEV_z,[N_d2,N_a1,1,N_a2]); % (d,a1prime,1,a2)

                % n-Monotonicity
                ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, [n_d2,1], special_n_semiz, d23_gridvals, a1_grid, a1_grid(level1ii), a2_grid, z_val, ReturnFnParamsVec,1);

                entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedentireEV_z;

                % First, we want a1prime conditional on (d,1,a)
                [~,maxindex1]=max(entireRHS_ii_z,[],2);

                % Now, get and store the full (d,aprime)
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_z,[N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

                % Store
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V_ford3_jj(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_jj(curraindex,z_c,d3_c)=shiftdim(maxindex2,1);

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
                        ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, [n_d2,1], special_n_semiz, d23_gridvals, a1_grid(a1primeindexes), a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, ReturnFnParamsVec,2);
                        daprime=(1:1:N_d2)'+N_d2*repelem(a1primeindexes-1,1,1,level1iidiff(ii),1)+N_d2*N_a1*N_a1*shiftdim((0:1:N_a2-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedentireEV_z(reshape(daprime,[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2]));
                        [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                        V_ford3_jj(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                        % maxindex does not need reworking, as with expasset there is no a2prime
                        %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                        dind=(rem(maxindex-1,N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                        allind=dind+N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                        Policy_ford3_jj(curraindex,z_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, [n_d2,1], special_n_semiz, d23_gridvals, a1_grid(loweredge), a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, ReturnFnParamsVec,2);
                        daprime=(1:1:N_d2)'+N_d2*repelem(loweredge-1,1,1,level1iidiff(ii),1)+N_d2*N_a1*N_a1*shiftdim((0:1:N_a2-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedentireEV_z(reshape(daprime,[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2]));
                        [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                        V_ford3_jj(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                        % maxindex does not need reworking, as with expasset there is no a2prime
                        %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                        dind=(rem(maxindex-1,N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                        allind=dind+N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                        Policy_ford3_jj(curraindex,z_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                    end
                end
            end
        end
    end
    
    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_jj,[],3); % max over d3
    V(:,:,jj)=V_jj;
    Policy3(2,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d2a1prime_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
    Policy3(1,:,:,jj)=rem(d2a1prime_ind-1,N_d2)+1; % d2
    Policy3(3,:,:,jj)=ceil(d2a1prime_ind/N_d2); % a1prime

end


%% For experience asset, just output Policy as is and then use Case2 to UnKron

end
