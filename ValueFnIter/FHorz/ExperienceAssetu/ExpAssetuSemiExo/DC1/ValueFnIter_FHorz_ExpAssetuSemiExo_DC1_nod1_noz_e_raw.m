function [V,Policy3]=ValueFnIter_FHorz_ExpAssetuSemiExo_DC1_nod1_noz_e_raw(n_d2,n_d3,n_a1,n_a2,n_semiz,n_e,n_u,N_j, d2_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, e_gridvals_J, u_grid, pi_semiz_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d2 determines experience asset, d3 determines semi-exog state
% a is endogenous state, a2 is experience asset
% z is exogenous state, semiz is semi-exog state

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_e=prod(n_e);
N_u=prod(n_u);

V=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d1,d2,d3,a1prime seperately
Policy3=zeros(4,N_a,N_semiz,N_e,N_j,'gpuArray');

pi_u=shiftdim(pi_u,-2); % put it into third dimension

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_gridvals=CreateGridvals(n_d23,[d2_grid;d3_grid],1);

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
else
    % precompute
    eind=shiftdim((0:1:N_e-1),-2); % already includes -1
end

if vfoptions.lowmemory>1
    special_n_semiz=ones(1,length(n_semiz));
else
    % precompute
    semizind=shiftdim((0:1:N_semiz-1),-1); % already includes -1
end

% Preallocate
V_ford3_jj=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');
Policy_ford3_jj=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');

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
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,n_d23,n_a1,vfoptions.level1n,n_a2,n_semiz,n_e, d23_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        % First, we want a1prime conditional on (d,1,a)
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d23*N_a1,vfoptions.level1n*N_a2,N_semiz,N_e]),[],1);

        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
        dind=rem(maxindex2-1,N_d23)+1; % Do I need this shiftdim(), can probably delete all these
        Policy3(1,curraindex,:,:,N_j)=rem(dind-1,N_d2)+1; % d2
        Policy3(2,curraindex,:,:,N_j)=ceil(dind/N_d2); % d3
        Policy3(3,curraindex,:,:,N_j)=ceil(maxindex2/N_d23); % aprime

        % Attempt for improved version
        maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z-by-n_e
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_z-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,n_d23,maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,n_e, d23_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d23)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=dind+N_d23*a2ind+N_d23*N_a2*semizind+N_d23*N_a2*N_semiz*eind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_a2-by-n_z-by-n_e
                Policy3(1,curraindex,:,:,N_j)=rem(dind-1,N_d2)+1; % d2
                Policy3(2,curraindex,:,:,N_j)=ceil(dind/N_d2); % d3
                Policy3(3,curraindex,:,:,N_j)=ceil(maxindex/N_d23+loweredge(allind)-1); % aprime
            else
                loweredge=maxindex1(:,1,ii,:,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,n_d23,1,level1iidiff(ii),n_a2,n_semiz,n_e, d23_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d23)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=dind+N_d23*a2ind+N_d23*N_a2*semizind+N_d23*N_a2*N_semiz*eind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_z-by-n_e
                Policy3(1,curraindex,:,:,N_j)=rem(dind-1,N_d2)+1; % d2
                Policy3(2,curraindex,:,:,N_j)=ceil(dind/N_d2); % d3
                Policy3(3,curraindex,:,:,N_j)=ceil(maxindex/N_d23+loweredge(allind)-1); % aprime
            end
        end

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);

            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,n_d23,n_a1,vfoptions.level1n,n_a2,n_semiz,special_n_e, d23_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d23*N_a1,vfoptions.level1n*N_a2,N_semiz]),[],1);

            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
            dind=rem(maxindex2-1,N_d23)+1; % Do I need this shiftdim(), can probably delete all these
            Policy3(1,curraindex,:,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
            Policy3(2,curraindex,:,e_c,N_j)=ceil(dind/N_d2); % d3
            Policy3(3,curraindex,:,e_c,N_j)=ceil(maxindex2/N_d23); % a1prime

            % Attempt for improved version
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_z
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,n_d23,maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,special_n_e, d23_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d23)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d23*a2ind+N_d23*N_a2*semizind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_a2-by-n_z
                    Policy3(1,curraindex,:,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
                    Policy3(2,curraindex,:,e_c,N_j)=ceil(dind/N_d2); % d3
                    Policy3(3,curraindex,:,e_c,N_j)=ceil(maxindex/N_d23+loweredge(allind)-1); % a1prime
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,n_d23,1,level1iidiff(ii),n_a2,n_semiz,special_n_e, d23_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d23)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d23*a2ind+N_d23*N_a2*semizind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_z
                    Policy3(1,curraindex,:,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
                    Policy3(2,curraindex,:,e_c,N_j)=ceil(dind/N_d2); % d3
                    Policy3(3,curraindex,:,e_c,N_j)=ceil(maxindex/N_d23+loweredge(allind)-1); % a1prime
                end
            end
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_semiz
            z_val=semiz_gridvals_J(z_c,:,N_j);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                % n-Monotonicity
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,n_d23,n_a1,vfoptions.level1n,n_a2,special_n_semiz,special_n_e, d23_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1);

                % First, we want a1prime conditional on (d,1,a)
                [~,maxindex1]=max(ReturnMatrix_ii,[],2);

                % Now, get and store the full (d,aprime)
                [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d23*N_a1,vfoptions.level1n*N_a2]),[],1);

                % Store
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                dind=rem(maxindex2-1,N_d23)+1; % Do I need this shiftdim(), can probably delete all these
                Policy3(1,curraindex,z_c,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
                Policy3(2,curraindex,z_c,e_c,N_j)=ceil(dind/N_d2); % d3
                Policy3(3,curraindex,z_c,e_c,N_j)=ceil(maxindex2/N_d23); % a1prime

                % Attempt for improved version
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2
                        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,n_d23,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_semiz,special_n_e, d23_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                        V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        % maxindex does not need reworking, as with expasset there is no a2prime
                        %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                        dind=(rem(maxindex-1,N_d23)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                        allind=dind+N_d23*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_a2
                        Policy3(1,curraindex,z_c,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
                        Policy3(2,curraindex,z_c,e_c,N_j)=ceil(dind/N_d2); % d3
                        Policy3(3,curraindex,z_c,e_c,N_j)=ceil(maxindex/N_d23+loweredge(allind)-1); % a1prime
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,n_d23,1,level1iidiff(ii),n_a2,special_n_semiz,special_n_e, d23_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                        V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        % maxindex does not need reworking, as with expasset there is no a2prime
                        %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                        dind=(rem(maxindex-1,N_d23)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                        allind=dind+N_d23*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                        Policy3(1,curraindex,z_c,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
                        Policy3(2,curraindex,z_c,e_c,N_j)=ceil(dind/N_d2); % d3
                        Policy3(3,curraindex,z_c,e_c,N_j)=ceil(maxindex/N_d23+loweredge(allind)-1); % d4
                    end
                end
            end
        end
    end
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetuFnMatrix_Case1(aprimeFn, n_d2, n_a2, n_u, d2_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_u], whereas aprimeProbs is [N_d2,N_a2,N_u]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1,N_semiz);  % [N_d2*N_a1,N_a2,N_u,N_semiz]
    
    EVpre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]).*shiftdim(pi_e_J(:,N_j),-2),3);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            % d3_val=d3_grid(d3_c);
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_bothz_d3',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,z), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,z), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % Already applied the probabilities from interpolating onto grid
            EV=squeeze(sum((EV.*pi_u),3)); % (d2,a1prime,a2,semiz)

            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2,N_semiz]); % (d2,a1prime,1,a2,zprime)

            % n-Monotonicity
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_semiz,n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

            entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV;

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1,vfoptions.level1n*N_a2,N_semiz,N_e]),[],1);

            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V_ford3_jj(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
            Policy_ford3_jj(curraindex,:,:,d3_c)=shiftdim(maxindex2,1);

            % Attempt for improved version
            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z-by-n_e
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_z-by-n_e
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                    daprime=(1:1:N_d2)'+N_d2*repelem(a1primeindexes-1,1,1,level1iidiff(ii),1,1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_semiz-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii_d3+DiscountedEV(reshape(daprime,[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_semiz,N_e]));
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_jj(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d2*a2ind+N_d2*N_a2*semizind+N_d2*N_a2*N_semiz*eind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_a2-by-n_z-by-n_e
                    Policy_ford3_jj(curraindex,:,:,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,[n_d2,1],1,level1iidiff(ii),n_a2,n_semiz,n_e, d23_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                    daprime=(1:1:N_d2)'+N_d2*repelem(loweredge-1,1,1,level1iidiff(ii),1,1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_semiz-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii_d3+DiscountedEV(reshape(daprime,[N_d2*1,level1iidiff(ii)*N_a2,N_semiz,N_e]));
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_jj(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d2*a2ind+N_d2*N_a2*semizind+N_d2*N_a2*N_semiz*eind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_z-by-n_e
                    Policy_ford3_jj(curraindex,:,:,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                end
            end
        end
        

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            % d3_val=d3_grid(d3_c);
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_bothz_d3',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,z), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,z), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % Already applied the probabilities from interpolating onto grid
            EV=squeeze(sum((EV.*pi_u),3)); % (d2,a1prime,a2,semiz)

            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2,N_semiz]); % (d2,a1prime,1,a2,zprime)

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                % n-Monotonicity
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_semiz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);

                entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV;

                % First, we want a1prime conditional on (d,1,a)
                [~,maxindex1]=max(entireRHS_ii_d3,[],2);

                % Now, get and store the full (d,aprime)
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1,vfoptions.level1n*N_a2,N_semiz]),[],1);

                % Store
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(maxindex2,1);

                % Attempt for improved version
                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_z
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                        daprime=(1:1:N_d2)'+N_d2*repelem(a1primeindexes-1,1,1,level1iidiff(ii),1,1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_semiz-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_ii_d3+DiscountedEV(reshape(daprime,[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_semiz]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        % maxindex does not need reworking, as with expasset there is no a2prime
                        %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                        dind=(rem(maxindex-1,N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                        allind=dind+N_d2*a2ind+N_d2*N_a2*semizind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_a2-by-n_z
                        Policy_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,[n_d2,1],1,level1iidiff(ii),n_a2,n_semiz,special_n_e, d23_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                        daprime=(1:1:N_d2)'+N_d2*repelem(loweredge-1,1,1,level1iidiff(ii),1,1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_semiz-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_ii_d3+DiscountedEV(reshape(daprime,[N_d2*1,level1iidiff(ii)*N_a2,N_semiz]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        % maxindex does not need reworking, as with expasset there is no a2prime
                        %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                        dind=(rem(maxindex-1,N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                        allind=dind+N_d2*a2ind+N_d2*N_a2*semizind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_z
                        Policy_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                    end
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_bothz_d3',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,z), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,z), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % Already applied the probabilities from interpolating onto grid
            EV=squeeze(sum((EV.*pi_u),3)); % (d2,a1prime,a2,semiz)

            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2,N_semiz]); % (d2,a1prime,1,a2,zprime)

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    % n-Monotonicity
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_semiz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1);

                    entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_z;

                    % First, we want a1prime conditional on (d,1,a)
                    [~,maxindex1]=max(entireRHS_ii_d3,[],2);

                    % Now, get and store the full (d,aprime)
                    [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

                    % Store
                    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                    V_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    Policy_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex2,1);

                    % Attempt for improved version
                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                            % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2
                            ReturnMatrix_ii_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_semiz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,2);
                            daprime=(1:1:N_d2)'+N_d2*repelem(a1primeindexes-1,1,1,level1iidiff(ii),1,1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                            entireRHS_ii=ReturnMatrix_ii_d3+DiscountedEV_z(reshape(daprime,[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2]));
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            % maxindex does not need reworking, as with expasset there is no a2prime
                            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                            dind=(rem(maxindex-1,N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                            allind=dind+N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_a2
                            Policy_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                        else
                            loweredge=maxindex1(:,1,ii,:,:);
                            % Just use aprime(ii) for everything
                            ReturnMatrix_ii_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,[n_d2,1],1,level1iidiff(ii),n_a2,special_n_semiz,special_n_e, d23_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,2);
                            daprime=(1:1:N_d2)'+N_d2*repelem(loweredge-1,1,1,level1iidiff(ii),1,1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                            entireRHS_ii=ReturnMatrix_ii_d3+DiscountedEV_z(reshape(daprime,[N_d2*1,level1iidiff(ii)*N_a2]));
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            % maxindex does not need reworking, as with expasset there is no a2prime
                            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                            dind=(rem(maxindex-1,N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                            allind=dind+N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                            Policy_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                        end
                    end
                end
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_jj,[],4); % max over d2
    V(:,:,:,N_j)=V_jj;
    Policy3(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d2a1prime_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    Policy3(1,:,:,:,N_j)=rem(d2a1prime_ind-1,N_d2)+1; % d2
    Policy3(3,:,:,:,N_j)=ceil(d2a1prime_ind/N_d2); % a1prime
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
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetuFnMatrix_Case1(aprimeFn, n_d2, n_a2, n_u, d2_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_u], whereas aprimeProbs is [N_d2,N_a2,N_u]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1,N_semiz);  % [N_d2*N_a1,N_a2,N_u,N_semiz]
    
    EVpre=sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj),-2),3);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            % d3_val=d3_grid(d3_c);
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_bothz_d3',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,z), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,z), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % Already applied the probabilities from interpolating onto grid
            EV=squeeze(sum((EV.*pi_u),3)); % (d2,a1prime,a2,semiz)

            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2,N_semiz]); % (d2,a1prime,1,a2,zprime)

            % n-Monotonicity
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_semiz,n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

            entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV;

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1,vfoptions.level1n*N_a2,N_semiz,N_e]),[],1);

            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V_ford3_jj(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
            Policy_ford3_jj(curraindex,:,:,d3_c)=shiftdim(maxindex2,1);

            % Attempt for improved version
            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z-by-n_e
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_z-by-n_e
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                    daprime=(1:1:N_d2)'+N_d2*repelem(a1primeindexes-1,1,1,level1iidiff(ii),1,1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_semiz-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii_d3+DiscountedEV(reshape(daprime,[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_semiz,N_e]));
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_jj(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d2*a2ind+N_d2*N_a2*semizind+N_d2*N_a2*N_semiz*eind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_a2-by-n_z-by-n_e
                    Policy_ford3_jj(curraindex,:,:,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,[n_d2,1],1,level1iidiff(ii),n_a2,n_semiz,n_e, d23_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                    daprime=(1:1:N_d2)'+N_d2*repelem(loweredge-1,1,1,level1iidiff(ii),1,1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_semiz-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii_d3+DiscountedEV(reshape(daprime,[N_d2*1,level1iidiff(ii)*N_a2,N_semiz,N_e]));
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_jj(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d2*a2ind+N_d2*N_a2*semizind+N_d2*N_a2*N_semiz*eind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_z-by-n_e
                    Policy_ford3_jj(curraindex,:,:,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                end
            end
        end
        
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_bothz_d3',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,z), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,z), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % Already applied the probabilities from interpolating onto grid
            EV=squeeze(sum((EV.*pi_u),3)); % (d2,a1prime,a2,semiz)

            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2,N_semiz]); % (d2,a1prime,1,a2,zprime)

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                % n-Monotonicity
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_semiz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1);

                entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV;

                % First, we want a1prime conditional on (d,1,a)
                [~,maxindex1]=max(entireRHS_ii_d3,[],2);

                % Now, get and store the full (d,aprime)
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1,vfoptions.level1n*N_a2,N_semiz]),[],1);

                % Store
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(maxindex2,1);

                % Attempt for improved version
                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_z
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
                        daprime=(1:1:N_d2)'+N_d2*repelem(a1primeindexes-1,1,1,level1iidiff(ii),1,1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_semiz-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_ii_d3+DiscountedEV(reshape(daprime,[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_semiz]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        % maxindex does not need reworking, as with expasset there is no a2prime
                        %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                        dind=(rem(maxindex-1,N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                        allind=dind+N_d2*a2ind+N_d2*N_a2*semizind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_a2-by-n_z
                        Policy_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,[n_d2,1],1,level1iidiff(ii),n_a2,n_semiz,special_n_e, d23_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
                        daprime=(1:1:N_d2)'+N_d2*repelem(loweredge-1,1,1,level1iidiff(ii),1,1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_semiz-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_ii_d3+DiscountedEV(reshape(daprime,[N_d2*1,level1iidiff(ii)*N_a2,N_semiz]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        % maxindex does not need reworking, as with expasset there is no a2prime
                        %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                        dind=(rem(maxindex-1,N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                        allind=dind+N_d2*a2ind+N_d2*N_a2*semizind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_z
                        Policy_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                    end
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_bothz_d3',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,z), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,z), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % Already applied the probabilities from interpolating onto grid
            EV=squeeze(sum((EV.*pi_u),3)); % (d2,a1prime,a2,semiz)

            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2,N_semiz]); % (d2,a1prime,1,a2,zprime)

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);
                DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);

                    % n-Monotonicity
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_semiz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1);

                    entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_z;

                    % First, we want a1prime conditional on (d,1,a)
                    [~,maxindex1]=max(entireRHS_ii_d3,[],2);

                    % Now, get and store the full (d,aprime)
                    [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

                    % Store
                    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                    V_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    Policy_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex2,1);

                    % Attempt for improved version
                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                            % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2
                            ReturnMatrix_ii_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_semiz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,2);
                            daprime=(1:1:N_d2)'+N_d2*repelem(a1primeindexes-1,1,1,level1iidiff(ii),1,1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                            entireRHS_ii=ReturnMatrix_ii_d3+DiscountedEV_z(reshape(daprime,[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2]));
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            % maxindex does not need reworking, as with expasset there is no a2prime
                            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                            dind=(rem(maxindex-1,N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                            allind=dind+N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_a2
                            Policy_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                        else
                            loweredge=maxindex1(:,1,ii,:,:);
                            % Just use aprime(ii) for everything
                            ReturnMatrix_ii_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, 0,[n_d2,1],1,level1iidiff(ii),n_a2,special_n_semiz,special_n_e, d23_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,2);
                            daprime=(1:1:N_d2)'+N_d2*repelem(loweredge-1,1,1,level1iidiff(ii),1,1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                            entireRHS_ii=ReturnMatrix_ii_d3+DiscountedEV_z(reshape(daprime,[N_d2*1,level1iidiff(ii)*N_a2]));
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            % maxindex does not need reworking, as with expasset there is no a2prime
                            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                            dind=(rem(maxindex-1,N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                            allind=dind+N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                            Policy_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                        end
                    end
                end
            end
        end
    end


    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_jj,[],4); % max over d3
    V(:,:,:,jj)=V_jj;
    Policy3(2,:,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d2a1prime_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    Policy3(1,:,:,:,jj)=rem(d2a1prime_ind-1,N_d2)+1; % d2
    Policy3(3,:,:,:,jj)=ceil(d2a1prime_ind/N_d2); % a1prime
end


%% For experience asset, just output Policy as is and then use Case2 to UnKron


end
