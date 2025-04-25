function [V,Policy3]=ValueFnIter_Case1_FHorz_SemiExo_DC1_nod1_e_raw(n_d2,n_a,n_z,n_semiz, n_e,N_j, d2_grid, a_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J,pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

error('This needs to be tested')

n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

V=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d,d2,aprime seperately
Policy3=zeros(2,N_a,N_semiz*N_z,N_e,N_j,'gpuArray');

%%
d2_grid=gpuArray(d2_grid);
a_grid=gpuArray(a_grid);

special_n_d2=ones(1,length(n_d2));
d2_gridvals=CreateGridvals(n_d2,d2_grid,1);

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
else
    eind=shiftdim((0:1:N_e-1),-2); % already includes -1
end
if vfoptions.lowmemory>1
    special_n_bothz=ones(1,length(n_z)+length(n_semiz));
else
    bothzind=shiftdim((0:1:N_bothz-1),-1);
end
bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

% Preallocate
if vfoptions.lowmemory==0
    V_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
    Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
elseif vfoptions.lowmemory==1 % loops over e
    V_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
    Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
end

pi_e_J=shiftdim(pi_e_J,-2); % Move to third dimension

% n-Monotonicity
% vfoptions.level1n=5;
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')

    if vfoptions.lowmemory==0

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d2, n_bothz, n_e, d2_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        % First, we want aprime conditional on (d,1,a,z,e)
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d2*N_a,vfoptions.level1n,N_bothz,N_e]),[],1);

        % Store
        V(level1ii,:,:,N_j)=shiftdim(Vtempii,1);
        Policytemp(level1ii,:,:)=shiftdim(maxindex2,1); % d,aprime
        
        % Second level based on montonicity
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_bothz-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_bothz-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d2, n_bothz, n_e, d2_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                d2ind=(rem(maxindex-1,N_d2)+1);
                allind=d2ind+N_d2*bothzind+N_d2*N_bothz*eind; % loweredge is n_d-by-1-by-1-by-n_bothz-by-n_e
                Policytemp(curraindex,:,:)=shiftdim(maxindex+N_d2*(loweredge(allind)-1)); % loweredge(given the d and z)
            else
                loweredge=maxindex1(:,1,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d2, n_bothz, n_e, d2_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                d2ind=(rem(maxindex-1,N_d2)+1);
                allind=d2ind+N_d2*bothzind+N_d2*N_bothz*eind; % loweredge is n_d-by-1-by-1-by-n_bothz-by-n_e
                Policytemp(curraindex,:,:)=shiftdim(maxindex+N_d2*(loweredge(allind)-1)); % loweredge(given the d and z)
            end
        end

        % Deal with policy for semi-exo
        Policy3(1,:,:,:,N_j)=shiftdim(rem(Policytemp-1,N_d2)+1,-1);
        Policy3(2,:,:,:,N_j)=shiftdim(ceil(Policytemp/N_d2),-1);

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % n-Monotonicity
            ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d2, n_bothz, special_n_e, d2_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);

            % First, we want aprime conditional on (d,1,a,z,e)
            [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_e,[N_d2*N_a,vfoptions.level1n,N_bothz]),[],1);

            % Store
            V(level1ii,:,e_c,N_j)=shiftdim(Vtempii,1);
            Policytemp(level1ii,:,e_c)=shiftdim(maxindex2,1); % d,aprime

            % Second level based on montonicity
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_bothz
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_bothz
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d2, n_bothz, special_n_e, d2_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
                    V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    d2ind=(rem(maxindex-1,N_d2)+1);
                    allind=d2ind+N_d2*bothzind; % loweredge is n_d-by-1-by-1-by-n_bothz
                    Policytemp(curraindex,:,e_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1)); % loweredge(given the d and z)
                else
                    loweredge=maxindex1(:,1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d2, n_bothz, special_n_e, d2_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
                    V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    d2ind=(rem(maxindex-1,N_d2)+1);
                    allind=d2ind+N_d2*bothzind; % loweredge is n_d-by-1-by-1-by-n_bothz
                    Policytemp(curraindex,:,e_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1)); % loweredge(given the d and z)
                end
            end

            % Deal with policy for semi-exo
            Policy3(1,:,:,e_c,N_j)=shiftdim(rem(Policytemp-1,N_d2)+1,-1);
            Policy3(2,:,:,e_c,N_j)=shiftdim(ceil(Policytemp/N_d2),-1);
        end
    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_semiz*N_z,N_e]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    V_Jplus1=sum(V_Jplus1.*pi_e_J(1,1,:,N_j),3);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j)); % reverse order

            EV=V_Jplus1.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % n-Monotonicity
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_bothz,n_e, d2_gridvals(d2_c,:)', a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2); % effecitely nod (as just a single d2 point), so use level=2 even for first level

            entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*EV;

            % First, we want aprime conditional on (1,a,z,e)
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);

            % Store
            V_ford2_jj(level1ii,:,:,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(level1ii,:,:,d2_c)=shiftdim(maxindex,1); % d,aprime

            % Second level based on montonicity
            maxgap=squeeze(max(max(maxindex(1,2:end,:,:)-maxindex(1,1:end-1,:,:),[],5),[],4);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d1-by-1-by-n_bothz-by-n_e
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_bothz-by-n_e
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_bothz, n_e, d2_gridvals(d2_c,:)', a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                    aprimez=repelem(aprimeindexes,1,1,level1iidiff(ii),1)+N_a*shiftdim((0:1:N_bothz-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV(aprimez(:)),[(maxgap(ii)+1),level1iidiff(ii),N_bothz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    allind=(bothzind+1)+N_bothz*eind; % loweredge is n_d1-by-1-by-1-by-n_bothz-by-n_e
                    Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex+(loweredge(allind)-1)); % loweredge(given the d and z)
                else
                    loweredge=maxindex1(1,ii,:,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_gridvals(d2_c,:)', a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                    aprimez=repelem(loweredge,1,1,level1iidiff(ii),1)+N_a*shiftdim((0:1:N_bothz-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV(aprimez(:)),[1,level1iidiff(ii),N_bothz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    allind=(bothzind+1)+N_bothz*eind; % loweredge is n_d1-by-1-by-1-by-n_bothz-by-n_e
                    Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex+(loweredge(allind)-1)); % loweredge(given the d and z)
                end
            end

        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,N_j)=V_jj;
        Policy3(1,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
        Policy3(2,:,:,:,N_j)=shiftdim(aprime_ind,-1);

    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));

            EV=V_Jplus1.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            entireEV=repelem(EV,N_d1,1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                % n-Monotonicity
                ReturnMatrix_d2iie=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_gridvals(d2_c,:)', a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2); % effecitely nod (as just a single d2 point), so use level=2 even for first level

                entireRHS_ii=ReturnMatrix_d2iie+DiscountFactorParamsVec*EV;

                % First, we want aprime conditional on (d,1,a,z,e)
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);

                % Store
                V_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(maxindex,1); % d,aprime

                % Second level based on montonicity
                maxgap=squeeze(max(maxindex(1,2:end,:)-maxindex(1,1:end-1,:),[],4);
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d1-by-1-by-n_bothz
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_bothz
                        ReturnMatrix_iie=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_gridvals(d2_c,:)', a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                        aprimez=repelem(aprimeindexes,1,1,level1iidiff(ii),1)+N_a*shiftdim((0:1:N_bothz-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_iie+DiscountFactorParamsVec*reshape(EV(aprimez(:)),[(maxgap(ii)+1),level1iidiff(ii),N_bothz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        allind=bothzind+1; % loweredge is n_d1-by-1-by-1-by-n_bothz
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex+(loweredge(allind)-1)); % loweredge(given the d and z)
                    else
                        loweredge=maxindex(1,ii,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_gridvals(d2_c,:)', a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                        aprimez=repelem(loweredge,1,1,level1iidiff(ii),1)+N_a*shiftdim((0:1:N_bothz-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV(aprimez(:)),[1,level1iidiff(ii),N_bothz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        allind=bothzind+1; % loweredge is n_d1-by-1-by-1-by-n_bothz
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex+(loweredge(allind)-1)); % loweredge(given the d and z)
                    end
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy3(1,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
        Policy3(2,:,:,:,N_j)=shiftdim(aprime_ind,-1);

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
    
    VKronNext_j=V(:,:,:,jj+1);
        
    VKronNext_j=sum(VKronNext_j.*pi_e_J(1,1,:,jj),3);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d2_c,jj)); % reverse order

            EV=VKronNext_j.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % n-Monotonicity
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_bothz,n_e, d2_gridvals(d2_c,:)', a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2); % effecitely nod (as just a single d2 point), so use level=2 even for first level

            entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*EV;

            % First, we want aprime conditional on (1,a,z,e)
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            % Store
            V_ford2_jj(level1ii,:,:,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(level1ii,:,:,d2_c)=shiftdim(maxindex,1); % d,aprime

            % Second level based on montonicity
            maxgap=squeeze(max(max(maxindex(1,2:end,:,:)-maxindex(1,1:end-1,:,:),[],5),[],4);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d1-by-1-by-n_bothz-by-n_e
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_bothz-by-n_e
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_bothz, n_e, d2_gridvals(d2_c,:)', a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                    aprimez=repelem(aprimeindexes-1,1,1,level1iidiff(ii),1)+N_a*shiftdim((0:1:N_bothz-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV(aprimez(:)),[(maxgap(ii)+1),level1iidiff(ii),N_bothz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    allind=(bothzind+1)+N_bothz*eind; % loweredge is n_d1-by-1-by-1-by-n_bothz-by-n_e
                    Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex+(loweredge(allind)-1)); % loweredge(given the d and z)
                else
                    loweredge=maxindex1(1,ii,:,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_bothz, n_e, d2_gridvals(d2_c,:)', a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                    aprimez=repelem(loweredge,1,1,level1iidiff(ii),1)+N_a*shiftdim((0:1:N_bothz-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV(aprimez(:)),[1,level1iidiff(ii),N_bothz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    allind=(bothzind+1)+N_bothz*eind; % loweredge is n_d1-by-1-by-1-by-n_bothz-by-n_e
                    Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex+(loweredge(allind)-1)); % loweredge(given the d and z)
                end
            end

        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy3(1,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
        Policy3(1,:,:,:,jj)=shiftdim(aprime_ind,-1);

    elseif vfoptions.lowmemory==1 
        for d2_c=1:N_d2
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));

            EV=VKronNext_j.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                % n-Monotonicity
                ReturnMatrix_d2iie=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_gridvals(d2_c,:)', a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2); % effecitely nod (as just a single d2 point), so use level=2 even for first level

                entireRHS_ii=ReturnMatrix_d2iie+DiscountFactorParamsVec*EV;

                % First, we want aprime conditional on (1,a,z,e)
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);

                % Store
                V_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(maxindex,1); % d,aprime

                % Second level based on montonicity
                maxgap=squeeze(max(maxindex(1,2:end,:)-maxindex(1,1:end-1,:),[],4);
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d1-by-1-by-n_bothz
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_bothz
                        ReturnMatrix_iie=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_gridvals(d2_c,:)', a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
                        aprimez=repelem(aprimeindexes,1,1,level1iidiff(ii),1)+N_a*shiftdim((0:1:N_bothz-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_iie+DiscountFactorParamsVec*reshape(EV(aprimez(:)),[(maxgap(ii)+1),level1iidiff(ii),N_bothz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        allind=bothzind+1; % loweredge is n_d1-by-1-by-1-by-n_bothz
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex+(loweredge(allind)-1)); % loweredge(given the d and z)
                    else
                        loweredge=maxindex(1,ii,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_gridvals(d2_c,:)', a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
                        aprimez=repelem(loweredge,1,1,level1iidiff(ii),1)+N_a*shiftdim((0:1:N_bothz-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV(aprimez(:)),[1,level1iidiff(ii),N_bothz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        allind=bothzind+1; % loweredge is n_d1-by-1-by-1-by-n_bothz
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex+(loweredge(allind)-1)); % loweredge(given the d and z)
                    end
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy3(1,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
        Policy3(2,:,:,:,jj)=shiftdim(aprime_ind,-1);

    end
end



end