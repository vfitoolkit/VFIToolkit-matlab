function [V,Policy2]=ValueFnIter_FHorz_DC1_e_raw(n_d,n_a,n_z,n_e,N_j, d_grid, a_grid, z_gridvals_J, e_gridvals_J,pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
d_grid=gpuArray(d_grid);
d_gridvals=CreateGridvals(n_d,d_grid,1);
a_grid=gpuArray(a_grid);

if vfoptions.lowmemory>=1
    special_n_e=ones(1,length(n_e));
else
    eind=shiftdim((0:1:N_e-1),-2); % already includes -1
end
if vfoptions.lowmemory==2
    special_n_z=ones(1,length(n_z));
else
    zind=shiftdim((0:1:N_z-1),-1); % already includes -1
end

% n-Monotonicity
% vfoptions.level1n=5;
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

pi_e_J=shiftdim(pi_e_J,-2); % Move to third dimension

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        % First, we want aprime conditional on (d,1,a,z,e)
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a,vfoptions.level1n,N_z,N_e]),[],1);

        % Store
        V(level1ii,:,:,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,:,:,N_j)=shiftdim(maxindex2,1); % d,aprime

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1)); % max over d,z,e
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-1-by-n_z-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_z-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*zind+N_d*N_z*eind; % loweredge is n_d-by-1-by-1-by-n_z-by-n_e
                Policy(curraindex,:,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*zind+N_d*N_z*eind; % loweredge is n_d-by-1-by-1-by-n_z-by-n_e
                Policy(curraindex,:,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            end
        end
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);

            % First, we want aprime conditional on (d,1,a,z,e)
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a,vfoptions.level1n,N_z]),[],1);

            % Store
            V(level1ii,:,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(level1ii,:,e_c,N_j)=shiftdim(maxindex2,1); % d,aprime

            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1)); % max over d,z,e
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-1-by-n_z
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_z
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    allind=dind+N_d*zind; % loweredge is n_d-by-1-by-1-by-n_z
                    Policy(curraindex,:,e_c,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    allind=dind+N_d*zind; % loweredge is n_d-by-1-by-1-by-n_z
                    Policy(curraindex,:,e_c,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
                end
            end
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                % n-Monotonicity
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,1);

                % First, we want aprime conditional on (d,1,a,z,e)
                [~,maxindex1]=max(ReturnMatrix_ii,[],2);

                % Now, get and store the full (d,aprime)
                [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a,vfoptions.level1n]),[],1);

                % Store
                V(level1ii,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                Policy(level1ii,z_c,e_c,N_j)=shiftdim(maxindex2,1); % d,aprime

                % Attempt for improved version
                maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1)); % max over d,z,e
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d-by-1-by-1
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d-by-maxgap(ii)+1-by-1
                        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                        V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d)+1);
                        allind=dind; % loweredge is n_d-by-1-by-1
                        Policy(curraindex,z_c,e_c,N_j)=shiftdim(maxindex'+N_d*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                        V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d)+1);
                        allind=dind; % loweredge is n_d-by-1-by-1
                        Policy(curraindex,z_c,e_c,N_j)=shiftdim(maxindex'+N_d*(loweredge(allind)-1),1);
                    end
                end
            end

        end
    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    V_Jplus1=sum(V_Jplus1.*pi_e_J(1,1,:,N_j),3);

    EV=V_Jplus1.*shiftdim(pi_z_J(:,:,N_j)',-1);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    entireEV=repmat(shiftdim(EV,-1),N_d,1,1,1); % [d,aprime,1,z]

    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV; %repmat(entireEV,1,N_a,1,N_e);

        % First, we want aprime conditional on (d,1,a,z,e)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n,N_z,N_e]),[],1);

        % Store
        V(level1ii,:,:,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,:,:,N_j)=shiftdim(maxindex2,1); % d,aprime

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1)); % max over d,z,e
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-1-by-n_z-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_z-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                daprimez=(1:1:N_d)'+N_d*repelem(aprimeindexes-1,1,1,level1iidiff(ii),1)+N_d*N_a*shiftdim((0:1:N_z-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV(reshape(daprimez,[N_d*(maxgap(ii)+1),level1iidiff(ii),N_z,N_e]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*zind+N_d*N_z*eind; % loweredge is n_d-by-1-by-1-by-n_z-by-n_e
                Policy(curraindex,:,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                daprimez=(1:1:N_d)'+N_d*repelem(loweredge-1,1,1,level1iidiff(ii),1)+N_d*N_a*shiftdim((0:1:N_z-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV(reshape(daprimez,[N_d*1,level1iidiff(ii),N_z,N_e]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*zind+N_d*N_z*eind; % loweredge is n_d-by-1-by-1-by-n_z-by-n_e
                Policy(curraindex,:,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            end
        end
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);

            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV;

            % First, we want aprime conditional on (d,1,a,z,e)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n,N_z]),[],1);

            % Store
            V(level1ii,:,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(level1ii,:,e_c,N_j)=shiftdim(maxindex2,1); % d,aprime

            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1)); % max over d,z,e
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-1-by-n_z
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_z
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                    daprimez=(1:1:N_d)'+N_d*repelem(aprimeindexes-1,1,1,level1iidiff(ii),1)+N_d*N_a*shiftdim((0:1:N_z-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV(reshape(daprimez,[N_d*(maxgap(ii)+1),level1iidiff(ii),N_z]));
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    allind=dind+N_d*zind; % loweredge is n_d-by-1-by-1-by-n_z
                    Policy(curraindex,:,e_c,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                    daprimez=(1:1:N_d)'+N_d*repelem(loweredge-1,1,1,level1iidiff(ii),1)+N_d*N_a*shiftdim((0:1:N_z-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV(reshape(daprimez,[N_d*1,level1iidiff(ii),N_z]));
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    allind=dind+N_d*zind; % loweredge is n_d-by-1-by-1-by-n_z
                    Policy(curraindex,:,e_c,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
                end
            end
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            entireEV_z=entireEV(:,:,z_c);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                % n-Monotonicity
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,1);

                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV_z;

                % First, we want aprime conditional on (d,1,a,z,e)
                [~,maxindex1]=max(entireRHS_ii,[],2);

                % Now, get and store the full (d,aprime)
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n,N_z]),[],1);

                % Store
                V(level1ii,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                Policy(level1ii,z_c,e_c,N_j)=shiftdim(maxindex2,1); % d,aprime

                % Attempt for improved version
                maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1)); % max over d,z,e
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d-by-1-by-1
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d-by-maxgap(ii)+1-by-1
                        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,2);
                        daprimez=(1:1:N_d)'+N_d*repelem(aprimeindexes-1,1,1,level1iidiff(ii),1); % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV_z(reshape(daprimez,[N_d*(maxgap(ii)+1),level1iidiff(ii)]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d)+1);
                        allind=dind; % loweredge is n_d-by-1-by-1
                        Policy(curraindex,z_c,e_c,N_j)=shiftdim(maxindex'+N_d*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,2);
                        daprimez=(1:1:N_d)'+N_d*repelem(loweredge-1,1,1,level1iidiff(ii),1); % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV_z(reshape(daprimez,[N_d*1,level1iidiff(ii)]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d)+1);
                        allind=dind; % loweredge is n_d-by-1-by-1
                        Policy(curraindex,z_c,e_c,N_j)=shiftdim(maxindex'+N_d*(loweredge(allind)-1),1);
                    end
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
    
    VKronNext_j=V(:,:,:,jj+1);
    VKronNext_j=sum(VKronNext_j.*pi_e_J(1,1,:,jj),3);

    EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    entireEV=repmat(shiftdim(EV,-1),N_d,1,1,1); % [d,aprime,1,z]

    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV;

        % First, we want aprime conditional on (d,1,a,z,e)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n,N_z,N_e]),[],1);

        % Store
        V(level1ii,:,:,jj)=shiftdim(Vtempii,1);
        Policy(level1ii,:,:,jj)=shiftdim(maxindex2,1); % d,aprime

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1)); % max over d,z,e
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-1-by-n_z-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii)); % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_z-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                daprimez=(1:1:N_d)'+N_d*repelem(aprimeindexes-1,1,1,level1iidiff(ii),1)+N_d*N_a*shiftdim((0:1:N_z-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV(reshape(daprimez,[N_d*(maxgap(ii)+1),level1iidiff(ii),N_z,N_e]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,jj)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*zind+N_d*N_z*eind; % loweredge is n_d-by-1-by-1-by-n_z-by-n_e
                Policy(curraindex,:,:,jj)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                daprimez=(1:1:N_d)'+N_d*repelem(loweredge-1,1,1,level1iidiff(ii),1)+N_d*N_a*shiftdim((0:1:N_z-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV(reshape(daprimez,[N_d*1,level1iidiff(ii),N_z,N_e]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,jj)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*zind+N_d*N_z*eind; % loweredge is n_d-by-1-by-1-by-n_z-by-n_e
                Policy(curraindex,:,:,jj)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            end
        end
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1);

            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV;

            % First, we want aprime conditional on (d,1,a,z,e)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n,N_z]),[],1);

            % Store
            V(level1ii,:,e_c,jj)=shiftdim(Vtempii,1);
            Policy(level1ii,:,e_c,jj)=shiftdim(maxindex2,1); % d,aprime

            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1)); % max over d,z,e
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-1-by-n_z
                    aprimeindexes=loweredge+(0:1:maxgap(ii)); % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_z
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
                    daprimez=(1:1:N_d)'+N_d*repelem(aprimeindexes-1,1,1,level1iidiff(ii),1)+N_d*N_a*shiftdim((0:1:N_z-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV(reshape(daprimez,[N_d*(maxgap(ii)+1),level1iidiff(ii),N_z]));
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curraindex,:,e_c,jj)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    allind=dind+N_d*zind; % loweredge is n_d-by-1-by-1-by-n_z
                    Policy(curraindex,:,e_c,jj)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
                    daprimez=(1:1:N_d)'+N_d*repelem(loweredge-1,1,1,level1iidiff(ii),1)+N_d*N_a*shiftdim((0:1:N_z-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV(reshape(daprimez,[N_d*1,level1iidiff(ii),N_z]));
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curraindex,:,e_c,jj)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    allind=dind+N_d*zind; % loweredge is n_d-by-1-by-1-by-n_z
                    Policy(curraindex,:,e_c,jj)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
                end
            end
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            entireEV_z=entireEV(:,:,z_c);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                % n-Monotonicity
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,1);

                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV_z;

                % First, we want aprime conditional on (d,1,a,z,e)
                [~,maxindex1]=max(entireRHS_ii,[],2);

                % Now, get and store the full (d,aprime)
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);

                % Store
                V(level1ii,z_c,e_c,jj)=shiftdim(Vtempii,1);
                Policy(level1ii,z_c,e_c,jj)=shiftdim(maxindex2,1); % d,aprime

                % Attempt for improved version
                maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1)); % max over d,z,e
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d-by-1-by-1
                        aprimeindexes=loweredge+(0:1:maxgap(ii)); % aprime possibilities are n_d-by-maxgap(ii)+1-by-1
                        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,2);
                        daprimez=(1:1:N_d)'+N_d*repelem(aprimeindexes-1,1,1,level1iidiff(ii),1); % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV_z(reshape(daprimez,[N_d*(maxgap(ii)+1),level1iidiff(ii)]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V(curraindex,z_c,e_c,jj)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d)+1);
                        allind=dind; % loweredge is n_d-by-1-by-1
                        Policy(curraindex,z_c,e_c,jj)=shiftdim(maxindex'+N_d*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,2);
                        daprimez=(1:1:N_d)'+N_d*repelem(loweredge-1,1,1,level1iidiff(ii),1); % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV_z(reshape(daprimez,[N_d*1,level1iidiff(ii)]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V(curraindex,z_c,e_c,jj)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d)+1);
                        allind=dind; % loweredge is n_d-by-1-by-1
                        Policy(curraindex,z_c,e_c,jj)=shiftdim(maxindex'+N_d*(loweredge(allind)-1),1);
                    end
                end
            end
        end
    end
end

%%
Policy2=zeros(2,N_a,N_z,N_e,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
Policy2(1,:,:,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:,:,:)=shiftdim(ceil(Policy/N_d),-1);

end
