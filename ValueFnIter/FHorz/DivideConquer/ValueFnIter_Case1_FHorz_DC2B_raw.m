function [V,Policy2]=ValueFnIter_Case1_FHorz_DC2B_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
d_grid=gpuArray(d_grid);
d_gridvals=CreateGridvals(n_d,d_grid,1);
a_grid=gpuArray(a_grid);

N_a1=n_a(1);
N_a2=n_a(2);
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% n-Monotonicity
% vfoptions.level1n=[21,21];
level1ii=round(linspace(1,n_a(1),vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;
Policytemp=zeros(N_a,N_z,'gpuArray');

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_Par2(ReturnFn, n_d, n_z, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);

    % size(ReturnMatrix_ii) % (d, aprime, a,z)
    % [n_d,n_a,vfoptions.level1n,n_z]

    % First, we want a1prime conditional on (d,1,a2prime,a,z)
    [~,maxindex1]=max(ReturnMatrix_ii,[],2);

    % Now, get and store the full (d,aprime)
    [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2,N_z]),[],1);
    % Store
    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
    V(curraindex,:,N_j)=shiftdim(Vtempii,1);
    Policytemp(curraindex,:)=shiftdim(maxindex2,1);
    
    % Attempt for improved version
    maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1-by-n_a2-by-1-by-n_a2-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_Par2(ReturnFn, n_d, n_z, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            V(curraindex,:,N_j)=shiftdim(Vtempii,1);
            % maxindex needs to be reworked:
            %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
            dind=(rem(maxindex-1,N_d)+1);
            a1primeind=rem(ceil(maxindex/N_d)-1,maxgap(ii)+1)+1-1; % already includes -1
            a2primeind=ceil(maxindex/(N_d*(maxgap(ii)+1)))-1; % already includes -1
            maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
            zind=shiftdim((0:1:N_z-1),-1); % already includes -1
            allind=dind+N_d*a2primeind+N_d*N_a2*a2ind+N_d*N_a2*N_a2*zind; % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
            Policytemp(curraindex,:)=shiftdim(maxindexfix+N_d*(loweredge(allind)-1),1);
        else
            loweredge=maxindex1(:,1,:,ii,:,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_Par2(ReturnFn, n_d, n_z, d_gridvals, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            V(curraindex,:,N_j)=shiftdim(Vtempii,1);
            % maxindex needs to be reworked:
            %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
            dind=(rem(maxindex-1,N_d)+1);
            a1primeind=0; %1-1; % already includes -1
            a2primeind=ceil(maxindex/N_d)-1; % already includes -1 % divide by (N_d*1)
            maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
            zind=shiftdim((0:1:N_z-1),-1); % already includes -1
            allind=dind+N_d*a2primeind+N_d*N_a2*a2ind+N_d*N_a2*N_a2*zind; % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
            Policytemp(curraindex,:)=shiftdim(maxindexfix+N_d*(loweredge(allind)-1),1);
        end

    end

    Policy(:,:,N_j)=Policytemp;

else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_z]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    EV=V_Jplus1.*shiftdim(pi_z_J(:,:,N_j)',-1);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension
    DiscountedentireEV=DiscountFactorParamsVec*reshape(repmat(shiftdim(EV,-1),N_d,1,1,1),[N_d,N_a1,N_a2,1,1,N_z]); % [d,aprime,1,1,z]

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_Par2(ReturnFn, n_d, n_z, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);

    entireRHS_ii=ReturnMatrix_ii+DiscountedentireEV;

    % size(ReturnMatrix_ii) % (d, aprime, a,z)
    % [n_d,n_a,vfoptions.level1n,n_z]

    % First, we want a1prime conditional on (d,1,a2prime,a,z)
    [~,maxindex1]=max(entireRHS_ii,[],2);

    % Now, get and store the full (d,aprime)
    [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2,N_z]),[],1);
    % Store
    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
    V(curraindex,:,N_j)=shiftdim(Vtempii,1);
    Policytemp(curraindex,:)=shiftdim(maxindex2,1);
    
    % Attempt for improved version
    maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1-by-n_a2-by-1-by-n_a2-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_Par2(ReturnFn, n_d, n_z, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            daprimez=(1:1:N_d)'+N_d*repelem(a1primeindexes-1,1,1,1,level1iidiff(ii),1,1)+N_d*N_a1*shiftdim((0:1:N_a2-1),-1)+N_d*N_a*shiftdim((0:1:N_z-1),-4); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountedentireEV(reshape(daprimez,[N_d*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_z]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,:,N_j)=shiftdim(Vtempii,1);
            % maxindex needs to be reworked:
            %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
            dind=(rem(maxindex-1,N_d)+1);
            a1primeind=rem(ceil(maxindex/N_d)-1,maxgap(ii)+1)+1-1; % already includes -1
            a2primeind=ceil(maxindex/(N_d*(maxgap(ii)+1)))-1; % already includes -1
            maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
            zind=shiftdim((0:1:N_z-1),-1); % already includes -1
            allind=dind+N_d*a2primeind+N_d*N_a2*a2ind+N_d*N_a2*N_a2*zind; % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
            Policytemp(curraindex,:)=shiftdim(maxindexfix+N_d*(loweredge(allind)-1),1);
        else
            loweredge=maxindex1(:,1,:,ii,:,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_Par2(ReturnFn, n_d, n_z, d_gridvals, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            daprimez=(1:1:N_d)'+N_d*repelem(loweredge-1,1,1,1,level1iidiff(ii),1,1)+N_d*1*shiftdim((0:1:N_a2-1),-1)+N_d*N_a*shiftdim((0:1:N_z-1),-4); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountedentireEV(reshape(daprimez,[N_d*1*N_a2,level1iidiff(ii)*N_a2,N_z]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,:,N_j)=shiftdim(Vtempii,1);
            % maxindex needs to be reworked:
            %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
            dind=(rem(maxindex-1,N_d)+1);
            a1primeind=0; %1-1; % already includes -1
            a2primeind=ceil(maxindex/N_d)-1; % already includes -1 % divide by (N_d*1)
            maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
            zind=shiftdim((0:1:N_z-1),-1); % already includes -1
            allind=dind+N_d*a2primeind+N_d*N_a2*a2ind+N_d*N_a2*N_a2*zind; % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
            Policytemp(curraindex,:)=shiftdim(maxindexfix+N_d*(loweredge(allind)-1),1);
        end

    end

    Policy(:,:,N_j)=Policytemp;

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

    EV=V(:,:,jj+1).*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension
    DiscountedentireEV=DiscountFactorParamsVec*reshape(repmat(shiftdim(EV,-1),N_d,1,1,1),[N_d,N_a1,N_a2,1,1,N_z]); % [d,aprime,1,1,z]

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_Par2(ReturnFn, n_d, n_z, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);

    entireRHS_ii=ReturnMatrix_ii+DiscountedentireEV;

    % size(ReturnMatrix_ii) % (d, aprime, a,z)
    % [n_d,n_a,vfoptions.level1n,n_z]

    % First, we want a1prime conditional on (d,1,a2prime,a,z)
    [~,maxindex1]=max(entireRHS_ii,[],2);

    % Now, get and store the full (d,aprime)
    [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2,N_z]),[],1);
    % Store
    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
    V(curraindex,:,jj)=shiftdim(Vtempii,1);
    Policytemp(curraindex,:)=shiftdim(maxindex2,1);
    
    % Attempt for improved version
    maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1-by-n_a2-by-1-by-n_a2-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_Par2(ReturnFn, n_d, n_z, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
            daprimez=(1:1:N_d)'+N_d*repelem(a1primeindexes-1,1,1,1,level1iidiff(ii),1,1)+N_d*N_a1*shiftdim((0:1:N_a2-1),-1)+N_d*N_a*shiftdim((0:1:N_z-1),-4); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountedentireEV(reshape(daprimez,[N_d*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_z]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,:,jj)=shiftdim(Vtempii,1);
            % maxindex needs to be reworked:
            %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
            dind=(rem(maxindex-1,N_d)+1);
            a1primeind=rem(ceil(maxindex/N_d)-1,maxgap(ii)+1)+1-1; % already includes -1
            a2primeind=ceil(maxindex/(N_d*(maxgap(ii)+1)))-1; % already includes -1
            maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
            zind=shiftdim((0:1:N_z-1),-1); % already includes -1
            allind=dind+N_d*a2primeind+N_d*N_a2*a2ind+N_d*N_a2*N_a2*zind; % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
            Policytemp(curraindex,:)=shiftdim(maxindexfix+N_d*(loweredge(allind)-1),1);
        else
            loweredge=maxindex1(:,1,:,ii,:,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_Par2(ReturnFn, n_d, n_z, d_gridvals, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
            daprimez=(1:1:N_d)'+N_d*repelem(loweredge-1,1,1,1,level1iidiff(ii),1,1)+N_d*1*shiftdim((0:1:N_a2-1),-1)+N_d*N_a*shiftdim((0:1:N_z-1),-4); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountedentireEV(reshape(daprimez,[N_d*1*N_a2,level1iidiff(ii)*N_a2,N_z]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,:,jj)=shiftdim(Vtempii,1);
            % maxindex needs to be reworked:
            %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
            dind=(rem(maxindex-1,N_d)+1);
            a1primeind=0; %1-1; % already includes -1
            a2primeind=ceil(maxindex/N_d)-1; % already includes -1 % divide by (N_d*1)
            maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
            zind=shiftdim((0:1:N_z-1),-1); % already includes -1
            allind=dind+N_d*a2primeind+N_d*N_a2*a2ind+N_d*N_a2*N_a2*zind; % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
            Policytemp(curraindex,:)=shiftdim(maxindexfix+N_d*(loweredge(allind)-1),1);
        end

    end

    Policy(:,:,jj)=Policytemp;

end


%%
Policy2=zeros(2,N_a,N_z,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
Policy2(1,:,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:,:)=shiftdim(ceil(Policy/N_d),-1);

end