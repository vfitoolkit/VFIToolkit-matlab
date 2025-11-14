function [V,Policy2]=ValueFnIter_FHorz_DC2B_noz_e_raw(n_d,n_a,n_e,N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% divide-and-conquer in the first endo state

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z 

%%
n_a1=n_a(1);
n_a2=n_a(2:end);
N_a1=n_a1;
N_a2=n_a2;
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% n-Monotonicity
% vfoptions.level1n=[21,21];
level1ii=round(linspace(1,n_a(1),vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

pi_e_J=shiftdim(pi_e_J,-1);

% precompute
a2ind=gpuArray(0:1:N_a2-1); % already includes -1
eind=shiftdim(gpuArray(0:1:N_e-1),-1); % already includes -1
a2Bind=shiftdim(gpuArray(0:1:N_a2-1),-1); % already includes -1

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_Par2(ReturnFn, n_d, n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);

    % size(ReturnMatrix_ii) % (d, aprime, a,z)
    % [n_d,n_a,vfoptions.level1n,n_z]

    % First, we want a1prime conditional on (d,1,a2prime,a)
    [~,maxindex1]=max(ReturnMatrix_ii,[],2);

    % Now, get and store the full (d,aprime)
    [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2,N_e]),[],1);
    % Store
    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
    V(curraindex,:,N_j)=shiftdim(Vtempii,1);
    Policy(curraindex,:,N_j)=shiftdim(maxindex2,1);
    
    % Attempt for improved version
    maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_e
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1-by-n_a2-by-1-by-n_a2-by-n_e
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_Par2(ReturnFn, n_d, n_e, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            V(curraindex,:,N_j)=shiftdim(Vtempii,1);
            % maxindex needs to be reworked:
            %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
            dind=(rem(maxindex-1,N_d)+1);
            a1primeind=rem(ceil(maxindex/N_d)-1,maxgap(ii)+1)+1-1; % already includes -1
            a2primeind=ceil(maxindex/(N_d*(maxgap(ii)+1)))-1; % already includes -1
            maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            allind=dind+N_d*a2primeind+N_d*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d*N_a2*N_a2*eind; % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_e
            Policy(curraindex,:,N_j)=shiftdim(maxindexfix+N_d*(loweredge(allind)-1),1);
        else
            loweredge=maxindex1(:,1,:,ii,:,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_Par2(ReturnFn, n_d, n_e, d_gridvals, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            V(curraindex,:,N_j)=shiftdim(Vtempii,1);
            % maxindex needs to be reworked:
            %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
            dind=(rem(maxindex-1,N_d)+1);
            a1primeind=0; %1-1; % already includes -1
            a2primeind=ceil(maxindex/N_d)-1; % already includes -1 % divide by (N_d*1)
            maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            allind=dind+N_d*a2primeind+N_d*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d*N_a2*N_a2*eind; % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_e
            Policy(curraindex,:,N_j)=shiftdim(maxindexfix+N_d*(loweredge(allind)-1),1);
        end

    end
    
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=sum(reshape(vfoptions.V_Jplus1,[N_a,N_e]).*pi_e_J(1,:,N_j),2); % Using V_Jplus1
    
    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[1,N_a1,N_a2,1,1]); % will autoexpand d in 1st-dim

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_Par2(ReturnFn, n_d, n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);

    entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

    % size(ReturnMatrix_ii) % (d, aprime, a,z,e)
    % [n_d,n_a,vfoptions.level1n,n_z,n_e]

    % First, we want a1prime conditional on (d,1,a2prime,a,z,e)
    [~,maxindex1]=max(entireRHS_ii,[],2);

    % Now, get and store the full (d,aprime)
    [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2,N_e]),[],1);
    % Store
    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
    V(curraindex,:,N_j)=shiftdim(Vtempii,1);
    Policy(curraindex,:,N_j)=shiftdim(maxindex2,1);
    
    % Attempt for improved version
    maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_e
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1-by-n_a2-by-1-by-n_a2-by-n_e
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_Par2(ReturnFn, n_d, n_e, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            aprime=repelem(a1primeindexes,1,1,1,level1iidiff(ii),1,1)+N_a1*a2Bind;
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprime,[N_d*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_e]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,:,N_j)=shiftdim(Vtempii,1);
            % maxindex needs to be reworked:
            %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
            dind=(rem(maxindex-1,N_d)+1);
            a1primeind=rem(ceil(maxindex/N_d)-1,maxgap(ii)+1)+1-1; % already includes -1
            a2primeind=ceil(maxindex/(N_d*(maxgap(ii)+1)))-1; % already includes -1
            maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            allind=dind+N_d*a2primeind+N_d*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d*N_a2*N_a2*eind; % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_e
            Policy(curraindex,:,N_j)=shiftdim(maxindexfix+N_d*(loweredge(allind)-1),1);
        else
            loweredge=maxindex1(:,1,:,ii,:,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_Par2(ReturnFn, n_d, n_e, d_gridvals, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            aprime=repelem(loweredge,1,1,1,level1iidiff(ii),1,1)+N_a1*a2Bind;
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprime,[N_d*1*N_a2,level1iidiff(ii)*N_a2,N_e]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,:,N_j)=shiftdim(Vtempii,1);
            % maxindex needs to be reworked:
            %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
            dind=(rem(maxindex-1,N_d)+1);
            a1primeind=0; %1-1; % already includes -1
            a2primeind=ceil(maxindex/N_d)-1; % already includes -1 % divide by (N_d*1)
            maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            allind=dind+N_d*a2primeind+N_d*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d*N_a2*N_a2*eind; % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_e
            Policy(curraindex,:,N_j)=shiftdim(maxindexfix+N_d*(loweredge(allind)-1),1);
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
    
    EV=sum(V(:,:,jj+1).*pi_e_J(1,:,jj),2);
    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[1,N_a1,N_a2,1,1]); % will autoexpand d in 1st-dim

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_Par2(ReturnFn, n_d, n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);

    entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

    % size(ReturnMatrix_ii) % (d, aprime, a,z)
    % [n_d,n_a,vfoptions.level1n,n_z]

    % First, we want a1prime conditional on (d,1,a2prime,a,z)
    [~,maxindex1]=max(entireRHS_ii,[],2);

    % Now, get and store the full (d,aprime)
    [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2,N_e]),[],1);
    % Store
    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
    V(curraindex,:,jj)=shiftdim(Vtempii,1);
    Policy(curraindex,:,jj)=shiftdim(maxindex2,1);
    
    % Attempt for improved version
    maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1-by-n_a2-by-1-by-n_a2-by-n_e
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_Par2(ReturnFn, n_d, n_e, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
            aprime=repelem(a1primeindexes,1,1,1,level1iidiff(ii),1,1)+N_a1*a2Bind;
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprime,[N_d*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_e]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,:,jj)=shiftdim(Vtempii,1);
            % maxindex needs to be reworked:
            %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
            dind=(rem(maxindex-1,N_d)+1);
            a1primeind=rem(ceil(maxindex/N_d)-1,maxgap(ii)+1)+1-1; % already includes -1
            a2primeind=ceil(maxindex/(N_d*(maxgap(ii)+1)))-1; % already includes -1
            maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            allind=dind+N_d*a2primeind+N_d*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d*N_a2*N_a2*eind; % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_e
            Policy(curraindex,:,jj)=shiftdim(maxindexfix+N_d*(loweredge(allind)-1),1);
        else
            loweredge=maxindex1(:,1,:,ii,:,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_Par2(ReturnFn, n_d, n_e, d_gridvals, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
            aprime=repelem(loweredge,1,1,1,level1iidiff(ii),1,1)+N_a1*a2Bind;
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprime,[N_d*1*N_a2,level1iidiff(ii)*N_a2,N_e]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,:,jj)=shiftdim(Vtempii,1);
            % maxindex needs to be reworked:
            %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
            dind=(rem(maxindex-1,N_d)+1);
            a1primeind=0; %1-1; % already includes -1
            a2primeind=ceil(maxindex/N_d)-1; % already includes -1 % divide by (N_d*1)
            maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            allind=dind+N_d*a2primeind+N_d*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d*N_a2*N_a2*eind; % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_e
            Policy(curraindex,:,jj)=shiftdim(maxindexfix+N_d*(loweredge(allind)-1),1);
        end
    end

end

%%
Policy2=zeros(2,N_a,N_e,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
Policy2(1,:,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:,:)=shiftdim(ceil(Policy/N_d),-1);

end
