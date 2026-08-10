function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicS_DC2A_noz_raw(n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Sophisticated quasi-hyperbolic discounting variant of ValueFnIter_FHorz_DC2A_noz_raw.
% divide-and-conquer in the first endo state. Has d. No z. No e. GPU (parallel==2 only).
%
% Sophisticated: Vhat_j = max u + beta_0*beta*E[Vunderbar_{j+1}]
%                Vunderbar_j = Vhat_j + (beta - beta_0*beta)*EV_at_optimal_aprime

N_d=prod(n_d);
N_a=prod(n_a);

Vhat=zeros(N_a,N_j,'gpuArray');
Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
n_a1=n_a(1);
n_a2=n_a(2:end);
N_a1=n_a1;
N_a2=n_a2;
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% n-Monotonicity
level1ii=round(linspace(1,n_a(1),vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% precompute
a2ind=gpuArray(0:1:N_a2-1); % already includes -1
a2Bind=shiftdim(gpuArray(0:1:N_a2-1),-1); % already includes -1


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_noz(ReturnFn, n_d, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, ReturnFnParamsVec,1,0);

    % size(ReturnMatrix_ii) % (d, aprime, a,z)
    % [n_d,n_a,vfoptions.level1n,n_z]

    % First, we want a1prime conditional on (d,1,a2prime,a)
    [~,maxindex1]=max(ReturnMatrix_ii,[],2);

    % Now, get and store the full (d,aprime)
    [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2]),[],1);
    % Store
    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
    Vhat(curraindex,N_j)=shiftdim(Vtempii,1);
    Policy(curraindex,N_j)=shiftdim(maxindex2,1);

    % Attempt for improved version
    maxgap=squeeze(max(max(max(maxindex1(:,1,:,2:end,:)-maxindex1(:,1,:,1:end-1,:),[],5),[],3),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,:,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1-by-n_a2-by-1-by-n_a2
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_noz(ReturnFn, n_d, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, ReturnFnParamsVec,2,0);
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            Vhat(curraindex,N_j)=shiftdim(Vtempii,1);
            % maxindex needs to be reworked:
            %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
            dind=(rem(maxindex-1,N_d)+1);
            a1primeind=rem(ceil(maxindex/N_d)-1,maxgap(ii)+1)+1-1; % already includes -1
            a2primeind=ceil(maxindex/(N_d*(maxgap(ii)+1)))-1; % already includes -1
            maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            allind=dind+N_d*a2primeind+N_d*N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
            Policy(curraindex,N_j)=shiftdim(maxindexfix+N_d*(loweredge(allind)-1),1);
        else
            loweredge=maxindex1(:,1,:,ii,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_noz(ReturnFn, n_d, d_gridvals, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, ReturnFnParamsVec,2,0);
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            Vhat(curraindex,N_j)=shiftdim(Vtempii,1);
            % maxindex needs to be reworked:
            %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
            dind=(rem(maxindex-1,N_d)+1);
            a1primeind=0; %1-1; % already includes -1
            a2primeind=ceil(maxindex/N_d)-1; % already includes -1 % divide by (N_d*1)
            maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            allind=dind+N_d*a2primeind+N_d*N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
            Policy(curraindex,N_j)=shiftdim(maxindexfix+N_d*(loweredge(allind)-1),1);
        end

    end

    Vunderbar=Vhat; % terminal period: no continuation

else
    % Using V_Jplus1
    EV=reshape(vfoptions.V_Jplus1,[N_a,1]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    Vunderbar=zeros(N_a,N_j,'gpuArray');

    DiscountedEV=beta0beta*reshape(EV,[1,N_a1,N_a2,1,1]); % will autoexand d in 1st-dim

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_noz(ReturnFn, n_d, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, ReturnFnParamsVec,1,0);

    entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

    % size(ReturnMatrix_ii) % (d, aprime, a,z)
    % [n_d,n_a,vfoptions.level1n,n_z]

    % First, we want a1prime conditional on (d,1,a2prime,a,z)
    [~,maxindex1]=max(entireRHS_ii,[],2);

    % Now, get and store the full (d,aprime)
    [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2]),[],1);
    % Store
    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
    Vhat(curraindex,N_j)=shiftdim(Vtempii,1);
    Policy(curraindex,N_j)=shiftdim(maxindex2,1);

    % Attempt for improved version
    maxgap=squeeze(max(max(max(maxindex1(:,1,:,2:end,:)-maxindex1(:,1,:,1:end-1,:),[],5),[],3),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,:,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1-by-n_a2-by-1-by-n_a2
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_noz(ReturnFn, n_d, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, ReturnFnParamsVec,2,0);
            aprime=repelem(a1primeindexes,1,1,1,level1iidiff(ii),1,1)+N_a1*a2Bind;
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprime,[N_d*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            Vhat(curraindex,N_j)=shiftdim(Vtempii,1);
            % maxindex needs to be reworked:
            %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
            dind=(rem(maxindex-1,N_d)+1);
            a1primeind=rem(ceil(maxindex/N_d)-1,maxgap(ii)+1)+1-1; % already includes -1
            a2primeind=ceil(maxindex/(N_d*(maxgap(ii)+1)))-1; % already includes -1
            maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            allind=dind+N_d*a2primeind+N_d*N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
            Policy(curraindex,N_j)=shiftdim(maxindexfix+N_d*(loweredge(allind)-1),1);
        else
            loweredge=maxindex1(:,1,:,ii,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_noz(ReturnFn, n_d, d_gridvals, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, ReturnFnParamsVec,2,0);
            aprime=repelem(loweredge,1,1,1,level1iidiff(ii),1,1)+N_a1*a2Bind;
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprime,[N_d*1*N_a2,level1iidiff(ii)*N_a2]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            Vhat(curraindex,N_j)=shiftdim(Vtempii,1);
            % maxindex needs to be reworked:
            %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
            dind=(rem(maxindex-1,N_d)+1);
            a1primeind=0; %1-1; % already includes -1
            a2primeind=ceil(maxindex/N_d)-1; % already includes -1 % divide by (N_d*1)
            maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            allind=dind+N_d*a2primeind+N_d*N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
            Policy(curraindex,N_j)=shiftdim(maxindexfix+N_d*(loweredge(allind)-1),1);
        end

    end

    % Sophisticated: re-evaluate continuation at chosen policy
    aprime_ind=ceil(Policy(:,N_j)/N_d);
    EV_at_policy=EV(aprime_ind);
    Vunderbar(:,N_j)=Vhat(:,N_j)+(beta-beta0beta)*EV_at_policy;

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
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    EV=Vunderbar(:,jj+1); % sophisticated continuation
    DiscountedEV=beta0beta*reshape(EV,[1,N_a1,N_a2,1,1]); % will autoexand d in 1st-dim

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_noz(ReturnFn, n_d, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, ReturnFnParamsVec,1,0);

    entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

    % size(ReturnMatrix_ii) % (d, aprime, a,z)
    % [n_d,n_a,vfoptions.level1n,n_z]

    % First, we want a1prime conditional on (d,1,a2prime,a,z)
    [~,maxindex1]=max(entireRHS_ii,[],2);

    % Now, get and store the full (d,aprime)
    [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2]),[],1);
    % Store
    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
    Vhat(curraindex,jj)=shiftdim(Vtempii,1);
    Policy(curraindex,jj)=shiftdim(maxindex2,1);

    % Attempt for improved version
    maxgap=squeeze(max(max(max(maxindex1(:,1,:,2:end,:)-maxindex1(:,1,:,1:end-1,:),[],5),[],3),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,:,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1-by-n_a2-by-1-by-n_a2
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_noz(ReturnFn, n_d, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, ReturnFnParamsVec,2,0);
            aprime=repelem(a1primeindexes,1,1,1,level1iidiff(ii),1,1)+N_a1*a2Bind;
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprime,[N_d*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            Vhat(curraindex,jj)=shiftdim(Vtempii,1);
            % maxindex needs to be reworked:
            %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
            dind=(rem(maxindex-1,N_d)+1);
            a1primeind=rem(ceil(maxindex/N_d)-1,maxgap(ii)+1)+1-1; % already includes -1
            a2primeind=ceil(maxindex/(N_d*(maxgap(ii)+1)))-1; % already includes -1
            maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            allind=dind+N_d*a2primeind+N_d*N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
            Policy(curraindex,jj)=shiftdim(maxindexfix+N_d*(loweredge(allind)-1),1);
        else
            loweredge=maxindex1(:,1,:,ii,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_noz(ReturnFn, n_d, d_gridvals, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, ReturnFnParamsVec,2,0);
            aprime=repelem(loweredge,1,1,1,level1iidiff(ii),1,1)+N_a1*a2Bind;
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprime,[N_d*1*N_a2,level1iidiff(ii)*N_a2]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            Vhat(curraindex,jj)=shiftdim(Vtempii,1);
            % maxindex needs to be reworked:
            %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
            dind=(rem(maxindex-1,N_d)+1);
            a1primeind=0; %1-1; % already includes -1
            a2primeind=ceil(maxindex/N_d)-1; % already includes -1 % divide by (N_d*1)
            maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            allind=dind+N_d*a2primeind+N_d*N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
            Policy(curraindex,jj)=shiftdim(maxindexfix+N_d*(loweredge(allind)-1),1);
        end

    end

    % Sophisticated: re-evaluate continuation at chosen policy
    aprime_ind=ceil(Policy(:,jj)/N_d);
    EV_at_policy=EV(aprime_ind);
    Vunderbar(:,jj)=Vhat(:,jj)+(beta-beta0beta)*EV_at_policy;

end

%%
Policy=shiftdim(Policy,-1);


end
