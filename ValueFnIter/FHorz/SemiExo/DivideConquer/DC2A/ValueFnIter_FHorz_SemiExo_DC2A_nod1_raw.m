function [V, Policy]=ValueFnIter_FHorz_SemiExo_DC2A_nod1_raw(n_d2,n_a,n_z,n_semiz, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);

V=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d,d2,aprime seperately
Policy=zeros(2,N_a,N_semiz*N_z,N_j,'gpuArray');


%%
special_n_d2=ones(1,length(n_d2));

n_a1=n_a(1);
n_a2=n_a(2:end);
N_a1=n_a1;
N_a2=n_a2;
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

% n-Monotonicity
level1ii=round(linspace(1,N_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% precompute
a2ind=gpuArray(0:1:N_a2-1); % already includes -1
bothzind=shiftdim(gpuArray(0:1:N_bothz-1),-1); % already includes -1
bothzBind=shiftdim(gpuArray(0:1:N_bothz-1),-3); % already includes -1

% Preallocate
V_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')

    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,1);

        %Calc the max and it's index
        [~,maxindex1]=max(ReturnMatrix_ii,[],1);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_bothz]),[],1);
        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
        V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
        Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_bothz
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,5,0);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                a2primeind=ceil(maxindex/(maxgap(ii)+1));
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*bothzind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            else
                loweredge=maxindex1(1,:,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,5,0);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=1;
                a2primeind=maxindex;
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*bothzind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            end
        end
    end
    % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
    V(:,:,N_j)=V_jj;
    Policy(1,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]); % This is the value of d that corresponds, make it this shape for addition just below
    Policy(2,:,:,N_j)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);

else
    % Using V_Jplus1
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);    % First, switch V_Jplus1 into Kron form

    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        % Note: By definition V_Jplus1 does not depend on d (only aprime)
        pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j)); % reverse order

        EV_d2=EV.*shiftdim(pi_bothz',-1);
        EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension
        DiscountedEV_d2=DiscountFactorParamsVec*reshape(EV_d2,[N_a1,N_a2,1,1,N_bothz]); % autoexpand (a,bothz)

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,1);

        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2;

        %Calc the max and it's index
        [~,maxindex1]=max(entireRHS_ii,[],1);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_bothz]),[],1);
        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
        V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
        Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_bothz
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,5,0);
                aprime=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*bothzBind; % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2(reshape(aprime,[(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_bothz]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                a2primeind=ceil(maxindex/(maxgap(ii)+1));
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*bothzind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            else
                loweredge=maxindex1(1,:,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,5,0);
                aprime=repelem(loweredge,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*bothzBind; % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2(reshape(aprime,[1*N_a2,level1iidiff(ii)*N_a2,N_bothz]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=1;
                a2primeind=maxindex;
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*bothzind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            end
        end
    end
    % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
    V(:,:,N_j)=V_jj;
    Policy(1,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]); % This is the value of d that corresponds, make it this shape for addition just below
    Policy(2,:,:,N_j)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);

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

    EV=V(:,:,jj+1);

    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj)); % reverse order

        EV_d2=EV.*shiftdim(pi_bothz',-1);
        EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension
        DiscountedEV_d2=DiscountFactorParamsVec*reshape(EV_d2,[N_a1,N_a2,1,1,N_bothz]); % autoexpand (a,bothz)

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,1,1);

        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2;

        %Calc the max and it's index
        [~,maxindex1]=max(entireRHS_ii,[],1);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_bothz]),[],1);
        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
        V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
        Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_bothz
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,5,0);
                aprime=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*bothzBind; % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2(reshape(aprime,[(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_bothz]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                a2primeind=ceil(maxindex/(maxgap(ii)+1));
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*bothzind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            else
                loweredge=maxindex1(1,:,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,5,0);
                aprime=repelem(loweredge,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*bothzBind; % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2(reshape(aprime,[1*N_a2,level1iidiff(ii)*N_a2,N_bothz]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=1;
                a2primeind=maxindex;
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*bothzind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            end
        end
    end
    % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
    V(:,:,jj)=V_jj;
    Policy(1,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]); % This is the value of d that corresponds, make it this shape for addition just below
    Policy(2,:,:,jj)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
end


end
