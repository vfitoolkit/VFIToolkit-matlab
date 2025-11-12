function [V,Policy]=ValueFnIter_FHorz_DC2B_nod_noz_e_lowmem_raw(n_a, n_e, N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% lowmem=loop over e
special_n_e=ones(1,length(n_e),'gpuArray');

N_a=prod(n_a);
N_e=prod(n_e);

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%%
N_a1=n_a(1);
N_a2=n_a(2);
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% n-Monotonicity
% vfoptions.level1n=7;
level1ii=round(linspace(1,N_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

pi_e_J=shiftdim(pi_e_J,-1);

% precompute
a2ind=gpuArray(0:1:N_a2-1); % already includes -1
% lowmem, so no longer need eind

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')

    for e_c=1:N_e
        e_vals=e_gridvals_J(e_c,:,N_j);
        % n-Monotonicity
        ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, special_n_e, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, e_vals, ReturnFnParamsVec,1);

        %Calc the max and it's index
        [~,maxindex1]=max(ReturnMatrix_ii_e,[],1);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_e,[N_a1*N_a2,vfoptions.level1n*N_a2]),[],1);
        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
        V(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
        Policy(curraindex,e_c,N_j)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(maxindex1(1,:,2:end,:)-maxindex1(1,:,1:end-1,:),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,:,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_e
                ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, special_n_e, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, e_vals, ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
                V(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                a2primeind=ceil(maxindex/(maxgap(ii)+1));
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_e
                Policy(curraindex,e_c,N_j)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            else
                loweredge=maxindex1(1,:,ii,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, special_n_e, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, e_vals, ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
                V(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=1;
                a2primeind=maxindex;
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_e
                Policy(curraindex,e_c,N_j)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            end
        end
    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_e]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    V_Jplus1=sum(V_Jplus1.*pi_e_J(1,:,N_j),2);
    DiscountedEV=DiscountFactorParamsVec*reshape(V_Jplus1,[N_a1,N_a2,1,1]);  % autoexpand (a)

    for e_c=1:N_e
        e_vals=e_gridvals_J(e_c,:,N_j);
        % n-Monotonicity
        ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, special_n_e, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, e_vals, ReturnFnParamsVec,1);

        entireRHS_ii=ReturnMatrix_ii_e+DiscountedEV;

        %Calc the max and it's index
        [~,maxindex1]=max(entireRHS_ii,[],1);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2]),[],1);
        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
        V(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
        Policy(curraindex,e_c,N_j)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(maxindex1(1,:,2:end,:)-maxindex1(1,:,1:end-1,:),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,:,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_e
                ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, special_n_e, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, e_vals, ReturnFnParamsVec,2);
                aprime=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1)+N_a1*a2ind; % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii_e+DiscountedEV(reshape(aprime,[(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                a2primeind=ceil(maxindex/(maxgap(ii)+1));
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_e
                Policy(curraindex,e_c,N_j)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            else
                loweredge=maxindex1(1,:,ii,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, special_n_e, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, e_vals, ReturnFnParamsVec,2);
                aprime=repelem(loweredge,1,1,level1iidiff(ii),1,1)+N_a1*a2ind; % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii_e+DiscountedEV(reshape(aprime,[1*N_a2,level1iidiff(ii)*N_a2]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=1;
                a2primeind=maxindex;
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_e
                Policy(curraindex,e_c,N_j)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            end
        end
    end
end


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
    end
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    EV=sum(V(:,:,jj+1).*pi_e_J(1,:,jj),2);
    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_a1,N_a2,1,1]);  % autoexpand (a)

    for e_c=1:N_e
        e_vals=e_gridvals_J(e_c,:,jj);
        % n-Monotonicity
        ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, special_n_e, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, e_vals, ReturnFnParamsVec,1);

        entireRHS_ii=ReturnMatrix_ii_e+DiscountedEV; % autofill e

        %Calc the max and it's index
        [~,maxindex1]=max(entireRHS_ii,[],1);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2]),[],1);
        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
        V(curraindex,e_c,jj)=shiftdim(Vtempii,1);
        Policy(curraindex,e_c,jj)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(maxindex1(1,:,2:end,:)-maxindex1(1,:,1:end-1,:),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,:,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_e
                ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, special_n_e, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, e_vals, ReturnFnParamsVec,2);
                aprime=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1)+N_a1*a2ind; % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii_e+DiscountedEV(reshape(aprime,[(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,e_c,jj)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                a2primeind=ceil(maxindex/(maxgap(ii)+1));
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_e
                Policy(curraindex,e_c,jj)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            else
                loweredge=maxindex1(1,:,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, special_n_e, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, e_vals, ReturnFnParamsVec,2);
                aprime=repelem(loweredge,1,1,level1iidiff(ii),1,1)+N_a1*a2ind; % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii_e+DiscountedEV(reshape(aprime,[1*N_a2,level1iidiff(ii)*N_a2]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,e_c,jj)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=1;
                a2primeind=maxindex;
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_e
                Policy(curraindex,e_c,jj)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            end
        end
    end
end





end
