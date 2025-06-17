function [V, Policy]=ValueFnIter_FHorz_DC2B_nod_lowmem_raw(n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% lowmem=loop over z
special_n_z=ones(1,length(n_z),'gpuArray');

N_a=prod(n_a);
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%%
N_a1=n_a(1);
N_a2=n_a(2);
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% n-Monotonicity
% vfoptions.level1n=7;
level1ii=round(linspace(1,N_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% lowmem so no longer need zind

%% j=N_j
% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    for z_c=1:N_z
        z_vals=z_gridvals_J(z_c,:,N_j);
        % n-Monotonicity
        ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, special_n_z, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_vals, ReturnFnParamsVec,1);

        % size(ReturnMatrix_ii) % (aprime, a,z)
        % [n_a,vfoptions.level1n,n_z]

        %Calc the max and it's index
        [~,maxindex1]=max(ReturnMatrix_ii_z,[],1);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_z,[N_a1*N_a2,vfoptions.level1n*N_a2]),[],1);
        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
        Policy(curraindex,z_c,N_j)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(maxindex1(1,:,2:end,:)-maxindex1(1,:,1:end-1,:),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,:,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_z
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_z
                ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, special_n_z, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii_z,[],1);
                V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                a2primeind=ceil(maxindex/(maxgap(ii)+1));
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=a2primeind+N_a2*a2ind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_z
                Policy(curraindex,z_c,N_j)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            else
                loweredge=maxindex1(1,:,ii,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, special_n_z, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii_z,[],1);
                V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=1;
                a2primeind=maxindex;
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=a2primeind+N_a2*a2ind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_z
                Policy(curraindex,z_c,N_j)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            end
        end
    end

else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_z]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    EV=V_Jplus1.*shiftdim(pi_z_J(:,:,N_j)',-1);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension
    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_a1,N_a2,1,1,N_z]);  % autoexpand (a,z)

    for z_c=1:N_z
        z_vals=z_gridvals_J(z_c,:,N_j);
        DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);
        % n-Monotonicity
        ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, special_n_z, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_vals, ReturnFnParamsVec,1);

        % size(ReturnMatrix_ii) % (aprime, a,z)
        % [n_a,vfoptions.level1n,n_z]

        entireRHS_ii=ReturnMatrix_ii_z+DiscountedEV_z;

        %Calc the max and it's index
        [~,maxindex1]=max(entireRHS_ii,[],1);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2]),[],1);
        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
        Policy(curraindex,z_c,N_j)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(maxindex1(1,:,2:end,:)-maxindex1(1,:,1:end-1,:),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,:,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_z
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_z
                ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, special_n_z, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, ReturnFnParamsVec,2);
                aprime=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1)+N_a1*(0:1:N_a2-1); % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii_z+DiscountedEV_z(reshape(aprime,[(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                a2primeind=ceil(maxindex/(maxgap(ii)+1));
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=a2primeind+N_a2*a2ind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_z
                Policy(curraindex,z_c,N_j)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            else
                loweredge=maxindex1(1,:,ii,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, special_n_z, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, ReturnFnParamsVec,2);
                aprime=repelem(loweredge,1,1,level1iidiff(ii),1,1)+N_a1*(0:1:N_a2-1); % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii_z+DiscountedEV_z(reshape(aprime,[1*N_a2,level1iidiff(ii)*N_a2]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=1;
                a2primeind=maxindex;
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=a2primeind+N_a2*a2ind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_z
                Policy(curraindex,z_c,N_j)=shiftdim(maxindexfix+loweredge(allind)-1,1);
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
    
    VKronNext_j=V(:,:,jj+1);
    
    % Use sparse for a few lines until sum over zprime
    EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension
    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_a1,N_a2,1,1,N_z]);  % autoexpand (a,z)
        
    for z_c=1:N_z
        z_vals=z_gridvals_J(z_c,:,jj);
        DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);
        % n-Monotonicity
        ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, special_n_z, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_vals, ReturnFnParamsVec,1);

        % size(ReturnMatrix_ii) % (aprime, a,z)
        % [n_a,vfoptions.level1n,n_z]

        entireRHS_ii=ReturnMatrix_ii_z+DiscountedEV_z;

        %Calc the max and it's index
        [~,maxindex1]=max(entireRHS_ii,[],1);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2]),[],1);
        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V(curraindex,z_c,jj)=shiftdim(Vtempii,1);
        Policy(curraindex,z_c,jj)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(maxindex1(1,:,2:end,:)-maxindex1(1,:,1:end-1,:),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,:,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_z
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_z
                ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, special_n_z, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, ReturnFnParamsVec,2);
                aprime=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1)+N_a1*(0:1:N_a2-1); % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii_z+DiscountedEV(reshape(aprime,[(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,z_c,jj)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                a2primeind=ceil(maxindex/(maxgap(ii)+1));
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=a2primeind+N_a2*a2ind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_z
                Policy(curraindex,z_c,jj)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            else
                loweredge=maxindex1(1,:,ii,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, special_n_z, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, ReturnFnParamsVec,2);
                aprime=repelem(loweredge,1,1,level1iidiff(ii),1,1)+N_a1*(0:1:N_a2-1); % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii_z+DiscountedEV(reshape(aprime,[1*N_a2,level1iidiff(ii)*N_a2]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,z_c,jj)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=1;
                a2primeind=maxindex;
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=a2primeind+N_a2*a2ind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_z
                Policy(curraindex,z_c,jj)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            end
        end
    end
end


end
