function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_DC2A_nod1_e_raw(n_d2,n_a,n_z,n_semiz,n_e, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive QH + SemiExo + DC2A (divide-and-conquer in the first endo state, a2 enumerated in full), no d1, with z, with e.
%
% Naive: Valt_j   = max u + beta*E[V_{j+1}]         (used as EVsource)
%        Vtilde_j = max u + beta_0*beta*E[V_{j+1}]  (agent's choice)

n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

Valt=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
Vtilde=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d,d2,aprime seperately
Policy=zeros(2,N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
Policyalt=zeros(2,N_a,N_semiz*N_z,N_e,N_j,'gpuArray'); % exponential discounter optimal (d2, aprime)


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
eind=shiftdim(gpuArray(0:1:N_e-1),-2); % already includes -1
bothzBind=shiftdim(gpuArray(0:1:N_bothz-1),-3); % already includes -1

% lowmemory: which shocks are looped vs vectorised (spec: =1 loop e; =2 outer z/inner e, vec semiz; =3 joint bothz/inner e)
if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory==2
    special_n_z=ones(1,length(n_z));
    special_n_e=ones(1,length(n_e));
    semizind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % semiz-block analogue of bothzind (L2)
    semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-3); % semiz-block analogue of bothzBind (L2)
elseif vfoptions.lowmemory==3
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
    special_n_e=ones(1,length(n_e));
end

pi_e_J=shiftdim(pi_e_J,-2); % Move to third dimension

% Preallocate
V_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
Vtilde_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
Policy_V_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')

  if vfoptions.lowmemory==0
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,1);

        %Calc the max and it's index
        [~,maxindex1]=max(ReturnMatrix_ii,[],1);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_bothz,N_e]),[],1);
        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
        V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
        Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(max(maxindex1(1,:,2:end,:,:,:)-maxindex1(1,:,1:end-1,:,:,:),[],6),[],5),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,:,ii,:,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,5,0);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                a2primeind=ceil(maxindex/(maxgap(ii)+1));
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*bothzind+N_a2*N_a2*N_bothz*eind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            else
                loweredge=maxindex1(1,:,ii,:,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,5,0);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=1;
                a2primeind=maxindex;
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*bothzind+N_a2*N_a2*N_bothz*eind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            end
        end
    end

  elseif vfoptions.lowmemory==1 % loop e, vectorise bothz
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,1);
            [~,maxindex1]=max(ReturnMatrix_ii,[],1);
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_bothz]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
            V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex2,1);
            maxgap=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,5,0);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                    a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                    a2primeind=ceil(maxindex/(maxgap(ii)+1));
                    maxindexfix=a1primeind+N_a1*(a2primeind-1);
                    allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*bothzind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz
                    Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                else
                    loweredge=maxindex1(1,:,ii,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,5,0);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                    a1primeind=1;
                    a2primeind=maxindex;
                    maxindexfix=a1primeind+N_a1*(a2primeind-1);
                    allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*bothzind;
                    Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                end
            end
        end
    end

  elseif vfoptions.lowmemory==2 % outer z / inner e, vectorise semiz
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_valblock, e_val, ReturnFnParamsVec,1,1);
                [~,maxindex1]=max(ReturnMatrix_ii,[],1);
                [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_semiz]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
                V_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindex2,1);
                maxgap=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap(ii));
                        aprimeindexes=loweredge+(0:1:maxgap(ii))';
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_valblock, e_val, ReturnFnParamsVec,5,0);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                        V_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                        a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                        a2primeind=ceil(maxindex/(maxgap(ii)+1));
                        maxindexfix=a1primeind+N_a1*(a2primeind-1);
                        allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*semizind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_semiz
                        Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                    else
                        loweredge=maxindex1(1,:,ii,:,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_valblock, e_val, ReturnFnParamsVec,5,0);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                        V_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                        a1primeind=1;
                        a2primeind=maxindex;
                        maxindexfix=a1primeind+N_a1*(a2primeind-1);
                        allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*semizind;
                        Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                    end
                end
            end
        end
    end

  elseif vfoptions.lowmemory==3 % joint bothz / inner e
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_val, e_val, ReturnFnParamsVec,1,1);
                [~,maxindex1]=max(ReturnMatrix_ii,[],1);
                [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_a1*N_a2,vfoptions.level1n*N_a2]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
                V_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindex2,1);
                maxgap=squeeze(max(max(maxindex1(1,:,2:end,:)-maxindex1(1,:,1:end-1,:),[],4),[],2));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(1,:,ii,:),N_a1-maxgap(ii));
                        aprimeindexes=loweredge+(0:1:maxgap(ii))';
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, e_val, ReturnFnParamsVec,5,0);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                        V_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                        a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                        a2primeind=ceil(maxindex/(maxgap(ii)+1));
                        maxindexfix=a1primeind+N_a1*(a2primeind-1);
                        allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is 1-by-n_a2-by-1-by-n_a2
                        Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                    else
                        loweredge=maxindex1(1,:,ii,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, e_val, ReturnFnParamsVec,5,0);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                        V_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                        a1primeind=1;
                        a2primeind=maxindex;
                        maxindexfix=a1primeind+N_a1*(a2primeind-1);
                        allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii));
                        Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                    end
                end
            end
        end
    end
  end
    % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
    Valt(:,:,:,N_j)=V_jj;
    Vtilde(:,:,:,N_j)=V_jj; % terminal period: QH and exponential discounter coincide
    Policy(1,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
    Policy(2,:,:,:,N_j)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
    Policyalt(:,:,:,:,N_j)=Policy(:,:,:,:,N_j); % terminal period: QH and exponential discounter coincide

else
    % Using V_Jplus1
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*pi_e_J(1,1,:,N_j+1),3); % First, switch V_Jplus1 into Kron form and integrate over e'

    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        % Note: By definition V_Jplus1 does not depend on d (only aprime)
        pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j)); % reverse order

      if vfoptions.lowmemory==0
        EV_d2=EV.*shiftdim(pi_bothz',-1);
        EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension
        EV_d2=reshape(EV_d2,[N_a1,N_a2,1,1,N_bothz]); % autoexpand (a,bothz,e)
        DiscountedEV_d2=beta*EV_d2;
        DiscountedEV_d2_tilde=beta0beta*EV_d2;

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,1);

        %% Valt (beta)
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2;

        %Calc the max and it's index
        [~,maxindex1]=max(entireRHS_ii,[],1);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_bothz,N_e]),[],1);
        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
        V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
        Policy_V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap_V=squeeze(max(max(max(max(maxindex1(1,:,2:end,:,:,:)-maxindex1(1,:,1:end-1,:,:,:),[],6),[],5),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
            if maxgap_V(ii)>0
                loweredge=min(maxindex1(1,:,ii,:,:,:),N_a1-maxgap_V(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap_V(ii) points
                % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                % aprime possibilities are (maxgap_V(ii)+1)-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,5,0);
                aprime=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1,1)+N_a1*a2ind+N_a*bothzBind; % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2(reshape(aprime,[(maxgap_V(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_bothz,N_e]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap_V(ii)+1, but needs to be after N_a1'
                a1primeind=rem(maxindex-1,maxgap_V(ii)+1)+1;
                a2primeind=ceil(maxindex/(maxgap_V(ii)+1));
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap_V(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*bothzind+N_a2*N_a2*N_bothz*eind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                Policy_V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            else
                loweredge=maxindex1(1,:,ii,:,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,5,0);
                aprime=repelem(loweredge,1,1,level1iidiff(ii),1,1,1)+N_a1*a2ind+N_a*bothzBind; % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2(reshape(aprime,[1*N_a2,level1iidiff(ii)*N_a2,N_bothz,N_e]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap_V(ii)+1, but needs to be after N_a1'
                a1primeind=1;
                a2primeind=maxindex;
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap_V(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*bothzind+N_a2*N_a2*N_bothz*eind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                Policy_V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            end
        end

        %% Vtilde (beta0*beta)
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2_tilde;

        %Calc the max and it's index
        [~,maxindex1]=max(entireRHS_ii,[],1);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_bothz,N_e]),[],1);
        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
        Vtilde_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
        Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(max(maxindex1(1,:,2:end,:,:,:)-maxindex1(1,:,1:end-1,:,:,:),[],6),[],5),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,:,ii,:,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,5,0);
                aprime=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1,1)+N_a1*a2ind+N_a*bothzBind; % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2_tilde(reshape(aprime,[(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_bothz,N_e]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vtilde_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                a2primeind=ceil(maxindex/(maxgap(ii)+1));
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*bothzind+N_a2*N_a2*N_bothz*eind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            else
                loweredge=maxindex1(1,:,ii,:,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,5,0);
                aprime=repelem(loweredge,1,1,level1iidiff(ii),1,1,1)+N_a1*a2ind+N_a*bothzBind; % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2_tilde(reshape(aprime,[1*N_a2,level1iidiff(ii)*N_a2,N_bothz,N_e]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vtilde_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=1;
                a2primeind=maxindex;
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*bothzind+N_a2*N_a2*N_bothz*eind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            end
        end

      elseif vfoptions.lowmemory==1 % loop e, vectorise bothz
        EV_d2=EV.*shiftdim(pi_bothz',-1);
        EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension
        EV_d2=reshape(EV_d2,[N_a1,N_a2,1,1,N_bothz]); % autoexpand (a,bothz)
        DiscountedEV_d2=beta*EV_d2;
        DiscountedEV_d2_tilde=beta0beta*EV_d2;
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,1);
            %% Valt (beta)
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2;
            [~,maxindex1]=max(entireRHS_ii,[],1);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_bothz]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
            V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
            Policy_V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex2,1);
            maxgap_V=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap_V(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap_V(ii) points
                    % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz
                    aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                    % aprime possibilities are (maxgap_V(ii)+1)-n_a2-by-1-by-n_a2-by-n_bothz
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,5,0);
                    aprime=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*bothzBind; % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2(reshape(aprime,[(maxgap_V(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_bothz]));
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                    a1primeind=rem(maxindex-1,maxgap_V(ii)+1)+1;
                    a2primeind=ceil(maxindex/(maxgap_V(ii)+1));
                    maxindexfix=a1primeind+N_a1*(a2primeind-1);
                    allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*bothzind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz
                    Policy_V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                else
                    loweredge=maxindex1(1,:,ii,:,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,5,0);
                    aprime=repelem(loweredge,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*bothzBind; % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2(reshape(aprime,[1*N_a2,level1iidiff(ii)*N_a2,N_bothz]));
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                    a1primeind=1;
                    a2primeind=maxindex;
                    maxindexfix=a1primeind+N_a1*(a2primeind-1);
                    allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*bothzind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz
                    Policy_V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                end
            end

            %% Vtilde (beta0*beta)
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2_tilde;
            [~,maxindex1]=max(entireRHS_ii,[],1);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_bothz]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
            Vtilde_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex2,1);
            maxgap=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_bothz
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,5,0);
                    aprime=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*bothzBind; % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2_tilde(reshape(aprime,[(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_bothz]));
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vtilde_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                    a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                    a2primeind=ceil(maxindex/(maxgap(ii)+1));
                    maxindexfix=a1primeind+N_a1*(a2primeind-1);
                    allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*bothzind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz
                    Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                else
                    loweredge=maxindex1(1,:,ii,:,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,5,0);
                    aprime=repelem(loweredge,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*bothzBind; % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2_tilde(reshape(aprime,[1*N_a2,level1iidiff(ii)*N_a2,N_bothz]));
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vtilde_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                    a1primeind=1;
                    a2primeind=maxindex;
                    maxindexfix=a1primeind+N_a1*(a2primeind-1);
                    allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*bothzind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz
                    Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                end
            end
        end

      elseif vfoptions.lowmemory==2 % outer z / inner e, vectorise semiz
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,N_j);
            EV_d2z=EV.*shiftdim(pi_bothz(semizblock,:)',-1);
            EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2z=sum(EV_d2z,2); % sum over z', leaving a singular second dimension
            EV_d2z=reshape(EV_d2z,[N_a1,N_a2,1,1,N_semiz]); % autoexpand (a,semiz)
            DiscountedEV_d2z=beta*EV_d2z;
            DiscountedEV_d2z_tilde=beta0beta*EV_d2z;
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                % n-Monotonicity
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_valblock, e_val, ReturnFnParamsVec,1,1);
                %% Valt (beta)
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2z;
                [~,maxindex1]=max(entireRHS_ii,[],1);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_semiz]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
                V_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_V_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindex2,1);
                maxgap_V=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                    if maxgap_V(ii)>0
                        loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap_V(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap_V(ii) points
                        % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_semiz
                        aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                        % aprime possibilities are (maxgap_V(ii)+1)-n_a2-by-1-by-n_a2-by-n_semiz
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_valblock, e_val, ReturnFnParamsVec,5,0);
                        aprime=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*semizBind; % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2z(reshape(aprime,[(maxgap_V(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_semiz]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                        a1primeind=rem(maxindex-1,maxgap_V(ii)+1)+1;
                        a2primeind=ceil(maxindex/(maxgap_V(ii)+1));
                        maxindexfix=a1primeind+N_a1*(a2primeind-1);
                        allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*semizind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_semiz
                        Policy_V_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                    else
                        loweredge=maxindex1(1,:,ii,:,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_valblock, e_val, ReturnFnParamsVec,5,0);
                        aprime=repelem(loweredge,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*semizBind; % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2z(reshape(aprime,[1*N_a2,level1iidiff(ii)*N_a2,N_semiz]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                        a1primeind=1;
                        a2primeind=maxindex;
                        maxindexfix=a1primeind+N_a1*(a2primeind-1);
                        allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*semizind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_semiz
                        Policy_V_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                    end
                end

                %% Vtilde (beta0*beta)
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2z_tilde;
                [~,maxindex1]=max(entireRHS_ii,[],1);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_semiz]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
                Vtilde_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindex2,1);
                maxgap=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_semiz
                        aprimeindexes=loweredge+(0:1:maxgap(ii))';
                        % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_semiz
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_valblock, e_val, ReturnFnParamsVec,5,0);
                        aprime=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*semizBind; % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2z_tilde(reshape(aprime,[(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_semiz]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vtilde_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                        a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                        a2primeind=ceil(maxindex/(maxgap(ii)+1));
                        maxindexfix=a1primeind+N_a1*(a2primeind-1);
                        allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*semizind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_semiz
                        Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                    else
                        loweredge=maxindex1(1,:,ii,:,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_valblock, e_val, ReturnFnParamsVec,5,0);
                        aprime=repelem(loweredge,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*semizBind; % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2z_tilde(reshape(aprime,[1*N_a2,level1iidiff(ii)*N_a2,N_semiz]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vtilde_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                        a1primeind=1;
                        a2primeind=maxindex;
                        maxindexfix=a1primeind+N_a1*(a2primeind-1);
                        allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*semizind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_semiz
                        Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                    end
                end
            end
        end

      elseif vfoptions.lowmemory==3 % joint bothz / inner e
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            EV_d2z=EV.*shiftdim(pi_bothz(z_c,:)',-1);
            EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2z=sum(EV_d2z,2); % sum over z', leaving a singular second dimension
            EV_d2z=reshape(EV_d2z,[N_a1,N_a2]); % autoexpand (a)
            DiscountedEV_d2z=beta*EV_d2z;
            DiscountedEV_d2z_tilde=beta0beta*EV_d2z;
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                % n-Monotonicity
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_val, e_val, ReturnFnParamsVec,1,1);
                %% Valt (beta)
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2z;
                [~,maxindex1]=max(entireRHS_ii,[],1);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
                V_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_V_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindex2,1);
                maxgap_V=squeeze(max(max(maxindex1(1,:,2:end,:)-maxindex1(1,:,1:end-1,:),[],4),[],2));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                    if maxgap_V(ii)>0
                        loweredge=min(maxindex1(1,:,ii,:),N_a1-maxgap_V(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap_V(ii) points
                        % loweredge is 1-by-n_a2-by-1-by-n_a2
                        aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                        % aprime possibilities are (maxgap_V(ii)+1)-n_a2-by-1-by-n_a2
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, e_val, ReturnFnParamsVec,5,0);
                        aprime=repelem(aprimeindexes,1,1,level1iidiff(ii),1)+N_a1*a2ind; % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2z(reshape(aprime,[(maxgap_V(ii)+1)*N_a2,level1iidiff(ii)*N_a2]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                        a1primeind=rem(maxindex-1,maxgap_V(ii)+1)+1;
                        a2primeind=ceil(maxindex/(maxgap_V(ii)+1));
                        maxindexfix=a1primeind+N_a1*(a2primeind-1);
                        allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is 1-by-n_a2-by-1-by-n_a2
                        Policy_V_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                    else
                        loweredge=maxindex1(1,:,ii,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, e_val, ReturnFnParamsVec,5,0);
                        aprime=repelem(loweredge,1,1,level1iidiff(ii),1)+N_a1*a2ind; % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2z(reshape(aprime,[1*N_a2,level1iidiff(ii)*N_a2]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                        a1primeind=1;
                        a2primeind=maxindex;
                        maxindexfix=a1primeind+N_a1*(a2primeind-1);
                        allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is 1-by-n_a2-by-1-by-n_a2
                        Policy_V_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                    end
                end

                %% Vtilde (beta0*beta)
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2z_tilde;
                [~,maxindex1]=max(entireRHS_ii,[],1);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
                Vtilde_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindex2,1);
                maxgap=squeeze(max(max(maxindex1(1,:,2:end,:)-maxindex1(1,:,1:end-1,:),[],4),[],2));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(1,:,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is 1-by-n_a2-by-1-by-n_a2
                        aprimeindexes=loweredge+(0:1:maxgap(ii))';
                        % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, e_val, ReturnFnParamsVec,5,0);
                        aprime=repelem(aprimeindexes,1,1,level1iidiff(ii),1)+N_a1*a2ind; % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2z_tilde(reshape(aprime,[(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vtilde_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                        a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                        a2primeind=ceil(maxindex/(maxgap(ii)+1));
                        maxindexfix=a1primeind+N_a1*(a2primeind-1);
                        allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is 1-by-n_a2-by-1-by-n_a2
                        Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                    else
                        loweredge=maxindex1(1,:,ii,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, e_val, ReturnFnParamsVec,5,0);
                        aprime=repelem(loweredge,1,1,level1iidiff(ii),1)+N_a1*a2ind; % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2z_tilde(reshape(aprime,[1*N_a2,level1iidiff(ii)*N_a2]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vtilde_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                        a1primeind=1;
                        a2primeind=maxindex;
                        maxindexfix=a1primeind+N_a1*(a2primeind-1);
                        allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is 1-by-n_a2-by-1-by-n_a2
                        Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                    end
                end
            end
        end
      end
    end
    % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    [V1_jj,maxindex]=max(Vtilde_ford2_jj,[],4); % max over d2
    Vtilde(:,:,:,N_j)=V1_jj;
    Policy(1,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
    maxindex_lin=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
    Policy(2,:,:,:,N_j)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex_lin-1)),[1,N_a,N_semiz*N_z,N_e]);
    % Valt at exponential discounter optimum (full max over d2 and aprime)
    [V_jj,maxindexalt_d2]=max(V_ford2_jj,[],4); % max over d2
    Valt(:,:,:,N_j)=V_jj;
    Policyalt(1,:,:,:,N_j)=shiftdim(maxindexalt_d2,-1); % d2 is just maxindexalt_d2
    maxindexalt_lin=reshape(maxindexalt_d2,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
    Policyalt(2,:,:,:,N_j)=reshape(Policy_V_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindexalt_lin-1)),[1,N_a,N_semiz*N_z,N_e]);

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

    EV=sum(Valt(:,:,:,jj+1).*pi_e_J(1,1,:,jj+1),3); % integrate over e'

    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj)); % reverse order

      if vfoptions.lowmemory==0
        EV_d2=EV.*shiftdim(pi_bothz',-1);
        EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension
        EV_d2=reshape(EV_d2,[N_a1,N_a2,1,1,N_bothz]); % autoexpand (a,bothz,e)
        DiscountedEV_d2=beta*EV_d2;
        DiscountedEV_d2_tilde=beta0beta*EV_d2;

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,1);

        %% Valt (beta)
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2;

        %Calc the max and it's index
        [~,maxindex1]=max(entireRHS_ii,[],1);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_bothz,N_e]),[],1);
        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
        V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
        Policy_V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap_V=squeeze(max(max(max(max(maxindex1(1,:,2:end,:,:,:)-maxindex1(1,:,1:end-1,:,:,:),[],6),[],5),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
            if maxgap_V(ii)>0
                loweredge=min(maxindex1(1,:,ii,:,:,:),N_a1-maxgap_V(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap_V(ii) points
                % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                % aprime possibilities are (maxgap_V(ii)+1)-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,5,0);
                aprime=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1,1)+N_a1*a2ind+N_a*bothzBind; % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2(reshape(aprime,[(maxgap_V(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_bothz,N_e]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap_V(ii)+1, but needs to be after N_a1'
                a1primeind=rem(maxindex-1,maxgap_V(ii)+1)+1;
                a2primeind=ceil(maxindex/(maxgap_V(ii)+1));
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap_V(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*bothzind+N_a2*N_a2*N_bothz*eind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                Policy_V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            else
                loweredge=maxindex1(1,:,ii,:,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,5,0);
                aprime=repelem(loweredge,1,1,level1iidiff(ii),1,1,1)+N_a1*a2ind+N_a*bothzBind; % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2(reshape(aprime,[1*N_a2,level1iidiff(ii)*N_a2,N_bothz,N_e]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap_V(ii)+1, but needs to be after N_a1'
                a1primeind=1;
                a2primeind=maxindex;
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap_V(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*bothzind+N_a2*N_a2*N_bothz*eind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                Policy_V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            end
        end

        %% Vtilde (beta0*beta)
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2_tilde;

        %Calc the max and it's index
        [~,maxindex1]=max(entireRHS_ii,[],1);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_bothz,N_e]),[],1);
        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
        Vtilde_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
        Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(max(maxindex1(1,:,2:end,:,:,:)-maxindex1(1,:,1:end-1,:,:,:),[],6),[],5),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,:,ii,:,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,5,0);
                aprime=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1,1)+N_a1*a2ind+N_a*bothzBind; % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2_tilde(reshape(aprime,[(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_bothz,N_e]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vtilde_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                a2primeind=ceil(maxindex/(maxgap(ii)+1));
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*bothzind+N_a2*N_a2*N_bothz*eind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            else
                loweredge=maxindex1(1,:,ii,:,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,5,0);
                aprime=repelem(loweredge,1,1,level1iidiff(ii),1,1,1)+N_a1*a2ind+N_a*bothzBind; % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2_tilde(reshape(aprime,[1*N_a2,level1iidiff(ii)*N_a2,N_bothz,N_e]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vtilde_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=1;
                a2primeind=maxindex;
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*bothzind+N_a2*N_a2*N_bothz*eind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            end
        end

      elseif vfoptions.lowmemory==1 % loop e, vectorise bothz
        EV_d2=EV.*shiftdim(pi_bothz',-1);
        EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension
        EV_d2=reshape(EV_d2,[N_a1,N_a2,1,1,N_bothz]); % autoexpand (a,bothz)
        DiscountedEV_d2=beta*EV_d2;
        DiscountedEV_d2_tilde=beta0beta*EV_d2;
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1,1);
            %% Valt (beta)
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2;
            [~,maxindex1]=max(entireRHS_ii,[],1);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_bothz]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
            V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
            Policy_V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex2,1);
            maxgap_V=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap_V(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap_V(ii) points
                    % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz
                    aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                    % aprime possibilities are (maxgap_V(ii)+1)-n_a2-by-1-by-n_a2-by-n_bothz
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,5,0);
                    aprime=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*bothzBind; % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2(reshape(aprime,[(maxgap_V(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_bothz]));
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                    a1primeind=rem(maxindex-1,maxgap_V(ii)+1)+1;
                    a2primeind=ceil(maxindex/(maxgap_V(ii)+1));
                    maxindexfix=a1primeind+N_a1*(a2primeind-1);
                    allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*bothzind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz
                    Policy_V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                else
                    loweredge=maxindex1(1,:,ii,:,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,5,0);
                    aprime=repelem(loweredge,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*bothzBind; % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2(reshape(aprime,[1*N_a2,level1iidiff(ii)*N_a2,N_bothz]));
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                    a1primeind=1;
                    a2primeind=maxindex;
                    maxindexfix=a1primeind+N_a1*(a2primeind-1);
                    allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*bothzind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz
                    Policy_V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                end
            end

            %% Vtilde (beta0*beta)
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2_tilde;
            [~,maxindex1]=max(entireRHS_ii,[],1);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_bothz]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
            Vtilde_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex2,1);
            maxgap=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_bothz
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,5,0);
                    aprime=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*bothzBind; % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2_tilde(reshape(aprime,[(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_bothz]));
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vtilde_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                    a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                    a2primeind=ceil(maxindex/(maxgap(ii)+1));
                    maxindexfix=a1primeind+N_a1*(a2primeind-1);
                    allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*bothzind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz
                    Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                else
                    loweredge=maxindex1(1,:,ii,:,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,5,0);
                    aprime=repelem(loweredge,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*bothzBind; % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2_tilde(reshape(aprime,[1*N_a2,level1iidiff(ii)*N_a2,N_bothz]));
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vtilde_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                    a1primeind=1;
                    a2primeind=maxindex;
                    maxindexfix=a1primeind+N_a1*(a2primeind-1);
                    allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*bothzind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_bothz
                    Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                end
            end
        end

      elseif vfoptions.lowmemory==2 % outer z / inner e, vectorise semiz
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,jj);
            EV_d2z=EV.*shiftdim(pi_bothz(semizblock,:)',-1);
            EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2z=sum(EV_d2z,2); % sum over z', leaving a singular second dimension
            EV_d2z=reshape(EV_d2z,[N_a1,N_a2,1,1,N_semiz]); % autoexpand (a,semiz)
            DiscountedEV_d2z=beta*EV_d2z;
            DiscountedEV_d2z_tilde=beta0beta*EV_d2z;
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                % n-Monotonicity
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_valblock, e_val, ReturnFnParamsVec,1,1);
                %% Valt (beta)
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2z;
                [~,maxindex1]=max(entireRHS_ii,[],1);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_semiz]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
                V_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_V_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindex2,1);
                maxgap_V=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                    if maxgap_V(ii)>0
                        loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap_V(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap_V(ii) points
                        % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_semiz
                        aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                        % aprime possibilities are (maxgap_V(ii)+1)-n_a2-by-1-by-n_a2-by-n_semiz
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_valblock, e_val, ReturnFnParamsVec,5,0);
                        aprime=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*semizBind; % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2z(reshape(aprime,[(maxgap_V(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_semiz]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                        a1primeind=rem(maxindex-1,maxgap_V(ii)+1)+1;
                        a2primeind=ceil(maxindex/(maxgap_V(ii)+1));
                        maxindexfix=a1primeind+N_a1*(a2primeind-1);
                        allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*semizind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_semiz
                        Policy_V_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                    else
                        loweredge=maxindex1(1,:,ii,:,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_valblock, e_val, ReturnFnParamsVec,5,0);
                        aprime=repelem(loweredge,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*semizBind; % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2z(reshape(aprime,[1*N_a2,level1iidiff(ii)*N_a2,N_semiz]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                        a1primeind=1;
                        a2primeind=maxindex;
                        maxindexfix=a1primeind+N_a1*(a2primeind-1);
                        allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*semizind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_semiz
                        Policy_V_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                    end
                end

                %% Vtilde (beta0*beta)
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2z_tilde;
                [~,maxindex1]=max(entireRHS_ii,[],1);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_semiz]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
                Vtilde_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindex2,1);
                maxgap=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_semiz
                        aprimeindexes=loweredge+(0:1:maxgap(ii))';
                        % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_semiz
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_valblock, e_val, ReturnFnParamsVec,5,0);
                        aprime=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*semizBind; % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2z_tilde(reshape(aprime,[(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_semiz]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vtilde_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                        a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                        a2primeind=ceil(maxindex/(maxgap(ii)+1));
                        maxindexfix=a1primeind+N_a1*(a2primeind-1);
                        allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*semizind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_semiz
                        Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                    else
                        loweredge=maxindex1(1,:,ii,:,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_valblock, e_val, ReturnFnParamsVec,5,0);
                        aprime=repelem(loweredge,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*semizBind; % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2z_tilde(reshape(aprime,[1*N_a2,level1iidiff(ii)*N_a2,N_semiz]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vtilde_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                        a1primeind=1;
                        a2primeind=maxindex;
                        maxindexfix=a1primeind+N_a1*(a2primeind-1);
                        allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*semizind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_semiz
                        Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                    end
                end
            end
        end

      elseif vfoptions.lowmemory==3 % joint bothz / inner e
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,jj);
            EV_d2z=EV.*shiftdim(pi_bothz(z_c,:)',-1);
            EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2z=sum(EV_d2z,2); % sum over z', leaving a singular second dimension
            EV_d2z=reshape(EV_d2z,[N_a1,N_a2]); % autoexpand (a)
            DiscountedEV_d2z=beta*EV_d2z;
            DiscountedEV_d2z_tilde=beta0beta*EV_d2z;
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                % n-Monotonicity
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_val, e_val, ReturnFnParamsVec,1,1);
                %% Valt (beta)
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2z;
                [~,maxindex1]=max(entireRHS_ii,[],1);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
                V_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_V_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindex2,1);
                maxgap_V=squeeze(max(max(maxindex1(1,:,2:end,:)-maxindex1(1,:,1:end-1,:),[],4),[],2));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                    if maxgap_V(ii)>0
                        loweredge=min(maxindex1(1,:,ii,:),N_a1-maxgap_V(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap_V(ii) points
                        % loweredge is 1-by-n_a2-by-1-by-n_a2
                        aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                        % aprime possibilities are (maxgap_V(ii)+1)-n_a2-by-1-by-n_a2
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, e_val, ReturnFnParamsVec,5,0);
                        aprime=repelem(aprimeindexes,1,1,level1iidiff(ii),1)+N_a1*a2ind; % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2z(reshape(aprime,[(maxgap_V(ii)+1)*N_a2,level1iidiff(ii)*N_a2]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                        a1primeind=rem(maxindex-1,maxgap_V(ii)+1)+1;
                        a2primeind=ceil(maxindex/(maxgap_V(ii)+1));
                        maxindexfix=a1primeind+N_a1*(a2primeind-1);
                        allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is 1-by-n_a2-by-1-by-n_a2
                        Policy_V_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                    else
                        loweredge=maxindex1(1,:,ii,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, e_val, ReturnFnParamsVec,5,0);
                        aprime=repelem(loweredge,1,1,level1iidiff(ii),1)+N_a1*a2ind; % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2z(reshape(aprime,[1*N_a2,level1iidiff(ii)*N_a2]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                        a1primeind=1;
                        a2primeind=maxindex;
                        maxindexfix=a1primeind+N_a1*(a2primeind-1);
                        allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is 1-by-n_a2-by-1-by-n_a2
                        Policy_V_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                    end
                end

                %% Vtilde (beta0*beta)
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2z_tilde;
                [~,maxindex1]=max(entireRHS_ii,[],1);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
                Vtilde_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindex2,1);
                maxgap=squeeze(max(max(maxindex1(1,:,2:end,:)-maxindex1(1,:,1:end-1,:),[],4),[],2));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(1,:,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is 1-by-n_a2-by-1-by-n_a2
                        aprimeindexes=loweredge+(0:1:maxgap(ii))';
                        % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, e_val, ReturnFnParamsVec,5,0);
                        aprime=repelem(aprimeindexes,1,1,level1iidiff(ii),1)+N_a1*a2ind; % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2z_tilde(reshape(aprime,[(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vtilde_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                        a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                        a2primeind=ceil(maxindex/(maxgap(ii)+1));
                        maxindexfix=a1primeind+N_a1*(a2primeind-1);
                        allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is 1-by-n_a2-by-1-by-n_a2
                        Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                    else
                        loweredge=maxindex1(1,:,ii,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, e_val, ReturnFnParamsVec,5,0);
                        aprime=repelem(loweredge,1,1,level1iidiff(ii),1)+N_a1*a2ind; % the current aprimeii(ii):aprimeii(ii+1)
                        entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2z_tilde(reshape(aprime,[1*N_a2,level1iidiff(ii)*N_a2]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vtilde_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                        a1primeind=1;
                        a2primeind=maxindex;
                        maxindexfix=a1primeind+N_a1*(a2primeind-1);
                        allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is 1-by-n_a2-by-1-by-n_a2
                        Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                    end
                end
            end
        end
      end
    end
    % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    [V1_jj,maxindex]=max(Vtilde_ford2_jj,[],4); % max over d2
    Vtilde(:,:,:,jj)=V1_jj;
    Policy(1,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
    maxindex_lin=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
    Policy(2,:,:,:,jj)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex_lin-1)),[1,N_a,N_semiz*N_z,N_e]);
    % Valt at exponential discounter optimum (full max over d2 and aprime)
    [V_jj,maxindexalt_d2]=max(V_ford2_jj,[],4); % max over d2
    Valt(:,:,:,jj)=V_jj;
    Policyalt(1,:,:,:,jj)=shiftdim(maxindexalt_d2,-1); % d2 is just maxindexalt_d2
    maxindexalt_lin=reshape(maxindexalt_d2,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
    Policyalt(2,:,:,:,jj)=reshape(Policy_V_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindexalt_lin-1)),[1,N_a,N_semiz*N_z,N_e]);
end


end
