function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_DC2A_e_raw(n_d1,n_d2,n_a,n_z,n_semiz,n_e, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Sophisticated QH + SemiExo + DC2A (divide-and-conquer in the first endo state, a2 enumerated in full), with d1, with z, with e.
%
% Sophisticated: Vhat_j = max u + beta_0*beta*E[Vunderbar_{j+1}]
%                Vunderbar_j = Vhat_j + (beta - beta_0*beta)*EV_at_optimal_aprime

n_d=[n_d1,n_d2];
n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=prod(n_d); % Needed for indexing the joint d output
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

Vhat=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d1,d2,aprime seperately
Policy=zeros(3,N_a,N_semiz*N_z,N_e,N_j,'gpuArray');


%%
special_n_d=[n_d1,ones(1,length(n_d2))];
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];
d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_d2,length(n_d1)+length(n_d2)]),[1,3,2]); % version to use when looping over d2

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
a2Bind=shiftdim(gpuArray(0:1:N_a2-1),-1); % already includes -1
bothzind=shiftdim(gpuArray(0:1:N_bothz-1),-1); % already includes -1
eind=shiftdim(gpuArray(0:1:N_e-1),-2); % already includes -1
bothzBind=shiftdim(gpuArray(0:1:N_bothz-1),-4); % already includes -1

% lowmemory: which shocks are looped vs vectorised (spec: =1 loop e; =2 outer z/inner e, vec semiz; =3 joint bothz/inner e)
if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory==2
    special_n_z=ones(1,length(n_z));
    special_n_e=ones(1,length(n_e));
    semizind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % semiz-block analogue of bothzind (L2)
    semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-4); % semiz-block analogue of bothzBind (L2)
elseif vfoptions.lowmemory==3
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
    special_n_e=ones(1,length(n_e));
end

pi_e_J=shiftdim(pi_e_J,-2); % Move to third dimension

% Preallocate
Vhat_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
Vunderbar_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')

  if vfoptions.lowmemory==0
    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

        % First, we want a1prime conditional on (d,1,a2prime,a1,a2,bothz,e)
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d1*N_a1*N_a2,vfoptions.level1n*N_a2,N_bothz,N_e]),[],1);
        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
        Vhat_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
        Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex2,1); % packed (d1, a1prime, a2prime)

        % Attempt for improved version
        maxgap=squeeze(max(max(max(max(max(maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d1-by-(maxgap(ii)+1)-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Vhat_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                dind=(rem(maxindex-1,N_d1)+1);
                a1primeind=rem(ceil(maxindex/N_d1)-1,maxgap(ii)+1)+1-1; % already includes -1
                a2primeind=ceil(maxindex/(N_d1*(maxgap(ii)+1)))-1; % already includes -1
                maxindexfix=dind+N_d1*a1primeind+N_d1*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=dind+N_d1*a2primeind+N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d1*N_a2*N_a2*bothzind+N_d1*N_a2*N_a2*N_bothz*eind; % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindexfix+N_d1*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Vhat_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                dind=(rem(maxindex-1,N_d1)+1);
                a1primeind=0; %1-1; % already includes -1
                a2primeind=ceil(maxindex/N_d1)-1; % already includes -1 % divide by (N_d1*1)
                maxindexfix=dind+N_d1*a1primeind+N_d1*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=dind+N_d1*a2primeind+N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d1*N_a2*N_a2*bothzind+N_d1*N_a2*N_a2*N_bothz*eind; % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindexfix+N_d1*(loweredge(allind)-1),1);
            end
        end
    end

  elseif vfoptions.lowmemory==1 % loop e, vectorise bothz
    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d1*N_a1*N_a2,vfoptions.level1n*N_a2,N_bothz]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
            Vhat_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex2,1);
            maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii)); % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_bothz
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    Vhat_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    a1primeind=rem(ceil(maxindex/N_d1)-1,maxgap(ii)+1)+1-1; % already includes -1
                    a2primeind=ceil(maxindex/(N_d1*(maxgap(ii)+1)))-1; % already includes -1
                    maxindexfix=dind+N_d1*a1primeind+N_d1*N_a1*a2primeind;
                    allind=dind+N_d1*a2primeind+N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d1*N_a2*N_a2*bothzind; % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_bothz
                    Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindexfix+N_d1*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    Vhat_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    a1primeind=0; %1-1; % already includes -1
                    a2primeind=ceil(maxindex/N_d1)-1; % already includes -1 % divide by (N_d1*1)
                    maxindexfix=dind+N_d1*a1primeind+N_d1*N_a1*a2primeind;
                    allind=dind+N_d1*a2primeind+N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d1*N_a2*N_a2*bothzind; % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_bothz
                    Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindexfix+N_d1*(loweredge(allind)-1),1);
                end
            end
        end
    end

  elseif vfoptions.lowmemory==2 % outer z / inner e, vectorise semiz
    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_valblock, e_val, ReturnFnParamsVec,1,0);
                [~,maxindex1]=max(ReturnMatrix_ii,[],2);
                [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d1*N_a1*N_a2,vfoptions.level1n*N_a2,N_semiz]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
                Vhat_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindex2,1);
                maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii)); % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_semiz
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_valblock, e_val, ReturnFnParamsVec,2,0);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                        Vhat_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        a1primeind=rem(ceil(maxindex/N_d1)-1,maxgap(ii)+1)+1-1; % already includes -1
                        a2primeind=ceil(maxindex/(N_d1*(maxgap(ii)+1)))-1; % already includes -1
                        maxindexfix=dind+N_d1*a1primeind+N_d1*N_a1*a2primeind;
                        allind=dind+N_d1*a2primeind+N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d1*N_a2*N_a2*semizind; % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_semiz
                        Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindexfix+N_d1*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_valblock, e_val, ReturnFnParamsVec,2,0);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                        Vhat_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        a1primeind=0; %1-1; % already includes -1
                        a2primeind=ceil(maxindex/N_d1)-1; % already includes -1 % divide by (N_d1*1)
                        maxindexfix=dind+N_d1*a1primeind+N_d1*N_a1*a2primeind;
                        allind=dind+N_d1*a2primeind+N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d1*N_a2*N_a2*semizind; % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_semiz
                        Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindexfix+N_d1*(loweredge(allind)-1),1);
                    end
                end
            end
        end
    end

  elseif vfoptions.lowmemory==3 % joint bothz / inner e
    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_val, e_val, ReturnFnParamsVec,1,0);
                [~,maxindex1]=max(ReturnMatrix_ii,[],2);
                [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d1*N_a1*N_a2,vfoptions.level1n*N_a2]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
                Vhat_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindex2,1);
                maxgap=squeeze(max(max(max(maxindex1(:,1,:,2:end,:)-maxindex1(:,1,:,1:end-1,:),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:),N_a1-maxgap(ii)); % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, e_val, ReturnFnParamsVec,2,0);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                        Vhat_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        a1primeind=rem(ceil(maxindex/N_d1)-1,maxgap(ii)+1)+1-1; % already includes -1
                        a2primeind=ceil(maxindex/(N_d1*(maxgap(ii)+1)))-1; % already includes -1
                        maxindexfix=dind+N_d1*a1primeind+N_d1*N_a1*a2primeind;
                        allind=dind+N_d1*a2primeind+N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2
                        Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindexfix+N_d1*(reshape(loweredge(allind),size(maxindex))-1),1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, e_val, ReturnFnParamsVec,2,0);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                        Vhat_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        a1primeind=0; %1-1; % already includes -1
                        a2primeind=ceil(maxindex/N_d1)-1; % already includes -1 % divide by (N_d1*1)
                        maxindexfix=dind+N_d1*a1primeind+N_d1*N_a1*a2primeind;
                        allind=dind+N_d1*a2primeind+N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2
                        Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindexfix+N_d1*(reshape(loweredge(allind),size(maxindex))-1),1);
                    end
                end
            end
        end
    end
  end
    % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    [V_jj,maxindex]=max(Vhat_ford2_jj,[],4); % max over d2
    Vhat(:,:,:,N_j)=V_jj;
    Vunderbar(:,:,:,N_j)=V_jj; % terminal period: no continuation
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
    Policy(1,:,:,:,N_j)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1); % d1
    Policy(3,:,:,:,N_j)=shiftdim(ceil(d1aprime_ind/N_d1),-1); % joint(a1prime, a2prime)

else
    % Using V_Jplus1
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*pi_e_J(1,1,:,N_j+1),3); % First, switch V_Jplus1 into Kron form and integrate over e'

    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        % Note: By definition V_Jplus1 does not depend on d (only aprime)
        pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j)); % reverse order

      if vfoptions.lowmemory==0
        EV_d2=EV.*shiftdim(pi_bothz',-1);
        EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension
        EV_d2=reshape(EV_d2,[1,N_a1,N_a2,1,1,N_bothz]); % will autoexpand d in 1st-dim, and e
        DiscountedEV_d2=beta0beta*EV_d2;

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

        %% Vhat (beta0*beta)
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2; % autofill e

        % First, we want a1prime conditional on (d,1,a2prime,a1,a2,bothz,e)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_a1*N_a2,vfoptions.level1n*N_a2,N_bothz,N_e]),[],1);
        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
        Vhat_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
        Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(max(max(maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d1-by-(maxgap(ii)+1)-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
                aprime=repelem(a1primeindexes,1,1,1,level1iidiff(ii),1,1,1)+N_a1*a2Bind+N_a*bothzBind;
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2(reshape(aprime,[N_d1*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_bothz,N_e]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                dind=(rem(maxindex-1,N_d1)+1);
                a1primeind=rem(ceil(maxindex/N_d1)-1,maxgap(ii)+1)+1-1; % already includes -1
                a2primeind=ceil(maxindex/(N_d1*(maxgap(ii)+1)))-1; % already includes -1
                maxindexfix=dind+N_d1*a1primeind+N_d1*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=dind+N_d1*a2primeind+N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d1*N_a2*N_a2*bothzind+N_d1*N_a2*N_a2*N_bothz*eind; % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindexfix+N_d1*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
                aprime=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1)+N_a1*a2Bind+N_a*bothzBind;
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2(reshape(aprime,[N_d1*1*N_a2,level1iidiff(ii)*N_a2,N_bothz,N_e]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                dind=(rem(maxindex-1,N_d1)+1);
                a1primeind=0; %1-1; % already includes -1
                a2primeind=ceil(maxindex/N_d1)-1; % already includes -1 % divide by (N_d1*1)
                maxindexfix=dind+N_d1*a1primeind+N_d1*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=dind+N_d1*a2primeind+N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d1*N_a2*N_a2*bothzind+N_d1*N_a2*N_a2*N_bothz*eind; % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindexfix+N_d1*(loweredge(allind)-1),1);
            end
        end

        % Vunderbar = Vhat + (beta - beta0beta)*EV_at_policy
        aprime_ind_d2=ceil(Policy_ford2_jj(:,:,:,d2_c)/N_d1);
        EVd2_2d=reshape(EV_d2,[N_a,N_bothz]);
        EV_at_policy_d2=reshape(EVd2_2d(aprime_ind_d2+N_a*reshape(0:1:N_bothz-1,1,N_bothz)),[N_a,N_bothz,N_e]);
        Vunderbar_ford2_jj(:,:,:,d2_c)=Vhat_ford2_jj(:,:,:,d2_c)+(beta-beta0beta)*EV_at_policy_d2;
      elseif vfoptions.lowmemory==1 % loop e, vectorise bothz
        EV_d2=EV.*shiftdim(pi_bothz',-1);
        EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension
        EV_d2=reshape(EV_d2,[1,N_a1,N_a2,1,1,N_bothz]); % will autoexpand d in 1st-dim
        DiscountedEV_d2=beta0beta*EV_d2;
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);
            %% Vhat (beta0*beta)
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2;
            [~,maxindex1]=max(entireRHS_ii,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_a1*N_a2,vfoptions.level1n*N_a2,N_bothz]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
            Vhat_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex2,1);
            maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii)); % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_bothz
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                    aprime=repelem(a1primeindexes,1,1,1,level1iidiff(ii),1,1)+N_a1*a2Bind+N_a*bothzBind;
                    entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2(reshape(aprime,[N_d1*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_bothz]));
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vhat_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    a1primeind=rem(ceil(maxindex/N_d1)-1,maxgap(ii)+1)+1-1; % already includes -1
                    a2primeind=ceil(maxindex/(N_d1*(maxgap(ii)+1)))-1; % already includes -1
                    maxindexfix=dind+N_d1*a1primeind+N_d1*N_a1*a2primeind;
                    allind=dind+N_d1*a2primeind+N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d1*N_a2*N_a2*bothzind; % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_bothz
                    Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindexfix+N_d1*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                    aprime=repelem(loweredge,1,1,1,level1iidiff(ii),1,1)+N_a1*a2Bind+N_a*bothzBind;
                    entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2(reshape(aprime,[N_d1*1*N_a2,level1iidiff(ii)*N_a2,N_bothz]));
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vhat_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    a1primeind=0; %1-1; % already includes -1
                    a2primeind=ceil(maxindex/N_d1)-1; % already includes -1 % divide by (N_d1*1)
                    maxindexfix=dind+N_d1*a1primeind+N_d1*N_a1*a2primeind;
                    allind=dind+N_d1*a2primeind+N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d1*N_a2*N_a2*bothzind; % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_bothz
                    Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindexfix+N_d1*(loweredge(allind)-1),1);
                end
            end
        end

        % Vunderbar = Vhat + (beta - beta0beta)*EV_at_policy
        aprime_ind_d2=ceil(Policy_ford2_jj(:,:,:,d2_c)/N_d1);
        EVd2_2d=reshape(EV_d2,[N_a,N_bothz]);
        EV_at_policy_d2=reshape(EVd2_2d(aprime_ind_d2+N_a*reshape(0:1:N_bothz-1,1,N_bothz)),[N_a,N_bothz,N_e]);
        Vunderbar_ford2_jj(:,:,:,d2_c)=Vhat_ford2_jj(:,:,:,d2_c)+(beta-beta0beta)*EV_at_policy_d2;
      elseif vfoptions.lowmemory==2 % outer z / inner e, vectorise semiz
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,N_j);
            EV_d2z=EV.*shiftdim(pi_bothz(semizblock,:)',-1);
            EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2z=sum(EV_d2z,2); % sum over z', leaving a singular second dimension
            EV_d2z=reshape(EV_d2z,[1,N_a1,N_a2,1,1,N_semiz]); % will autoexpand d in 1st-dim
            DiscountedEV_d2z=beta0beta*EV_d2z;
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_valblock, e_val, ReturnFnParamsVec,1,0);
                %% Vhat (beta0*beta)
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2z;
                [~,maxindex1]=max(entireRHS_ii,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_a1*N_a2,vfoptions.level1n*N_a2,N_semiz]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
                Vhat_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindex2,1);
                maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii)); % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_semiz
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_valblock, e_val, ReturnFnParamsVec,2,0);
                        aprime=repelem(a1primeindexes,1,1,1,level1iidiff(ii),1,1)+N_a1*a2Bind+N_a*semizBind;
                        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2z(reshape(aprime,[N_d1*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_semiz]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vhat_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        a1primeind=rem(ceil(maxindex/N_d1)-1,maxgap(ii)+1)+1-1; % already includes -1
                        a2primeind=ceil(maxindex/(N_d1*(maxgap(ii)+1)))-1; % already includes -1
                        maxindexfix=dind+N_d1*a1primeind+N_d1*N_a1*a2primeind;
                        allind=dind+N_d1*a2primeind+N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d1*N_a2*N_a2*semizind; % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_semiz
                        Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindexfix+N_d1*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_valblock, e_val, ReturnFnParamsVec,2,0);
                        aprime=repelem(loweredge,1,1,1,level1iidiff(ii),1,1)+N_a1*a2Bind+N_a*semizBind;
                        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2z(reshape(aprime,[N_d1*1*N_a2,level1iidiff(ii)*N_a2,N_semiz]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vhat_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        a1primeind=0; %1-1; % already includes -1
                        a2primeind=ceil(maxindex/N_d1)-1; % already includes -1 % divide by (N_d1*1)
                        maxindexfix=dind+N_d1*a1primeind+N_d1*N_a1*a2primeind;
                        allind=dind+N_d1*a2primeind+N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d1*N_a2*N_a2*semizind; % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_semiz
                        Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindexfix+N_d1*(loweredge(allind)-1),1);
                    end
                end
            end
            % Vunderbar = Vhat + (beta - beta0beta)*EV_at_policy
            aprime_ind_d2=ceil(Policy_ford2_jj(:,semizblock,:,d2_c)/N_d1);
            EVd2z_2d=reshape(EV_d2z,[N_a,N_semiz]);
            EV_at_policy_d2=reshape(EVd2z_2d(aprime_ind_d2+N_a*reshape(0:1:N_semiz-1,1,N_semiz)),[N_a,N_semiz,N_e]);
            Vunderbar_ford2_jj(:,semizblock,:,d2_c)=Vhat_ford2_jj(:,semizblock,:,d2_c)+(beta-beta0beta)*EV_at_policy_d2;
        end

      elseif vfoptions.lowmemory==3 % joint bothz / inner e
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            EV_d2z=EV.*shiftdim(pi_bothz(z_c,:)',-1);
            EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2z=sum(EV_d2z,2); % sum over z', leaving a singular second dimension
            EV_d2z=reshape(EV_d2z,[1,N_a1,N_a2]); % will autoexpand d in 1st-dim
            DiscountedEV_d2z=beta0beta*EV_d2z;
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_val, e_val, ReturnFnParamsVec,1,0);
                %% Vhat (beta0*beta)
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2z;
                [~,maxindex1]=max(entireRHS_ii,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_a1*N_a2,vfoptions.level1n*N_a2]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
                Vhat_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindex2,1);
                maxgap=squeeze(max(max(max(maxindex1(:,1,:,2:end,:)-maxindex1(:,1,:,1:end-1,:),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:),N_a1-maxgap(ii)); % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, e_val, ReturnFnParamsVec,2,0);
                        aprime=repelem(a1primeindexes,1,1,1,level1iidiff(ii),1)+N_a1*a2Bind;
                        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2z(reshape(aprime,[N_d1*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vhat_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        a1primeind=rem(ceil(maxindex/N_d1)-1,maxgap(ii)+1)+1-1; % already includes -1
                        a2primeind=ceil(maxindex/(N_d1*(maxgap(ii)+1)))-1; % already includes -1
                        maxindexfix=dind+N_d1*a1primeind+N_d1*N_a1*a2primeind;
                        allind=dind+N_d1*a2primeind+N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2
                        Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindexfix+N_d1*(reshape(loweredge(allind),size(maxindex))-1),1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, e_val, ReturnFnParamsVec,2,0);
                        aprime=repelem(loweredge,1,1,1,level1iidiff(ii),1)+N_a1*a2Bind;
                        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2z(reshape(aprime,[N_d1*1*N_a2,level1iidiff(ii)*N_a2]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vhat_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        a1primeind=0; %1-1; % already includes -1
                        a2primeind=ceil(maxindex/N_d1)-1; % already includes -1 % divide by (N_d1*1)
                        maxindexfix=dind+N_d1*a1primeind+N_d1*N_a1*a2primeind;
                        allind=dind+N_d1*a2primeind+N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2
                        Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindexfix+N_d1*(reshape(loweredge(allind),size(maxindex))-1),1);
                    end
                end
            end
            % Vunderbar = Vhat + (beta - beta0beta)*EV_at_policy
            aprime_ind_d2=ceil(Policy_ford2_jj(:,z_c,:,d2_c)/N_d1);
            EV_at_policy_d2=reshape(EV_d2z(aprime_ind_d2),[N_a,1,N_e]);
            Vunderbar_ford2_jj(:,z_c,:,d2_c)=Vhat_ford2_jj(:,z_c,:,d2_c)+(beta-beta0beta)*EV_at_policy_d2;
        end
      end
    end
    % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    [V1_jj,maxindex]=max(Vhat_ford2_jj,[],4); % max over d2
    Vhat(:,:,:,N_j)=V1_jj;
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
    maxindex_lin=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
    Vunderbar(:,:,:,N_j)=reshape(Vunderbar_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex_lin-1)),[N_a,N_semiz*N_z,N_e]);
    d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex_lin-1)),[1,N_a,N_semiz*N_z,N_e]);
    Policy(1,:,:,:,N_j)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1); % d1
    Policy(3,:,:,:,N_j)=shiftdim(ceil(d1aprime_ind/N_d1),-1); % joint(a1prime, a2prime)

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

    EV=sum(Vunderbar(:,:,:,jj+1).*pi_e_J(1,1,:,jj+1),3); % integrate over e'

    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj)); % reverse order

      if vfoptions.lowmemory==0
        EV_d2=EV.*shiftdim(pi_bothz',-1);
        EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension
        EV_d2=reshape(EV_d2,[1,N_a1,N_a2,1,1,N_bothz]); % will autoexpand d in 1st-dim, and e
        DiscountedEV_d2=beta0beta*EV_d2;

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);

        %% Vhat (beta0*beta)
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2; % autofill e

        % First, we want a1prime conditional on (d,1,a2prime,a1,a2,bothz,e)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_a1*N_a2,vfoptions.level1n*N_a2,N_bothz,N_e]),[],1);
        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
        Vhat_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
        Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(max(max(maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d1-by-(maxgap(ii)+1)-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0);
                aprime=repelem(a1primeindexes,1,1,1,level1iidiff(ii),1,1,1)+N_a1*a2Bind+N_a*bothzBind;
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2(reshape(aprime,[N_d1*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_bothz,N_e]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                dind=(rem(maxindex-1,N_d1)+1);
                a1primeind=rem(ceil(maxindex/N_d1)-1,maxgap(ii)+1)+1-1; % already includes -1
                a2primeind=ceil(maxindex/(N_d1*(maxgap(ii)+1)))-1; % already includes -1
                maxindexfix=dind+N_d1*a1primeind+N_d1*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=dind+N_d1*a2primeind+N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d1*N_a2*N_a2*bothzind+N_d1*N_a2*N_a2*N_bothz*eind; % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindexfix+N_d1*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0);
                aprime=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1)+N_a1*a2Bind+N_a*bothzBind;
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2(reshape(aprime,[N_d1*1*N_a2,level1iidiff(ii)*N_a2,N_bothz,N_e]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                dind=(rem(maxindex-1,N_d1)+1);
                a1primeind=0; %1-1; % already includes -1
                a2primeind=ceil(maxindex/N_d1)-1; % already includes -1 % divide by (N_d1*1)
                maxindexfix=dind+N_d1*a1primeind+N_d1*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=dind+N_d1*a2primeind+N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d1*N_a2*N_a2*bothzind+N_d1*N_a2*N_a2*N_bothz*eind; % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_bothz-by-n_e
                Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindexfix+N_d1*(loweredge(allind)-1),1);
            end
        end

        % Vunderbar = Vhat + (beta - beta0beta)*EV_at_policy
        aprime_ind_d2=ceil(Policy_ford2_jj(:,:,:,d2_c)/N_d1);
        EVd2_2d=reshape(EV_d2,[N_a,N_bothz]);
        EV_at_policy_d2=reshape(EVd2_2d(aprime_ind_d2+N_a*reshape(0:1:N_bothz-1,1,N_bothz)),[N_a,N_bothz,N_e]);
        Vunderbar_ford2_jj(:,:,:,d2_c)=Vhat_ford2_jj(:,:,:,d2_c)+(beta-beta0beta)*EV_at_policy_d2;
      elseif vfoptions.lowmemory==1 % loop e, vectorise bothz
        EV_d2=EV.*shiftdim(pi_bothz',-1);
        EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension
        EV_d2=reshape(EV_d2,[1,N_a1,N_a2,1,1,N_bothz]); % will autoexpand d in 1st-dim
        DiscountedEV_d2=beta0beta*EV_d2;
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1,0);
            %% Vhat (beta0*beta)
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2;
            [~,maxindex1]=max(entireRHS_ii,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_a1*N_a2,vfoptions.level1n*N_a2,N_bothz]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
            Vhat_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex2,1);
            maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii)); % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_bothz
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2,0);
                    aprime=repelem(a1primeindexes,1,1,1,level1iidiff(ii),1,1)+N_a1*a2Bind+N_a*bothzBind;
                    entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2(reshape(aprime,[N_d1*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_bothz]));
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vhat_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    a1primeind=rem(ceil(maxindex/N_d1)-1,maxgap(ii)+1)+1-1; % already includes -1
                    a2primeind=ceil(maxindex/(N_d1*(maxgap(ii)+1)))-1; % already includes -1
                    maxindexfix=dind+N_d1*a1primeind+N_d1*N_a1*a2primeind;
                    allind=dind+N_d1*a2primeind+N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d1*N_a2*N_a2*bothzind; % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_bothz
                    Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindexfix+N_d1*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2,0);
                    aprime=repelem(loweredge,1,1,1,level1iidiff(ii),1,1)+N_a1*a2Bind+N_a*bothzBind;
                    entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2(reshape(aprime,[N_d1*1*N_a2,level1iidiff(ii)*N_a2,N_bothz]));
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vhat_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    a1primeind=0; %1-1; % already includes -1
                    a2primeind=ceil(maxindex/N_d1)-1; % already includes -1 % divide by (N_d1*1)
                    maxindexfix=dind+N_d1*a1primeind+N_d1*N_a1*a2primeind;
                    allind=dind+N_d1*a2primeind+N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d1*N_a2*N_a2*bothzind; % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_bothz
                    Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindexfix+N_d1*(loweredge(allind)-1),1);
                end
            end
        end

        % Vunderbar = Vhat + (beta - beta0beta)*EV_at_policy
        aprime_ind_d2=ceil(Policy_ford2_jj(:,:,:,d2_c)/N_d1);
        EVd2_2d=reshape(EV_d2,[N_a,N_bothz]);
        EV_at_policy_d2=reshape(EVd2_2d(aprime_ind_d2+N_a*reshape(0:1:N_bothz-1,1,N_bothz)),[N_a,N_bothz,N_e]);
        Vunderbar_ford2_jj(:,:,:,d2_c)=Vhat_ford2_jj(:,:,:,d2_c)+(beta-beta0beta)*EV_at_policy_d2;
      elseif vfoptions.lowmemory==2 % outer z / inner e, vectorise semiz
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,jj);
            EV_d2z=EV.*shiftdim(pi_bothz(semizblock,:)',-1);
            EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2z=sum(EV_d2z,2); % sum over z', leaving a singular second dimension
            EV_d2z=reshape(EV_d2z,[1,N_a1,N_a2,1,1,N_semiz]); % will autoexpand d in 1st-dim
            DiscountedEV_d2z=beta0beta*EV_d2z;
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_valblock, e_val, ReturnFnParamsVec,1,0);
                %% Vhat (beta0*beta)
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2z;
                [~,maxindex1]=max(entireRHS_ii,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_a1*N_a2,vfoptions.level1n*N_a2,N_semiz]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
                Vhat_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindex2,1);
                maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii)); % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_semiz
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_valblock, e_val, ReturnFnParamsVec,2,0);
                        aprime=repelem(a1primeindexes,1,1,1,level1iidiff(ii),1,1)+N_a1*a2Bind+N_a*semizBind;
                        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2z(reshape(aprime,[N_d1*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_semiz]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vhat_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        a1primeind=rem(ceil(maxindex/N_d1)-1,maxgap(ii)+1)+1-1; % already includes -1
                        a2primeind=ceil(maxindex/(N_d1*(maxgap(ii)+1)))-1; % already includes -1
                        maxindexfix=dind+N_d1*a1primeind+N_d1*N_a1*a2primeind;
                        allind=dind+N_d1*a2primeind+N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d1*N_a2*N_a2*semizind; % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_semiz
                        Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindexfix+N_d1*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_valblock, e_val, ReturnFnParamsVec,2,0);
                        aprime=repelem(loweredge,1,1,1,level1iidiff(ii),1,1)+N_a1*a2Bind+N_a*semizBind;
                        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2z(reshape(aprime,[N_d1*1*N_a2,level1iidiff(ii)*N_a2,N_semiz]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vhat_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        a1primeind=0; %1-1; % already includes -1
                        a2primeind=ceil(maxindex/N_d1)-1; % already includes -1 % divide by (N_d1*1)
                        maxindexfix=dind+N_d1*a1primeind+N_d1*N_a1*a2primeind;
                        allind=dind+N_d1*a2primeind+N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d1*N_a2*N_a2*semizind; % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2-by-n_semiz
                        Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindexfix+N_d1*(loweredge(allind)-1),1);
                    end
                end
            end
            % Vunderbar = Vhat + (beta - beta0beta)*EV_at_policy
            aprime_ind_d2=ceil(Policy_ford2_jj(:,semizblock,:,d2_c)/N_d1);
            EVd2z_2d=reshape(EV_d2z,[N_a,N_semiz]);
            EV_at_policy_d2=reshape(EVd2z_2d(aprime_ind_d2+N_a*reshape(0:1:N_semiz-1,1,N_semiz)),[N_a,N_semiz,N_e]);
            Vunderbar_ford2_jj(:,semizblock,:,d2_c)=Vhat_ford2_jj(:,semizblock,:,d2_c)+(beta-beta0beta)*EV_at_policy_d2;
        end

      elseif vfoptions.lowmemory==3 % joint bothz / inner e
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,jj);
            EV_d2z=EV.*shiftdim(pi_bothz(z_c,:)',-1);
            EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2z=sum(EV_d2z,2); % sum over z', leaving a singular second dimension
            EV_d2z=reshape(EV_d2z,[1,N_a1,N_a2]); % will autoexpand d in 1st-dim
            DiscountedEV_d2z=beta0beta*EV_d2z;
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_val, e_val, ReturnFnParamsVec,1,0);
                %% Vhat (beta0*beta)
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2z;
                [~,maxindex1]=max(entireRHS_ii,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_a1*N_a2,vfoptions.level1n*N_a2]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
                Vhat_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindex2,1);
                maxgap=squeeze(max(max(max(maxindex1(:,1,:,2:end,:)-maxindex1(:,1,:,1:end-1,:),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:),N_a1-maxgap(ii)); % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, e_val, ReturnFnParamsVec,2,0);
                        aprime=repelem(a1primeindexes,1,1,1,level1iidiff(ii),1)+N_a1*a2Bind;
                        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2z(reshape(aprime,[N_d1*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vhat_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        a1primeind=rem(ceil(maxindex/N_d1)-1,maxgap(ii)+1)+1-1; % already includes -1
                        a2primeind=ceil(maxindex/(N_d1*(maxgap(ii)+1)))-1; % already includes -1
                        maxindexfix=dind+N_d1*a1primeind+N_d1*N_a1*a2primeind;
                        allind=dind+N_d1*a2primeind+N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2
                        Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindexfix+N_d1*(reshape(loweredge(allind),size(maxindex))-1),1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, e_val, ReturnFnParamsVec,2,0);
                        aprime=repelem(loweredge,1,1,1,level1iidiff(ii),1)+N_a1*a2Bind;
                        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2z(reshape(aprime,[N_d1*1*N_a2,level1iidiff(ii)*N_a2]));
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vhat_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        a1primeind=0; %1-1; % already includes -1
                        a2primeind=ceil(maxindex/N_d1)-1; % already includes -1 % divide by (N_d1*1)
                        maxindexfix=dind+N_d1*a1primeind+N_d1*N_a1*a2primeind;
                        allind=dind+N_d1*a2primeind+N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is n_d1-by-1-by-n_a2-by-1-by-n_a2
                        Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindexfix+N_d1*(reshape(loweredge(allind),size(maxindex))-1),1);
                    end
                end
            end
            % Vunderbar = Vhat + (beta - beta0beta)*EV_at_policy
            aprime_ind_d2=ceil(Policy_ford2_jj(:,z_c,:,d2_c)/N_d1);
            EV_at_policy_d2=reshape(EV_d2z(aprime_ind_d2),[N_a,1,N_e]);
            Vunderbar_ford2_jj(:,z_c,:,d2_c)=Vhat_ford2_jj(:,z_c,:,d2_c)+(beta-beta0beta)*EV_at_policy_d2;
        end
      end
    end
    % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    [V1_jj,maxindex]=max(Vhat_ford2_jj,[],4); % max over d2
    Vhat(:,:,:,jj)=V1_jj;
    Policy(2,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
    maxindex_lin=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
    Vunderbar(:,:,:,jj)=reshape(Vunderbar_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex_lin-1)),[N_a,N_semiz*N_z,N_e]);
    d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex_lin-1)),[1,N_a,N_semiz*N_z,N_e]);
    Policy(1,:,:,:,jj)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1); % d1
    Policy(3,:,:,:,jj)=shiftdim(ceil(d1aprime_ind/N_d1),-1); % joint(a1prime, a2prime)
end


end
