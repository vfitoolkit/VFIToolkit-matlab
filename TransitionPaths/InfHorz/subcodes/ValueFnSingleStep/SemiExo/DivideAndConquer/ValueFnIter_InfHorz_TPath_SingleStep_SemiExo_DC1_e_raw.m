function [V,Policy]=ValueFnIter_InfHorz_TPath_SingleStep_SemiExo_DC1_e_raw(Vnext,n_d1,n_d2,n_a,n_z,n_semiz, n_e, d1_gridvals, d2_gridvals, a_grid, z_gridvals, semiz_gridvals, e_gridvals, pi_z, pi_semiz, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Single Bellman step, divide-and-conquer (one endogenous state), with semi-exogenous state semiz, with d1 and d2 (d2 determines the transition of semiz), with z and iid e.
% Vnext is [N_a,N_bothz,N_e] (next period value fn), bothz=(semiz,z) with semiz first
% pi_z is [N_z,N_zprime]; pi_semiz is [N_semiz,N_semizprime,N_d2] (this period's transition); pi_e is [N_e,1], the distribution of next period e.
% Policy is returned as kron rows [d1; d2; aprime], [3,N_a,N_bothz,N_e]
% lowmemory: 0 is everything at once; 1 loops over e.

n_d=[n_d1,n_d2];
n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

if vfoptions.lowmemory>=2
    error('vfoptions.lowmemory=2 not supported with semi-exogenous states')
end

%%
special_n_d=[n_d1,ones(1,length(n_d2))];
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];

d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_d2,length(n_d)]),[1,3,2]); % version to use when looping over d2

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
eind=shiftdim(gpuArray(0:1:N_e-1),-2); % [1,1,1,N_e], already includes -1
bothzind=shiftdim(gpuArray(0:1:N_bothz-1),-1); % [1,1,N_bothz], already includes -1
bothzBind=shiftdim(gpuArray(0:1:N_bothz-1),-2); % [1,1,1,N_bothz], already includes -1

bothz_gridvals=[repmat(semiz_gridvals,N_z,1),repelem(z_gridvals,N_semiz,1)];

% Preallocate
V_ford2=zeros(N_a,N_bothz,N_e,N_d2,'gpuArray');
Policy_ford2=zeros(N_a,N_bothz,N_e,N_d2,'gpuArray');

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

% Calc the condl expectation term (except beta), computed once and used (or sliced) by every lowmemory branch
% First e', then z', then semiz' (given d2)
EV=reshape(Vnext,[N_a,N_bothz,N_e]).*reshape(pi_e,[1,1,N_e]); % [N_a,N_bothz,N_e], aprime-by-(semizprime,zprime)-by-eprime
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
EV=sum(EV,3); % sum over e', [N_a,N_bothz]
EV=reshape(EV,[N_a,N_semiz,N_z]).*shiftdim(pi_z',-2); % [N_a,N_semiz,N_zprime,N_z], aprime-by-semizprime-by-zprime-by-z
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=sum(EV,3); % sum over z', [N_a,N_semiz,1,N_z], aprime-by-semizprime-by-1-by-z
EV=EV.*permute(pi_semiz,[4,2,1,5,3]); % [N_a,N_semizprime,N_semiz,N_z,N_d2]
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=sum(EV,2); % sum over semiz', [N_a,1,N_semiz,N_z,N_d2]
DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_a,N_bothz,N_d2]); % aprime-by-(semiz,z)-by-d2

%%
if vfoptions.lowmemory==0
    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        DiscountedEV_d2=reshape(DiscountedEV(:,:,d2_c),[1,N_a,1,N_bothz]); % broadcasts over d1 (and level1n, and e); linear index of (1,aprime,1,bothz) is aprime+N_a*(bothz-1)

        % n-Monotonicity
        ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, a_grid, a_grid(level1ii), bothz_gridvals, e_gridvals, ReturnFnParamsVec,1); % [N_d1,N_a,level1n,N_bothz,N_e], d1-by-aprime-by-a-by-(semiz,z)-by-e

        entireRHS_ii=ReturnMatrix_d2ii+DiscountedEV_d2;

        % First, we want aprime conditional on (d1,1,a,bothz,e)
        [~,maxindex1]=max(entireRHS_ii,[],2); % [N_d1,1,level1n,N_bothz,N_e]

        % Now, get and store the full (d1,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_a,vfoptions.level1n,N_bothz,N_e]),[],1); % [1,level1n,N_bothz,N_e]

        % Store
        V_ford2(level1ii,:,:,d2_c)=reshape(Vtempii,[vfoptions.level1n,N_bothz,N_e]);
        Policy_ford2(level1ii,:,:,d2_c)=reshape(maxindex2,[vfoptions.level1n,N_bothz,N_e]); % d1,aprime

        % Second level based on monotonicity
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1)); % max over d1,bothz,e
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            npts=length(curraindex);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d1-by-1-by-1-by-n_bothz-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d1-by-maxgap(ii)+1-by-1-by-n_bothz-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, reshape(a_grid(aprimeindexes),[N_d1,maxgap(ii)+1,1,N_bothz,N_e]), a_grid(curraindex), bothz_gridvals, e_gridvals, ReturnFnParamsVec,2); % [N_d1*(maxgap(ii)+1),npts,N_bothz,N_e]
                aprimez=aprimeindexes+N_a*bothzBind; % [N_d1,maxgap(ii)+1,1,N_bothz,N_e], index into DiscountedEV_d2
                entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV_d2(aprimez),[N_d1*(maxgap(ii)+1),1,N_bothz,N_e]); % autoexpand level1iidiff(ii) in 2nd-dim
                [Vtempii,maxindex]=max(entireRHS_ii,[],1); % [1,npts,N_bothz,N_e], maxindex=d1+N_d1*(aprime-loweredge(d1,bothz,e))
                V_ford2(curraindex,:,:,d2_c)=reshape(Vtempii,[npts,N_bothz,N_e]);
                dind=(rem(maxindex-1,N_d1)+1);
                allind=dind+N_d1*bothzind+N_d1*N_bothz*eind; % loweredge is n_d1-by-1-by-1-by-n_bothz-by-n_e, so linear index is d1+N_d1*(bothz-1)+N_d1*N_bothz*(e-1)
                Policy_ford2(curraindex,:,:,d2_c)=reshape(maxindex+N_d1*(reshape(loweredge(allind),size(allind))-1),[npts,N_bothz,N_e]); % loweredge(given the d1, bothz and e)
            else
                loweredge=maxindex1(:,1,ii,:,:);
                % loweredge is n_d1-by-1-by-1-by-n_bothz-by-n_e
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, reshape(a_grid(loweredge),[N_d1,1,1,N_bothz,N_e]), a_grid(curraindex), bothz_gridvals, e_gridvals, ReturnFnParamsVec,2); % [N_d1,npts,N_bothz,N_e]
                aprimez=loweredge+N_a*bothzBind; % [N_d1,1,1,N_bothz,N_e], index into DiscountedEV_d2
                entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV_d2(aprimez),[N_d1,1,N_bothz,N_e]); % autoexpand level1iidiff(ii) in 2nd-dim
                [Vtempii,maxindex]=max(entireRHS_ii,[],1); % [1,npts,N_bothz,N_e], maxindex=d1
                V_ford2(curraindex,:,:,d2_c)=reshape(Vtempii,[npts,N_bothz,N_e]);
                dind=(rem(maxindex-1,N_d1)+1);
                allind=dind+N_d1*bothzind+N_d1*N_bothz*eind; % loweredge is n_d1-by-1-by-1-by-n_bothz-by-n_e, so linear index is d1+N_d1*(bothz-1)+N_d1*N_bothz*(e-1)
                Policy_ford2(curraindex,:,:,d2_c)=reshape(maxindex+N_d1*(reshape(loweredge(allind),size(allind))-1),[npts,N_bothz,N_e]); % loweredge(given the d1, bothz and e)
            end
        end
    end

elseif vfoptions.lowmemory==1
    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        DiscountedEV_d2=reshape(DiscountedEV(:,:,d2_c),[1,N_a,1,N_bothz]); % broadcasts over d1 (and level1n); linear index of (1,aprime,1,bothz) is aprime+N_a*(bothz-1)

        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);

            % n-Monotonicity
            ReturnMatrix_d2iie=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, a_grid, a_grid(level1ii), bothz_gridvals, e_val, ReturnFnParamsVec,1); % [N_d1,N_a,level1n,N_bothz], d1-by-aprime-by-a-by-(semiz,z)

            entireRHS_ii=ReturnMatrix_d2iie+DiscountedEV_d2;

            % First, we want aprime conditional on (d1,1,a,bothz)
            [~,maxindex1]=max(entireRHS_ii,[],2); % [N_d1,1,level1n,N_bothz]

            % Now, get and store the full (d1,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_a,vfoptions.level1n,N_bothz]),[],1); % [1,level1n,N_bothz]

            % Store
            V_ford2(level1ii,:,e_c,d2_c)=reshape(Vtempii,[vfoptions.level1n,N_bothz]);
            Policy_ford2(level1ii,:,e_c,d2_c)=reshape(maxindex2,[vfoptions.level1n,N_bothz]); % d1,aprime

            % Second level based on monotonicity
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1)); % max over d1,bothz
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                npts=length(curraindex);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d1-by-1-by-1-by-n_bothz
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d1-by-maxgap(ii)+1-by-1-by-n_bothz
                    ReturnMatrix_iie=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, reshape(a_grid(aprimeindexes),[N_d1,maxgap(ii)+1,1,N_bothz]), a_grid(curraindex), bothz_gridvals, e_val, ReturnFnParamsVec,2); % [N_d1*(maxgap(ii)+1),npts,N_bothz]
                    aprimez=aprimeindexes+N_a*bothzBind; % [N_d1,maxgap(ii)+1,1,N_bothz], index into DiscountedEV_d2
                    entireRHS_ii=ReturnMatrix_iie+reshape(DiscountedEV_d2(aprimez),[N_d1*(maxgap(ii)+1),1,N_bothz]); % autoexpand level1iidiff(ii) in 2nd-dim
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1); % [1,npts,N_bothz], maxindex=d1+N_d1*(aprime-loweredge(d1,bothz))
                    V_ford2(curraindex,:,e_c,d2_c)=reshape(Vtempii,[npts,N_bothz]);
                    dind=(rem(maxindex-1,N_d1)+1);
                    allind=dind+N_d1*bothzind; % loweredge is n_d1-by-1-by-1-by-n_bothz, so linear index is d1+N_d1*(bothz-1)
                    Policy_ford2(curraindex,:,e_c,d2_c)=reshape(maxindex+N_d1*(reshape(loweredge(allind),size(allind))-1),[npts,N_bothz]); % loweredge(given the d1 and bothz)
                else
                    loweredge=maxindex1(:,1,ii,:);
                    % loweredge is n_d1-by-1-by-1-by-n_bothz
                    % Just use aprime(ii) for everything
                    ReturnMatrix_iie=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, reshape(a_grid(loweredge),[N_d1,1,1,N_bothz]), a_grid(curraindex), bothz_gridvals, e_val, ReturnFnParamsVec,2); % [N_d1,npts,N_bothz]
                    aprimez=loweredge+N_a*bothzBind; % [N_d1,1,1,N_bothz], index into DiscountedEV_d2
                    entireRHS_ii=ReturnMatrix_iie+reshape(DiscountedEV_d2(aprimez),[N_d1,1,N_bothz]); % autoexpand level1iidiff(ii) in 2nd-dim
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1); % [1,npts,N_bothz], maxindex=d1
                    V_ford2(curraindex,:,e_c,d2_c)=reshape(Vtempii,[npts,N_bothz]);
                    dind=(rem(maxindex-1,N_d1)+1);
                    allind=dind+N_d1*bothzind; % loweredge is n_d1-by-1-by-1-by-n_bothz, so linear index is d1+N_d1*(bothz-1)
                    Policy_ford2(curraindex,:,e_c,d2_c)=reshape(maxindex+N_d1*(reshape(loweredge(allind),size(allind))-1),[npts,N_bothz]); % loweredge(given the d1 and bothz)
                end
            end
        end
    end
end

% Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
[V,maxindex]=max(V_ford2,[],4); % max over d2, [N_a,N_bothz,N_e]
d1aprime_ind=reshape(Policy_ford2((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(reshape(maxindex,[N_a*N_bothz*N_e,1])-1)),[1,N_a,N_bothz,N_e]);
Policy=[rem(d1aprime_ind-1,N_d1)+1; reshape(maxindex,[1,N_a,N_bothz,N_e]); ceil(d1aprime_ind/N_d1)]; % [d1; d2; aprime]


end
