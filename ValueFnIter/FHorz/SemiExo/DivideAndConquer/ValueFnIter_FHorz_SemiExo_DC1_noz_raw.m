function [V,Policy3]=ValueFnIter_FHorz_SemiExo_DC1_noz_raw(n_d1,n_d2,n_a,n_semiz,N_j, d1_grid, d2_grid, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

n_d=[n_d1,n_d2];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=prod(n_d); % Needed for N_j when converting to form of Policy3
N_a=prod(n_a);
N_semiz=prod(n_semiz);

V=zeros(N_a,N_semiz,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d,d2,aprime seperately
Policy3=zeros(3,N_a,N_semiz,N_j,'gpuArray');

%%
d1_grid=gpuArray(d1_grid);
d2_grid=gpuArray(d2_grid);
a_grid=gpuArray(a_grid);

special_n_d=[n_d1,ones(1,length(n_d2))];
d_gridvals=CreateGridvals(n_d,[d1_grid; d2_grid],1);

d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_d2,length(n_d1)+length(n_d2)]),[1,3,2]); % version to use when looping over d2

semizind=shiftdim((0:1:N_semiz-1),-1);

% Preallocate
V_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');
Policytemp=zeros(N_a,N_semiz,'gpuArray');

% n-Monotonicity
% vfoptions.level1n=5;
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);


if ~isfield(vfoptions,'V_Jplus1')

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_semiz, d_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        % First, we want aprime conditional on (d,1,a,z)
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a,vfoptions.level1n,N_semiz]),[],1);

        % Store
        V(level1ii,:,N_j)=shiftdim(Vtempii,1);
        Policytemp(level1ii,:)=shiftdim(maxindex2,1); % d,aprime

        % Second level based on montonicity
        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_z
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_z
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_semiz, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*semizind; % loweredge is n_d-by-1-by-1-by-n_z
                Policytemp(curraindex,:)=shiftdim(maxindex+N_d*(loweredge(allind)-1)); % loweredge(given the d and z)
            else
                loweredge=maxindex1(:,1,ii,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_semiz, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*semizind; % loweredge is n_d-by-1-by-1-by-n_z
                Policytemp(curraindex,:)=shiftdim(maxindex+N_d*(loweredge(allind)-1)); % loweredge(given the d and z)
            end
        end

        % Deal with policy for semi-exo
        d_ind=shiftdim(rem(Policytemp-1,N_d)+1,-1);
        Policy3(1,:,:,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
        Policy3(2,:,:,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
        Policy3(3,:,:,N_j)=shiftdim(ceil(Policytemp/N_d),-1);

else
    % Using V_Jplus1
    EV=reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_z]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        % Note: By definition V_Jplus1 does not depend on d (only aprime)
        pi_bothz=pi_semiz_J(:,:,d2_c,N_j); % reverse order

        EV_d2=EV.*shiftdim(pi_bothz',-1);
        EV_d2(isnan(EV_d2))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

        entireEV=repmat(shiftdim(EV_d2,-1),N_d1,1,1,1); % [d1,aprime,1,z]

        % n-Monotonicity
        ReturnMatrix_d2ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*entireEV;

        % First, we want aprime conditional on (d,1,a,z)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_a,vfoptions.level1n,N_semiz]),[],1);

        % Store
        V_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,2);
        Policy_ford2_jj(level1ii,:,d2_c)=shiftdim(maxindex2,2); % d,aprime

        % Second level based on montonicity
        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxaprimeii(ii+1)>minaprimeii(ii)
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_z
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_z
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                daprimez=(repmat(1:1:N_d1,1,maxaprimeii(ii+1)-minaprimeii(ii)+1)+N_d1*repelem(minaprimeii(ii)-1:1:maxaprimeii(ii+1)-1,1,N_d1))'+N_d1*N_a*(0:1:N_semiz-1); % all the d, with the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEV(daprimez(:)),[N_d1*(maxaprimeii(ii+1)-minaprimeii(ii)+1),1,N_semiz]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex,1)+N_d1*(minaprimeii(ii)-1);
            else
                loweredge=maxindex1(:,1,ii,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                daprimez=((1:1:N_d1)+N_d1*(minaprimeii(ii)-1))'+N_d1*N_a*(0:1:N_semiz-1); % all the d, with the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEV(daprimez(:)),[N_d1,1,N_semiz]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex,1)+N_d1*(minaprimeii(ii)-1);
            end
        end
    end
    % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
    V(:,:,N_j)=V_jj;
    Policy3(2,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
    Policy3(1,:,:,N_j)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
    Policy3(3,:,:,N_j)=shiftdim(ceil(d1aprime_ind/N_d1),-1);

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
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
        pi_bothz=pi_semiz_J(:,:,d2_c,jj); % reverse order

        EV_d2=EV.*shiftdim(pi_bothz',-1);
        EV_d2(isnan(EV_d2))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

        entireEV=repmat(shiftdim(EV_d2,-1),N_d1,1,1,1); % [d1,aprime,1,z]

        % n-Monotonicity
        ReturnMatrix_d2ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*entireEV;

        % First, we want aprime conditional on (d,1,a,z)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_a,vfoptions.level1n,N_semiz]),[],1);

        % Store
        V_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,1);
        Policy_ford2_jj(level1ii,:,d2_c)=shiftdim(maxindex2,1); % d,aprime

        % Second level based on montonicity
        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_z
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_z
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                daprimez=(1:1:N_d1)'+N_d1*repelem(aprimeindexes-1,1,1,level1iidiff(ii),1)+N_d1*N_a*shiftdim((0:1:N_semiz-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEV(daprimez(:)),[N_d1*(maxgap(ii)+1),level1iidiff(ii),N_semiz]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d1)+1);
                allind=dind+N_d1*semizind; % loweredge is n_d-by-1-by-1-by-n_z
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1)); % loweredge(given the d and z)
            else
                loweredge=maxindex1(:,1,ii,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                daprimez=(1:1:N_d1)'+N_d1*repelem(loweredge-1,1,1,level1iidiff(ii),1)+N_d1*N_a*shiftdim((0:1:N_semiz-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEV(daprimez(:)),[N_d1,level1iidiff(ii),N_semiz]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d1)+1);
                allind=dind+N_d1*semizind; % loweredge is n_d-by-1-by-1-by-n_z
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1)); % loweredge(given the d and z)
            end
        end
    end
    % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
    V(:,:,jj)=V_jj;
    Policy3(2,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
    Policy3(1,:,:,jj)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
    Policy3(3,:,:,jj)=shiftdim(ceil(d1aprime_ind/N_d1),-1);

end


end
