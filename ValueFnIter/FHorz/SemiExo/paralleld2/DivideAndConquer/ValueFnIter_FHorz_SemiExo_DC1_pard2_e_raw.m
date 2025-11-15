function [V,Policy]=ValueFnIter_FHorz_SemiExo_DC1_pard2_e_raw(n_d1,n_d2,n_a,n_z,n_semiz, n_e,N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J,pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

n_bothz=[n_semiz,n_z]; % These are the return function arguments

n_d=[n_d1,n_d2];
N_d=prod(n_d);

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

V=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');

%%
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
else
    eind=shiftdim((0:1:N_e-1),-2); % already includes -1
end
bothzind=shiftdim((0:1:N_bothz-1),-1);
bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

% pi_bothz_J=repmat(permute(pi_semiz_J,[3,2,1,4]),1,N_z,N_z,1).*repelem(permute(pi_z_J,[4,2,1,3]),1,N_semiz,N_semiz,1);
pi_semiz_J_permute=permute(pi_semiz_J,[3,2,1,4]); % (d2,semizprime,semiz,j)
pi_z_J_permute=permute(pi_z_J,[4,2,1,3]);% (1,zprime,z,j)

pi_e_J=shiftdim(pi_e_J,-2); % Move to third dimension

% n-Monotonicity
% vfoptions.level1n=5;
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_bothz, n_e, d_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        % First, we want aprime conditional on (d,1,a,z,e)
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a,vfoptions.level1n,N_bothz,N_e]),[],1);

        % Store
        V(level1ii,:,:,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,:,:,N_j)=shiftdim(maxindex2,1); % d,aprime
        
        % Second level based on montonicity
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_bothz-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_bothz-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_bothz, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*bothzind+N_d*N_bothz*eind; % loweredge is n_d-by-1-by-1-by-n_bothz-by-n_e
                Policy(curraindex,:,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1)); % loweredge(given the d and z)
            else
                loweredge=maxindex1(:,1,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_bothz, n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*bothzind+N_d*N_bothz*eind; % loweredge is n_d-by-1-by-1-by-n_bothz-by-n_e
                Policy(curraindex,:,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1)); % loweredge(given the d and z)
            end
        end

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % n-Monotonicity
            ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_bothz, special_n_e, d_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);

            % First, we want aprime conditional on (d,1,a,z,e)
            [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_e,[N_d*N_a,vfoptions.level1n,N_bothz]),[],1);

            % Store
            V(level1ii,:,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(level1ii,:,e_c,N_j)=shiftdim(maxindex2,1); % d,aprime

            % Second level based on montonicity
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_bothz
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_bothz
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_bothz, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
                    V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    allind=dind+N_d*bothzind; % loweredge is n_d-by-1-by-1-by-n_bothz
                    Policy(curraindex,:,e_c,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1)); % loweredge(given the d and z)
                else
                    loweredge=maxindex1(:,1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_bothz, special_n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
                    V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    allind=dind+N_d*bothzind; % loweredge is n_d-by-1-by-1-by-n_bothz
                    Policy(curraindex,:,e_c,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1)); % loweredge(given the d and z)
                end
            end

        end
    end

else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_semiz*N_z,N_e]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    V_Jplus1=sum(V_Jplus1.*pi_e_J(1,1,:,N_j),3);

    if vfoptions.lowmemory==0

        % pi_bothz=repmat(pi_semiz_J_permute(:,:,:,N_j),1,N_z,N_z).*repelem(pi_z_J_permute(1,:,:,N_j),1,N_semiz,N_semiz);
        % % pi_bothz=pi_bothz_J(:,:,:,N_j);
        % % (d2,zprime,z)
        % 
        % ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_bothz, n_e, d_grid, a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        % % (d & aprime,a,z,e)
        % 
        % EV=replem(V_Jplus1,N_d2,1).*repmat(pi_bothz,N_a,1,1);
        % EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        % EV=sum(EV,2); % sum over z', leaving a singular second dimension
        % 
        % entireEV=repelem(EV,N_d1,1,1);
        % entireRHS=ReturnMatrix+DiscountFactorParamsVec*entireEV;
        % 
        % % Calc the max and it's index
        % [Vtemp,maxindex]=max(entireRHS,[],1);
        % V(:,:,:,N_j)=Vtemp;
        % Policy(:,:,:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1

        % pi_bothz=repmat(pi_semiz_J_permute(:,:,:,N_j),1,N_z,N_z).*repelem(pi_z_J_permute(1,:,:,N_j),1,N_semiz,N_semiz);
        % % (d2,zprime,z)
        % 
        % EV=replem(V_Jplus1,N_d2,1).*repmat(pi_bothz,N_a,1,1);
        % EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        % EV=sum(EV,2); % sum over z', leaving a singular second dimension
        % 
        % entireEV=repelem(EV,N_d1,1,1);
        % 
        % for e_c=1:N_e
        %     e_val=e_gridvals_J(e_c,:,N_j);
        %     ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_bothz, special_n_e, d_grid, a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
        %     % (d & aprime,a,z)
        % 
        %     entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*entireEV;
        % 
        %     % Calc the max and it's index
        %     [Vtemp,maxindex]=max(entireRHS_e,[],1);
        %     V(:,:,e_c,N_j)=Vtemp;
        %     Policy(:,:,e_c,N_j)=maxindex;
        % end
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
    
    VKronNext_j=V(:,:,:,jj+1);
        
    VKronNext_j=sum(VKronNext_j.*pi_e_J(1,1,:,jj),3);
    % (aprime,zprime)

    if vfoptions.lowmemory==0

        pi_bothz=repmat(pi_semiz_J_permute(:,:,:,jj),1,N_z,N_z).*repelem(pi_z_J_permute(1,:,:,jj),1,N_semiz,N_semiz);

        EV=repelem(VKronNext_j,N_d2,1).*repmat(pi_bothz,N_a,1,1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension

        entireEV=repelem(reshape(EV,[N_d2,N_a,1,N_bothz]),N_d1,1,1); % (d,aprime,1,bothz) [1 is for 'a', to be filled in later]

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_bothz, n_e, d_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV;

        % First, we want aprime conditional on (d,1,a,z,e)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n,N_bothz,N_e]),[],1);

        % Store
        V(level1ii,:,:,jj)=shiftdim(Vtempii,1);
        Policy(level1ii,:,:,jj)=shiftdim(maxindex2,1); % d,aprime

        % Second level based on montonicity
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_bothz-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_bothz-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_bothz, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                daprimez=(1:1:N_d)'+N_d*repelem(aprimeindexes-1,1,1,level1iidiff(ii),1)+N_d*N_a*shiftdim((0:1:N_bothz-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEV(daprimez(:)),[N_d*(maxgap(ii)+1),level1iidiff(ii),N_bothz,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,jj)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*bothzind+N_d*N_bothz*eind; % loweredge is n_d1-by-1-by-1-by-n_bothz-by-n_e
                Policy(curraindex,:,:,jj)=shiftdim(maxindex+N_d*(loweredge(allind)-1)); % loweredge(given the d and z)
            else
                loweredge=maxindex1(:,1,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_bothz, n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                daprimez=(1:1:N_d)'+N_d*repelem(loweredge-1,1,1,level1iidiff(ii),1)+N_d*N_a*shiftdim((0:1:N_bothz-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEV(daprimez(:)),[N_d,level1iidiff(ii),N_bothz,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,jj)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*bothzind+N_d*N_bothz*eind; % loweredge is n_d1-by-1-by-1-by-n_bothz-by-n_e
                Policy(curraindex,:,:,jj)=shiftdim(maxindex+N_d*(loweredge(allind)-1)); % loweredge(given the d and z)
            end
        end
    elseif vfoptions.lowmemory==1

        % pi_bothz=repmat(pi_semiz_J_permute(:,:,:,jj),1,N_z,N_z).*repelem(pi_z_J_permute(1,:,:,jj),1,N_semiz,N_semiz);
        % timer(1)=toc;
        % 
        % EV=replem(VKronNext_j,N_d2,1).*repmat(pi_bothz,N_a,1,1);
        % EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        % EV=sum(EV,2); % sum over z', leaving a singular second dimension
        % 
        % entireEV=repelem(EV,N_d1,1,1);
        % 
        % for e_c=1:N_e
        %     e_val=e_gridvals_J(e_c,:,jj);
        %     ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_bothz, special_n_e, d_grid, a_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec);
        %     % (d & aprime,a,z)
        % 
        %     entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*entireEV;
        % 
        %     % Calc the max and it's index
        %     [Vtemp,maxindex]=max(entireRHS_e,[],1);
        %     V(:,:,e_c,jj)=Vtemp;
        %     Policy(:,:,e_c,jj)=maxindex;
        % end
    end

end



end