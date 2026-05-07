function [V,Policy2]=ValueFnIter_Case1_FHorz_DC2_lowmem_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% lowmem=loop over z
special_n_z=ones(1,length(n_z));

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

N_a1=n_a(1);
N_a2=n_a(2);

V=zeros(N_a1,N_a2,N_z,N_j,'gpuArray');
Policy=zeros(N_a1,N_a2,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
d_grid=gpuArray(d_grid);
d_gridvals=CreateGridvals(n_d,d_grid,1);
a_grid=gpuArray(a_grid);

a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% n-Monotonicity
% vfoptions.level1n=[21,21];
level11ii=round(linspace(1,n_a(1),vfoptions.level1n(1)));
level12kk=round(linspace(1,n_a(2),vfoptions.level1n(2)));

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    for z_c=1:N_z
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, special_n_z, d_gridvals, a1_grid, a2_grid, a1_grid(level11ii), a2_grid(level12kk), z_gridvals_J(z_c,:,N_j), ReturnFnParamsVec, 1);
        % (d,a1a2prime,a1,a2,z)
            
        % First, we want a1a2prime conditional on (d,1,a1,a2,z)
        % We would just do
        % [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        % But there is an error in Matlab for max in second dimesion on GPU: https://au.mathworks.com/matlabcentral/answers/2152160-error-in-index-returned-by-max-in-the-second-dimension-in-obscure-case
        % So instead for now we instead do following two lines
        [~,maxindex1]=max(permute(ReturnMatrix_ii,[2,1,3,4]),[],1);
        maxindex1=permute(maxindex1,[2,1,3,4]);

        % % Store
        % V(level11ii,level12kk,:)=shiftdim(Vtempii,1);
        % Policy(level11ii,level12kk,:)=shiftdim(maxindex2,1);

        %% Level 2
        % Split maxindex1 into a1prime and a2prime
        maxindex11=reshape(rem(maxindex1-1,N_a1)+1,[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2)]);
        maxindex12=reshape(ceil(maxindex1/N_a1),[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2)]);
        
        % Attempt for improved version
        maxgap1=squeeze(max(maxindex11(:,1,2:end,2:end)-maxindex11(:,1,1:end-1,1:end-1),[],1));
        maxgap2=squeeze(max(maxindex12(:,1,2:end,2:end)-maxindex12(:,1,1:end-1,1:end-1),[],1));
        for ii=1:(vfoptions.level1n(1)-1)
            for kk=1:(vfoptions.level1n(2)-1)
                curra1index=(level11ii(ii):1:level11ii(ii+1)); % Note: in 2D we need to include the edges
                curra2index=(level12kk(kk):1:level12kk(kk+1)); % Note: in 2D we need to include the edges

                if maxgap1(ii,kk)>0 && maxgap2(ii,kk)>0
                    loweredge1=min(maxindex11(:,1,ii,kk),N_a1-maxgap1(ii,kk)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    loweredge2=min(maxindex12(:,1,ii,kk),N_a2-maxgap2(ii,kk)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-1-by-1-by-n_z
                    a1primeindexes=loweredge1+repmat((0:1:maxgap1(ii,kk)),1,maxgap2(ii,kk)+1);
                    a2primeindexes=loweredge2+repelem((0:1:maxgap2(ii,kk)),1,maxgap1(ii,kk)+1);
                    % aprime possibilities are n_d-by-(maxgap1(ii)+1)-(maxgap2(ii)+1)-by-1-by-n_a2-by-n_z
                    ReturnMatrix_iikk=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, special_n_z, d_gridvals, a1_grid(a1primeindexes), a2_grid(a2primeindexes), a1_grid(level11ii(ii):level11ii(ii+1)), a2_grid(level12kk(kk):level12kk(kk+1)), z_gridvals_J(z_c,:,N_j), ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_iikk,[],1);
                    V(curra1index,curra2index,z_c,N_j)=shiftdim(Vtempii,1);
                    % maxindex needs to be reworked:
                    %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                    dind=rem(maxindex-1,N_d)+1;
                    a1primeind=rem(ceil(maxindex/N_d)-1,maxgap1(ii,kk)+1)+1-1; % already includes -1
                    a2primeind=ceil(maxindex/(N_d*(maxgap1(ii,kk)+1)))-1; % already includes -1
                    maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                    %  the a1prime is relative to loweredge1(dind & zind), need to 'add'  the loweredge1
                    %  the a2prime is relative to loweredge2(dind & zind), need to 'add'  the loweredge2
                    Policy(curra1index,curra2index,z_c,N_j)=shiftdim(maxindexfix+N_d*(loweredge1(dind)-1)+N_d*N_a1*(loweredge2(dind)-1),1);

                elseif maxgap1(ii,kk)>0 && maxgap2(ii,kk)==0
                    loweredge1=min(maxindex11(:,1,ii,kk),N_a1-maxgap1(ii,kk)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    loweredge2=maxindex12(:,1,ii,kk);
                    % loweredge is n_d-by-1-by-1-by-1-by-n_z
                    a1primeindexes=loweredge1+(0:1:maxgap1(ii,kk));
                    ReturnMatrix_iikk=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, special_n_z, d_gridvals, a1_grid(a1primeindexes), reshape(a2_grid(loweredge2),[N_d,1,1,1]), a1_grid(level11ii(ii):level11ii(ii+1)), a2_grid(level12kk(kk):level12kk(kk+1)), z_gridvals_J(z_c,:,N_j), ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_iikk,[],1);
                    V(curra1index,curra2index,z_c,N_j)=shiftdim(Vtempii,1);
                    % no need to rework maxindex in this case
                    dind=rem(maxindex-1,N_d)+1;
                    %  the a1prime is relative to loweredge1(dind & zind), need to 'add'  the loweredge1
                    %  the a2prime is relative to loweredge2(dind & zind), need to 'add'  the loweredge2
                    Policy(curra1index,curra2index,z_c,N_j)=shiftdim(maxindex+N_d*(loweredge1(dind)-1)+N_d*N_a1*(loweredge2(dind)-1),1);

                elseif maxgap1(ii,kk)==0 && maxgap2(ii,kk)>0
                    loweredge1=maxindex11(:,1,ii,kk);
                    loweredge2=min(maxindex12(:,1,ii,kk),N_a2-maxgap2(ii,kk)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-1-by-1-by-n_z (a1prime&a2prime,a1,a2,z)
                    a2primeindexes=loweredge2+(0:1:maxgap2(ii,kk));
                    ReturnMatrix_iikk=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, special_n_z, d_gridvals, reshape(a1_grid(loweredge1),[N_d,1,1,1]), a2_grid(a2primeindexes), a1_grid(level11ii(ii):level11ii(ii+1)), a2_grid(level12kk(kk):level12kk(kk+1)), z_gridvals_J(z_c,:,N_j), ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_iikk,[],1);
                    V(curra1index,curra2index,z_c,N_j)=shiftdim(Vtempii,1);
                    %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                    %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                    dind=rem(maxindex-1,N_d)+1;
                    % a1primeind=1;
                    % a2primeind=ceil(maxindex/N_d);
                    maxindexfix=dind+N_d*N_a1*(ceil(maxindex/N_d)-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                    %  the a1prime is relative to loweredge1(dind & zind), need to 'add'  the loweredge1
                    %  the a2prime is relative to loweredge2(dind & zind), need to 'add'  the loweredge2
                    Policy(curra1index,curra2index,z_c,N_j)=shiftdim(maxindexfix+N_d*(loweredge1(dind)-1)+N_d*N_a1*(loweredge2(dind)-1),1);

                else % maxgap1(ii,kk)==0 && maxgap2(ii,kk)==0
                    loweredge1=maxindex11(:,1,ii,kk);
                    loweredge2=maxindex12(:,1,ii,kk);
                    % loweredge is n_d-by-1-by-1-by-1-by-n_z (a1prime&a2prime,a1,a2,z)
                    ReturnMatrix_iikk=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, special_n_z, d_gridvals, reshape(a1_grid(loweredge1),[N_d,1,1,1]), reshape(a2_grid(loweredge2),[N_d,1,1,1]), a1_grid(level11ii(ii):level11ii(ii+1)), a2_grid(level12kk(kk):level12kk(kk+1)), z_gridvals_J(z_c,:,N_j), ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_iikk,[],1);
                    V(curra1index,curra2index,z_c,N_j)=shiftdim(Vtempii,1);
                    % maxindex is anyway just the index for d, so
                    dind=maxindex;
                    %  the a1prime is relative to loweredge1(zind), need to 'add'  the loweredge1
                    %  the a2prime is relative to loweredge2(zind), need to 'add'  the loweredge2
                    Policy(curra1index,curra2index,z_c,N_j)=shiftdim(maxindex+N_d*(loweredge1(dind)-1)+N_d*N_a1*(loweredge2(dind)-1),1);
                end

            end
        end
    end

else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a1,N_a2,N_z]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    EV=V_Jplus1.*shiftdim(pi_z_J(:,:,N_j)',-2);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,3); % sum over z', leaving a singular second dimension
    DiscountedentireEV=DiscountFactorParamsVec*reshape(repmat(shiftdim(EV,-1),N_d,1,1,1),[N_d,N_a1*N_a2,1,1,N_z]); % [d,aprime,1,1,z]

    for z_c=1:N_z
        DiscountedentireEV_z=DiscountedentireEV(:,:,1,1,z_c);

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, special_n_z, d_gridvals, a1_grid, a2_grid, a1_grid(level11ii), a2_grid(level12kk), z_gridvals_J(z_c,:,N_j), ReturnFnParamsVec, 1);
        % (d,a1a2prime,a1,a2,z)

        entireRHS_ii=ReturnMatrix_ii+DiscountedentireEV_z;

        % First, we want a1a2prime conditional on (d,1,a,z)
        % We would just do
        % [~,maxindex1]=max(entireRHS_ii,[],2);
        % But there is an error in Matlab for max in second dimesion on GPU: https://au.mathworks.com/matlabcentral/answers/2152160-error-in-index-returned-by-max-in-the-second-dimension-in-obscure-case
        % So instead for now we instead do following two lines
        [~,maxindex1]=max(permute(entireRHS_ii,[2,1,3,4]),[],1);
        maxindex1=permute(maxindex1,[2,1,3,4]);
        
        % In 2D, this all gets overwritten as each layer2-ii-kk includes the edges, so I can skip it to save time
        % % Now, get and store the full (d,aprime)
        % [~,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a1*N_a2,vfoptions.level1n(1),vfoptions.level1n(2),N_z]),[],1);
        %
        % % Store
        % V(level11ii,level12kk,:)=shiftdim(Vtempii,1);
        % Policy(level11ii,level12kk,:)=shiftdim(maxindex2,1);

        %% Level 2
        % Split maxindex1 into a1prime and a2prime
        maxindex11=reshape(rem(maxindex1-1,N_a1)+1,[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2)]);
        maxindex12=reshape(ceil(maxindex1/N_a1),[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2)]);

        % Attempt for improved version
        maxgap1=squeeze(max(maxindex11(:,1,2:end,2:end)-maxindex11(:,1,1:end-1,1:end-1),[],1));
        maxgap2=squeeze(max(maxindex12(:,1,2:end,2:end)-maxindex12(:,1,1:end-1,1:end-1),[],1));
        for ii=1:(vfoptions.level1n(1)-1)
            for kk=1:(vfoptions.level1n(2)-1)
                curra1index=(level11ii(ii):1:level11ii(ii+1)); % Note: in 2D we need to include the edges
                curra2index=(level12kk(kk):1:level12kk(kk+1)); % Note: in 2D we need to include the edges

                if maxgap1(ii,kk)>0 && maxgap2(ii,kk)>0
                    loweredge1=min(maxindex11(:,1,ii,kk),N_a1-maxgap1(ii,kk)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    loweredge2=min(maxindex12(:,1,ii,kk),N_a2-maxgap2(ii,kk)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-1-by-1-by-n_z
                    a1primeindexes=loweredge1+repmat((0:1:maxgap1(ii,kk)),1,maxgap2(ii,kk)+1);
                    a2primeindexes=loweredge2+repelem((0:1:maxgap2(ii,kk)),1,maxgap1(ii,kk)+1);
                    % aprime possibilities are maxgap(ii)+1-n_a2-by-1-by-n_a2-by-n_z
                    ReturnMatrix_iikk=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, special_n_z, d_gridvals, a1_grid(a1primeindexes), a2_grid(a2primeindexes), a1_grid(level11ii(ii):level11ii(ii+1)), a2_grid(level12kk(kk):level12kk(kk+1)), z_gridvals_J(z_c,:,N_j), ReturnFnParamsVec,2);
                    daprime=(1:1:N_d)'+N_d*(a1primeindexes-1)+N_d*N_a1*(a2primeindexes-1); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_iikk+DiscountedentireEV_z(reshape(daprime,[N_d*(maxgap1(ii,kk)+1)*(maxgap2(ii,kk)+1),1,1]));  % Autofill level11iidiff(ii),level12kkdiff(kk) in the 2nd and 3rd dimensions
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curra1index,curra2index,z_c,N_j)=shiftdim(Vtempii,1);
                    % maxindex needs to be reworked:
                    %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                    dind=rem(maxindex-1,N_d)+1;
                    a1primeind=rem(ceil(maxindex/N_d)-1,maxgap1(ii,kk)+1)+1-1; % already includes -1
                    a2primeind=ceil(maxindex/(N_d*(maxgap1(ii,kk)+1)))-1; % already includes -1
                    maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                    %  the a1prime is relative to loweredge1(dind & zind), need to 'add'  the loweredge1
                    %  the a2prime is relative to loweredge2(dind & zind), need to 'add'  the loweredge2
                    Policy(curra1index,curra2index,z_c,N_j)=shiftdim(maxindexfix+N_d*(loweredge1(dind)-1)+N_d*N_a1*(loweredge2(dind)-1),1);

                elseif maxgap1(ii,kk)>0 && maxgap2(ii,kk)==0
                    loweredge1=min(maxindex11(:,1,ii,kk),N_a1-maxgap1(ii,kk)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    loweredge2=maxindex12(:,1,ii,kk);
                    % loweredge is n_d-by-1-by-1-by-1-by-n_z
                    a1primeindexes=loweredge1+(0:1:maxgap1(ii,kk));
                    ReturnMatrix_iikk=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, special_n_z, d_gridvals, a1_grid(a1primeindexes), reshape(a2_grid(loweredge2),[N_d,1,1,1]), a1_grid(level11ii(ii):level11ii(ii+1)), a2_grid(level12kk(kk):level12kk(kk+1)), z_gridvals_J(z_c,:,N_j), ReturnFnParamsVec,2);
                    daprime=(1:1:N_d)'+N_d*(a1primeindexes-1)+N_d*N_a1*(loweredge2-1); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_iikk+DiscountedentireEV_z(reshape(daprime,[N_d*(maxgap1(ii,kk)+1)*1,1,1]));  % Autofill level11iidiff(ii),level12kkdiff(kk) in the 2nd and 3rd dimensions
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curra1index,curra2index,z_c,N_j)=shiftdim(Vtempii,1);
                    % no need to rework maxindex in this case
                    dind=rem(maxindex-1,N_d)+1;
                    %  the a1prime is relative to loweredge1(dind & zind), need to 'add'  the loweredge1
                    %  the a2prime is relative to loweredge2(dind & zind), need to 'add'  the loweredge2
                    Policy(curra1index,curra2index,z_c,N_j)=shiftdim(maxindex+N_d*(loweredge1(dind)-1)+N_d*N_a1*(loweredge2(dind)-1),1);

                elseif maxgap1(ii,kk)==0 && maxgap2(ii,kk)>0
                    loweredge1=maxindex11(:,1,ii,kk);
                    loweredge2=min(maxindex12(:,1,ii,kk),N_a2-maxgap2(ii,kk)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-1-by-1-by-n_z (a1prime&a2prime,a1,a2,z)
                    a2primeindexes=loweredge2+(0:1:maxgap2(ii,kk));
                    ReturnMatrix_iikk=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, special_n_z, d_gridvals, reshape(a1_grid(loweredge1),[N_d,1,1,1]), a2_grid(a2primeindexes), a1_grid(level11ii(ii):level11ii(ii+1)), a2_grid(level12kk(kk):level12kk(kk+1)), z_gridvals_J(z_c,:,N_j), ReturnFnParamsVec,2);
                    daprime=(1:1:N_d)'+N_d*(loweredge1-1)+N_d*N_a1*(a2primeindexes-1); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_iikk+DiscountedentireEV_z(reshape(daprime,[N_d*(maxgap1(ii,kk)+1)*(maxgap2(ii,kk)+1),1,1]));  % Autofill level11iidiff(ii),level12kkdiff(kk) in the 2nd and 3rd dimensions
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curra1index,curra2index,z_c,N_j)=shiftdim(Vtempii,1);
                    %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                    %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                    dind=rem(maxindex-1,N_d)+1;
                    % a1primeind=1;
                    % a2primeind=ceil(maxindex/N_d);
                    maxindexfix=dind+N_d*N_a1*(ceil(maxindex/N_d)-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                    %  the a1prime is relative to loweredge1(dind & zind), need to 'add'  the loweredge1
                    %  the a2prime is relative to loweredge2(dind & zind), need to 'add'  the loweredge2
                    Policy(curra1index,curra2index,z_c,N_j)=shiftdim(maxindexfix+N_d*(loweredge1(dind)-1)+N_d*N_a1*(loweredge2(dind)-1),1);

                else % maxgap1(ii,kk)==0 && maxgap2(ii,kk)==0
                    loweredge1=maxindex11(:,1,ii,kk);
                    loweredge2=maxindex12(:,1,ii,kk);
                    % loweredge is n_d-by-1-by-1-by-1-by-n_z (a1prime&a2prime,a1,a2,z)
                    ReturnMatrix_iikk=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, special_n_z, d_gridvals, reshape(a1_grid(loweredge1),[N_d,1,1,1]), reshape(a2_grid(loweredge2),[N_d,1,1,1]), a1_grid(level11ii(ii):level11ii(ii+1)), a2_grid(level12kk(kk):level12kk(kk+1)), z_gridvals_J(z_c,:,N_j), ReturnFnParamsVec,2);
                    daprime=(1:1:N_d)'+N_d*(loweredge1-1)+N_d*N_a1*(loweredge2-1); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_iikk+DiscountedentireEV_z(reshape(daprime,[N_d*1*1,1,1]));  % Autofill level11iidiff(ii),level12kkdiff(kk) in the 2nd and 3rd dimensions
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curra1index,curra2index,z_c,N_j)=shiftdim(Vtempii,1);
                    % maxindex is anyway just the index for d, so
                    dind=maxindex;
                    %  the a1prime is relative to loweredge1(zind), need to 'add'  the loweredge1
                    %  the a2prime is relative to loweredge2(zind), need to 'add'  the loweredge2
                    Policy(curra1index,curra2index,z_c,N_j)=shiftdim(maxindex+N_d*(loweredge1(dind)-1)+N_d*N_a1*(loweredge2(dind)-1),1);
                end
            end
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

    EV=V(:,:,:,jj+1).*shiftdim(pi_z_J(:,:,jj)',-2);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,3); % sum over z', leaving a singular second dimension
    DiscountedentireEV=DiscountFactorParamsVec*reshape(repmat(shiftdim(EV,-1),N_d,1,1,1),[N_d,N_a1*N_a2,1,1,N_z]); % [d,aprime,1,1,z]

    for z_c=1:N_z
        DiscountedentireEV_z=DiscountedentireEV(:,:,1,1,z_c);
        
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, special_n_z, d_gridvals, a1_grid, a2_grid, a1_grid(level11ii), a2_grid(level12kk), z_gridvals_J(z_c,:,jj), ReturnFnParamsVec, 1);
        % (d,a1a2prime,a1,a2,z)

        entireRHS_ii=ReturnMatrix_ii+DiscountedentireEV_z;

        % First, we want a1a2prime conditional on (d,1,a,z)
        % We would just do
        % [~,maxindex1]=max(entireRHS_ii,[],2);
        % But there is an error in Matlab for max in second dimesion on GPU: https://au.mathworks.com/matlabcentral/answers/2152160-error-in-index-returned-by-max-in-the-second-dimension-in-obscure-case
        % So instead for now we instead do following two lines
        [~,maxindex1]=max(permute(entireRHS_ii,[2,1,3,4]),[],1);
        maxindex1=permute(maxindex1,[2,1,3,4]);


        % In 2D, this all gets overwritten as each layer2-ii-kk includes the edges, so I can skip it to save time
        % % Now, get and store the full (d,aprime)
        % [~,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a1*N_a2,vfoptions.level1n(1),vfoptions.level1n(2),N_z]),[],1);
        %
        % % Store
        % V(level11ii,level12kk,:)=shiftdim(Vtempii,1);
        % Policy(level11ii,level12kk,:)=shiftdim(maxindex2,1);

        %% Level 2
        % Split maxindex1 into a1prime and a2prime
        maxindex11=reshape(rem(maxindex1-1,N_a1)+1,[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2)]);
        maxindex12=reshape(ceil(maxindex1/N_a1),[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2)]);

        % Attempt for improved version
        maxgap1=squeeze(max(maxindex11(:,1,2:end,2:end)-maxindex11(:,1,1:end-1,1:end-1),[],1));
        maxgap2=squeeze(max(maxindex12(:,1,2:end,2:end)-maxindex12(:,1,1:end-1,1:end-1),[],1));
        for ii=1:(vfoptions.level1n(1)-1)
            for kk=1:(vfoptions.level1n(2)-1)
                curra1index=(level11ii(ii):1:level11ii(ii+1)); % Note: in 2D we need to include the edges
                curra2index=(level12kk(kk):1:level12kk(kk+1)); % Note: in 2D we need to include the edges

                if maxgap1(ii,kk)>0 && maxgap2(ii,kk)>0
                    loweredge1=min(maxindex11(:,1,ii,kk),N_a1-maxgap1(ii,kk)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    loweredge2=min(maxindex12(:,1,ii,kk),N_a2-maxgap2(ii,kk)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-1-by-1-by-n_z
                    a1primeindexes=loweredge1+repmat((0:1:maxgap1(ii,kk)),1,maxgap2(ii,kk)+1);
                    a2primeindexes=loweredge2+repelem((0:1:maxgap2(ii,kk)),1,maxgap1(ii,kk)+1);
                    % aprime possibilities are maxgap(ii)+1-n_a2-by-1-by-n_a2-by-n_z
                    ReturnMatrix_iikk=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, special_n_z, d_gridvals, a1_grid(a1primeindexes), a2_grid(a2primeindexes), a1_grid(level11ii(ii):level11ii(ii+1)), a2_grid(level12kk(kk):level12kk(kk+1)), z_gridvals_J(z_c,:,jj), ReturnFnParamsVec,2);
                    daprime=(1:1:N_d)'+N_d*(a1primeindexes-1)+N_d*N_a1*(a2primeindexes-1); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_iikk+DiscountedentireEV_z(reshape(daprime,[N_d*(maxgap1(ii,kk)+1)*(maxgap2(ii,kk)+1),1,1]));  % Autofill level11iidiff(ii),level12kkdiff(kk) in the 2nd and 3rd dimensions
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curra1index,curra2index,z_c,jj)=shiftdim(Vtempii,1);
                    % maxindex needs to be reworked:
                    %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                    dind=rem(maxindex-1,N_d)+1;
                    a1primeind=rem(ceil(maxindex/N_d)-1,maxgap1(ii,kk)+1)+1-1; % already includes -1
                    a2primeind=ceil(maxindex/(N_d*(maxgap1(ii,kk)+1)))-1; % already includes -1
                    maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                    %  the a1prime is relative to loweredge1(dind & zind), need to 'add'  the loweredge1
                    %  the a2prime is relative to loweredge2(dind & zind), need to 'add'  the loweredge2
                    Policy(curra1index,curra2index,z_c,jj)=shiftdim(maxindexfix+N_d*(loweredge1(dind)-1)+N_d*N_a1*(loweredge2(dind)-1),1);

                elseif maxgap1(ii,kk)>0 && maxgap2(ii,kk)==0
                    loweredge1=min(maxindex11(:,1,ii,kk),N_a1-maxgap1(ii,kk)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    loweredge2=maxindex12(:,1,ii,kk);
                    % loweredge is n_d-by-1-by-1-by-1-by-n_z
                    a1primeindexes=loweredge1+(0:1:maxgap1(ii,kk));
                    ReturnMatrix_iikk=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, special_n_z, d_gridvals, a1_grid(a1primeindexes), reshape(a2_grid(loweredge2),[N_d,1,1,1]), a1_grid(level11ii(ii):level11ii(ii+1)), a2_grid(level12kk(kk):level12kk(kk+1)), z_gridvals_J(z_c,:,jj), ReturnFnParamsVec,2);
                    daprime=(1:1:N_d)'+N_d*(a1primeindexes-1)+N_d*N_a1*(loweredge2-1); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_iikk+DiscountedentireEV(reshape(daprime,[N_d*(maxgap1(ii,kk)+1)*1,1,1]));  % Autofill level11iidiff(ii),level12kkdiff(kk) in the 2nd and 3rd dimensions
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curra1index,curra2index,z_c,jj)=shiftdim(Vtempii,1);
                    % no need to rework maxindex in this case
                    dind=rem(maxindex-1,N_d)+1;
                    %  the a1prime is relative to loweredge1(dind & zind), need to 'add'  the loweredge1
                    %  the a2prime is relative to loweredge2(dind & zind), need to 'add'  the loweredge2
                    Policy(curra1index,curra2index,z_c,jj)=shiftdim(maxindex+N_d*(loweredge1(dind)-1)+N_d*N_a1*(loweredge2(dind)-1),1);

                elseif maxgap1(ii,kk)==0 && maxgap2(ii,kk)>0
                    loweredge1=maxindex11(:,1,ii,kk);
                    loweredge2=min(maxindex12(:,1,ii,kk),N_a2-maxgap2(ii,kk)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-1-by-1-by-n_z (a1prime&a2prime,a1,a2,z)
                    a2primeindexes=loweredge2+(0:1:maxgap2(ii,kk));
                    ReturnMatrix_iikk=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, special_n_z, d_gridvals, reshape(a1_grid(loweredge1),[N_d,1,1,1]), a2_grid(a2primeindexes), a1_grid(level11ii(ii):level11ii(ii+1)), a2_grid(level12kk(kk):level12kk(kk+1)), z_gridvals_J(z_c,:,jj), ReturnFnParamsVec,2);
                    daprime=(1:1:N_d)'+N_d*(loweredge1-1)+N_d*N_a1*(a2primeindexes-1); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_iikk+DiscountedentireEV(reshape(daprime,[N_d*(maxgap1(ii,kk)+1)*(maxgap2(ii,kk)+1),1,1]));  % Autofill level11iidiff(ii),level12kkdiff(kk) in the 2nd and 3rd dimensions
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curra1index,curra2index,z_c,jj)=shiftdim(Vtempii,1);
                    %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                    %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                    dind=rem(maxindex-1,N_d)+1;
                    % a1primeind=1;
                    % a2primeind=ceil(maxindex/N_d);
                    maxindexfix=dind+N_d*N_a1*(ceil(maxindex/N_d)-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                    %  the a1prime is relative to loweredge1(dind & zind), need to 'add'  the loweredge1
                    %  the a2prime is relative to loweredge2(dind & zind), need to 'add'  the loweredge2
                    Policy(curra1index,curra2index,z_c,jj)=shiftdim(maxindexfix+N_d*(loweredge1(dind)-1)+N_d*N_a1*(loweredge2(dind)-1),1);

                else % maxgap1(ii,kk)==0 && maxgap2(ii,kk)==0
                    loweredge1=maxindex11(:,1,ii,kk);
                    loweredge2=maxindex12(:,1,ii,kk);
                    % loweredge is n_d-by-1-by-1-by-1-by-n_z (a1prime&a2prime,a1,a2,z)
                    ReturnMatrix_iikk=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, special_n_z, d_gridvals, reshape(a1_grid(loweredge1),[N_d,1,1,1]), reshape(a2_grid(loweredge2),[N_d,1,1,1]), a1_grid(level11ii(ii):level11ii(ii+1)), a2_grid(level12kk(kk):level12kk(kk+1)), z_gridvals_J(z_c,:,jj), ReturnFnParamsVec,2);
                    daprime=(1:1:N_d)'+N_d*(loweredge1-1)+N_d*N_a1*(loweredge2-1); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_iikk+DiscountedentireEV(reshape(daprime,[N_d*1*1,1,1]));  % Autofill level11iidiff(ii),level12kkdiff(kk) in the 2nd and 3rd dimensions
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curra1index,curra2index,z_c,jj)=shiftdim(Vtempii,1);
                    % maxindex is anyway just the index for d, so
                    dind=maxindex;
                    %  the a1prime is relative to loweredge1(zind), need to 'add'  the loweredge1
                    %  the a2prime is relative to loweredge2(zind), need to 'add'  the loweredge2
                    Policy(curra1index,curra2index,z_c,jj)=shiftdim(maxindex+N_d*(loweredge1(dind)-1)+N_d*N_a1*(loweredge2(dind)-1),1);
                end
            end
        end
    end

end

% Can skip V reshape as code works without this, but needs to be done for Policy (or more precisely for Policy2)
Policy=reshape(Policy,[N_a,N_z,N_j]);

%%
Policy2=zeros(2,N_a,N_z,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
Policy2(1,:,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:,:)=shiftdim(ceil(Policy/N_d),-1);


end