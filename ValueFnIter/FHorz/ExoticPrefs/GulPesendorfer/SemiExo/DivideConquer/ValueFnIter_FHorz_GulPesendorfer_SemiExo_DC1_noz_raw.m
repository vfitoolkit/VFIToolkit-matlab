function [V,Policy3]=ValueFnIter_FHorz_GulPesendorfer_SemiExo_DC1_noz_raw(n_d1,n_d2,n_a,n_semiz,N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions)
% Gul-Pesendorfer with a semi-exogenous state and divide-and-conquer on a. The tempted
% objective u+v+beta*EV goes through the standard per-d2 DC machinery (an inner solve per d2
% value, then a max over d2, so the windows/argmax are those of the TEMPTED objective); the
% most-tempting term is a max over the FULL (d1,d2,aprime) choice set: a per-d2 full (d1,aprime)
% max is collected alongside each inner solve (from the level-1 temptation matrix at the level-1
% stations, and from full-column temptation matrices one a-slab at a time for the intermediate a,
% never over a window) and the max over d2 is subtracted from V after the outer max (the '-max v'
% term is constant w.r.t. the choice given the state, so the subtraction after the d2-max is exact).

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
special_n_d=[n_d1,ones(1,length(n_d2))];
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];

d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_d2,length(n_d1)+length(n_d2)]),[1,3,2]); % version to use when looping over d2

if vfoptions.lowmemory==1
    special_n_semiz=ones(1,length(n_semiz));
end

semizind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % already includes -1
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-2); % already includes -1

% Preallocate
V_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');
MostTempting_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray'); % per-d2 full (d1,aprime) max of the temptation

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames,N_j);


if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            % n-Monotonicity
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
            TemptationMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(TemptationFn, special_n_d, n_semiz, d12c_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), TemptationFnParamsVec,1);
            MostTempting_ford2_jj(level1ii,:,d2_c)=shiftdim(max(reshape(TemptationMatrix_d2ii,[N_d1*N_a,vfoptions.level1n,N_semiz]),[],1),1); % full (d1,aprime) grid at the level-1 stations

            entireRHS_ii=ReturnMatrix_d2ii+TemptationMatrix_d2ii;

            % First, we want aprime conditional on (d,1,a,z)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_a,vfoptions.level1n,N_semiz]),[],1);

            % Store
            V_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(level1ii,:,d2_c)=shiftdim(maxindex2,1); % d,aprime


            % Second level based on monotonicity
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                % Most-tempting term over the FULL (d1,aprime) grid for these a (never just the window)
                TemptationMatrix_full=CreateReturnFnMatrix_Disc_DC1(TemptationFn, special_n_d, n_semiz, d12c_gridvals, a_grid, a_grid(curraindex), semiz_gridvals_J(:,:,N_j), TemptationFnParamsVec,1);
                MostTempting_ford2_jj(curraindex,:,d2_c)=shiftdim(max(reshape(TemptationMatrix_full,[N_d1*N_a,length(curraindex),N_semiz]),[],1),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_semiz
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_semiz
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                    TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1(TemptationFn, special_n_d, n_semiz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), TemptationFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii;
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    allind=dind+N_d1*semizind; % loweredge is n_d-by-1-by-1-by-n_semiz
                    Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+N_d1*(reshape(loweredge(allind),size(maxindex))-1)); % loweredge(given the d and z)
                else
                    loweredge=maxindex1(:,1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                    TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1(TemptationFn, special_n_d, n_semiz, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), TemptationFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii;
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    allind=dind+N_d1*semizind; % loweredge is n_d-by-1-by-1-by-n_semiz
                    Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+N_d1*(reshape(loweredge(allind),size(maxindex))-1)); % loweredge(given the d and z)
                end
            end
        end

    elseif vfoptions.lowmemory==1 % loop semiz

        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            for semiz_c=1:N_semiz
                semiz_val=semiz_gridvals_J(semiz_c,:,N_j);

                % n-Monotonicity
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid, a_grid(level1ii), semiz_val, ReturnFnParamsVec,1);
                TemptationMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(TemptationFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid, a_grid(level1ii), semiz_val, TemptationFnParamsVec,1);
                MostTempting_ford2_jj(level1ii,semiz_c,d2_c)=shiftdim(max(reshape(TemptationMatrix_d2ii,[N_d1*N_a,vfoptions.level1n]),[],1),1); % full (d1,aprime) grid at the level-1 stations

                entireRHS_ii=ReturnMatrix_d2ii+TemptationMatrix_d2ii;

                % First, we want aprime conditional on (d,1,a,z)
                [~,maxindex1]=max(entireRHS_ii,[],2);

                % Now, get and store the full (d,aprime)
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_a,vfoptions.level1n]),[],1);

                % Store
                V_ford2_jj(level1ii,semiz_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(level1ii,semiz_c,d2_c)=shiftdim(maxindex2,1); % d,aprime


                % Second level based on monotonicity
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    % Most-tempting term over the FULL (d1,aprime) grid for these a (never just the window)
                    TemptationMatrix_full=CreateReturnFnMatrix_Disc_DC1(TemptationFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid, a_grid(curraindex), semiz_val, TemptationFnParamsVec,1);
                    MostTempting_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(max(reshape(TemptationMatrix_full,[N_d1*N_a,length(curraindex)]),[],1),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d-by-1
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d-by-maxgap(ii)+1
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_val, ReturnFnParamsVec,2);
                        TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1(TemptationFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_val, TemptationFnParamsVec,2);
                        entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii;
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        allind=dind; % single semiz
                        Policy_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(maxindex+N_d1*(reshape(loweredge(allind),size(maxindex))-1)); % loweredge(given the d)
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_val, ReturnFnParamsVec,2);
                        TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1(TemptationFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_val, TemptationFnParamsVec,2);
                        entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii;
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        allind=dind; % single semiz
                        Policy_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(maxindex+N_d1*(reshape(loweredge(allind),size(maxindex))-1)); % loweredge(given the d)
                    end
                end
            end
        end

    end
    % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
    V(:,:,N_j)=V_jj-max(MostTempting_ford2_jj,[],3); % subtract the full (d1,d2,aprime) most-tempting term
    Policy3(2,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
    Policy3(1,:,:,N_j)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
    Policy3(3,:,:,N_j)=shiftdim(ceil(d1aprime_ind/N_d1),-1);

else
    % Using V_Jplus1
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);    % First, switch V_Jplus1 into Kron form

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=pi_semiz_J(:,:,d2_c,N_j); % reverse order

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            % n-Monotonicity
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
            TemptationMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(TemptationFn, special_n_d, n_semiz, d12c_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), TemptationFnParamsVec,1);
            MostTempting_ford2_jj(level1ii,:,d2_c)=shiftdim(max(reshape(TemptationMatrix_d2ii,[N_d1*N_a,vfoptions.level1n,N_semiz]),[],1),1); % full (d1,aprime) grid at the level-1 stations

            entireRHS_ii=ReturnMatrix_d2ii+TemptationMatrix_d2ii+DiscountFactorParamsVec*shiftdim(EV_d2,-1);

            % First, we want aprime conditional on (d,1,a,z)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_a,vfoptions.level1n,N_semiz]),[],1);

            % Store
            V_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(level1ii,:,d2_c)=shiftdim(maxindex2,1); % d,aprime

            % Second level based on monotonicity
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                % Most-tempting term over the FULL (d1,aprime) grid for these a (never just the window)
                TemptationMatrix_full=CreateReturnFnMatrix_Disc_DC1(TemptationFn, special_n_d, n_semiz, d12c_gridvals, a_grid, a_grid(curraindex), semiz_gridvals_J(:,:,N_j), TemptationFnParamsVec,1);
                MostTempting_ford2_jj(curraindex,:,d2_c)=shiftdim(max(reshape(TemptationMatrix_full,[N_d1*N_a,length(curraindex),N_semiz]),[],1),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_semiz
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_semiz
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                    TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1(TemptationFn, special_n_d, n_semiz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), TemptationFnParamsVec,2);
                    aprimez=aprimeindexes+N_a*semizBind;
                    entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii+DiscountFactorParamsVec*reshape(EV_d2(aprimez),[N_d1*(maxgap(ii)+1),1,N_semiz]); % autoexpand level1iidiff(ii) in 2nd-dim
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    allind=dind+N_d1*semizind; % loweredge is n_d-by-1-by-1-by-n_semiz
                    Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+N_d1*(reshape(loweredge(allind),size(maxindex))-1)); % loweredge(given the d and z)
                else
                    loweredge=maxindex1(:,1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                    TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1(TemptationFn, special_n_d, n_semiz, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), TemptationFnParamsVec,2);
                    aprimez=loweredge+N_a*semizBind;
                    entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii+DiscountFactorParamsVec*reshape(EV_d2(aprimez),[N_d1,1,N_semiz]); % autoexpand level1iidiff(ii) in 2nd-dim
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    allind=dind+N_d1*semizind; % loweredge is n_d-by-1-by-1-by-n_semiz
                    Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+N_d1*(reshape(loweredge(allind),size(maxindex))-1)); % loweredge(given the d and z)
                end
            end
        end

    elseif vfoptions.lowmemory==1 % loop semiz
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j); % reverse order

            for semiz_c=1:N_semiz
                semiz_val=semiz_gridvals_J(semiz_c,:,N_j);

                % Calc the condl expectation term (except beta), which depends on semiz but not on control variables
                EV_d2z=EV.*pi_semiz(semiz_c,:);
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2); % sum over semiz', leaving a singular second dimension

                % n-Monotonicity
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid, a_grid(level1ii), semiz_val, ReturnFnParamsVec,1);
                TemptationMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(TemptationFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid, a_grid(level1ii), semiz_val, TemptationFnParamsVec,1);
                MostTempting_ford2_jj(level1ii,semiz_c,d2_c)=shiftdim(max(reshape(TemptationMatrix_d2ii,[N_d1*N_a,vfoptions.level1n]),[],1),1); % full (d1,aprime) grid at the level-1 stations

                entireRHS_ii=ReturnMatrix_d2ii+TemptationMatrix_d2ii+DiscountFactorParamsVec*shiftdim(EV_d2z,-1);

                % First, we want aprime conditional on (d,1,a,z)
                [~,maxindex1]=max(entireRHS_ii,[],2);

                % Now, get and store the full (d,aprime)
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_a,vfoptions.level1n]),[],1);

                % Store
                V_ford2_jj(level1ii,semiz_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(level1ii,semiz_c,d2_c)=shiftdim(maxindex2,1); % d,aprime

                % Second level based on monotonicity
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    % Most-tempting term over the FULL (d1,aprime) grid for these a (never just the window)
                    TemptationMatrix_full=CreateReturnFnMatrix_Disc_DC1(TemptationFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid, a_grid(curraindex), semiz_val, TemptationFnParamsVec,1);
                    MostTempting_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(max(reshape(TemptationMatrix_full,[N_d1*N_a,length(curraindex)]),[],1),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d-by-1
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d-by-maxgap(ii)+1
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_val, ReturnFnParamsVec,2);
                        TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1(TemptationFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_val, TemptationFnParamsVec,2);
                        aprimez=aprimeindexes; % single semiz
                        entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii+DiscountFactorParamsVec*reshape(EV_d2z(aprimez),[N_d1*(maxgap(ii)+1),1]); % autoexpand level1iidiff(ii) in 2nd-dim
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        allind=dind; % single semiz
                        Policy_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(maxindex+N_d1*(reshape(loweredge(allind),size(maxindex))-1)); % loweredge(given the d)
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_val, ReturnFnParamsVec,2);
                        TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1(TemptationFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_val, TemptationFnParamsVec,2);
                        aprimez=loweredge; % single semiz
                        entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii+DiscountFactorParamsVec*reshape(EV_d2z(aprimez),[N_d1,1]); % autoexpand level1iidiff(ii) in 2nd-dim
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        allind=dind; % single semiz
                        Policy_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(maxindex+N_d1*(reshape(loweredge(allind),size(maxindex))-1)); % loweredge(given the d)
                    end
                end
            end
        end

    end
    % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
    V(:,:,N_j)=V_jj-max(MostTempting_ford2_jj,[],3); % subtract the full (d1,d2,aprime) most-tempting term
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
    TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=V(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_bothz=pi_semiz_J(:,:,d2_c,jj); % reverse order

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            % n-Monotonicity
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,1);
            TemptationMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(TemptationFn, special_n_d, n_semiz, d12c_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,jj), TemptationFnParamsVec,1);
            MostTempting_ford2_jj(level1ii,:,d2_c)=shiftdim(max(reshape(TemptationMatrix_d2ii,[N_d1*N_a,vfoptions.level1n,N_semiz]),[],1),1); % full (d1,aprime) grid at the level-1 stations

            entireRHS_ii=ReturnMatrix_d2ii+TemptationMatrix_d2ii+DiscountFactorParamsVec*shiftdim(EV_d2,-1);

            % First, we want aprime conditional on (d,1,a,z)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_a,vfoptions.level1n,N_semiz]),[],1);

            % Store
            V_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(level1ii,:,d2_c)=shiftdim(maxindex2,1); % d,aprime

            % Second level based on monotonicity
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                % Most-tempting term over the FULL (d1,aprime) grid for these a (never just the window)
                TemptationMatrix_full=CreateReturnFnMatrix_Disc_DC1(TemptationFn, special_n_d, n_semiz, d12c_gridvals, a_grid, a_grid(curraindex), semiz_gridvals_J(:,:,jj), TemptationFnParamsVec,1);
                MostTempting_ford2_jj(curraindex,:,d2_c)=shiftdim(max(reshape(TemptationMatrix_full,[N_d1*N_a,length(curraindex),N_semiz]),[],1),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_semiz
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_semiz
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                    TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1(TemptationFn, special_n_d, n_semiz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,jj), TemptationFnParamsVec,2);
                    aprimez=aprimeindexes+N_a*semizBind;
                    entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii+DiscountFactorParamsVec*reshape(EV_d2(aprimez),[N_d1*(maxgap(ii)+1),1,N_semiz]); % autoexpand level1iidiff(ii) in 2nd-dim
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    allind=dind+N_d1*semizind; % loweredge is n_d-by-1-by-1-by-n_semiz
                    Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+N_d1*(reshape(loweredge(allind),size(maxindex))-1)); % loweredge(given the d and z)
                else
                    loweredge=maxindex1(:,1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                    TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1(TemptationFn, special_n_d, n_semiz, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,jj), TemptationFnParamsVec,2);
                    aprimez=loweredge+N_a*semizBind;
                    entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii+DiscountFactorParamsVec*reshape(EV_d2(aprimez),[N_d1,1,N_semiz]); % autoexpand level1iidiff(ii) in 2nd-dim
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    allind=dind+N_d1*semizind; % loweredge is n_d-by-1-by-1-by-n_semiz
                    Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+N_d1*(reshape(loweredge(allind),size(maxindex))-1)); % loweredge(given the d and z)
                end
            end
        end

    elseif vfoptions.lowmemory==1 % loop semiz
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_semiz=pi_semiz_J(:,:,d2_c,jj); % reverse order

            for semiz_c=1:N_semiz
                semiz_val=semiz_gridvals_J(semiz_c,:,jj);

                % Calc the condl expectation term (except beta), which depends on semiz but not on control variables
                EV_d2z=EV.*pi_semiz(semiz_c,:);
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2); % sum over semiz', leaving a singular second dimension

                % n-Monotonicity
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid, a_grid(level1ii), semiz_val, ReturnFnParamsVec,1);
                TemptationMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(TemptationFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid, a_grid(level1ii), semiz_val, TemptationFnParamsVec,1);
                MostTempting_ford2_jj(level1ii,semiz_c,d2_c)=shiftdim(max(reshape(TemptationMatrix_d2ii,[N_d1*N_a,vfoptions.level1n]),[],1),1); % full (d1,aprime) grid at the level-1 stations

                entireRHS_ii=ReturnMatrix_d2ii+TemptationMatrix_d2ii+DiscountFactorParamsVec*shiftdim(EV_d2z,-1);

                % First, we want aprime conditional on (d,1,a,z)
                [~,maxindex1]=max(entireRHS_ii,[],2);

                % Now, get and store the full (d,aprime)
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_a,vfoptions.level1n]),[],1);

                % Store
                V_ford2_jj(level1ii,semiz_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(level1ii,semiz_c,d2_c)=shiftdim(maxindex2,1); % d,aprime

                % Second level based on monotonicity
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    % Most-tempting term over the FULL (d1,aprime) grid for these a (never just the window)
                    TemptationMatrix_full=CreateReturnFnMatrix_Disc_DC1(TemptationFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid, a_grid(curraindex), semiz_val, TemptationFnParamsVec,1);
                    MostTempting_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(max(reshape(TemptationMatrix_full,[N_d1*N_a,length(curraindex)]),[],1),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d-by-1
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d-by-maxgap(ii)+1
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_val, ReturnFnParamsVec,2);
                        TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1(TemptationFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_val, TemptationFnParamsVec,2);
                        aprimez=aprimeindexes; % single semiz
                        entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii+DiscountFactorParamsVec*reshape(EV_d2z(aprimez),[N_d1*(maxgap(ii)+1),1]); % autoexpand level1iidiff(ii) in 2nd-dim
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        allind=dind; % single semiz
                        Policy_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(maxindex+N_d1*(reshape(loweredge(allind),size(maxindex))-1)); % loweredge(given the d)
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_val, ReturnFnParamsVec,2);
                        TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1(TemptationFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_val, TemptationFnParamsVec,2);
                        aprimez=loweredge; % single semiz
                        entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii+DiscountFactorParamsVec*reshape(EV_d2z(aprimez),[N_d1,1]); % autoexpand level1iidiff(ii) in 2nd-dim
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        allind=dind; % single semiz
                        Policy_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(maxindex+N_d1*(reshape(loweredge(allind),size(maxindex))-1)); % loweredge(given the d)
                    end
                end
            end
        end

    end
    % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
    V(:,:,jj)=V_jj-max(MostTempting_ford2_jj,[],3); % subtract the full (d1,d2,aprime) most-tempting term
    Policy3(2,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
    Policy3(1,:,:,jj)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
    Policy3(3,:,:,jj)=shiftdim(ceil(d1aprime_ind/N_d1),-1);

end


end
