function [V,Policy]=ValueFnIter_FHorz_EpsteinZin_DC1_noz_e_raw(n_d,n_a,n_e,N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7, ezc8)
% Divide-and-conquer version of ValueFnIter_FHorz_EpsteinZin_noz_e_raw.
% Grafts the Epstein-Zin transforms onto ValueFnIter_FHorz_DC1_noz_e_raw: temp4
% (the post-certainty-equivalent continuation) is pointwise in aprime, so it is
% computed once per age over the full aprime grid and indexed exactly where the
% vNM code indexes EV; the return transform and the final ^ezc7 wrap each
% level's entireRHS before its max (a monotone transform, so the
% divide-and-conquer monotonicity logic is unaffected).

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
else
    eind=shiftdim((0:1:N_e-1),-1); % already includes -1
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
if vfoptions.EZoneminusbeta==1
    ezc1=1-DiscountFactorParamsVec; % Just in case it depends on age
elseif vfoptions.EZoneminusbeta==2
    ezc1=1-sj(N_j)*DiscountFactorParamsVec;
end

pi_e_J=shiftdim(pi_e_J,-1); % Move to second dimension (normally -2, but no z so -1)

% If there is a warm-glow at end of the final period, evaluate the warmglowfn
if warmglow==1
    WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,N_j);
    WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a, a_grid, WGParamsVec);
    WGmatrix=WGmatrixraw;
    WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5(N_j);
    WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
    if ~isfield(vfoptions,'V_Jplus1')
        becareful=(WGmatrix==0);
        WGmatrix(isfinite(WGmatrix))=ezc3*DiscountFactorParamsVec*(((1-sj(N_j))*WGmatrix(isfinite(WGmatrix)).^ezc8(N_j)).^ezc6(N_j));
        WGmatrix(becareful)=0;
    end
    % WGmatrix is a column over the full aprime grid; it is indexed by aprime below
else
    WGmatrix=zeros(N_a,1,'gpuArray');
end

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_e, d_gridvals, a_grid, a_grid(level1ii), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
        ReturnMatrix_ii(becareful)=(ezc1*ReturnMatrix_ii(becareful).^ezc2(N_j)).^ezc7(N_j);
        ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
        entireRHS_ii=ReturnMatrix_ii+shiftdim(WGmatrix,-1); % warm-glow (zero if not using)

        % First, we want aprime conditional on (d,1,a,e)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n,N_e]),[],1);

        % Store
        V(level1ii,:,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,:,N_j)=shiftdim(maxindex2,1); % d,aprime

        % Attempt for improved version
        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1)); % max over d,e
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-1-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                ReturnMatrix_ii(becareful)=(ezc1*ReturnMatrix_ii(becareful).^ezc2(N_j)).^ezc7(N_j);
                ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
                entireRHS_ii=ReturnMatrix_ii+reshape(WGmatrix(aprimeindexes),[N_d*(maxgap(ii)+1),1,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*eind; % loweredge is n_d-by-1-by-1-by-n_e
                Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                ReturnMatrix_ii(becareful)=(ezc1*ReturnMatrix_ii(becareful).^ezc2(N_j)).^ezc7(N_j);
                ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
                entireRHS_ii=ReturnMatrix_ii+reshape(WGmatrix(loweredge),[N_d*1,1,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*eind; % loweredge is n_d-by-1-by-1-by-n_e
                Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            end
        end
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_e, d_gridvals, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec,1);

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
            ReturnMatrix_ii(becareful)=(ezc1*ReturnMatrix_ii(becareful).^ezc2(N_j)).^ezc7(N_j);
            ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
            entireRHS_ii=ReturnMatrix_ii+shiftdim(WGmatrix,-1); % warm-glow (zero if not using)

            % First, we want aprime conditional on (d,1,a,e)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);

            % Store
            V(level1ii,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(level1ii,e_c,N_j)=shiftdim(maxindex2,1); % d,aprime

            % Attempt for improved version
            maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1)); % max over d,e
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-1
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,2);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                    ReturnMatrix_ii(becareful)=(ezc1*ReturnMatrix_ii(becareful).^ezc2(N_j)).^ezc7(N_j);
                    ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
                    entireRHS_ii=ReturnMatrix_ii+reshape(WGmatrix(aprimeindexes),[N_d*(maxgap(ii)+1),1]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    allind=dind; % loweredge is n_d-by-1-by-1
                    Policy(curraindex,e_c,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,2);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                    ReturnMatrix_ii(becareful)=(ezc1*ReturnMatrix_ii(becareful).^ezc2(N_j)).^ezc7(N_j);
                    ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
                    entireRHS_ii=ReturnMatrix_ii+reshape(WGmatrix(loweredge),[N_d*1,1]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    % allind=dind; % loweredge is n_d-by-1-by-1
                    Policy(curraindex,e_c,N_j)=maxindex'+N_d*(loweredge(dind)-1);
                end
            end
        end
    end

else

    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_e]);    % First, switch V_Jplus1 into Kron form

    % Part of Epstein-Zin is before taking expectation
    temp=V_Jplus1;
    temp(isfinite(V_Jplus1))=(ezc4*V_Jplus1(isfinite(V_Jplus1))).^ezc5(N_j);
    temp(V_Jplus1==0)=0;

    % Take expectation over e (e is iid, so this is the whole expectation)
    EV=sum(temp.*pi_e_J(1,:,N_j+1),2);

    % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over aprime
    temp4=EV;
    if warmglow==1
        becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
        temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrix(becareful).^ezc8(N_j)).^ezc6(N_j);
        temp4((EV==0)&(WGmatrix==0))=0; % Is actually zero
    else % not using warmglow
        temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4)).^ezc8(N_j)).^ezc6(N_j);
        temp4(EV==0)=0;
    end

    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_e, d_gridvals, a_grid, a_grid(level1ii), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
        temp2_ii=ReturnMatrix_ii;
        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
        temp2_ii(ReturnMatrix_ii==0)=-Inf;

        entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*shiftdim(temp4,-1);

        temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
        entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
        entireRHS_ii(entireRHS_ii==0)=-Inf;

        % First, we want aprime conditional on (d,1,a,z)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n,N_e]),[],1);

        % Store
        V(level1ii,:,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,:,N_j)=shiftdim(maxindex2,1); % d,aprime

        % Attempt for improved version
        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1)); % max over d,e
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-1-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                temp2_ii=ReturnMatrix_ii;
                temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                temp2_ii(ReturnMatrix_ii==0)=-Inf;
                entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimeindexes),[N_d*(maxgap(ii)+1),1,N_e]); % autoexpand level1iidiff(ii) in 2nd-dim
                temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                entireRHS_ii(entireRHS_ii==0)=-Inf;
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*eind; % loweredge is n_d-by-1-by-1-by-n_e
                Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                temp2_ii=ReturnMatrix_ii;
                temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                temp2_ii(ReturnMatrix_ii==0)=-Inf;
                entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(loweredge),[N_d*1,1,N_e]); % autoexpand level1iidiff(ii) in 2nd-dim
                temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                entireRHS_ii(entireRHS_ii==0)=-Inf;
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*eind; % loweredge is n_d-by-1-by-1-by-n_e
                Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            end
        end
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);

            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_e, d_gridvals, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec,1);

            becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
            temp2_ii=ReturnMatrix_ii;
            temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
            temp2_ii(ReturnMatrix_ii==0)=-Inf;

            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*shiftdim(temp4,-1);

            temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
            entireRHS_ii(entireRHS_ii==0)=-Inf;

            % First, we want aprime conditional on (d,1,a,z)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);

            % Store
            V(level1ii,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(level1ii,e_c,N_j)=shiftdim(maxindex2,1); % d,aprime

            % Attempt for improved version
            maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1)); % max over d,e
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-1
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,2);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                    temp2_ii(ReturnMatrix_ii==0)=-Inf;
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimeindexes),[N_d*(maxgap(ii)+1),1]);  % autoexpand level1iidiff(ii) in 2nd-dim
                    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                    entireRHS_ii(entireRHS_ii==0)=-Inf;
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    % allind=dind; % loweredge is n_d-by-1-by-1
                    Policy(curraindex,e_c,N_j)=maxindex'+N_d*(loweredge(dind)-1);
                else
                    loweredge=maxindex1(:,1,ii);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,2);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                    temp2_ii(ReturnMatrix_ii==0)=-Inf;
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*temp4(loweredge); % autoexpand level1iidiff(ii) in 2nd-dim
                    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                    entireRHS_ii(entireRHS_ii==0)=-Inf;
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    % allind=dind; % loweredge is n_d-by-1-by-1
                    Policy(curraindex,e_c,N_j)=maxindex'+N_d*(loweredge(dind)-1);
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
    if vfoptions.EZoneminusbeta==1
        ezc1=1-DiscountFactorParamsVec; % Just in case it depends on age
    elseif vfoptions.EZoneminusbeta==2
        ezc1=1-sj(jj)*DiscountFactorParamsVec;
    end

    % If there is a warm-glow, evaluate the warmglowfn
    if warmglow==1
        WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
        WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a, a_grid, WGParamsVec);
        WGmatrix=WGmatrixraw;
        WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5(jj);
        WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
        % WGmatrix is a column over the full aprime grid; combined into temp4 below
    end

    EVpre=V(:,:,jj+1);
    % Part of Epstein-Zin is before taking expectation
    temp=EVpre;
    temp(isfinite(EVpre))=(ezc4*EVpre(isfinite(EVpre))).^ezc5(jj);
    temp(EVpre==0)=0;

    % Take expectation over e (e is iid, so this is the whole expectation)
    EV=sum(temp.*pi_e_J(1,:,jj+1),2);

    % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over aprime
    temp4=EV;
    if warmglow==1
        becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
        temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrix(becareful).^ezc8(jj)).^ezc6(jj);
        temp4((EV==0)&(WGmatrix==0))=0; % Is actually zero
    else % not using warmglow
        temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
        temp4(EV==0)=0;
    end

    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_e, d_gridvals, a_grid, a_grid(level1ii), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
        temp2_ii=ReturnMatrix_ii;
        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
        temp2_ii(ReturnMatrix_ii==0)=-Inf;

        entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*shiftdim(temp4,-1);

        temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
        entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
        entireRHS_ii(entireRHS_ii==0)=-Inf;

        % First, we want aprime conditional on (d,1,a,z)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n,N_e]),[],1);

        % Store
        V(level1ii,:,jj)=shiftdim(Vtempii,1);
        Policy(level1ii,:,jj)=shiftdim(maxindex2,1); % d,aprime

        % Attempt for improved version
        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1)); % max over d,e
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-1-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                temp2_ii=ReturnMatrix_ii;
                temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                temp2_ii(ReturnMatrix_ii==0)=-Inf;
                entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimeindexes),[N_d*(maxgap(ii)+1),1,N_e]);  % autoexpand level1iidiff(ii) in 2nd-dim
                temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                entireRHS_ii(entireRHS_ii==0)=-Inf;
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,jj)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*eind; % loweredge is n_d-by-1-by-1-by-n_e
                Policy(curraindex,:,jj)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                temp2_ii=ReturnMatrix_ii;
                temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                temp2_ii(ReturnMatrix_ii==0)=-Inf;
                entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(loweredge),[N_d*1,1,N_e]); % autoexpand level1iidiff(ii) in 2nd-dim
                temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                entireRHS_ii(entireRHS_ii==0)=-Inf;
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,jj)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*eind; % loweredge is n_d-by-1-by-1-by-n_e
                Policy(curraindex,:,jj)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            end
        end
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);

            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_e, d_gridvals, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec,1);

            becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
            temp2_ii=ReturnMatrix_ii;
            temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
            temp2_ii(ReturnMatrix_ii==0)=-Inf;

            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*shiftdim(temp4,-1);

            temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
            entireRHS_ii(entireRHS_ii==0)=-Inf;

            % First, we want aprime conditional on (d,1,a,z)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);

            % Store
            V(level1ii,e_c,jj)=shiftdim(Vtempii,1);
            Policy(level1ii,e_c,jj)=shiftdim(maxindex2,1); % d,aprime

            % Attempt for improved version
            maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1)); % max over d,e
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-1
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,2);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                    temp2_ii(ReturnMatrix_ii==0)=-Inf;
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimeindexes),[N_d*(maxgap(ii)+1),1]);  % autoexpand level1iidiff(ii) in 2nd-dim
                    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                    entireRHS_ii(entireRHS_ii==0)=-Inf;
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curraindex,e_c,jj)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    % allind=dind; % loweredge is n_d-by-1-by-1
                    Policy(curraindex,e_c,jj)=maxindex'+N_d*(loweredge(dind)-1);
                else
                    loweredge=maxindex1(:,1,ii);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,2);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                    temp2_ii(ReturnMatrix_ii==0)=-Inf;
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*temp4(loweredge);  % autoexpand level1iidiff(ii) in 2nd-dim
                    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                    entireRHS_ii(entireRHS_ii==0)=-Inf;
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curraindex,e_c,jj)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    % allind=dind; % loweredge is n_d-by-1-by-1
                    Policy(curraindex,e_c,jj)=maxindex'+N_d*(loweredge(dind)-1);
                end
            end
        end
    end
end

%%
Policy=shiftdim(Policy,-1);

end
