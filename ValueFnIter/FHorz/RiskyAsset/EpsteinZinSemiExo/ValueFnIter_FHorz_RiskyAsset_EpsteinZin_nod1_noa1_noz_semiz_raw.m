function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_nod1_noa1_noz_semiz_raw(n_d2,n_d3,n_d4,n_a,n_semiz,n_u,N_j, d2_grid, d3_grid, d4_grid, a_grid, semiz_gridvals_J, u_grid, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9)
% Epstein-Zin preferences
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% d4: ReturnFn but not aprimeFn, and determines semiz transitions
% noa1; no z (only semiz)

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_d4=prod(n_d4);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_u=prod(n_u);

special_n_d4=ones(1,length(n_d4));
d4_gridvals=CreateGridvals(n_d4,d4_grid,1);

n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_semiz,N_j,'gpuArray');
Policy3=zeros(3,N_a,N_semiz,N_j,'gpuArray'); % d2, d3, d4

%%
d23_grid=gpuArray(d23_grid);
a_grid=gpuArray(a_grid);
u_grid=gpuArray(u_grid);

d3d4_gridvals=gpuArray(CreateGridvals([n_d3,n_d4],[d3_grid;d4_grid],1));
a_gridvals=gpuArray(CreateGridvals(n_a,a_grid,1));

if vfoptions.lowmemory>0
    special_n_semiz=ones(1,length(n_semiz));
end

semizind=shiftdim(0:1:N_semiz-1,-1);

V_ford4_jj=zeros(N_a,N_semiz,N_d4,'gpuArray');
Policy_ford4_jj=zeros(N_a,N_semiz,N_d4,'gpuArray');
d2index_ford4_jj=zeros(N_d3,N_semiz,N_d4,'gpuArray');


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

% If there is a warm-glow at end of the final period, evaluate the warmglowfn
if warmglow==1
    WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,N_j);
    WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a, a_grid, WGParamsVec); % This depends on aprime
    WGmatrix=WGmatrixraw;
    WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5(N_j);
    WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
    %  Switch WGmatrix from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [aprimeIndex,aprimeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a, n_u, d23_grid, a_grid, u_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d*N_u,1], whereas aprimeProbs is [N_d,N_u]
    WG1=WGmatrix(aprimeIndex); % (d,u), the lower aprime
    WG2=WGmatrix(aprimeIndex+1); % (d,u), the upper aprime
    % Apply the aprimeProbs
    WG1=reshape(WG1,[N_d23,N_u]).*aprimeProbs; % probability of lower grid point
    WG2=reshape(WG2,[N_d23,N_u]).*(1-aprimeProbs); % probability of upper grid point
    % If WG1 or WG2 is infinite, and probability is zero, we will get a nan, so get rid of these
    WG1(isnan(WG1))=0;
    WG2(isnan(WG2))=0;
    % Expectation over u (using pi_u), and then add the lower and upper
    WGmatrix=sum((WG1.*pi_u'),2)+sum((WG2.*pi_u'),2); % (d,1), sum over u
    % WGmatrix is over (d,1)
    if ~isfield(vfoptions,'V_Jplus1')
        becareful=(WGmatrix==0);
        WGmatrix(isfinite(WGmatrix))=ezc3*DiscountFactorParamsVec*(((1-sj(N_j))*WGmatrix(isfinite(WGmatrix)).^ezc8(N_j)).^ezc6(N_j));
        WGmatrix(becareful)=0;
    else
        if vfoptions.lowmemory==0
            WGmatrix=repmat(WGmatrix,1,1,N_semiz);
        end
    end
else
    WGmatrix=0;
end

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d3,n_d4], n_a, n_semiz, d3d4_gridvals, a_gridvals, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec);

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0)); % finite and not zero
        ReturnMatrix(becareful)=(ezc1*ReturnMatrix(becareful).^ezc2(N_j)).^ezc7(N_j); % Otherwise can get things like 0 to negative power equals infinity
        ReturnMatrix(ReturnMatrix==0)=-Inf;

        if warmglow==1
            % Time to refine
            % First: ReturnMatrix, we can refine out d1
            % no d1 here
            % Second: WGmatrix, we can refine out d2
            [WGmatrix_onlyd3,d2index]=max(ezc9*reshape((~isinf(WGmatrix)).*WGmatrix,[N_d2,N_d3]),[],1);
            % Now put together entireRHS, which just depends on d3 (and d4)
            entireRHS=ReturnMatrix+ezc9*shiftdim(repmat(WGmatrix_onlyd3,1,N_d4),1);

            [Vtemp,maxindex]=max(entireRHS,[],1);
            V(:,:,N_j)=Vtemp;
            d3index=rem(maxindex-1,N_d3)+1;
            Policy3(1,:,:,N_j)=d2index(d3index); % d2, note: no a nor semiz in WGmatrix
            Policy3(2,:,:,N_j)=d3index; % d3
            Policy3(3,:,:,N_j)=shiftdim(ceil(maxindex/N_d3),-1); % d4
        elseif warmglow==0
            [Vtemp,maxindex]=max(ReturnMatrix,[],1);
            V(:,:,N_j)=Vtemp;
            Policy3(1,:,:,N_j)=1; % d2, is meaningless anyway
            Policy3(2,:,:,N_j)=rem(maxindex-1,N_d3)+1; % d3
            Policy3(3,:,:,N_j)=shiftdim(ceil(maxindex/N_d3),-1); % d4
        end

    elseif vfoptions.lowmemory==1
        for z_c=1:N_semiz
            z_val=semiz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d3,n_d4], n_a, special_n_semiz, d3d4_gridvals, a_gridvals, z_val, ReturnFnParamsVec);

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_z).*(ReturnMatrix_z~=0)); % finite and not zero
            ReturnMatrix_z(becareful)=(ezc1*ReturnMatrix_z(becareful).^ezc2(N_j)).^ezc7(N_j); % Otherwise can get things like 0 to negative power equals infinity
            ReturnMatrix_z(ReturnMatrix_z==0)=-Inf;

            if warmglow==1
                % Time to refine
                % First: ReturnMatrix, we can refine out d1
                % no d1 here
                % Second: WGmatrix, we can refine out d2
                [WGmatrix_onlyd3,d2index]=max(ezc9*reshape((~isinf(WGmatrix)).*WGmatrix,[N_d2,N_d3]),[],1);
                % Now put together entireRHS, which just depends on d3 (and d4)
                entireRHS=ReturnMatrix_z+ezc9*shiftdim(repmat(WGmatrix_onlyd3,1,N_d4),1);

                [Vtemp,maxindex]=max(entireRHS,[],1);
                V(:,z_c,N_j)=Vtemp;
                d3index=rem(maxindex-1,N_d3)+1;
                Policy3(1,:,z_c,N_j)=d2index(d3index); % d2, note: no a nor semiz in WGmatrix
                Policy3(2,:,z_c,N_j)=d3index; % d3
                Policy3(3,:,z_c,N_j)=shiftdim(ceil(maxindex/N_d3),-1); % d4
            elseif warmglow==0
                [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
                V(:,z_c,N_j)=Vtemp;
                Policy3(1,:,z_c,N_j)=1; % d2, is meaningless anyway
                Policy3(2,:,z_c,N_j)=rem(maxindex-1,N_d3)+1; % d3
                Policy3(3,:,z_c,N_j)=shiftdim(ceil(maxindex/N_d3),-1); % d4
            end
        end
    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);    % First, switch V_Jplus1 into Kron form

    if warmglow==0 % if warmglow==1 these were already created above
        aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
        [aprimeIndex,aprimeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a, n_u, d23_grid, a_grid, u_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
        % Note: aprimeIndex is [N_d*N_u,1], whereas aprimeProbs is [N_d,N_u]
    end

    % Part of Epstein-Zin is before taking expectation
    temp=V_Jplus1;
    temp(isfinite(V_Jplus1))=(ezc4*V_Jplus1(isfinite(V_Jplus1))).^ezc5(N_j);
    temp(V_Jplus1==0)=0; % otherwise zero to negative power is set to infinity

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(N_j)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,N_j);
    end

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d3_special_d4_gridvals=gpuArray(CreateGridvals([n_d3,special_n_d4], [d3_grid; d4_gridvals(d4_c,:)'], 1));
            ReturnMatrix_d4=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d3,special_n_d4], n_a, n_semiz, d3_special_d4_gridvals, a_gridvals, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec);

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_d4).*(ReturnMatrix_d4~=0)); % finite and not zero
            temp2=ReturnMatrix_d4;
            temp2(becareful)=ReturnMatrix_d4(becareful).^ezc2(N_j);
            temp2(ReturnMatrix_d4==0)=-Inf;

            EV=temp.*shiftdim(pi_semizd4',-1);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
            EV1=EV(aprimeIndex+N_a*((1:1:N_semiz)-1)); % (d,u,semiz), the lower aprime
            EV2=EV((aprimeIndex+1)+N_a*((1:1:N_semiz)-1)); % (d,u,semiz), the upper aprime

            % Apply the aprimeProbs
            EV1=reshape(EV1,[N_d23,N_u,N_semiz]).*aprimeProbs; % probability of lower grid point
            EV2=reshape(EV2,[N_d23,N_u,N_semiz]).*(1-aprimeProbs); % probability of upper grid point

            % Expectation over u (using pi_u), and then add the lower and upper
            EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2); % (d,1,semiz), sum over u
            % EV is over (d,1,semiz)

            % Part of Epstein-Zin is after taking expectation
            temp4=EV;
            if warmglow==1
                becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
                temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrix(becareful).^ezc8(N_j)).^ezc6(N_j);
                temp4((EV==0)&(WGmatrix==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4)).^ezc8(N_j)).^ezc6(N_j);
                temp4(EV==0)=0;
            end

            % Time to refine
            % First: ReturnMatrix, we can refine out d1
            % no d1 here
            % Second: EV, we can refine out d2
            [temp4_onlyd3,d2index]=max(ezc9*ezc3*reshape((~isinf(temp4)).*temp4,[N_d2,N_d3,1,N_semiz]),[],1);
            % Now put together entireRHS, which just depends on d3
            entireRHS=ezc1*temp2+DiscountFactorParamsVec*ezc9*shiftdim(temp4_onlyd3,1);

            temp5=logical(isfinite(entireRHS).*(entireRHS~=0));
            entireRHS(temp5)=entireRHS(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V_ford4_jj(:,:,d4_c)=shiftdim(Vtemp,1);
            Policy_ford4_jj(:,:,d4_c)=shiftdim(maxindex,1);
            d2index_ford4_jj(:,:,d4_c)=squeeze(d2index);
        end

        % Now we just max over d4, and keep the policy that corresponded to that (including modify the policy to include the d4 decision)
        [V_jj,maxindex]=max(V_ford4_jj,[],3); % max over d4
        V(:,:,N_j)=V_jj;
        Policy3(3,:,:,N_j)=maxindex; % d4 is just maxindex
        maxindex_d4=reshape(maxindex,[N_a*N_semiz,1]);
        d3_ind=reshape(Policy_ford4_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_d4-1)),[1,N_a,N_semiz]);
        Policy3(2,:,:,N_j)=d3_ind;
        Policy3(1,:,:,N_j)=shiftdim(d2index_ford4_jj(d3_ind+N_d3*semizind+N_d3*N_semiz*shiftdim(maxindex-1,-1)),-1);

    elseif vfoptions.lowmemory==1
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d3_special_d4_gridvals=gpuArray(CreateGridvals([n_d3,special_n_d4], [d3_grid; d4_gridvals(d4_c,:)'], 1));
            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_d4z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d3,special_n_d4], n_a, special_n_semiz, d3_special_d4_gridvals, a_gridvals, z_val, ReturnFnParamsVec);

                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(ReturnMatrix_d4z).*(ReturnMatrix_d4z~=0)); % finite and not zero
                temp2=ReturnMatrix_d4z;
                temp2(becareful)=ReturnMatrix_d4z(becareful).^ezc2(N_j);
                temp2(ReturnMatrix_d4z==0)=-Inf;

                %Calc the condl expectation term (except beta), which depends on z but
                %not on control variables
                EV_z=temp.*pi_semizd4(z_c,:);
                EV_z(isnan(EV_z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_z=sum(EV_z,2);

                % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
                EV1_z=aprimeProbs.*reshape(EV_z(aprimeIndex),[N_d23,N_u]); % (d,u), the lower aprime
                EV2_z=(1-aprimeProbs).*reshape(EV_z(aprimeIndex+1),[N_d23,N_u]); % (d,u), the upper aprime
                % Already applied the probabilities from interpolating onto grid

                % Expectation over u (using pi_u), and then add the lower and upper
                EV_z=sum((EV1_z.*pi_u'),2)+sum((EV2_z.*pi_u'),2); % (d,1), sum over u
                % EV_z is over (d,1)

                % Part of Epstein-Zin is after taking expectation
                temp4=EV_z;
                if warmglow==1
                    becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
                    temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrix(becareful).^ezc8(N_j)).^ezc6(N_j);
                    temp4((EV_z==0)&(WGmatrix==0))=0; % Is actually zero
                else % not using warmglow
                    temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4)).^ezc8(N_j)).^ezc6(N_j);
                    temp4(EV_z==0)=0;
                end

                % Time to refine
                % First: ReturnMatrix, we can refine out d1
                % no d1 here
                % Second: EV, we can refine out d2
                [temp4_onlyd3,d2index]=max(ezc9*ezc3*reshape((~isinf(temp4)).*temp4,[N_d2,N_d3,1]),[],1);
                % Now put together entireRHS, which just depends on d3
                entireRHS_d4z=ezc1*temp2+DiscountFactorParamsVec*ezc9*shiftdim(temp4_onlyd3,1);

                temp5=logical(isfinite(entireRHS_d4z).*(entireRHS_d4z~=0));
                entireRHS_d4z(temp5)=entireRHS_d4z(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_d4z,[],1);

                V_ford4_jj(:,z_c,d4_c)=shiftdim(Vtemp,1);
                Policy_ford4_jj(:,z_c,d4_c)=shiftdim(maxindex,1);
                d2index_ford4_jj(:,z_c,d4_c)=shiftdim(d2index,1);
            end
        end

        % Now we just max over d4, and keep the policy that corresponded to that (including modify the policy to include the d4 decision)
        [V_jj,maxindex]=max(V_ford4_jj,[],3); % max over d4
        V(:,:,N_j)=V_jj;
        Policy3(3,:,:,N_j)=maxindex; % d4 is just maxindex
        maxindex_d4=reshape(maxindex,[N_a*N_semiz,1]);
        d3_ind=reshape(Policy_ford4_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_d4-1)),[1,N_a,N_semiz]);
        Policy3(2,:,:,N_j)=d3_ind;
        Policy3(1,:,:,N_j)=shiftdim(d2index_ford4_jj(d3_ind+N_d3*semizind+N_d3*N_semiz*shiftdim(maxindex-1,-1)),-1);
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

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [aprimeIndex,aprimeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a, n_u, d23_grid, a_grid, u_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d*N_u,1], whereas aprimeProbs is [N_d,N_u]

    % If there is a warm-glow, evaluate the warmglowfn
    if warmglow==1
        WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
        WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a, a_grid, WGParamsVec);
        WGmatrix=WGmatrixraw;
        WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5(jj);
        WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
        %  Switch WGmatrix from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
        WG1=WGmatrix(aprimeIndex); % (d,u), the lower aprime
        WG2=WGmatrix(aprimeIndex+1); % (d,u), the upper aprime
        % Apply the aprimeProbs
        WG1=reshape(WG1,[N_d23,N_u]).*aprimeProbs; % probability of lower grid point
        WG2=reshape(WG2,[N_d23,N_u]).*(1-aprimeProbs); % probability of upper grid point
        % If WG1 or WG2 is infinite, and probability is zero, we will get a nan, so get rid of these
        WG1(isnan(WG1))=0;
        WG2(isnan(WG2))=0;
        % Expectation over u (using pi_u), and then add the lower and upper
        WGmatrix=sum((WG1.*pi_u'),2)+sum((WG2.*pi_u'),2); % (d,1), sum over u
        % WGmatrix is over (d,1)
        % Now just make it the right shape (lowmemory==0 EV is vectorized over semiz)
        if vfoptions.lowmemory==0
            WGmatrix=WGmatrix.*ones(1,1,N_semiz);
        end
    end

    EVpre=V(:,:,jj+1);

    % Part of Epstein-Zin is before taking expectation
    temp=EVpre;
    temp(isfinite(EVpre))=(ezc4*EVpre(isfinite(EVpre))).^ezc5(jj);
    temp(EVpre==0)=0;

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(jj)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,jj);
    end

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d3_special_d4_gridvals=gpuArray(CreateGridvals([n_d3,special_n_d4], [d3_grid; d4_gridvals(d4_c,:)'], 1));
            ReturnMatrix_d4=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d3,special_n_d4], n_a, n_semiz, d3_special_d4_gridvals, a_gridvals, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec);

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_d4).*(ReturnMatrix_d4~=0)); % finite and not zero
            temp2=ReturnMatrix_d4;
            temp2(becareful)=ReturnMatrix_d4(becareful).^ezc2(jj);
            temp2(ReturnMatrix_d4==0)=-Inf;

            EV=temp.*shiftdim(pi_semizd4',-1);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
            EV1=EV(aprimeIndex+N_a*((1:1:N_semiz)-1)); % (d,u,semiz), the lower aprime
            EV2=EV((aprimeIndex+1)+N_a*((1:1:N_semiz)-1)); % (d,u,semiz), the upper aprime

            % Apply the aprimeProbs
            EV1=reshape(EV1,[N_d23,N_u,N_semiz]).*aprimeProbs; % probability of lower grid point
            EV2=reshape(EV2,[N_d23,N_u,N_semiz]).*(1-aprimeProbs); % probability of upper grid point

            % Expectation over u (using pi_u), and then add the lower and upper
            EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2); % (d,1,semiz), sum over u
            % EV is over (d,1,semiz)

            % Part of Epstein-Zin is after taking expectation
            temp4=EV;
            if warmglow==1
                becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
                temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrix(becareful).^ezc8(jj)).^ezc6(jj);
                temp4((EV==0)&(WGmatrix==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                temp4(EV==0)=0;
            end

            % Time to refine
            % First: ReturnMatrix, we can refine out d1
            % no d1 here
            % Second: EV, we can refine out d2
            [temp4_onlyd3,d2index]=max(ezc9*ezc3*reshape((~isinf(temp4)).*temp4,[N_d2,N_d3,1,N_semiz]),[],1);
            % Now put together entireRHS, which just depends on d3
            entireRHS=ezc1*temp2+DiscountFactorParamsVec*ezc9*shiftdim(temp4_onlyd3,1);

            temp5=logical(isfinite(entireRHS).*(entireRHS~=0));
            entireRHS(temp5)=entireRHS(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V_ford4_jj(:,:,d4_c)=shiftdim(Vtemp,1);
            Policy_ford4_jj(:,:,d4_c)=shiftdim(maxindex,1);
            d2index_ford4_jj(:,:,d4_c)=squeeze(d2index);
        end

        % Now we just max over d4, and keep the policy that corresponded to that (including modify the policy to include the d4 decision)
        [V_jj,maxindex]=max(V_ford4_jj,[],3); % max over d4
        V(:,:,jj)=V_jj;
        Policy3(3,:,:,jj)=maxindex; % d4 is just maxindex
        maxindex_d4=reshape(maxindex,[N_a*N_semiz,1]);
        d3_ind=reshape(Policy_ford4_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_d4-1)),[1,N_a,N_semiz]);
        Policy3(2,:,:,jj)=d3_ind;
        Policy3(1,:,:,jj)=shiftdim(d2index_ford4_jj(d3_ind+N_d3*semizind+N_d3*N_semiz*shiftdim(maxindex-1,-1)),-1);

    elseif vfoptions.lowmemory==1
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d3_special_d4_gridvals=gpuArray(CreateGridvals([n_d3,special_n_d4], [d3_grid; d4_gridvals(d4_c,:)'], 1));
            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);
                ReturnMatrix_d4z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d3,special_n_d4], n_a, special_n_semiz, d3_special_d4_gridvals, a_gridvals, z_val, ReturnFnParamsVec);

                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(ReturnMatrix_d4z).*(ReturnMatrix_d4z~=0)); % finite and not zero
                temp2=ReturnMatrix_d4z;
                temp2(becareful)=ReturnMatrix_d4z(becareful).^ezc2(jj);
                temp2(ReturnMatrix_d4z==0)=-Inf;

                %Calc the condl expectation term (except beta), which depends on z but
                %not on control variables
                EV_z=temp.*pi_semizd4(z_c,:);
                EV_z(isnan(EV_z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_z=sum(EV_z,2);

                % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
                EV1_z=aprimeProbs.*reshape(EV_z(aprimeIndex),[N_d23,N_u]); % (d,u), the lower aprime
                EV2_z=(1-aprimeProbs).*reshape(EV_z(aprimeIndex+1),[N_d23,N_u]); % (d,u), the upper aprime
                % Already applied the probabilities from interpolating onto grid

                % Expectation over u (using pi_u), and then add the lower and upper
                EV_z=sum((EV1_z.*pi_u'),2)+sum((EV2_z.*pi_u'),2); % (d,1), sum over u
                % EV_z is over (d,1)

                % Part of Epstein-Zin is after taking expectation
                temp4=EV_z;
                if warmglow==1
                    becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
                    temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrix(becareful).^ezc8(jj)).^ezc6(jj);
                    temp4((EV_z==0)&(WGmatrix==0))=0; % Is actually zero
                else % not using warmglow
                    temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                    temp4(EV_z==0)=0;
                end

                % Time to refine
                % First: ReturnMatrix, we can refine out d1
                % no d1 here
                % Second: EV, we can refine out d2
                [temp4_onlyd3,d2index]=max(ezc9*ezc3*reshape((~isinf(temp4)).*temp4,[N_d2,N_d3,1]),[],1);
                % Now put together entireRHS, which just depends on d3
                entireRHS_d4z=ezc1*temp2+DiscountFactorParamsVec*ezc9*shiftdim(temp4_onlyd3,1);

                temp5=logical(isfinite(entireRHS_d4z).*(entireRHS_d4z~=0));
                entireRHS_d4z(temp5)=entireRHS_d4z(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_d4z,[],1);

                V_ford4_jj(:,z_c,d4_c)=shiftdim(Vtemp,1);
                Policy_ford4_jj(:,z_c,d4_c)=shiftdim(maxindex,1);
                d2index_ford4_jj(:,z_c,d4_c)=shiftdim(d2index,1);
            end
        end

        % Now we just max over d4, and keep the policy that corresponded to that (including modify the policy to include the d4 decision)
        [V_jj,maxindex]=max(V_ford4_jj,[],3); % max over d4
        V(:,:,jj)=V_jj;
        Policy3(3,:,:,jj)=maxindex; % d4 is just maxindex
        maxindex_d4=reshape(maxindex,[N_a*N_semiz,1]);
        d3_ind=reshape(Policy_ford4_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_d4-1)),[1,N_a,N_semiz]);
        Policy3(2,:,:,jj)=d3_ind;
        Policy3(1,:,:,jj)=shiftdim(d2index_ford4_jj(d3_ind+N_d3*semizind+N_d3*N_semiz*shiftdim(maxindex-1,-1)),-1);
    end
end

Policy=Policy3; % rows: d2, d3, d4 (multi-row form, decoded by UnKronPolicyIndexes3 in the dispatcher, mirroring the vNM pairing)


end
