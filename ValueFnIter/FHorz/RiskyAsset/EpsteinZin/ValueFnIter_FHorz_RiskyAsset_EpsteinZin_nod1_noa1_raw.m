function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_nod1_noa1_raw(n_d2,n_d3,n_a2,n_z,n_u,N_j, d2_grid, d3_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9)
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a2=prod(n_a2);
N_z=prod(n_z);
N_u=prod(n_u);

% For ReturnFn
% n_d3
% N_d3
% d3_grid
% For aprimeFn
n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a2,N_z,N_j,'gpuArray');
Policy2=zeros(2,N_a2,N_z,N_j,'gpuArray'); % d2, d3

%%
d3_grid=gpuArray(d3_grid);
d23_grid=gpuArray(d23_grid);
a2_grid=gpuArray(a2_grid);
u_grid=gpuArray(u_grid);

if vfoptions.lowmemory>0
    special_n_z=ones(1,length(n_z));
end

aind=(0:1:N_a2-1);
zind=shiftdim(0:1:N_z-1,-1);

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
    WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a2, a2_grid, WGParamsVec); % This depends on aprime
    WGmatrix=WGmatrixraw;
    WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5(N_j);
    WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
    %  Switch WGmatrix from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [aprimeIndex,aprimeProbs]=CreateaprimeFnMatrix_RiskyAsset(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
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
            WGmatrix=repmat(WGmatrix,1,1,N_z);
        end
    end
else
    WGmatrix=0;
end


if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d3, n_a2, n_z, d3_grid, a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0)); % finite and not zero
        ReturnMatrix(becareful)=(ezc1*ReturnMatrix(becareful).^ezc2(N_j)).^ezc7(N_j); % Otherwise can get things like 0 to negative power equals infinity
        ReturnMatrix(ReturnMatrix==0)=-Inf;

        %Calc the max and it's index
        if warmglow==1
            % Time to refine
            % First: ReturnMatrix, we can refine out d1
            % no d1 here
            % Second: EV, we can refine out d2
            [WGmatrix_onlyd3,d2index]=max(ezc9*reshape((~isinf(WGmatrix)).*WGmatrix,[N_d2,N_d3]),[],1);
            % Now put together entireRHS, which just depends on d3
            entireRHS=ReturnMatrix+ezc9*shiftdim(WGmatrix_onlyd3,1);

            % no point in Refine
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V(:,:,N_j)=Vtemp;
            Policy2(1,:,:,N_j)=shiftdim(d2index(maxindex),1); % d2
            Policy2(2,:,:,N_j)=shiftdim(maxindex,1); % d3
        else
            % no point in Refine (as there effectively is no d2)
            [Vtemp,maxindex]=max(ReturnMatrix,[],1);

            V(:,:,N_j)=Vtemp;
            Policy2(1,:,:,N_j)=1; % d2, is anyway meaningless
            Policy2(2,:,:,N_j)=shiftdim(ceil(maxindex),1); % d3
        end

    elseif vfoptions.lowmemory==1

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d3, n_a2, special_n_z, d3_grid, a2_grid, z_val, ReturnFnParamsVec);

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_z).*(ReturnMatrix_z~=0)); % finite and not zero
            ReturnMatrix_z(becareful)=(ezc1*ReturnMatrix_z(becareful).^ezc2(N_j)).^ezc7(N_j); % Otherwise can get things like 0 to negative power equals infinity
            ReturnMatrix_z(ReturnMatrix_z==0)=-Inf;

            %Calc the max and it's index
            if warmglow==1
                % Time to refine
                % First: ReturnMatrix, we can refine out d1
                % no d1 here
                % Second: EV, we can refine out d2
                [WGmatrix_onlyd3,d2index]=max(ezc9*reshape((~isinf(WGmatrix)).*WGmatrix,[N_d2,N_d3]),[],1);
                % Now put together entireRHS, which just depends on d3
                entireRHS_z=shiftdim(ReturnMatrix_z+ezc9*WGmatrix_onlyd3,1);

                % no point in Refine
                [Vtemp,maxindex]=max(entireRHS_z,[],1);
            else
                % no point in Refine
                [Vtemp,maxindex]=max(ReturnMatrix_z+WGmatrix,[],1);
            end
            V(:,z_c,N_j)=Vtemp;
            Policy2(1,:,z_c,N_j)=shiftdim(d2index(maxindex),1);
            Policy2(2,:,z_c,N_j)=shiftdim(maxindex,1);
        end

    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a2,N_z]);    % First, switch V_Jplus1 into Kron form
    
    if warmglow==0 % if warmglow==1 these were already created above
        aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
        [aprimeIndex,aprimeProbs]=CreateaprimeFnMatrix_RiskyAsset(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
        % Note: aprimeIndex is [N_d*N_u,1], whereas aprimeProbs is [N_d,N_u]
    end

    % Part of Epstein-Zin is before taking expectation
    temp=V_Jplus1;
    temp(isfinite(V_Jplus1))=(ezc4*V_Jplus1(isfinite(V_Jplus1))).^ezc5(N_j);
    temp(V_Jplus1==0)=0; % otherwise zero to negative power is set to infinity
    
    if vfoptions.lowmemory==0
        
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d3, n_a2, n_z, d3_grid, a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        % (d,aprime,a,z)
        
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0)); % finite and not zero
        temp2=ReturnMatrix;
        temp2(becareful)=ReturnMatrix(becareful).^ezc2(N_j);
        temp2(ReturnMatrix==0)=-Inf;

        EV=temp.*shiftdim(pi_z_J(:,:,N_j)',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension

        % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
        EV1=EV(aprimeIndex+N_a2*((1:1:N_z)-1)); % (d,u,z), the lower aprime
        EV2=EV((aprimeIndex+1)+N_a2*((1:1:N_z)-1)); % (d,u,z), the upper aprime

        % Apply the aprimeProbs
        EV1=reshape(EV1,[N_d23,N_u,N_z]).*aprimeProbs; % probability of lower grid point
        EV2=reshape(EV2,[N_d23,N_u,N_z]).*(1-aprimeProbs); % probability of upper grid point

        % Expectation over u (using pi_u), and then add the lower and upper
        EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2); % (d,1,z), sum over u
        % EV is over (d,1,z)

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
        [temp4_onlyd3,d2index]=max(ezc9*ezc3*reshape((~isinf(temp4)).*temp4,[N_d2,N_d3,1,N_z]),[],1);
        % Now put together entireRHS, which just depends on d3
        entireRHS=ezc1*temp2+ezc9*DiscountFactorParamsVec*shiftdim(temp4_onlyd3,1);
        % entireRHS=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4;

        temp5=logical(isfinite(entireRHS).*(entireRHS~=0));
        entireRHS(temp5)=ezc1*entireRHS(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,N_j)=Vtemp;
        Policy2(1,:,:,N_j)=shiftdim(d2index(maxindex+N_d3*zind),1);
        Policy2(2,:,:,N_j)=shiftdim(maxindex,1);
        
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d3, n_a2, special_n_z, d3_grid, a2_grid, z_val, ReturnFnParamsVec);
            
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_z).*(ReturnMatrix_z~=0)); % finite and not zero
            temp2=ReturnMatrix_z;
            temp2(becareful)=ReturnMatrix_z(becareful).^ezc2(N_j);
            temp2(ReturnMatrix_z==0)=-Inf;

            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=temp.*(ones(N_a2,1,'gpuArray')*pi_z_J(z_c,:,N_j));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
            EV1_z=aprimeProbs.*reshape(EV_z(aprimeIndex),[N_d23,N_u]); % (d,u), the lower aprime
            EV2_z=(1-aprimeProbs).*reshape(EV_z(aprimeIndex+1),[N_d23,N_u]); % (d,u), the upper aprime
            % Already applied the probabilities from interpolating onto grid
            
            % Expectation over u (using pi_u), and then add the lower and upper
            EV_z=sum((EV1_z.*pi_u'),2)+sum((EV2_z.*pi_u'),2); % (d,1,z), sum over u
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
            entireRHS_z=ezc1*temp2+ezc9*DiscountFactorParamsVec*shiftdim(temp4_onlyd3,1);

            temp5=logical(isfinite(entireRHS_z).*(entireRHS_z~=0));
            entireRHS_z(temp5)=ezc1*entireRHS_z(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            Policy2(1,:,z_c,N_j)=shiftdim(d2index(maxindex+N_d3*zind),1);
            Policy2(2,:,z_c,N_j)=shiftdim(maxindex,1);
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
    
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [aprimeIndex,aprimeProbs]=CreateaprimeFnMatrix_RiskyAsset(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d*N_u,1], whereas aprimeProbs is [N_d,N_u]

    % If there is a warm-glow, evaluate the warmglowfn
    if warmglow==1
        WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
        WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a2, a2_grid, WGParamsVec);
        WGmatrix=WGmatrixraw;
        WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5(jj);
        WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
        %  Switch WGmatrix from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
        
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
        if vfoptions.lowmemory==0
            WGmatrix=repmat(WGmatrix,1,1,N_z);
        end
    end

    VKronNext_j=V(:,:,jj+1);
    
    % Part of Epstein-Zin is before taking expectation
    temp=VKronNext_j;
    temp(isfinite(VKronNext_j))=(ezc4*VKronNext_j(isfinite(VKronNext_j))).^ezc5(jj);
    temp(VKronNext_j==0)=0;

    if vfoptions.lowmemory==0
        
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d3, n_a2, n_z, d3_grid, a2_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec);
        % (d,aprime,a,z)
        
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0)); % finite and not zero
        temp2=ReturnMatrix;
        temp2(becareful)=ReturnMatrix(becareful).^ezc2(jj);
        temp2(ReturnMatrix==0)=-Inf;

        EV=temp.*shiftdim(pi_z_J(:,:,jj)',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension

        % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
        EV1=EV(aprimeIndex+N_a2*((1:1:N_z)-1)); % (d,u,z), the lower aprime
        EV2=EV((aprimeIndex+1)+N_a2*((1:1:N_z)-1)); % (d,u,z), the upper aprime

        % Apply the aprimeProbs
        EV1=reshape(EV1,[N_d23,N_u,N_z]).*aprimeProbs; % probability of lower grid point
        EV2=reshape(EV2,[N_d23,N_u,N_z]).*(1-aprimeProbs); % probability of upper grid point

        % Expectation over u (using pi_u), and then add the lower and upper
        EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2); % (d,1,z), sum over u
        % EV is over (d,1,z)

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
        [temp4_onlyd3,d2index]=max(ezc9*ezc3*reshape((~isinf(temp4)).*temp4,[N_d2,N_d3,1,N_z]),[],1);
        % Now put together entireRHS, which just depends on d3
        entireRHS=ezc1*temp2+ezc9*DiscountFactorParamsVec*shiftdim(temp4_onlyd3,1);
        % entireRHS=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4;

        temp5=logical(isfinite(entireRHS).*(entireRHS~=0));
        entireRHS(temp5)=ezc1*entireRHS(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,:,jj)=shiftdim(Vtemp,1);
        Policy2(1,:,:,jj)=shiftdim(d2index(maxindex+N_d3*zind),1);
        Policy2(2,:,:,jj)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a2, special_n_z, d_grid, a2_grid, z_val, ReturnFnParamsVec);
            
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_z).*(ReturnMatrix_z~=0)); % finite and not zero
            temp2=ReturnMatrix_z;
            temp2(becareful)=ReturnMatrix_z(becareful).^ezc2(jj);
            temp2(ReturnMatrix_z==0)=-Inf;

            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=temp.*(ones(N_a2,1,'gpuArray')*pi_z_J(z_c,:,jj));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
            EV1_z=aprimeProbs.*reshape(EV_z(aprimeIndex),[N_d23,N_u]); % (d,u), the lower aprime
            EV2_z=(1-aprimeProbs).*reshape(EV_z(aprimeIndex+1),[N_d23,N_u]); % (d,u), the upper aprime
            % Already applied the probabilities from interpolating onto grid
            
            % Expectation over u (using pi_u), and then add the lower and upper
            EV_z=sum((EV1_z.*pi_u'),2)+sum((EV2_z.*pi_u'),2); % (d,1,z), sum over u
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
            entireRHS_z=ezc1*temp2+ezc9*DiscountFactorParamsVec*shiftdim(temp4_onlyd3,1);

            temp5=logical(isfinite(entireRHS_z).*(entireRHS_z~=0));
            entireRHS_z(temp5)=ezc1*entireRHS_z(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,z_c,jj)=Vtemp;
            Policy2(1,:,z_c,jj)=shiftdim(d2index(maxindex+N_d3*zind),1);
            Policy2(2,:,z_c,jj)=shiftdim(maxindex,1);
        end
    end
end

Policy=Policy2(1,:,:,:)+N_d2*(Policy2(2,:,:,:)-1); % d2, d3


end
