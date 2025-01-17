function [V,Policy3]=ValueFnIter_FHorz_RiskyAsset_Refine_noa1_raw(n_d1,n_d2,n_d3,n_a,n_z,n_u,N_j, d1_grid, d2_grid, d3_grid, a_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a=prod(n_a);
N_z=prod(n_z);
N_u=prod(n_u);

% For ReturnFn
n_d13=[n_d1,n_d3];
N_d13=prod(n_d13);
d13_grid=[d1_grid; d3_grid];
% For aprimeFn
n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy3=zeros(3,N_a,N_z,N_j,'gpuArray'); % three: d1, d2, d3 

%%
d13_grid=gpuArray(d13_grid);
d23_grid=gpuArray(d23_grid);
a_grid=gpuArray(a_grid);
u_grid=gpuArray(u_grid);

if vfoptions.lowmemory>0
    special_n_z=ones(1,length(n_z));
end

aind=(0:1:N_a-1);
zind=shiftdim(0:1:N_z-1,-1);


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d13, n_a, n_z, d13_grid, a_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        % Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        Policy3(1,:,:,N_j)=shiftdim(rem(maxindex-1,N_d1)+1,-1);
        Policy3(2,:,:,N_j)=1; % is meaningless anyway
        Policy3(3,:,:,N_j)=shiftdim(ceil(maxindex/N_d1),-1);
        
    elseif vfoptions.lowmemory==1

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d13, n_a, special_n_z, d13_grid, a_grid, z_val, ReturnFnParamsVec);
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            Policy3(1,:,z_c,N_j)=shiftdim(rem(maxindex-1,N_d1)+1,-1);
            Policy3(2,:,z_c,N_j)=1; % is meaningless anyway
            Policy3(3,:,z_c,N_j)=shiftdim(ceil(maxindex/N_d1),-1);
        end

    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_z]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [aprimeIndex,aprimeProbs]=CreateaprimeFnMatrix_RiskyAsset(aprimeFn, n_d23, n_a, n_u, d23_grid, a_grid, u_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d*N_u,1], whereas aprimeProbs is [N_d,N_u]

    if vfoptions.lowmemory==0
        
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d13, n_a, n_z, d13_grid, a_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        % (d,aprime,a,z)

        EV=V_Jplus1.*shiftdim(pi_z_J(:,:,N_j)',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension

        % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
        EV1=EV(aprimeIndex+N_a*((1:1:N_z)-1)); % (d,u,z), the lower aprime
        EV2=EV((aprimeIndex+1)+N_a*((1:1:N_z)-1)); % (d,u,z), the upper aprime

        % Apply the aprimeProbs
        EV1=reshape(EV1,[N_d23,N_u,N_z]).*aprimeProbs; % probability of lower grid point
        EV2=reshape(EV2,[N_d23,N_u,N_z]).*(1-aprimeProbs); % probability of upper grid point

        % Expectation over u (using pi_u), and then add the lower and upper
        EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2); % (d,1,z), sum over u
        % EV is over (d,1,z)

        % Time to refine
        % First: ReturnMatrix, we can refine out d1
        [ReturnMatrix_onlyd3,d1index]=max(reshape(ReturnMatrix,[N_d1,N_d3,N_a,N_z]),[],1);
        % Second: EV, we can refine out d2
        [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3,1,N_z]),[],1);
        % Now put together entireRHS, which just depends on d3
        entireRHS=shiftdim(ReturnMatrix_onlyd3+DiscountFactorParamsVec*EV_onlyd3,1);
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,N_j)=shiftdim(Vtemp,1);
        Policy3(3,:,:,N_j)=shiftdim(maxindex,1);
        Policy3(1,:,:,N_j)=shiftdim(d1index(maxindex+N_d3*aind+N_d3*N_a*zind),1);
        Policy3(2,:,:,N_j)=shiftdim(d2index(maxindex+N_d3*zind),1);
                    
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d13, n_a, special_n_z, d13_grid, a_grid, z_val, ReturnFnParamsVec);
            
            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=V_Jplus1.*(ones(N_a,1,'gpuArray')*pi_z_J(z_c,:,N_j));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
            EV1_z=aprimeProbs.*reshape(EV_z(aprimeIndex),[N_d23,N_u]); % (d,u), the lower aprime
            EV2_z=(1-aprimeProbs).*reshape(EV_z(aprimeIndex+1),[N_d23,N_u]); % (d,u), the upper aprime
            % Already applied the probabilities from interpolating onto grid
            
            % Expectation over u (using pi_u), and then add the lower and upper
            EV_z=sum((EV1_z.*pi_u'),2)+sum((EV2_z.*pi_u'),2); % (d,1,z), sum over u
            % EV_z is over (d,1)
            
            % Time to refine
            % First: ReturnMatrix, we can refine out d1
            [ReturnMatrix_onlyd3,d1index]=max(reshape(ReturnMatrix_z,[N_d1,N_d3,N_a]),[],1);
            % Second: EV, we can refine out d2
            [EV_onlyd3,d2index]=max(reshape(EV_z,[N_d2,N_d3,1]),[],1);
            % Now put together entireRHS, which just depends on d3
            entireRHS_z=shiftdim(ReturnMatrix_onlyd3+DiscountFactorParamsVec*EV_onlyd3,1);
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            Policy3(3,:,z_c,N_j)=shiftdim(maxindex,1);
            Policy3(1,:,z_c,N_j)=shiftdim(d1index(maxindex+N_d3*aind),1);
            Policy3(2,:,z_c,N_j)=shiftdim(d2index(maxindex),1);
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
    
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [aprimeIndex,aprimeProbs]=CreateaprimeFnMatrix_RiskyAsset(aprimeFn, n_d23, n_a, n_u, d23_grid, a_grid, u_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d*N_u,1], whereas aprimeProbs is [N_d,N_u]

    VKronNext_j=V(:,:,jj+1);

    
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d13, n_a, n_z, d13_grid, a_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec);
        % (d,aprime,a,z)

        EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension

        % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
        EV1=EV(aprimeIndex+N_a*((1:1:N_z)-1)); % (d,u,z), the lower aprime
        EV2=EV((aprimeIndex+1)+N_a*((1:1:N_z)-1)); % (d,u,z), the upper aprime

        % Apply the aprimeProbs
        EV1=reshape(EV1,[N_d23,N_u,N_z]).*aprimeProbs; % probability of lower grid point
        EV2=reshape(EV2,[N_d23,N_u,N_z]).*(1-aprimeProbs); % probability of upper grid point

        % Expectation over u (using pi_u), and then add the lower and upper
        EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2); % (d,1,z), sum over u
        % EV is over (d,1,z)

        % Time to refine
        % First: ReturnMatrix, we can refine out d1
        [ReturnMatrix_onlyd3,d1index]=max(reshape(ReturnMatrix,[N_d1,N_d3,N_a,N_z]),[],1);
        % Second: EV, we can refine out d2
        [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3,1,N_z]),[],1);
        % Now put together entireRHS, which just depends on d3
        entireRHS=shiftdim(ReturnMatrix_onlyd3+DiscountFactorParamsVec*EV_onlyd3,1);

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,jj)=shiftdim(Vtemp,1);
        Policy3(3,:,:,jj)=shiftdim(maxindex,1);
        Policy3(1,:,:,jj)=shiftdim(d1index(maxindex+N_d3*aind+N_d3*N_a*zind),1);
        Policy3(2,:,:,jj)=shiftdim(d2index(maxindex+N_d3*zind),1);

        
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d13, n_a, special_n_z, d13_grid, a_grid, z_val, ReturnFnParamsVec);
            
            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=VKronNext_j.*pi_z_J(z_c,:,jj);
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
            EV1_z=aprimeProbs.*reshape(EV_z(aprimeIndex),[N_d23,N_u]); % (d,u), the lower aprime
            EV2_z=(1-aprimeProbs).*reshape(EV_z(aprimeIndex+1),[N_d23,N_u]); % (d,u), the upper aprime
            % Already applied the probabilities from interpolating onto grid
            
            % Expectation over u (using pi_u), and then add the lower and upper
            EV_z=sum((EV1_z.*pi_u'),2)+sum((EV2_z.*pi_u'),2); % (d,1,z), sum over u
            % EV_z is over (d,1)

            % Time to refine
            % First: ReturnMatrix, we can refine out d1
            [ReturnMatrix_onlyd3,d1index]=max(reshape(ReturnMatrix_z,[N_d1,N_d3,N_a]),[],1);
            % Second: EV, we can refine out d2
            [EV_onlyd3,d2index]=max(reshape(EV_z,[N_d2,N_d3,1]),[],1);
            % Now put together entireRHS, which just depends on d3
            entireRHS_z=shiftdim(ReturnMatrix_onlyd3+DiscountFactorParamsVec*EV_onlyd3,1);
                        
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,z_c,jj)=Vtemp;
            Policy3(3,:,z_c,jj)=shiftdim(maxindex,1);
            Policy3(1,:,z_c,jj)=shiftdim(d1index(maxindex+N_d3*aind),1);
            Policy3(2,:,z_c,jj)=shiftdim(d2index(maxindex),1);

        end
    end
end



end