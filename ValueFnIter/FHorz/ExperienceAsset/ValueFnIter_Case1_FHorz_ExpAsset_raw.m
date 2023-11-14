function [V,Policy]=ValueFnIter_Case1_FHorz_ExpAsset_raw(n_d1,n_d2,n_a1,n_a2,n_z,N_j, d1_grid, d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

%%
d1_grid=gpuArray(d1_grid);
d2_grid=gpuArray(d2_grid);
a1_grid=gpuArray(a1_grid);
a2_grid=gpuArray(a2_grid);

% For the return function we just want (I'm just guessing that as I need them N_j times it will be fractionally faster to put them together now)
n_d=[n_d1,n_d2];
d_grid=[d1_grid;d2_grid];

if vfoptions.lowmemory>0
    l_z=length(n_z);
    special_n_z=ones(1,length(n_z));
    % z_gridvals is created below
end
if vfoptions.lowmemory>1
    special_n_a1=ones(1,length(n_a1));
    a1_gridvals=CreateGridvals(n_a1,a1_grid,1); % The 1 at end indicates want output in form of matrix.
end


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, n_d, n_a1,n_a2, n_z, d_grid, a1_grid, a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        Policy(:,:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, n_d, n_a1,n_a2, special_n_z, d_grid, a1_grid, a2_grid, z_val, ReturnFnParamsVec);
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            Policy(:,z_c,N_j)=maxindex;
        end

    elseif vfoptions.lowmemory==2

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for a1_c=1:N_a1
                a1_val=a1_gridvals(a1_c,:);
                ReturnMatrix_az=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, n_d, special_n_a1,n_a2, special_n_z, d_grid, a1_val, a2_grid, z_val, ReturnFnParamsVec);
                %Calc the max and it's index
                [Vtemp,maxindex]=max(ReturnMatrix_az);
                V(a1_c,:,z_c,N_j)=Vtemp;
                Policy(a1_c,:,z_c,N_j)=maxindex;

            end
        end

    end
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_Case1(aprimeFn, n_d2, n_a2, d2_grid, a2_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2*N_a2,1], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=kron(ones(N_d2*N_a2,1),(1:1:N_a1)')+N_a1*kron((a2primeIndex-1),ones(N_a1,1)); % [N_d2*N_a1*N_a2,1]
    if vfoptions.lowmemory>0 || vfoptions.paroverz==0
        aprimeProbs=kron(ones(N_a1,1),a2primeProbs);  % [N_d2*N_a1,N_a2]
    else % lowmemory=0 and paroverz=1
        aprimeProbs=repmat(kron(ones(N_a1,1),a2primeProbs),1,1,N_z);  % [N_d2*N_a1,N_a2,N_z]
    end


    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_z]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    if vfoptions.lowmemory==0
        
        ReturnMatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, n_d, n_a1,n_a2, n_z, d_grid, a1_grid, a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        % (d,aprime,a,z)
        
        if vfoptions.paroverz==1
            
            EV=V_Jplus1.*shiftdim(pi_z_J(:,:,N_j)',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=EV(aprimeIndex+N_a*((1:1:N_z)-1)); % (d2,a1prime,a2,z), the lower aprime
            EV2=EV((aprimeIndex+1)+N_a*((1:1:N_z)-1)); % (d2,a1prime,a2,z), the upper aprime

            % Apply the aprimeProbs
            entireEV=reshape(EV1,[N_d2*N_a1,N_a2,N_z]).*aprimeProbs+reshape(EV2,[N_d2*N_a1,N_a2,N_z]).*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % entireEV is (d,a1prime, a2,z)

%             entireEV=repelem(EV,n_d2,1,1); % I tried this instead but appears repelem() is slower than kron(). However kron() requires 2-D so here I just use repelem() anyway.
            entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(entireEV,N_d1,N_a1,1);
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);
            
            V(:,:,N_j)=shiftdim(Vtemp,1);
            Policy(:,:,N_j)=shiftdim(maxindex,1);
            
            
        elseif vfoptions.paroverz==0
            
            for z_c=1:N_z
                ReturnMatrix_z=ReturnMatrix(:,:,z_c);
                
                %Calc the condl expectation term (except beta), which depends on z but not on control variables
                EV_z=V_Jplus1.*(ones(N_a,1,'gpuArray')*pi_z_J(z_c,:,N_j));
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);

                % Switch EV_z from being in terms of aprime to being in terms of d and a
                EV1=EV_z(aprimeIndex); % (d2,a1prime,a2), the lower aprime
                EV2=EV_z(aprimeIndex+1); % (d2,a1prime,a2), the upper aprime

                % Apply the aprimeProbs
                entireEV_z=reshape(EV1,[N_d2*N_a1,N_a2]).*aprimeProbs+reshape(EV2,[N_d2*N_a1,N_a2]).*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
                % entireEV_z is (d,a1prime, a2)

                entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*kron(entireEV_z,ones(N_d1,N_a1));
                
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_z,[],1);
                V(:,z_c,N_j)=Vtemp;
                Policy(:,z_c,N_j)=maxindex;
            end
        end
        
    elseif vfoptions.lowmemory==1

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, n_d, n_a1,n_a2, special_n_z, d_grid, a1_grid, a2_grid, z_val, ReturnFnParamsVec);
            
            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=V_Jplus1.*(ones(N_a,1,'gpuArray')*pi_z_J(z_c,:,N_j));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);

            % Switch EV_z from being in terms of aprime to being in terms of d and a
            EV1=EV_z(aprimeIndex); % (d2,a1prime,a2), the lower aprime
            EV2=EV_z(aprimeIndex+1); % (d2,a1prime,a2), the upper aprime

            % Apply the aprimeProbs
            entireEV_z=reshape(EV1,[N_d2*N_a1,N_a2]).*aprimeProbs+reshape(EV2,[N_d2*N_a1,N_a2]).*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % entireEV_z is (d,a1prime, a2)

            entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*kron(entireEV_z,ones(N_d1,N_a1));
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            Policy(:,z_c,N_j)=maxindex;
        end
        
    elseif vfoptions.lowmemory==2
        error('lowmemory=2 not yet implemented (email me if you want/need it)')
        % if vfoptions.lowmemory>0
        %     if all(size(z_grid)==[sum(n_z),1])
        %         z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
        %     elseif all(size(z_grid)==[prod(n_z),l_z])
        %         z_gridvals=z_grid;
        %     end
        % end
        % for z_c=1:N_z
        %     %Calc the condl expectation term (except beta), which depends on z but
        %     %not on control variables
        %     EV_z=V_Jplus1.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
        %     EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        %     EV_z=sum(EV_z,2);
        % 
        %     entireEV_z=kron(EV_z,ones(N_d2,1));
        % 
        %     z_val=z_gridvals(z_c,:);
        %     for a1_c=1:N_a1
        %         a1_val=a1_gridvals(a1_c,:);
        %         ReturnMatrix_az=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, n_d2, special_n_a1,n_a2, special_n_z, d2_grid, a1_val, a2_grid, z_val, ReturnFnParamsVec);
        % 
        %         entireRHS_az=ReturnMatrix_az+DiscountFactorParamsVec*entireEV_z;
        %         %Calc the max and it's index
        %         [Vtemp,maxindex]=max(entireRHS_az);
        %         V(a_c,z_c,N_j)=Vtemp;
        %         Policy(a_c,z_c,N_j)=maxindex;
        %     end
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

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_Case1(aprimeFn, n_d2, n_a2, d2_grid, a2_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2*N_a2,1], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=kron(ones(N_d2*N_a2,1),(1:1:N_a1)')+N_a1*kron((a2primeIndex-1),ones(N_a1,1)); % [N_d2*N_a1*N_a2,1]
    if vfoptions.lowmemory>0 || vfoptions.paroverz==0
        aprimeProbs=kron(ones(N_a1,1),a2primeProbs);  % [N_d2*N_a1,N_a2]
    else % lowmemory=0 and paroverz=1
        aprimeProbs=repmat(kron(ones(N_a1,1),a2primeProbs),1,1,N_z);  % [N_d2*N_a1,N_a2,N_z]
    end

    VKronNext_j=V(:,:,jj+1);
    
    if vfoptions.lowmemory==0
        
        ReturnMatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, n_d, n_a1,n_a2, n_z, d_grid, a1_grid, a2_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec);
        % (d,aprime,a,z)
        
        if vfoptions.paroverz==1
            
            EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension
            
            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=EV(aprimeIndex+N_a*((1:1:N_z)-1)); % (d2,a1prime,a2,z), the lower aprime
            EV2=EV((aprimeIndex+1)+N_a*((1:1:N_z)-1)); % (d2,a1prime,a2,z), the upper aprime

            % Apply the aprimeProbs
            entireEV=reshape(EV1,[N_d2*N_a1,N_a2,N_z]).*aprimeProbs+reshape(EV2,[N_d2*N_a1,N_a2,N_z]).*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % entireEV is (d,a1prime, a2,z)

%             entireEV=repelem(EV,n_d2,1,1); % I tried this instead but appears repelem() is slower than kron(). However kron() requires 2-D so here I just use repelem() anyway.
            entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(entireEV,N_d1,N_a1,1);
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);
            
            V(:,:,jj)=shiftdim(Vtemp,1);
            Policy(:,:,jj)=shiftdim(maxindex,1);
            
            
        elseif vfoptions.paroverz==0
            
            for z_c=1:N_z
                ReturnMatrix_z=ReturnMatrix(:,:,z_c);
                
                %Calc the condl expectation term (except beta), which depends on z but not on control variables
                EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z_J(z_c,:,jj));
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);

                % Switch EV_z from being in terms of aprime to being in terms of d and a
                EV1=EV_z(aprimeIndex); % (d2,a1prime,a2), the lower aprime
                EV2=EV_z(aprimeIndex+1); % (d2,a1prime,a2), the upper aprime

                % Apply the aprimeProbs
                entireEV_z=reshape(EV1,[N_d2*N_a1,N_a2]).*aprimeProbs+reshape(EV2,[N_d2*N_a1,N_a2]).*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
                % entireEV_z is (d,a1prime, a2)

                entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*kron(entireEV_z,ones(N_d1,N_a1));
                
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_z,[],1);
                V(:,z_c,jj)=Vtemp;
                Policy(:,z_c,jj)=maxindex;
            end
        end
        
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, n_d, n_a1,n_a2, special_n_z, d_grid, a1_grid, a2_grid, z_val, ReturnFnParamsVec);
            
            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z_J(z_c,:,jj));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            % Switch EV_z from being in terms of aprime to being in terms of d and a
            EV1=EV_z(aprimeIndex); % (d2,a1prime,a2), the lower aprime
            EV2=EV_z(aprimeIndex+1); % (d2,a1prime,a2), the upper aprime

            % Apply the aprimeProbs
            entireEV_z=reshape(EV1,[N_d2*N_a1,N_a2]).*aprimeProbs+reshape(EV2,[N_d2*N_a1,N_a2]).*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % entireEV_z is (d,a1prime, a2)

            entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*kron(entireEV_z,ones(N_d1,N_a1));
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,z_c,jj)=Vtemp;
            Policy(:,z_c,jj)=maxindex;
        end
        
    elseif vfoptions.lowmemory==2
        % Note to self: one option would be to have outer loop over z,
        % inner loop over d1 (and then need to get the optimal d1 by a max
        % over that dimension).
        error('lowmemory=2 not yet implemented (email me if you want/need it)')
        % for z_c=1:N_z
        %     %Calc the condl expectation term (except beta), which depends on z but
        %     %not on control variables
        %     EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
        %     EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        %     EV_z=sum(EV_z,2);
        % 
        %     entireEV_z=kron(EV_z,ones(N_d2,1));
        % 
        %     z_val=z_gridvals(z_c,:);
        %     for a1_c=1:N_a1
        %         a1_val=a1_gridvals(a1_c,:);
        %         ReturnMatrix_az=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, n_d2, special_n_a1,n_a2, special_n_z, d2_grid, a1_val, a2_grid, z_val, ReturnFnParamsVec);
        % 
        %         entireRHS_az=ReturnMatrix_az+DiscountFactorParamsVec*entireEV_z;
        %         %Calc the max and it's index
        %         [Vtemp,maxindex]=max(entireRHS_az);
        %         V(a_c,z_c,jj)=Vtemp;
        %         Policy(a_c,z_c,jj)=maxindex;
        %     end
        % end
        
    end
end

%% For experience asset, just output Policy as is and then use Case2 to UnKron
% Policy2=zeros(2,N_a,N_z,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
% Policy2(1,:,:,:)=shiftdim(rem(Policy-1,N_d2)+1,-1);
% Policy2(2,:,:,:)=shiftdim(ceil(Policy/N_d2),-1);

end