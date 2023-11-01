function [V,Policy3]=ValueFnIter_Case1_FHorz_ExpAssetSemiExo_nod1_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,N_j, d2_grid, d3_grid, a1_grid, a2_grid, z_grid, semiz_grid, pi_z, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d2 determines experience asset, d3 determines semi-exog state
% a is endogenous state, a2 is experience asset
% z is exogenous state, semiz is semi-exog state

n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);

% disp('DEBUT')
% n_z
% n_semiz
% n_bothz
% N_bothz

V=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d2,d3,a1prime seperately
Policy3=zeros(3,N_a,N_semiz*N_z,N_j,'gpuArray');

%%
d2_grid=gpuArray(d2_grid);
d3_grid=gpuArray(d3_grid);
a1_grid=gpuArray(a1_grid);
a2_grid=gpuArray(a2_grid);
semiz_grid=gpuArray(semiz_grid);
z_grid=gpuArray(z_grid);

% For the return function we just want (I'm just guessing that as I need them N_j times it will be fractionally faster to put them together now)
n_d=[n_d2,n_d3];
N_d=prod(n_d);
d_grid=[d2_grid;d3_grid];

eval('fieldexists_ExogShockFn=1;vfoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;vfoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')
eval('fieldexists_pi_z_J=1;vfoptions.pi_z_J;','fieldexists_pi_z_J=0;')

if vfoptions.lowmemory>0
    l_z=length(n_z);
    % z_gridvals is created below

    % The grid for semiz is not allowed to depend on age (the way the transition probabilities are calculated does not allow for it)
    if all(size(semiz_grid)==[sum(n_semiz),1])
        semiz_gridvals=CreateGridvals(n_semiz,semiz_grid,1); % The 1 at end indicates want output in form of matrix.
    elseif all(size(semiz_grid)==[prod(n_semiz),l_semiz])
        semiz_gridvals=semiz_grid;
    end

    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

% Preallocate
V_ford3_jj=zeros(N_a,N_semiz*N_z,N_d3,'gpuArray');
Policy_ford3_jj=zeros(N_a,N_semiz*N_z,N_d3,'gpuArray');


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if fieldexists_pi_z_J==1
    z_grid=vfoptions.z_grid_J(:,N_j);
    pi_z=vfoptions.pi_z_J(:,:,N_j);
elseif fieldexists_ExogShockFn==1
    if fieldexists_ExogShockFnParamNames==1
        ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,N_j);
        ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
        for ii=1:length(ExogShockFnParamsVec)
            ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
        end
        [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
        z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
    else
        [z_grid,pi_z]=vfoptions.ExogShockFn(N_j);
        z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
    end
end
if vfoptions.lowmemory>0
    if all(size(z_grid)==[sum(n_z),1])
        z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
    elseif all(size(z_grid)==[prod(n_z),l_z])
        z_gridvals=z_grid;
    end
    bothz_gridvals=[kron(ones(N_z,1),semiz_gridvals),kron(z_gridvals,ones(N_semiz,1))];
else
    if all(size(z_grid)==[sum(n_z),1]) % if z are not using correlated/joint grid, then it is assumed semiz is not either
        bothz_grid=[semiz_grid; z_grid];
    elseif all(size(z_grid)==[prod(n_z),l_z])
        % Joint z_gridvals with semiz_gridvals (note that because z_grid is a joint/correlated grid z_gridvals is anyway just z_grid)
        bothz_grid=[kron(ones(N_z,1),semiz_gridvals), kron(z_grid,ones(N_semiz,1))];
        bothz_gridvals=both_zgrid;
    end
end

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, n_d, n_a1,n_a2, n_bothz, d_grid, a1_grid, a2_grid, bothz_grid, ReturnFnParamsVec); % [N_d*N_a1,N_a1*N_a2,N_z]
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        d_ind=shiftdim(rem(maxindex-1,N_d)+1,-1);
        Policy3(1,:,:,N_j)=shiftdim(rem(d_ind-1,N_d2)+1,-1);
        Policy3(2,:,:,N_j)=shiftdim(ceil(d_ind/N_d2),-1);
        Policy3(3,:,:,N_j)=shiftdim(ceil(maxindex/N_d),-1);

    elseif vfoptions.lowmemory==1

        for z_c=1:N_bothz
            z_val=bothz_gridvals(z_c,:);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, n_d, n_a1,n_a2, special_n_bothz, d_grid, a1_grid, a2_grid, z_val, ReturnFnParamsVec);
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            d_ind=shiftdim(rem(maxindex-1,N_d)+1,-1);
            Policy3(1,:,z_c,N_j)=shiftdim(rem(d_ind-1,N_d2)+1,-1);
            Policy3(2,:,z_c,N_j)=shiftdim(ceil(d_ind/N_d2),-1);
            Policy3(3,:,z_c,N_j)=shiftdim(ceil(maxindex/N_d),-1);
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
        aprimeProbs=repmat(kron(ones(N_a1,1),a2primeProbs),1,1,N_bothz);  % [N_d2*N_a1,N_a2,N_bothz]
    end


    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d3_val=d3_grid(d3_c);
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z,pi_semiz_J(:,:,d3_c,N_j));

            ReturnMatrix_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, [n_d2,1], n_a1,n_a2, n_bothz, [d2_grid;d3_val], a1_grid, a2_grid, bothz_grid, ReturnFnParamsVec);
            % (d,aprime,a,z)

            if vfoptions.paroverz==1

                EV=V_Jplus1.*shiftdim(pi_bothz',-1);
                EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV=sum(EV,2); % sum over z', leaving a singular second dimension

                % Switch EV from being in terms of aprime to being in terms of d and a
                EV1=EV(aprimeIndex+N_a*((1:1:N_bothz)-1)); % (d2,a1prime,a2,z), the lower aprime
                EV2=EV((aprimeIndex+1)+N_a*((1:1:N_bothz)-1)); % (d2,a1prime,a2,z), the upper aprime

                % Apply the aprimeProbs
                entireEV=reshape(EV1,[N_d2*N_a1,N_a2,N_bothz]).*aprimeProbs+reshape(EV2,[N_d2*N_a1,N_a2,N_bothz]).*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
                % entireEV is (d,a1prime, a2,z)

                %             entireEV=repelem(EV,n_d2,1,1); % I tried this instead but appears repelem() is slower than kron(). However kron() requires 2-D so here I just use repelem() anyway.
                entireRHS_d3=ReturnMatrix_d3+DiscountFactorParamsVec*repelem(entireEV,1,N_a1,1);

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_d3,[],1);

                V_ford3_jj(:,:,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,:,d3_c)=shiftdim(maxindex,1);

            elseif vfoptions.paroverz==0

                for z_c=1:N_bothz
                    ReturnMatrix_d3z=ReturnMatrix_d3(:,:,z_c);

                    %Calc the condl expectation term (except beta), which depends on z but not on control variables
                    EV_z=V_Jplus1.*(ones(N_a,1,'gpuArray')*pi_bothz(z_c,:));
                    EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                    EV_z=sum(EV_z,2);

                    % Switch EV_z from being in terms of aprime to being in terms of d and a
                    EV1=EV_z(aprimeIndex); % (d2,a1prime,a2), the lower aprime
                    EV2=EV_z(aprimeIndex+1); % (d2,a1prime,a2), the upper aprime

                    % Apply the aprimeProbs
                    entireEV_z=reshape(EV1,[N_d2*N_a1,N_a2]).*aprimeProbs+reshape(EV2,[N_d2*N_a1,N_a2]).*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
                    % entireEV_z is (d,a1prime, a2)

                    entireRHS_d3z=ReturnMatrix_d3z+DiscountFactorParamsVec*kron(entireEV_z,ones(1,N_a1));

                    %Calc the max and it's index
                    [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                    V_ford3_jj(:,z_c,d3_c)=Vtemp;
                    Policy_ford3_jj(:,z_c,d3_c)=maxindex;
                end
            end
        end
        
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d3_val=d3_grid(d3_c);
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_bothz=kron(pi_z,pi_semiz_J(:,:,d3_c,N_j));

            for z_c=1:N_bothz
                z_val=bothz_gridvals(z_c,:);
                ReturnMatrix_d3z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, [n_d2,1], n_a1,n_a2, special_n_bothz, [d2_grid;d3_val], a1_grid, a2_grid, z_val, ReturnFnParamsVec);

                %Calc the condl expectation term (except beta), which depends on z but not on control variables
                EV_z=V_Jplus1.*(ones(N_a,1,'gpuArray')*pi_bothz(z_c,:));
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);

                % Switch EV_z from being in terms of aprime to being in terms of d and a
                EV1=EV_z(aprimeIndex); % (d2,a1prime,a2), the lower aprime
                EV2=EV_z(aprimeIndex+1); % (d2,a1prime,a2), the upper aprime

                % Apply the aprimeProbs
                entireEV_z=reshape(EV1,[N_d2*N_a1,N_a2]).*aprimeProbs+reshape(EV2,[N_d2*N_a1,N_a2]).*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
                % entireEV_z is (d,a1prime, a2)

                entireRHS_d3z=ReturnMatrix_d3z+DiscountFactorParamsVec*kron(entireEV_z,ones(1,N_a1));

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                V_ford3_jj(:,z_c,d3_c)=Vtemp;
                Policy_ford3_jj(:,z_c,d3_c)=maxindex;
            end
        end
        
    elseif vfoptions.lowmemory==2
        error('lowmemory=2 not yet implemented (email me if you want/need it)')
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_jj,[],3); % max over d2
    V(:,:,N_j)=V_jj;
    Policy3(2,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d2a1prime_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    Policy3(1,:,:,N_j)=shiftdim(rem(d2a1prime_ind-1,N_d2)+1,-1); % d2
    Policy3(3,:,:,N_j)=shiftdim(ceil(d2a1prime_ind/N_d2),-1); % a1prime
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

    if fieldexists_pi_z_J==1
        z_grid=vfoptions.z_grid_J(:,jj);
        pi_z=vfoptions.pi_z_J(:,:,jj);
    elseif fieldexists_ExogShockFn==1
        if fieldexists_ExogShockFnParamNames==1
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
            ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
            for ii=1:length(ExogShockFnParamsVec)
                ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
            end
            [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
            z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
        else
            [z_grid,pi_z]=vfoptions.ExogShockFn(jj);
            z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
        end
    end
    if vfoptions.lowmemory>0 && (fieldexists_pi_z_J==1 || fieldexists_ExogShockFn==1)
        if all(size(z_grid)==[sum(n_z),1])
            z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
        elseif all(size(z_grid)==[prod(n_z),l_z])
            z_gridvals=z_grid;
        end
        bothz_gridvals=[kron(ones(N_z,1),semiz_gridvals),kron(z_gridvals,ones(N_semiz,1))];
    else
        if all(size(z_grid)==[sum(n_z),1]) % if z are not using correlated/joint grid, then it is assumed semiz is not either
            bothz_grid=[semiz_grid; z_grid];
        elseif all(size(z_grid)==[prod(n_z),l_z])
            % Joint z_gridvals with semiz_gridvals (note that because z_grid is a joint/correlated grid z_gridvals is anyway just z_grid)
            bothz_grid=[kron(ones(N_z,1),semiz_gridvals),kron(z_grid,ones(N_semiz,1))];
            bothz_gridvals=both_zgrid;
        end
    end

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_Case1(aprimeFn, n_d2, n_a2, d2_grid, a2_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2*N_a2,1], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=kron(ones(N_d2*N_a2,1),(1:1:N_a1)')+N_a1*kron((a2primeIndex-1),ones(N_a1,1)); % [N_d3*N_a1*N_a2,1]
    if vfoptions.lowmemory>0 || vfoptions.paroverz==0
        aprimeProbs=kron(ones(N_a1,1),a2primeProbs);  % [N_d2*N_a1,N_a2]
    else % lowmemory=0 and paroverz=1
        aprimeProbs=repmat(kron(ones(N_a1,1),a2primeProbs),1,1,N_bothz);  % [N_d3*N_a1,N_a2,N_z]
    end

    VKronNext_j=V(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            
            d3_val=d3_grid(d3_c);
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z,pi_semiz_J(:,:,d3_c,jj));

            ReturnMatrix_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, [n_d2,1], n_a1,n_a2, n_bothz, [d2_grid;d3_val], a1_grid, a2_grid, bothz_grid, ReturnFnParamsVec);
            % (d,aprime,a,z)

            if vfoptions.paroverz==1

                EV=VKronNext_j.*shiftdim(pi_bothz',-1);
                EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV=sum(EV,2); % sum over z', leaving a singular second dimension

                % Switch EV from being in terms of aprime to being in terms of d and a
                EV1=EV(aprimeIndex+N_a*((1:1:N_bothz)-1)); % (d2,a1prime,a2,z), the lower aprime
                EV2=EV((aprimeIndex+1)+N_a*((1:1:N_bothz)-1)); % (d2,a1prime,a2,z), the upper aprime

                % Apply the aprimeProbs
                entireEV=reshape(EV1,[N_d2*N_a1,N_a2,N_bothz]).*aprimeProbs+reshape(EV2,[N_d2*N_a1,N_a2,N_bothz]).*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
                % entireEV is (d,a1prime, a2,z)

                %             entireEV=repelem(EV,n_d2,1,1); % I tried this instead but appears repelem() is slower than kron(). However kron() requires 2-D so here I just use repelem() anyway.
                entireRHS=ReturnMatrix_d3+DiscountFactorParamsVec*repelem(entireEV,1,N_a1,1);

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS,[],1);

                V_ford3_jj(:,:,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,:,d3_c)=shiftdim(maxindex,1);

            elseif vfoptions.paroverz==0

                for z_c=1:N_bothz
                    ReturnMatrix_d3z=ReturnMatrix_d3(:,:,z_c);

                    %Calc the condl expectation term (except beta), which depends on z but not on control variables
                    EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_bothz(z_c,:));
                    EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                    EV_z=sum(EV_z,2);

                    % Switch EV_z from being in terms of aprime to being in terms of d and a
                    EV1=EV_z(aprimeIndex); % (d2,a1prime,a2), the lower aprime
                    EV2=EV_z(aprimeIndex+1); % (d2,a1prime,a2), the upper aprime

                    % Apply the aprimeProbs
                    entireEV_z=reshape(EV1,[N_d2*N_a1,N_a2]).*aprimeProbs+reshape(EV2,[N_d2*N_a1,N_a2]).*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
                    % entireEV_z is (d,a1prime, a2)

                    entireRHS_z=ReturnMatrix_d3z+DiscountFactorParamsVec*kron(entireEV_z,ones(1,N_a1));

                    %Calc the max and it's index
                    [Vtemp,maxindex]=max(entireRHS_z,[],1);

                    V_ford3_jj(:,z_c,d3_c)=shiftdim(Vtemp,1);
                    Policy_ford3_jj(:,z_c,d3_c)=shiftdim(maxindex,1);

                end
            end
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d3_val=d3_grid(d3_c);
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_bothz=kron(pi_z, pi_semiz_J(:,:,d3_c,jj));

            for z_c=1:N_bothz
                z_val=bothz_gridvals(z_c,:);
                ReturnMatrix_d3z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, [n_d2,1], n_a1,n_a2, special_n_bothz, [d2_grid;d3_val], a1_grid, a2_grid, z_val, ReturnFnParamsVec);

                %Calc the condl expectation term (except beta), which depends on z but not on control variables
                EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_bothz(z_c,:));
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);

                % Switch EV_z from being in terms of aprime to being in terms of d and a
                EV1=EV_z(aprimeIndex); % (d2,a1prime,a2), the lower aprime
                EV2=EV_z(aprimeIndex+1); % (d2,a1prime,a2), the upper aprime

                % Apply the aprimeProbs
                entireEV_z=reshape(EV1,[N_d2*N_a1,N_a2]).*aprimeProbs+reshape(EV2,[N_d2*N_a1,N_a2]).*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
                % entireEV_z is (d,a1prime, a2)

                entireRHS_z=ReturnMatrix_d3z+DiscountFactorParamsVec*kron(entireEV_z,ones(1,N_a1));

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_z,[],1);

                V_ford3_jj(:,z_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,z_c,d3_c)=shiftdim(maxindex,1);
            end
        end

    elseif vfoptions.lowmemory==2
        error('lowmemory=2 not yet implemented (email me if you want/need it)')
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_jj,[],3); % max over d3
    V(:,:,jj)=V_jj;
    Policy3(2,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d2a1prime_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    Policy3(1,:,:,jj)=shiftdim(rem(d2a1prime_ind-1,N_d2)+1,-1); % d2
    Policy3(3,:,:,jj)=shiftdim(ceil(d2a1prime_ind/N_d2),-1); % a1prime

end


%% For experience asset, just output Policy as is and then use Case2 to UnKron

end