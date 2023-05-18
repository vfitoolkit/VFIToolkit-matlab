function [V,Policy]=ValueFnIter_Case1_FHorz_ExpAsset_e_raw(n_d1,n_d2,n_a1,n_a2,n_z,n_e,N_j, d1_grid, d2_grid, a1_grid, a2_grid, z_grid, e_grid, pi_z, pi_e, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);
N_e=prod(n_e);

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

%%
d1_grid=gpuArray(d2_grid);
d2_grid=gpuArray(d2_grid);
a1_grid=gpuArray(a1_grid);
a2_grid=gpuArray(a2_grid);
z_grid=gpuArray(z_grid);
e_grid=gpuArray(e_grid);

% For the return function we just want (I'm just guessing that as I need them N_j times it will be fractionally faster to put them together now)
n_d=[n_d1,n_d2];
d_grid=[d1_grid;d2_grid];

% Z markov
eval('fieldexists_ExogShockFn=1;vfoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;vfoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')
eval('fieldexists_pi_z_J=1;vfoptions.pi_z_J;','fieldexists_pi_z_J=0;')
% E iid
eval('fieldexists_EiidShockFn=1;vfoptions.EiidShockFn;','fieldexists_EiidShockFn=0;')
eval('fieldexists_EiidShockFnParamNames=1;vfoptions.EiidShockFnParamNames;','fieldexists_EiidShockFnParamNames=0;')
eval('fieldexists_pi_e_J=1;vfoptions.pi_e_J;','fieldexists_pi_e_J=0;')

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
    % e_gridvals is created below
end
if vfoptions.lowmemory>1
    l_z=length(n_z);
    special_n_z=ones(1,l_z);
    % z_gridvals is created below
end

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
end

if fieldexists_pi_e_J==1
    e_grid=vfoptions.e_grid_J(:,N_j);
    pi_e=vfoptions.pi_e_J(:,N_j);
elseif fieldexists_EiidShockFn==1
    if fieldexists_EiidShockFnParamNames==1
        EiidShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.EiidShockFnParamNames,N_j);
        EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
        for ii=1:length(EiidShockFnParamsVec)
            EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
        end
        [e_grid,pi_e]=vfoptions.EiidShockFn(EiidShockFnParamsCell{:});
        e_grid=gpuArray(e_grid); pi_e=gpuArray(pi_e);
    else
        [e_grid,pi_e]=vfoptions.ExogShockFn(N_j);
        e_grid=gpuArray(e_grid); pi_e=gpuArray(pi_e);
    end
end
if vfoptions.lowmemory>0
    if all(size(e_grid)==[sum(n_e),1]) % kronecker (cross-product) grid
        e_gridvals=CreateGridvals(n_e,e_grid,1); % The 1 at end indicates want output in form of matrix.
    elseif all(size(e_grid)==[prod(n_e),length(n_e)]) % joint-grid
        e_gridvals=e_grid;
    end
end

pi_e=shiftdim(pi_e,-2); % Move to thrid dimension

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d, n_a1,n_a2, n_z, n_e, d_grid, a1_grid, a2_grid, z_grid, e_grid, ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=Vtemp;
        Policy(:,:,:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d, n_a1,n_a2, n_z, special_n_e, d_grid, a1_grid, a2_grid, z_grid,e_val, ReturnFnParamsVec);
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            Policy(:,:,e_c,N_j)=maxindex;
        end

    elseif vfoptions.lowmemory==2

        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            for z_c=1:N_z
                z_val=z_gridvals(z_c,:);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d, special_n_a1,n_a2, special_n_z, special_n_e, d_grid, a1_val, a2_grid, z_val, e_val, ReturnFnParamsVec);
                %Calc the max and it's index
                [Vtemp,maxindex]=max(ReturnMatrix_ze);
                V(:,z_c,e_c,N_j)=Vtemp;
                Policy(:,z_c,e_c,N_j)=maxindex;

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

    %% UP TO HERE %%

    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    V_Jplus1=sum(V_Jplus1.*pi_e,3);
    
    if vfoptions.lowmemory==0
        
        ReturnMatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d, n_a1,n_a2, n_z, n_e, d_grid, a1_grid, a2_grid, z_grid, e_grid, ReturnFnParamsVec);
        % (d,aprime,a,z,e)

        EV=V_Jplus1.*shiftdim(pi_z',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension

        % Switch EV from being in terms of aprime to being in terms of d and a
        EV1=EV(aprimeIndex+N_a*((1:1:N_z)-1)); % (d2,a1prime,a2,z), the lower aprime
        EV2=EV((aprimeIndex+1)+N_a*((1:1:N_z)-1)); % (d2,a1prime,a2,z), the upper aprime

        % Apply the aprimeProbs
        entireEV=reshape(EV1,[N_d2*N_a1,N_a2,N_z]).*aprimeProbs+reshape(EV2,[N_d2*N_a1,N_a2,N_z]).*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
        % entireEV is (d,a1prime, a2,z)

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(entireEV,N_d1,N_a1,1,N_e);

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,:,N_j)=shiftdim(maxindex,1);
        
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d, n_a1,n_a2, n_z, special_n_e, d_grid, a1_grid, a2_grid, z_grid,e_val, ReturnFnParamsVec);

            EV=V_Jplus1.*shiftdim(pi_z',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=EV(aprimeIndex+N_a*((1:1:N_z)-1)); % (d2,a1prime,a2,z), the lower aprime
            EV2=EV((aprimeIndex+1)+N_a*((1:1:N_z)-1)); % (d2,a1prime,a2,z), the upper aprime

            % Apply the aprimeProbs
            entireEV=reshape(EV1,[N_d2*N_a1,N_a2,N_z]).*aprimeProbs+reshape(EV2,[N_d2*N_a1,N_a2,N_z]).*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % entireEV is (d,a1prime, a2,z)

            entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*kron(kron(ones(N_d1,1),entireEV),ones(1,N_a1));
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            Policy(:,:,e_c,N_j)=maxindex;
        end
        
    elseif vfoptions.lowmemory==2
        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            for z_c=1:N_z
                z_val=z_gridvals(z_c,:);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d, n_a1,n_a2, special_n_z, special_n_e, d_grid, a1_grid, a2_grid, z_val,e_val, ReturnFnParamsVec);

                %Calc the condl expectation term (except beta), which depends on z but not on control variables
                EV_z=V_Jplus1.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);

                % Switch EV_z from being in terms of aprime to being in terms of d and a
                EV1=EV_z(aprimeIndex); % (d2,a1prime,a2), the lower aprime
                EV2=EV_z(aprimeIndex+1); % (d2,a1prime,a2), the upper aprime

                % Apply the aprimeProbs
                entireEV_z=reshape(EV1,[N_d2*N_a1,N_a2]).*aprimeProbs+reshape(EV2,[N_d2*N_a1,N_a2]).*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
                % entireEV_z is (d,a1prime, a2)

                entireRHS_ze=ReturnMatrix_ze+DiscountFactorParamsVec*kron(kron(ones(N_d1,1),entireEV_z),ones(1,N_a1));

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                V(:,z_c,e_c,N_j)=Vtemp;
                Policy(:,z_c,e_c,N_j)=maxindex;


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
    if vfoptions.lowmemory>1 && (fieldexists_pi_z_J==1 || fieldexists_ExogShockFn==1)
        if all(size(z_grid)==[sum(n_z),1])
            z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
        elseif all(size(z_grid)==[prod(n_z),l_z])
            z_gridvals=z_grid;
        end
    end

    if fieldexists_pi_e_J==1
        e_grid=vfoptions.e_grid_J(:,jj);
        pi_e=vfoptions.pi_e_J(:,jj);
    elseif fieldexists_EiidShockFn==1
        if fieldexists_EiidShockFnParamNames==1
            EiidShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.EiidShockFnParamNames,jj);
            EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
            for ii=1:length(EiidShockFnParamsVec)
                EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
            end
            [e_grid,pi_e]=vfoptions.EiidShockFn(EiidShockFnParamsCell{:});
            e_grid=gpuArray(e_grid); pi_e=gpuArray(pi_e);
        else
            [e_grid,pi_e]=vfoptions.ExogShockFn(jj);
            e_grid=gpuArray(e_grid); pi_e=gpuArray(pi_e);
        end
    end
    if vfoptions.lowmemory>0
        if all(size(e_grid)==[sum(n_e),1]) % kronecker (cross-product) grid
            e_gridvals=CreateGridvals(n_e,e_grid,1); % The 1 at end indicates want output in form of matrix.
        elseif all(size(e_grid)==[prod(n_e),length(n_e)]) % joint-grid
            e_gridvals=e_grid;
        end
    end

    pi_e=shiftdim(pi_e,-2); % Move to thrid dimension

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_Case1(aprimeFn, n_d2, n_a2, d2_grid, a2_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2*N_a2,1], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=kron(ones(N_d2*N_a2,1),(1:1:N_a1)')+N_a1*kron((a2primeIndex-1),ones(N_a1,1)); % [N_d2*N_a1*N_a2,1]
    if vfoptions.lowmemory==2
        aprimeProbs=kron(ones(N_a1,1),a2primeProbs);  % [N_d2*N_a1,N_a2]
    else % lowmemory=0 and  lowmemory=1
        aprimeProbs=repmat(kron(ones(N_a1,1),a2primeProbs),1,1,N_z);  % [N_d2*N_a1,N_a2,N_z]
    end

    VKronNext_j=V(:,:,jj+1);

    V_Jplus1=sum(V_Jplus1.*pi_e,3);
    
    if vfoptions.lowmemory==0
        
        ReturnMatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d, n_a1,n_a2, n_z,n_e, d_grid, a1_grid, a2_grid, z_grid,e_grid, ReturnFnParamsVec);
        % (d,aprime,a,z)

        EV=VKronNext_j.*shiftdim(pi_z',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension

        % Switch EV from being in terms of aprime to being in terms of d and a
        EV1=EV(aprimeIndex+N_a*((1:1:N_z)-1)); % (d2,a1prime,a2,z), the lower aprime
        EV2=EV((aprimeIndex+1)+N_a*((1:1:N_z)-1)); % (d2,a1prime,a2,z), the upper aprime

        % Apply the aprimeProbs
        entireEV=reshape(EV1,[N_d2*N_a1,N_a2,N_z]).*aprimeProbs+reshape(EV2,[N_d2*N_a1,N_a2,N_z]).*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
        % entireEV is (d,a1prime, a2,z)

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(entireEV,N_d1,N_a1,1,N_e);

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,jj)=shiftdim(maxindex,1);
                
    elseif vfoptions.lowmemory==1
        if vfoptions.lowmemory>0
            if all(size(z_grid)==[sum(n_z),1])
                z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
            elseif all(size(z_grid)==[prod(n_z),l_z])
                z_gridvals=z_grid;
            end
        end
        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d, n_a1,n_a2, n_z, special_n_e, d_grid, a1_grid, a2_grid, z_grid,e_val, ReturnFnParamsVec);

            EV=V_Jplus1.*shiftdim(pi_z',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=EV(aprimeIndex+N_a*((1:1:N_z)-1)); % (d2,a1prime,a2,z), the lower aprime
            EV2=EV((aprimeIndex+1)+N_a*((1:1:N_z)-1)); % (d2,a1prime,a2,z), the upper aprime

            % Apply the aprimeProbs
            entireEV=reshape(EV1,[N_d2*N_a1,N_a2,N_z]).*aprimeProbs+reshape(EV2,[N_d2*N_a1,N_a2,N_z]).*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % entireEV is (d,a1prime, a2,z)

            entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*kron(kron(ones(N_d1,1),entireEV),ones(1,N_a1));
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            V(:,:,e_c,jj)=Vtemp;
            Policy(:,:,e_c,jj)=maxindex;
        end
        
    elseif vfoptions.lowmemory==2
        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            for z_c=1:N_z
                z_val=z_gridvals(z_c,:);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d, n_a1,n_a2, special_n_z, special_n_e, d_grid, a1_grid, a2_grid, z_val,e_val, ReturnFnParamsVec);

                %Calc the condl expectation term (except beta), which depends on z but not on control variables
                EV_z=V_Jplus1.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);

                % Switch EV_z from being in terms of aprime to being in terms of d and a
                EV1=EV_z(aprimeIndex); % (d2,a1prime,a2), the lower aprime
                EV2=EV_z(aprimeIndex+1); % (d2,a1prime,a2), the upper aprime

                % Apply the aprimeProbs
                entireEV_z=reshape(EV1,[N_d2*N_a1,N_a2]).*aprimeProbs+reshape(EV2,[N_d2*N_a1,N_a2]).*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
                % entireEV_z is (d,a1prime, a2)

                entireRHS_ze=ReturnMatrix_ze+DiscountFactorParamsVec*kron(kron(ones(N_d1,1),entireEV_z),ones(1,N_a1));

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                V(:,z_c,e_c,jj)=Vtemp;
                Policy(:,z_c,e_c,jj)=maxindex;

            end
        end
    end
end

%% For experience asset, just output Policy as is and then use Case2 to UnKron
% Policy2=zeros(2,N_a,N_z,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
% Policy2(1,:,:,:)=shiftdim(rem(Policy-1,N_d2)+1,-1);
% Policy2(2,:,:,:)=shiftdim(ceil(Policy/N_d2),-1);

end