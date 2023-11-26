function [V,Policy2]=ValueFnIter_Case1_FHorz_ResidAsset_raw(n_d,n_a,n_r,n_z,N_j, d_grid, a_grid, r_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_r=prod(n_r);
N_z=prod(n_z);

V=zeros(N_a,N_r,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_r,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
r_grid=gpuArray(r_grid);

if vfoptions.lowmemory>0
    l_z=length(n_z);
    special_n_z=ones(1,length(n_z));
end
if vfoptions.lowmemory>1
    special_n_r=ones(1,length(n_r));
    r_gridvals=CreateGridvals(n_r,r_grid,1); % The 1 at end indicates want output in form of matrix.
end


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case1_ResidAsset_Disc_Par2(ReturnFn, n_d, n_a, n_r, n_z, d_grid, a_grid, r_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=Vtemp;
        Policy(:,:,:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_ResidAsset_Disc_Par2(ReturnFn, n_d, n_a, n_r, special_n_z, d_grid, a_grid, r_grid, z_val, ReturnFnParamsVec);
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,:,z_c,N_j)=Vtemp;
            Policy(:,:,z_c,N_j)=maxindex;
        end

    elseif vfoptions.lowmemory==2

        for r_c=1:N_r
            r_val=r_gridvals(r_c,:);
            ReturnMatrix_rz=CreateReturnFnMatrix_ResidAsset_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_r, n_z, d_grid, a_grid, r_val, z_gridvals_J(:,:,N_j), ReturnFnParamsVec);
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_rz);
            V(:,r_c,:,N_j)=Vtemp;
            Policy(:,r_c,:,N_j)=maxindex;
        end

    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_r,N_z]);    % First, switch V_Jplus1 into Kron form

    % Residual asset:
    % VKronNext_j is over (aprime,r,z)
    % Need to convert to be over (d,aprime,a,z)
    rprimeFnParamsVec=CreateVectorFromParams(Parameters, rprimeFnParamNames,N_j);
    [rprimeIndexes,rprimeProbs]=CreateResidualAssetFnMatrix_Case1(rprimeFn, n_d, n_a, n_r, n_z, d_grid, a_grid, r_grid, z_gridvals_J(:,:,N_j), rprimeFnParamsVec);  % Note, is actually rprime_grid (but r_grid is anyway same for all ages)
    % Note: rprimeIndex is [N_d*N_a*N_a*N_z,1], and rprimeProbs is [N_d*N_a*N_a*N_z,1]
    aprimeIndexes=kron(ones(N_a*N_z,1),(1:1:N_a)'); % aprime over (d,aprime,a,z)
    zprimeIndexes=kron((1:1:N_z)',ones(N_a*N_a,1)); % zprime over (d,aprime,a,z)

    % lower r index (size is N_d*N_a*N_a*N_z)
    fullindex=aprimeIndexes+N_a*(rprimeIndexes-1)+N_a*N_r*(zprimeIndexes-1); % index for (a',r',z'), as function of (d,a',a,z)
    Vindex1=V_Jplus1(fullindex);
    % upper r index
    fullindex=aprimeIndexes+N_a*rprimeIndexes+N_a*N_r*(zprimeIndexes-1); % index for (a',r',z'), as function of (d,a',a,z)
    Vindex2=V_Jplus1(fullindex);
    % So now we have next period value function, but with state (d,a',a,z)
    V_Jplus1=Vindex1.*rprimeProbs+Vindex2.*(1-rprimeProbs);
    V_Jplus1=reshape(V_Jplus1,[N_d*N_a,N_a,N_z]);
    % now, it is over (d,a',a,z)

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    if vfoptions.lowmemory==0
        
        ReturnMatrix=CreateReturnFnMatrix_Case1_ResidAsset_Disc_Par2(ReturnFn, n_d, n_a, n_r, n_z, d_grid, a_grid, r_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        % (d,aprime,a,z)
        
        if vfoptions.paroverz==1
            
            EV=V_Jplus1.*shiftdim(pi_z_J(:,:,N_j)',-2); % Note: shiftdim -3
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,3); % sum over z', leaving a singular second dimension
            
            entireRHS=ReturnMatrix+DiscountFactorParamsVec*(reshape(EV,[N_d*N_a,N_a,1,N_z]).*ones(1,1,N_r,1));
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);
            
            V(:,:,:,N_j)=shiftdim(Vtemp,1);
            Policy(:,:,:,N_j)=shiftdim(maxindex,1);
            
            
        elseif vfoptions.paroverz==0
            
            for z_c=1:N_z
                ReturnMatrix_z=ReturnMatrix(:,:,:,z_c);
                
                % Use sparse for a few lines until sum over zprime
                EV_z=V_Jplus1.*shiftdim(pi_z_J(z_c,:,N_j)',-2); % Note: shiftdim -3
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,3); % sum over z', leaving a singular second dimension
                
                entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*(reshape(EV_z,[N_d*N_a,N_a,1]).*ones(1,1,N_r));
                
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_z,[],1);
                V(:,:,z_c,N_j)=Vtemp;
                Policy(:,:,z_c,N_j)=maxindex;
            end
        end
        
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_ResidAsset_Disc_Par2(ReturnFn, n_d, n_a, n_r, special_n_z, d_grid, a_grid, r_grid, z_val, ReturnFnParamsVec);
            
            EV_z=V_Jplus1.*shiftdim(pi_z_J(z_c,:,N_j)',-2); % Note: shiftdim -3
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,3); % sum over z', leaving a singular second dimension
            
            entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*(reshape(EV_z,[N_d*N_a,N_a,1]).*ones(1,1,N_r));
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,:,z_c,N_j)=Vtemp;
            Policy(:,:,z_c,N_j)=maxindex;
        end
        
    elseif vfoptions.lowmemory==2
        % Loop over r

        EV=V_Jplus1.*shiftdim(pi_z_J(:,:,N_j)',-2); % Note: shiftdim -3
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,3); % sum over z', leaving a singular second dimension

        for r_c=1:N_r
            r_val=r_gridvals(r_c,:);
            ReturnMatrix_r=CreateReturnFnMatrix_Case1_ResidAsset_Disc_Par2(ReturnFn, n_d, n_a, special_n_r, n_z, d_grid, a_grid, r_val, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);

            entireRHS_r=ReturnMatrix_r+DiscountFactorParamsVec*reshape(EV,[N_d*N_a,N_a,1,N_z]);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_r);
            V(:,r_c,:,N_j)=Vtemp;
            Policy(:,r_c,:,N_j)=maxindex;
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
    
    VKronNext_j=V(:,:,jj+1);

    % Residual asset:
    % VKronNext_j is over (aprime,r,z)
    % Need to convert to be over (d,aprime,a,z)
    rprimeFnParamsVec=CreateVectorFromParams(Parameters, rprimeFnParamNames,jj);
    [rprimeIndexes,rprimeProbs]=CreateResidualAssetFnMatrix_Case1(rprimeFn, n_d, n_a, n_r, n_z, d_grid, a_grid, r_grid, z_gridvals_J(:,:,jj), rprimeFnParamsVec);  % Note, is actually rprime_grid (but r_grid is anyway same for all ages)
    % Note: rprimeIndex is [N_d*N_a*N_a*N_z,1], and rprimeProbs is [N_d*N_a*N_a*N_z,1]
    aprimeIndexes=kron(ones(N_a*N_z,1),(1:1:N_a)'); % aprime over (d,aprime,a,z)
    zprimeIndexes=kron((1:1:N_z)',ones(N_a*N_a,1)); % zprime over (d,aprime,a,z)

    % lower r index (size is N_d*N_a*N_a*N_z)
    fullindex=aprimeIndexes+N_a*(rprimeIndexes-1)+N_a*N_r*(zprimeIndexes-1); % index for (a',r',z'), as function of (d,a',a,z)
    Vindex1=VKronNext_j(fullindex);
    % upper r index
    fullindex=aprimeIndexes+N_a*rprimeIndexes+N_a*N_r*(zprimeIndexes-1); % index for (a',r',z'), as function of (d,a',a,z)
    Vindex2=VKronNext_j(fullindex);
    % So now we have next period value function, but with state (d,a',a,z)
    VKronNext_j=Vindex1.*rprimeProbs+Vindex2.*(1-rprimeProbs);
    VKronNext_j=reshape(VKronNext_j,[N_d*N_a,N_a,N_z]);
    % now, it is over (d,a',a,z)
    
    if vfoptions.lowmemory==0
        
        ReturnMatrix=CreateReturnFnMatrix_Case1_ResidAsset_Disc_Par2(ReturnFn, n_d, n_a, n_r, n_z, d_grid, a_grid, r_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec,0);
        % (d,aprime,a,r,z)

        if vfoptions.paroverz==1
            
            EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-2); % Note: shiftdim -3
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,3); % sum over z', leaving a singular second dimension
            
            entireRHS=ReturnMatrix+DiscountFactorParamsVec*(reshape(EV,[N_d*N_a,N_a,1,N_z]).*ones(1,1,N_r,1));
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);
            
            V(:,:,:,jj)=shiftdim(Vtemp,1);
            Policy(:,:,:,jj)=shiftdim(maxindex,1);
            
        elseif vfoptions.paroverz==0
            
            for z_c=1:N_z
                ReturnMatrix_z=ReturnMatrix(:,:,:,z_c);
                
                % Use sparse for a few lines until sum over zprime
                EV_z=VKronNext_j.*shiftdim(pi_z_J(z_c,:,jj)',-2); % Note: shiftdim -3
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,3); % sum over z', leaving a singular second dimension

                entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*(reshape(EV_z,[N_d*N_a,N_a,1]).*ones(1,1,N_r));
                
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_z,[],1);
                V(:,:,z_c,jj)=Vtemp;
                Policy(:,:,z_c,jj)=maxindex;
            end
        end
        
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_ResidAsset_Disc_Par2(ReturnFn, n_d, n_a, n_r, special_n_z, d_grid, a_grid, r_grid, z_val, ReturnFnParamsVec,0);
            
            EV_z=VKronNext_j.*shiftdim(pi_z_J(z_c,:,jj)',-2); % Note: shiftdim -3
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,3); % sum over z', leaving a singular second dimension
            
            entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*(reshape(EV_z,[N_d*N_a,N_a,1]).*ones(1,1,N_r));
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,:,z_c,jj)=Vtemp;
            Policy(:,:,z_c,jj)=maxindex;
        end
        
    elseif vfoptions.lowmemory==2
        % Loop over r

        EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-2); % Note: shiftdim -3
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,3); % sum over z', leaving a singular second dimension

        for r_c=1:N_r
            r_val=r_gridvals(r_c,:);
            ReturnMatrix_r=CreateReturnFnMatrix_Case1_ResidAsset_Disc_Par2(ReturnFn, n_d, n_a, special_n_r, n_z, d_grid, a_grid, r_val, z_gridvals_J(:,:,jj), ReturnFnParamsVec,0);

            entireRHS_r=ReturnMatrix_r+DiscountFactorParamsVec*reshape(EV,[N_d*N_a,N_a,1,N_z]);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_r);
            V(:,r_c,:,jj)=Vtemp;
            Policy(:,r_c,:,jj)=maxindex;
        end
        
    end
end

%%
Policy2=zeros(2,N_a,N_r,N_z,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
Policy2(1,:,:,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:,:,:)=shiftdim(ceil(Policy/N_d),-1);

end