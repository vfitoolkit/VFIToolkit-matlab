function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_nod1_e_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_e,n_u,N_j, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, e_gridvals_J, u_grid, pi_z_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_z=prod(n_z);
N_e=prod(n_e);
N_u=prod(n_u);

N_d=N_d2*N_d3;
N_a=N_a1*N_a2;

% For ReturnFn
% n_d3
% N_d13
% d3_grid
% For aprimeFn
n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy3=zeros(3,N_a,N_z,N_e,N_j,'gpuArray'); % d2,d3,a1prime

%%
pi_u=shiftdim(pi_u,-1); % 2nd dimension

d3a1_gridvals=CreateGridvals([n_d3,n_a1],[d3_grid;a1_grid],1);
a1a2_gridvals=CreateGridvals([n_a1,n_a2],[a1_grid;a2_grid],1);

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    l_z=length(n_z);
    special_n_z=ones(1,l_z);
end

% aind=(0:1:N_a-1);
zind=shiftdim(gpuArray(0:1:N_z-1),-1);
% eind=shiftdim(0:1:N_e-1,-2);


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

pi_e_J=shiftdim(pi_e_J,-2); % Move to third dimension

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2e(ReturnFn, [n_d3,n_a1], [n_a1,n_a2], n_z, n_e, d3a1_gridvals, a1a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=Vtemp;
        Policy3(1,:,:,:,N_j)=1; % is meaningless anyway
        Policy3(2,:,:,:,N_j)=shiftdim(rem(maxindex-1,N_d)+1,-1);
        Policy3(3,:,:,:,N_j)=shiftdim(ceil(maxindex/N_d),-1);

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_Par2e(ReturnFn, [n_d3,n_a1], [n_a1,n_a2], n_z, special_n_e, d3a1_gridvals, a1a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
            % Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            Policy3(1,:,:,e_c,N_j)=1; % is meaningless anyway
            Policy3(2,:,:,e_c,N_j)=shiftdim(rem(maxindex-1,N_d)+1,-1);
            Policy3(3,:,:,e_c,N_j)=shiftdim(ceil(maxindex/N_d),-1);
        end

    elseif vfoptions.lowmemory==2

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_z
                z_val=z_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_Par2e(ReturnFn, [n_d3,n_a1], [n_a1,n_a2], special_n_z, special_n_e, d3a1_gridvals, a1a2_gridvals, z_val, e_val, ReturnFnParamsVec);
                % Calc the max and it's index
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                V(:,z_c,e_c,N_j)=Vtemp;
                Policy3(1,:,z_c,e_c,N_j)=1; % is meaningless anyway
                Policy3(2,:,z_c,e_c,N_j)=shiftdim(rem(maxindex-1,N_d)+1,-1);
                Policy3(3,:,z_c,e_c,N_j)=shiftdim(ceil(maxindex/N_d),-1);
            end
        end

    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    EV=sum(reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]).*pi_e_J(1,1,:,N_j),3); % Using V_Jplus1

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateaprimeFnMatrix_RiskyAsset(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: a2primeIndex is [N_d,N_u], whereas a2primeProbs is [N_d,N_u]

    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1); % [N_d*N_a1,N_u]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d*N_a1,N_u]
    % aprimeProbs=repmat(a2primeProbs,N_a1,1);  % [N_d*N_a1,N_u]
    % Note: aprimeIndex corresponds to value of (a1, a2), but has dimension (d,a1)
    aprimeProbs=repmat(a2primeProbs,N_a1,1,N_z);  % [N_d*N_a1,N_u,N_z]

    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-1);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
    EVlower=reshape(EV(aprimeIndex(:),:),[N_d23*N_a1,N_u,N_z]); % the lower aprime
    EVupper=reshape(EV(aprimeplus1Index(:),:),[N_d23*N_a1,N_u,N_z]); % the upper aprime
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(EVlower==EVupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation
    
    %  Switch EV from being in terps of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*EVlower+(1-aprimeProbs).*EVupper; % (d23 & a1prime,u,zprime)
    % Already applied the probabilities from interpolating onto grid
    EV=squeeze(sum((EV.*pi_u),2)); % (d23 & a1prime,zprime)

    % Time to refine EV, we can refine out d2
    [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3*N_a1,1,N_z]),[],1);

    DiscountedEV_onlyd3=DiscountFactorParamsVec*shiftdim(EV_onlyd3,1);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2e(ReturnFn, [n_d3,n_a1], [n_a1,n_a2], n_z, n_e, d3a1_gridvals, a1a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        % (d,aprime,a,z,e)
        
        % Time to refine ReturnMatrix, we can refine out d1
        % no d1 here

        % Now put together entireRHS, which just depends on d3
        entireRHS=ReturnMatrix+DiscountedEV_onlyd3;
                
        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        
        V(:,:,:,N_j)=shiftdim(Vtemp,1);
        Policy3(2,:,:,:,N_j)=shiftdim(rem(maxindex-1,N_d3)+1,1);
        Policy3(3,:,:,:,N_j)=shiftdim(ceil(maxindex/N_d3),-1);
        Policy3(1,:,:,:,N_j)=shiftdim(d2index(maxindex+N_d3*zind),1);

    elseif vfoptions.lowmemory==1


        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);

            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_Par2e(ReturnFn, [n_d3,n_a1], [n_a1,n_a2], n_z, special_n_e, d3a1_gridvals, a1a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
            % (d,aprime,a,z)
            
            % Time to refine ReturnMatrix, we can refine out d1
            % no d1 here

            % Now put together entireRHS, which just depends on d3
            entireRHS_e=ReturnMatrix_e+DiscountedEV_onlyd3;
            
            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);

            V(:,:,e_c,N_j)=shiftdim(Vtemp,1);
            Policy3(2,:,:,e_c,N_j)=shiftdim(rem(maxindex-1,N_d3)+1,1);
            Policy3(3,:,:,e_c,N_j)=shiftdim(ceil(maxindex/N_d3),-1);
            Policy3(1,:,:,e_c,N_j)=shiftdim(d2index(maxindex+N_d3*zind),1);
        end
        
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            DiscountedEV_onlyd3_z=DiscountedEV_onlyd3(:,1,z_c);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                
                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_Par2e(ReturnFn, [n_d3,n_a1], [n_a1,n_a2], special_n_z, special_n_e, d3a1_gridvals, a1a2_gridvals, z_val, e_val, ReturnFnParamsVec);

                % Time to refine ReturnMatrix, we can refine out d1
                % no d1 here

                % Now put together entireRHS, which just depends on d3
                entireRHS_ze=ReturnMatrix_ze+DiscountedEV_onlyd3_z;
                
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_ze,[],1);

                V(:,z_c,e_c,N_j)=Vtemp;
                Policy3(2,:,z_c,e_c,N_j)=shiftdim(rem(maxindex-1,N_d3)+1,1);
                Policy3(3,:,z_c,e_c,N_j)=shiftdim(ceil(maxindex/N_d3),-1);
                Policy3(1,:,z_c,e_c,N_j)=shiftdim(d2index(maxindex),1);
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
    
    EV=sum(V(:,:,:,jj+1).*pi_e_J(1,1,:,jj),3);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateaprimeFnMatrix_RiskyAsset(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: a2primeIndex is [N_d,N_u], whereas a2primeProbs is [N_d,N_u]

    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1); % [N_d*N_a1,N_u]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d*N_a1,N_u]
    % aprimeProbs=repmat(a2primeProbs,N_a1,1);  % [N_d*N_a1,N_u]
    % Note: aprimeIndex corresponds to value of (a1, a2), but has dimension (d,a1)
    aprimeProbs=repmat(a2primeProbs,N_a1,1,N_z);  % [N_d*N_a1,N_u,N_z]

    EV=EV.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
    EVlower=reshape(EV(aprimeIndex(:),:),[N_d23*N_a1,N_u,N_z]); % the lower aprime
    EVupper=reshape(EV(aprimeplus1Index(:),:),[N_d23*N_a1,N_u,N_z]); % the upper aprime
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(EVlower==EVupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation
    
    %  Switch EV from being in terps of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*EVlower+(1-aprimeProbs).*EVupper; % (d23 & a1prime,u,zprime)
    % Already applied the probabilities from interpolating onto grid
    EV=squeeze(sum((EV.*pi_u),2)); % (d23 & a1prime,zprime)

    % Time to refine EV, we can refine out d2
    [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3*N_a1,1,N_z,1]),[],1);

    DiscountedEV_onlyd3=DiscountFactorParamsVec*shiftdim(EV_onlyd3,1);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2e(ReturnFn, [n_d3,n_a1], [n_a1,n_a2], n_z, n_e, d3a1_gridvals, a1a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec);
        % (d,aprime,a,z,e)
        
        % Time to refine ReturnMatrix, we can refine out d1
        % no d1 here

        % Now put together entireRHS, which just depends on d3
        entireRHS=ReturnMatrix+DiscountedEV_onlyd3;        
        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        
        V(:,:,:,jj)=shiftdim(Vtemp,1);
        Policy3(2,:,:,:,jj)=shiftdim(rem(maxindex-1,N_d3)+1,1);
        Policy3(3,:,:,:,jj)=shiftdim(ceil(maxindex/N_d3),-1);
        Policy3(1,:,:,:,jj)=shiftdim(d2index(maxindex+N_d3*zind),1);

    elseif vfoptions.lowmemory==1


        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_Par2e(ReturnFn, [n_d3,n_a1], [n_a1,n_a2], n_z, special_n_e, d3a1_gridvals, a1a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec);
            % (d,aprime,a,z)
            
            % Time to refine ReturnMatrix, we can refine out d1
            % no d1 here

            % Now put together entireRHS, which just depends on d3
            entireRHS_e=ReturnMatrix_e+DiscountedEV_onlyd3;            
            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            
            V(:,:,e_c,jj)=shiftdim(Vtemp,1);
            Policy3(2,:,:,e_c,jj)=shiftdim(rem(maxindex-1,N_d3)+1,1);
            Policy3(3,:,:,e_c,jj)=shiftdim(ceil(maxindex/N_d3),-1);
            Policy3(1,:,:,e_c,jj)=shiftdim(d2index(maxindex+N_d3*zind),1);
        end
        
    elseif vfoptions.lowmemory==2

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_onlyd3_z=DiscountedEV_onlyd3(:,1,z_c);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                
                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_Par2e(ReturnFn, [n_d3,n_a1], [n_a1,n_a2], special_n_z, special_n_e, d3a1_gridvals, a1a2_gridvals, z_val, e_val, ReturnFnParamsVec);
                
                % Time to refine ReturnMatrix, we can refine out d1
                % no d1 here

                % Now put together entireRHS, which just depends on d3
                entireRHS_ze=ReturnMatrix_ze+DiscountedEV_onlyd3_z;
                
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_ze,[],1);                

                V(:,z_c,e_c,jj)=Vtemp;
                Policy3(2,:,z_c,e_c,jj)=shiftdim(rem(maxindex-1,N_d3)+1,1);
                Policy3(3,:,z_c,e_c,jj)=shiftdim(ceil(maxindex/N_d3),-1);
                Policy3(1,:,z_c,e_c,jj)=shiftdim(d2index(maxindex),1);
            end
        end
    end
end

Policy=Policy3(1,:,:,:,:)+N_d2*(Policy3(2,:,:,:,:)-1)+N_d2*N_d3*(Policy3(3,:,:,:,:)-1); % d2, d3, a1prime


end
