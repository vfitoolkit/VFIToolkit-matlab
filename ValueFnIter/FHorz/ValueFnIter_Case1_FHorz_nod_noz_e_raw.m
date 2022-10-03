function  [V,Policy]=ValueFnIter_Case1_FHorz_nod_noz_e_raw(n_a, n_e, N_j, a_grid, e_grid, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
% Note: have no z variable, do have e variables

N_a=prod(n_a);
N_e=prod(n_e);

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_e,N_j,'gpuArray'); % no d variable

if ~isfield(vfoptions,'parallel_e')
    vfoptions.parallel_e=zeros(1,length(n_e));
    % Parallel e can have some elements (starting from the front end) equal to 1. I will parallelize over these.
end

%%
a_grid=gpuArray(a_grid);
e_grid=gpuArray(e_grid);

% Z markov
eval('fieldexists_ExogShockFn=1;vfoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;vfoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')
eval('fieldexists_pi_z_J=1;vfoptions.pi_z_J;','fieldexists_pi_z_J=0;')
% E iid
eval('fieldexists_EiidShockFn=1;vfoptions.EiidShockFn;','fieldexists_EiidShockFn=0;')
eval('fieldexists_EiidShockFnParamNames=1;vfoptions.EiidShockFnParamNames;','fieldexists_EiidShockFnParamNames=0;')
eval('fieldexists_pi_e_J=1;vfoptions.pi_e_J;','fieldexists_pi_e_J=0;')


%%
if all(vfoptions.parallel_e==0)
    %% Loop over all e
    special_n_e=ones(1,length(n_e));

    %% N_j
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);
    
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
            e_grid=gpuArray(e_grid); pi_z=gpuArray(pi_e);
        end
    end
    if all(size(e_grid)==[sum(n_e),1]) % kronecker (cross-product) grid
        e_gridvals=CreateGridvals(n_e,e_grid,1); % The 1 at end indicates want output in form of matrix.
    elseif all(size(e_grid)==[prod(n_e),length(n_e)]) % joint-grid
        e_gridvals=e_grid;
    end


    pi_e=shiftdim(pi_e,-1); % Move to second dimensionfor e_c=1:n_e (normally -2, but no z so -1)
    
    for e_c=1:N_e
        e_val=e_gridvals(e_c,:);
        ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_e, 0, a_grid, e_val, ReturnFnParamsVec); % Because no z, can treat e like z and call Par2 rather than Par2e
        % Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
        V(:,e_c,N_j)=Vtemp;
        Policy(:,e_c,N_j)=maxindex;
    end

    %% Loop backward over age
    for reverse_j=1:N_j-1
        jj=N_j-reverse_j;

        if vfoptions.verbose==1
            fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
        end

        % Create a vector containing all the return function parameters (in order)
        ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
        DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

        if fieldexists_pi_e_J==1
            e_grid=vfoptions.e_grid_J(:,jj);
            pi_e=vfoptions.pi_e_J(:,jj);
            pi_e=shiftdim(pi_e,-2); % Move to thrid dimension
            if all(size(e_grid)==[sum(n_e),1]) % kronecker (cross-product) grid
                e_gridvals=CreateGridvals(n_e,e_grid,1); % The 1 at end indicates want output in form of matrix.
            elseif all(size(e_grid)==[prod(n_e),length(n_e)]) % joint-grid
                e_gridvals=e_grid;
            end
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
                [e_grid,pi_e]=vfoptions.EiidShockFn(jj);
                e_grid=gpuArray(e_grid); pi_e=gpuArray(pi_e);
            end
            pi_e=shiftdim(pi_e,-1); % Move to second dimensionfor e_c=1:n_e (normally -2, but no z so -1)
            if all(size(e_grid)==[sum(n_e),1]) % kronecker (cross-product) grid
                e_gridvals=CreateGridvals(n_e,e_grid,1); % The 1 at end indicates want output in form of matrix.
            elseif all(size(e_grid)==[prod(n_e),length(n_e)]) % joint-grid
                e_gridvals=e_grid;
            end
        end

        VKronNext_j=V(:,:,jj+1);

        VKronNext_j=sum(VKronNext_j.*pi_e,2);

        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_e, 0, a_grid, e_val, ReturnFnParamsVec);

            EV_e=VKronNext_j; % Note, no z
            entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*EV_e.*ones(1,N_a,1);

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);

            V(:,e_c,jj)=shiftdim(Vtemp,1);
            Policy(:,e_c,jj)=shiftdim(maxindex,1);
        end
    end

elseif all(vfoptions.parallel_e==1)
    % Parallel over all e

    %% N_j
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);
    
    if fieldexists_pi_e_J==1
        e_grid=vfoptions.e_grid_J(:,N_j);
        pi_e=vfoptions.pi_e_J(:,N_j);
        if all(size(e_grid)==[sum(n_e),1]) % kronecker (cross-product) grid
            e_gridvals=CreateGridvals(n_e,e_grid,1); % The 1 at end indicates want output in form of matrix.
        elseif all(size(e_grid)==[prod(n_e),length(n_e)]) % joint-grid
            e_gridvals=e_grid;
        end
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
            e_grid=gpuArray(e_grid); pi_z=gpuArray(pi_e);
        end
        if all(size(e_grid)==[sum(n_e),1]) % kronecker (cross-product) grid
            e_gridvals=CreateGridvals(n_e,e_grid,1); % The 1 at end indicates want output in form of matrix.
        elseif all(size(e_grid)==[prod(n_e),length(n_e)]) % joint-grid
            e_gridvals=e_grid;
        end
    end

    pi_e=shiftdim(pi_e,-1); % Move to second dimensionfor e_c=1:n_e (normally -2, but no z so -1)
    
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_e, 0, a_grid, e_grid, ReturnFnParamsVec); % Because no z, can treat e like z and call Par2 rather than Par2e
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,:,N_j)=Vtemp;
    Policy(:,:,N_j)=maxindex;

    %% Loop backward over age
    for reverse_j=1:N_j-1
        jj=N_j-reverse_j;

        if vfoptions.verbose==1
            fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
        end

        % Create a vector containing all the return function parameters (in order)
        ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
        DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

        if fieldexists_pi_e_J==1
            e_grid=vfoptions.e_grid_J(:,jj);
            pi_e=vfoptions.pi_e_J(:,jj);
            pi_e=shiftdim(pi_e,-2); % Move to third dimension
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
                [e_grid,pi_e]=vfoptions.EiidShockFn(jj);
                e_grid=gpuArray(e_grid); pi_e=gpuArray(pi_e);
            end
            pi_e=shiftdim(pi_e,-1); % Move to second dimensionfor e_c=1:n_e (normally -2, but no z so -1)
        end

        VKronNext_j=V(:,:,jj+1);

        VKronNext_j=sum(VKronNext_j.*pi_e,2);

        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_e, 0, a_grid, e_grid, ReturnFnParamsVec);
        EV=VKronNext_j; % Note, no z

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV.*ones(1,N_a,N_e);
        
        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        
        V(:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,jj)=shiftdim(maxindex,1);
    end
    
else
    % Split e into e1 which I parallel over, and e2 which I loop over
    n_e1=n_e(logical(vfoptions.parallel_e));
    n_e2=n_e(~logical(vfoptions.parallel_e));
    N_e1=prod(n_e1);
    N_e2=prod(n_e2);
    if fieldexists_pi_e_J==1
        e_grid=vfoptions.e_grid_J;
        pi_e=vfoptions.pi_e;
    elseif fieldexists_EiidShockFn==1
        error('Cannot use EiidShockFn together with vfoptions.parallel_e (if this functionality is important to you please contact me and I can implement)')
    end
    e1_grid=e_grid(1:sum(n_e1),:); % Note, allows for dependence on age j
    e2_grid=e_grid(sum(n_e1)+1:end,:); % Note, allows for dependence on age j
    if size(pi_e,2)==1
        temp=reshape(pi_e,[N_e1,N_e2]);
        pi_e1=sum(temp,2); % Assumes that e1 and e2 are uncorrelated/independently distributed
        pi_e2=sum(temp,1)'; % Assumes that e1 and e2 are uncorrelated/independently distributed
    else
        % Same but looping over age j
        pi_e1=zeros(N_e1,N_J);
        pi_e2=zeros(N_e2,N_J);
        for jj=1:N_j
            temp=reshape(pi_e(:,jj),[N_e1,N_e2]);
            pi_e1(:,jj)=sum(temp,2); % Assumes that e1 and e2 are uncorrelated/independently distributed
            pi_e2(:,jj)=sum(temp,1)'; % Assumes that e1 and e2 are uncorrelated/independently distributed
        end
        vfoptions.e1_grid_J=e1_grid;
        vfoptions.pi_e1=pi_e1;
        vfoptions.e2_grid_J=e2_grid;
        vfoptions.pi_e2=pi_e2;
    end

    % Need to be a different size when do a mix of parallel and loop for e
    V=zeros(N_a,N_e1,N_e2,N_j,'gpuArray');
    Policy=zeros(N_a,N_e1,N_e2,N_j,'gpuArray');
    
    %% j=N_j

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

    if fieldexists_pi_e_J==1
        e1_grid=vfoptions.e1_grid_J(:,N_j);
        pi_e1=vfoptions.pi_e1_J(:,N_j);
        e2_grid=vfoptions.e2_grid_J(:,N_j);
        pi_e2=vfoptions.pi_e2_J(:,N_j);
    end

    % Now loop over e2, and within that parallelize over e1
    special_n_e2=ones(1,length(n_e2));
    if all(size(e2_grid)==[sum(n_e2),1]) % kronecker (cross-product) grid
        e2_gridvals=CreateGridvals(n_e2,e2_grid,1); % The 1 at end indicates want output in form of matrix.
    elseif all(size(e2_grid)==[prod(n_e2),length(n_e2)]) % joint-grid
        e2_gridvals=e2_grid;
    end
    pi_e1_prime=pi_e1'; % Move to second dimension
    pi_e2=shiftdim(pi_e2,-2); % Move to thrid dimension

    for e2_c=1:N_e2
        e2_val=e2_gridvals(e2_c,:);
        ReturnMatrix_e2=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, 0, n_a, n_e1, special_n_e2, 0, a_grid, e1_grid, e2_val, ReturnFnParamsVec); % Just treat e1 like z and e2 like e
        % Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix_e2,[],1);
        V(:,:,e2_c,N_j)=Vtemp;
        Policy(:,:,e2_c,N_j)=maxindex;
    end

    %% Iterate backwards through j.
    for reverse_j=1:N_j-1
        jj=N_j-reverse_j;

        if vfoptions.verbose==1
            fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
        end


        % Create a vector containing all the return function parameters (in order)
        ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
        DiscountFactorParamsVec=prod(DiscountFactorParamsVec);


        if fieldexists_pi_e_J==1
            e1_grid=vfoptions.e1_grid_J(:,jj);
            pi_e1=vfoptions.pi_e1_J(:,jj);
            e2_grid=vfoptions.e2_grid_J(:,jj);
            pi_e1_prime=pi_e1'; % Move to thrid dimension
            pi_e2=vfoptions.pi_e2_J(:,jj);
            pi_e2=shiftdim(pi_e2,-2); % Move to thrid dimension

            if all(size(e2_grid)==[sum(n_e2),1]) % kronecker (cross-product) grid
                e2_gridvals=CreateGridvals(n_e2,e2_grid,1); % The 1 at end indicates want output in form of matrix.
            elseif all(size(e2_grid)==[prod(n_e2),length(n_e2)]) % joint-grid
                e2_gridvals=e2_grid;
            end
        end

        VKronNext_j=V(:,:,:,jj+1);

        VKronNext_j=sum(VKronNext_j.*pi_e2,3);

        for e2_c=1:N_e2
            e2_val=e2_gridvals(e2_c,:);
            ReturnMatrix_e2=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, 0, n_a, n_e1, special_n_e2, 0, a_grid, e1_grid, e2_val, ReturnFnParamsVec); % Just treat e1 as z and e2 as e

            EV_e2=VKronNext_j.*pi_e1_prime;
            EV_e2(isnan(EV_e2))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_e2=sum(EV_e2,2); % sum over z', leaving a singular second dimension

            entireRHS_e2=ReturnMatrix_e2+DiscountFactorParamsVec*EV_e2.*ones(1,N_a,1);

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e2,[],1);

            V(:,:,e2_c,jj)=shiftdim(Vtemp,1);
            Policy(:,:,e2_c,jj)=shiftdim(maxindex,1);
        end
        
    end

    V=reshape(V,[N_a,N_e,N_j]);
    Policy=reshape(Policy,[N_a,N_e,N_j]);

end