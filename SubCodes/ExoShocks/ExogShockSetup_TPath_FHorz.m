function [z_gridvals_J, pi_z_J, pi_z_J_sim, e_gridvals_J, pi_e_J, pi_e_J_sim, ze_gridvals_J_fastOLG, transpathoptions, options]=ExogShockSetup_TPath_FHorz(n_z,z_grid,pi_z,N_a,N_j,Parameters,PricePathNames,ParamPathNames,transpathoptions,options,gridpiboth)
% Convert z and e to age-dependent joint-grids and transtion matrix
% Can input vfoptions OR simoptions
% output: z_gridvals_J, pi_z_J, e_gridvals_J, pi_e_J, transpathoptions, options
% pi_z_J_sim and pi_e_J_sim are for simoptions.fastOLG=1 (otherwise omitted)
% ze_gridvals_J_fastOLG is for the fastOLG version of AggVars

% Sets up
% transpathoptions.zpathtrivial=1; % z_gridvals_J and pi_z_J are not varying over the path
%                              =0; % they vary over path, so z_gridvals_J_T and pi_z_J_T
% transpathoptions.epathtrivial=1; % e_gridvals_J and pi_e_J are not varying over the path
%                              =0; % they vary over path, so e_gridvals_J_T and pi_e_J_T
% and
% transpathoptions.gridsinGE=1; % grids depend on a GE parameter and so need to be recomputed every iteration
%                           =0; % grids are exogenous
%
% transpathoptions.zepathtrivial=0 when either of zpathtrival and epathtrivial both are zero

% gridpiboth=4: sometimes (trans path GE) we want both grid and transition probabilties, including pi_z_J_sim alternative transition probs
% gridpiboth=3: sometimes (value fn iter) we want both grid and transition probabilties
% gridpiboth=2: sometimes (agent dist)    we want just transition probabilties, including pi_z_J_sim alternative transition probs
% gridpiboth=1: sometimes (FnsToEvaluate) we want just grid


%% Check basic setup
N_z=prod(n_z);

if ~isfield(options,'n_e')
    n_e=0;
else
    n_e=options.n_e;
    options=rmfield(options,'n_e');
end
N_e=prod(n_e);

transpathoptions.gridsinGE=0; % will be overwritten if appropriate
transpathoptions.zpathtrivial=1;  % will be overwritten if appropriate
transpathoptions.epathtrivial=1;  % will be overwritten if appropriate

%% Check if z_grid and/or pi_z depend on prices. If not then create pi_z_J and z_grid_J for the entire transition before we start
% If 'exogenous shock fn' is used, then precompute it to save evaluating it numerous times
% Check if using 'exogenous shock fn' (exogenous state has a grid and transition matrix that depends on age)

z_gridvals_J=[];
pi_z_J=[];
pi_z_J_sim=[];
l_z=0;
if N_z>0
    l_z=length(n_z);

    if isfield(options,'ExogShockFn') % Just calculate grid and transition probabilities anyway
        options.ExogShockFnParamNames=getAnonymousFnInputNames(options.ExogShockFn);
        % First, check if ExogShockFn depends on a PricePath parameter
        overlap=0;
        for ii=1:length(options.ExogShockFnParamNames)
            if any(strcmp(options.ExogShockFnParamNames{ii},PricePathNames))
                overlap=1;
            end
        end
        if overlap==1
            transpathoptions.gridsinGE=1;
            transpathoptions.zpathtrivial=0; % z_grid_J and pi_z_J vary over the path
            error('Not yet implemented to use ExogShockFn which includes parameters from PricePath (contact me)')
        else % overlap==0
            % Next,check if ExogShockFn depends on a ParamPath parameter
            overlap2=0;
            for ii=1:length(options.ExogShockFnParamNames)
                if strcmp(options.ExogShockFnParamNames{ii},ParamPathNames)
                    overlap2=1;
                end
            end
            if overlap2==0
                pi_z_J=zeros(N_z,N_z,N_j,'gpuArray');
                z_grid_J=zeros(N_z,N_j,'gpuArray');
                for jj=1:N_j
                    ExogShockFnParamsVec=CreateVectorFromParams(Parameters, options.ExogShockFnParamNames,jj);
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for ii=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                    end
                    [z_grid,pi_z]=options.ExogShockFn(ExogShockFnParamsCell{:});
                    pi_z_J(:,:,jj)=gpuArray(pi_z);
                    z_grid_J(:,jj)=gpuArray(z_grid);
                end
            elseif overlap2==1 % ExogShockFn depends on a ParamPath parameter
                transpathoptions.zpathtrivial=0; % z_grid_J and pi_z_J vary over the path
                transpathoptions.pi_z_J_T=zeros(N_z,N_z,N_j,T,'gpuArray');
                transpathoptions.z_grid_J_T=zeros(sum(n_z),N_j,T,'gpuArray');
                pi_z_J=zeros(N_z,N_z,N_j,'gpuArray');
                z_grid_J=zeros(sum(n_z),N_j,'gpuArray');
                for tt=1:T
                    for ii=1:length(ParamPathNames)
                        Parameters.(ParamPathNames{ii})=ParamPathStruct.(ParamPathNames{ii});
                    end
                    % Note, we know the PricePath is irrelevant for the current purpose
                    for jj=1:N_j
                        ExogShockFnParamsVec=CreateVectorFromParams(Parameters, options.ExogShockFnParamNames,jj);
                        ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                        for ii=1:length(ExogShockFnParamsVec)
                            ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                        end
                        [z_grid,pi_z]=options.ExogShockFn(ExogShockFnParamsCell{:});
                        pi_z_J(:,:,jj)=gpuArray(pi_z);
                        z_grid_J(:,jj)=gpuArray(z_grid);
                    end
                    transpathoptions.pi_z_J_T(:,:,:,tt)=pi_z_J;
                    transpathoptions.z_grid_J_T(:,:,tt)=z_grid_J;
                end
            end
        end

    else % Not ExogShockFn, or at least not any more
        if gridpiboth==1 % for most FnsToEvaluate, we don't use pi_z
            pi_z_J=[];
            % Now just do z_gridvals_J
            z_gridvals_J=zeros(prod(n_z),length(n_z),'gpuArray');
            if ndims(z_grid)==3 % already an age-dependent joint-grid
                if all(size(z_grid)==[prod(n_z),length(n_z),N_j])
                    z_gridvals_J=z_grid;
                end
            elseif all(size(z_grid)==[sum(n_z),N_j]) % age-dependent grid
                for jj=1:N_j
                    z_gridvals_J(:,:,jj)=CreateGridvals(n_z,z_grid(:,jj),1);
                end
            elseif all(size(z_grid)==[prod(n_z),length(n_z)]) % joint grid
                z_gridvals_J=z_grid.*ones(1,1,N_j,'gpuArray');
            elseif all(size(z_grid)==[sum(n_z),1]) % basic grid
                z_gridvals_J=CreateGridvals(n_z,z_grid,1).*ones(1,1,N_j,'gpuArray');
            end
        elseif gridpiboth==2 % For agent dist, we don't use grid
            z_gridvals_J=[];
            % Now just do pi_z_J
            if ndims(pi_z)==3
                if all(size(pi_z)==[N_z,N_z,N_j]) % age-dependent grid
                    pi_z_J=pi_z;
                end
            elseif all(size(pi_z)==[N_z,N_z])
                pi_z_J=pi_z.*ones(1,1,N_j,'gpuArray');
            end
        elseif gridpiboth==3 || gridpiboth==4 % For value fn, both z_gridvals_J and pi_z_J
            z_gridvals_J=zeros(prod(n_z),length(n_z),'gpuArray');
            pi_z_J=zeros(prod(n_z),prod(n_z),'gpuArray');
            if ndims(z_grid)==3 % already an age-dependent joint-grid
                if all(size(z_grid)==[prod(n_z),length(n_z),N_j])
                    z_gridvals_J=z_grid;
                end
                pi_z_J=pi_z;
            elseif all(size(z_grid)==[sum(n_z),N_j]) % age-dependent grid
                for jj=1:N_j
                    z_gridvals_J(:,:,jj)=CreateGridvals(n_z,z_grid(:,jj),1);
                end
                pi_z_J=pi_z;
            elseif all(size(z_grid)==[prod(n_z),length(n_z)]) % joint grid
                z_gridvals_J=z_grid.*ones(1,1,N_j,'gpuArray');
                pi_z_J=pi_z.*ones(1,1,N_j,'gpuArray');
            elseif all(size(z_grid)==[sum(n_z),1]) % basic grid
                z_gridvals_J=CreateGridvals(n_z,z_grid,1).*ones(1,1,N_j,'gpuArray');
                pi_z_J=pi_z.*ones(1,1,N_j,'gpuArray');
            end
        end
    end
    % Make sure they are on grid
    pi_z_J=gpuArray(pi_z_J);
    z_gridvals_J=gpuArray(z_gridvals_J);

    
    % When using fastOLG we want an alternative version of pi_z_J that we use for the agent distribution, call it pi_z_J_sim
    if gridpiboth==2 || gridpiboth==4
        if options.fastOLG==1
            % pi_z_J is currently (z,z',j) [When using options.fastOLG=1 it will get converted later to (j,z',z)]
            % We want pi_z_J_sim to map (j,z)-to-z', specifically [(N_j-1)*N_z,N_z] 
            pi_z_J_sim=gather(reshape(permute(pi_z_J(:,:,1:N_j-1),[3,1,2]),[(N_j-1)*N_z,N_z]));
            % Now extend it to map (j,z)-to-(j+1,z')
            II1=repmat(1:1:(N_j-1)*N_z,1,N_z);
            II2=repmat(1:1:(N_j-1),1,N_z*N_z)+repelem((N_j-1)*(0:1:N_z-1),1,N_z*(N_j-1));
            pi_z_J_sim=sparse(II1,II2,pi_z_J_sim,(N_j-1)*N_z,(N_j-1)*N_z);
        end
    end

    if transpathoptions.fastOLG==1 % Reshape grid and transtion matrix for use with fastOLG
        if gridpiboth==1 % for most FnsToEvaluate, we don't use pi_z
            z_gridvals_J=permute(z_gridvals_J,[3,1,2]); % Give it the size required for CreateReturnFnMatrix_Case1_Disc_Par2_fastOLG(): N_j-by-N_z-by-l_z
            if transpathoptions.zpathtrivial==0
                temp=transpathoptions.z_grid_J_T;
                transpathoptions=rmfield(transpathoptions,'z_grid_J_T');
                transpathoptions.z_gridvals_J_T=zeros(N_z,l_z,N_j,T,'gpuArray');
                for tt=1:T
                    for jj=1:N_j
                        z_gridvals_J(:,:,jj,tt)=CreateGridvals(n_z,temp(:,jj,tt),1);
                    end
                end
                transpathoptions.z_gridvals_J_T=permute(transpathoptions.z_gridvals_J_T,[3,1,2,4]); % from [N_z,l_z,N_j,T] to [N_j,N_z,l_z,T]
            end
        elseif gridpiboth==2 % For agent dist, we don't use grid
            pi_z_J=permute(pi_z_J,[3,2,1]); % We want it to be (j,z',z) for value function
            transpathoptions.pi_z_J_alt=permute(pi_z_J,[1,3,2]); % But is (j,z,z') for agent dist with fastOLG [note, this permute is off the previous one]
            if transpathoptions.zpathtrivial==0
                transpathoptions.pi_z_J_T=permute(transpathoptions.pi_z_J_T,[3,1,2,4]);  % We want it to be (j,z,z',t)
                transpathoptions.pi_z_J_T_alt=permute(transpathoptions.pi_z_J_T,[1,3,2,4]);  % We want it to be (j,z',z,t) [note, this permute is off the previous one]
            end
        elseif gridpiboth==3 || gridpiboth==4 % For value fn, both z_gridvals_J and pi_z_J
            z_gridvals_J=permute(z_gridvals_J,[3,1,2]); % Give it the size required for CreateReturnFnMatrix_Case1_Disc_Par2_fastOLG(): N_j-by-N_z-by-l_z
            pi_z_J=permute(pi_z_J,[3,2,1]); % We want it to be (j,z',z) for value function
            transpathoptions.pi_z_J_alt=permute(pi_z_J,[1,3,2]); % But is (j,z,z') for agent dist with fastOLG [note, this permute is off the previous one]
            if transpathoptions.zpathtrivial==0
                temp=transpathoptions.z_grid_J_T;
                transpathoptions=rmfield(transpathoptions,'z_grid_J_T');
                transpathoptions.z_gridvals_J_T=zeros(N_z,l_z,N_j,T,'gpuArray');
                for tt=1:T
                    for jj=1:N_j
                        z_gridvals_J(:,:,jj,tt)=CreateGridvals(n_z,temp(:,jj,tt),1);
                    end
                end
                transpathoptions.z_gridvals_J_T=permute(transpathoptions.z_gridvals_J_T,[3,1,2,4]); % from [N_z,l_z,N_j,T] to [N_j,N_z,l_z,T]
                transpathoptions.pi_z_J_T=permute(transpathoptions.pi_z_J_T,[3,1,2,4]);  % We want it to be (j,z,z',t)
                transpathoptions.pi_z_J_T_alt=permute(transpathoptions.pi_z_J_T,[1,3,2,4]);  % We want it to be (j,z',z,t) [note, this permute is off the previous one]
            end
        end
    end

    z_gridvals_J=gpuArray(z_gridvals_J);
    pi_z_J=gpuArray(pi_z_J);
    % z_gridvals_J is [N_z,l_z,N_j] if transpathoptions.fastOLG=0
    %              is [N_j,N_z,l_z] if transpathoptions.fastOLG=1
    % pi_z_J is [N_z,N_z,N_j]       if transpathoptions.fastOLG=0 (j,z,z')
    % pi_z_J is [N_j,N_z,N_z]       if transpathoptions.fastOLG=1 (j,z',z)
    % pi_z_J and z_gridvals_J are both gpuArrays
end

%% If using e variables do the same for e as we just did for z
e_gridvals_J=[];
pi_e_J=[];
pi_e_J_sim=[];
l_e=0;
if N_e>0
    l_e=length(n_e);
    % Check if e_grid and/or pi_e depend on prices. If not then create pi_e_J and e_grid_J for the entire transition before we start

    transpathoptions.epathprecomputed=0;

    % Just calculate grid and transition probabilities anyway
    if isfield(options,'EiidShockFn')
        options.EiidShockFnParamNames=getAnonymousFnInputNames(options.EiidShockFn);
        % First, check if EiidShockFn depends on a PricePath parameter
        overlap=0;
        for ii=1:length(options.EiidShockFnParamNames)
            if strcmp(options.EiidShockFnParamNames{ii},PricePathNames)
                overlap=1;
            end
        end
        if overlap==1
            transpathoptions.gridsinGE=1;
            transpathoptions.epathtrivial=0; % e_grid_J and pi_e_J vary over the path
            error('Not yet implemented to use EiidShockFn which includes parameters from PricePath (contact me)')
        else % overlap==0
            % Next,check if EiidShockFn depends on a ParamPath parameter
            overlap2=0;
            for ii=1:length(options.EiidShockFnParamNames)
                if strcmp(options.EiidShockFnParamNames{ii},ParamPathNames)
                    overlap2=1;
                end
            end
            if overlap2==0
                pi_e_J=zeros(N_e,N_e,N_j,'gpuArray');
                e_grid_J=zeros(N_e,N_j,'gpuArray');
                for jj=1:N_j
                    EiidShockFnParamsVec=CreateVectorFromParams(Parameters, options.EiidShockFnParamNames,jj);
                    EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                    for ii=1:length(EiidShockFnParamsVec)
                        EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                    end
                    [e_grid,pi_e]=options.EiidShockFn(EiidShockFnParamsCell{:});
                    pi_e_J(:,jj)=gpuArray(pi_e);
                    e_grid_J(:,jj)=gpuArray(e_grid);
                end
            elseif overlap2==1 % ExogShockFn depends on a ParamPath parameter
                transpathoptions.epathtrivial=0; % e_grid_J and pi_e_J vary over the path
                transpathoptions.pi_e_J_T=zeros(N_e,N_e,N_j,T,'gpuArray');
                transpathoptions.e_grid_J_T=zeros(sum(n_e),N_j,T,'gpuArray');
                pi_e_J=zeros(N_e,N_e,N_j,'gpuArray');
                e_grid_J=zeros(sum(n_e),N_j,'gpuArray');
                for tt=1:T
                    for ii=1:length(ParamPathNames)
                        Parameters.(ParamPathNames{ii})=ParamPathStruct.(ParamPathNames{ii});
                    end
                    % Note, we know the PricePath is irrelevant for the current purpose
                    for jj=1:N_j
                        EiidShockFnParamsVec=CreateVectorFromParams(Parameters, options.EiidShockFnParamNames,jj);
                        EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                        for ii=1:length(ExogShockFnParamsVec)
                            EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                        end
                        [e_grid,pi_e]=options.ExogShockFn(EiidShockFnParamsCell{:});
                        pi_e_J(:,jj)=gpuArray(pi_e);
                        e_grid_J(:,jj)=gpuArray(e_grid);
                    end
                    transpathoptions.pi_e_J_T(:,:,tt)=pi_e_J;
                    transpathoptions.e_grid_J_T(:,:,tt)=e_grid_J;
                end
            end
        end

    else % Not ExogShockFn, or at least not any more
        if gridpiboth==1 % for most FnsToEvaluate, we don't use pi_z
            if isfield(options,'e_grid')
                if ndims(options.e_grid)==3 % already age-dependent gridvals
                    e_gridvals_J=gpuArray(options.e_grid);
                elseif ndims(options.e_grid)==2
                    % Could be be age-dependent e_grid_J or e_gridvals
                    if size(options.e_grid,2)==N_j
                        if l_e==1
                            e_gridvals_J=reshape(options.e_grid,[N_e,l_e,N_j]); % Avoid Matlab getting annoyed about CreateGridVals() for no apparent reason
                        else
                            e_gridvals_J=zeros(N_e,l_e,N_j,'gpuArray');
                            for jj=1:N_j
                                e_gridvals_J(:,:,jj)=CreateGridVals(n_e,options.e_grid(:,jj),1);
                            end
                        end
                    else
                        e_gridvals_J=options.e_grid.*ones(1,1,N_j,'gpuArray');
                    end
                end
                options=rmfield(options,'e_grid');
            end
        elseif gridpiboth==2 % For agent dist, we don't use grid
            if isfield(options,'pi_e')
                pi_e_J=options.pi_e.*ones(1,N_j,'gpuArray'); % this works regardless of if pi_e depends on j or not
                options=rmfield(options,'pi_e');
            end
        elseif gridpiboth==3 || gridpiboth==4 % For value fn, both z_gridvals_J and pi_z_J
            if isfield(options,'pi_e')
                % Just assume there is also options.e_grid
                if ndims(options.e_grid)==3 % already age-dependent gridvals
                    e_gridvals_J=gpuArray(options.e_grid);
                elseif ndims(options.e_grid)==2
                    % Could be be age-dependent e_grid_J or e_gridvals
                    if size(options.e_grid,2)==N_j
                        if l_e==1
                            e_gridvals_J=reshape(options.e_grid,[N_e,l_e,N_j]); % Avoid Matlab getting annoyed about CreateGridVals() for no apparent reason
                        else
                            e_gridvals_J=zeros(N_e,l_e,N_j,'gpuArray');
                            for jj=1:N_j
                                e_gridvals_J(:,:,jj)=CreateGridVals(n_e,options.e_grid(:,jj),1);
                            end
                        end
                    else
                        e_gridvals_J=options.e_grid.*ones(1,1,N_j,'gpuArray');
                    end
                end
                options=rmfield(options,'e_grid');
                pi_e_J=options.pi_e.*ones(1,N_j,'gpuArray');  % this works regardless of if pi_e depends on j or not
                options=rmfield(options,'pi_e');
            end
        end
    end
    % Make sure they are on grid
    pi_e_J=gpuArray(pi_e_J);
    e_gridvals_J=gpuArray(e_gridvals_J);
    
    % When using fastOLG we want an alternative version of pi_e_J that we use for the agent distribution, call it pi_e_J_sim
    if gridpiboth==2 || gridpiboth==4
        if options.fastOLG==1
            % pi_e_J is current (e',j)  [When using options.fastOLG=1 it will get converted later to (j,e')]
            if N_z==0
                % pi_e_J_sim maps (a,j)-to-e (but only for jj=2:end), is [N_a*(N_j-1),N_e]
                pi_e_J_sim=repelem(pi_e_J(:,2:end)',N_a,1); % (a,j)-by-e (but only for jj=2:end)
            else
                % pi_e_J_sim maps (a,j,z)-to-e (but only for jj=2:end), is [N_a*(N_j-1)*N_z,N_e]
                pi_e_J_sim=repmat(repelem(pi_e_J(:,2:end)',N_a,1),N_z,1); % (a,j,z)-by-e (but only for jj=2:end)
            end
        end
    end
    
    if transpathoptions.fastOLG==1 % Reshape grid and transtion matrix for use with fastOLG
        if N_z>0
            if gridpiboth==1 % for most FnsToEvaluate, we don't use pi_e
                e_gridvals_J=permute(e_gridvals_J,[3,4,1,2]); % Give it the size required for CreateReturnFnMatrix_Case1_Disc_Par2_fastOLGe: (j,1,N_e,l_e)
                % transpathoptions.pi_e_J_alt=permute(pi_e_J,[1,3,2]); % But is (j,z,z') for agent dist with fastOLG [note, this permute is off the previous one]
                if transpathoptions.epathtrivial==0
                    temp=transpathoptions.e_grid_J_T;
                    transpathoptions=rmfield(transpathoptions,'e_grid_J_T');
                    transpathoptions.e_gridvals_J_T=zeros(N_e,l_e,N_j,T,'gpuArray');
                    for tt=1:T
                        for jj=1:N_j
                            e_gridvals_J(:,:,jj,tt)=CreateGridvals(n_e,temp(:,jj,tt),1);
                        end
                    end
                    transpathoptions.e_gridvals_J_T=permute(transpathoptions.e_gridvals_J_T,[3,5,1,2,4]); % from (e,j,t) to (j,e,t) [second dimension is singular, this is how I want it for fastOLG value fn where first dim is j, then second is z (which is not relevant to e)]
                end
            elseif gridpiboth==2 % For agent dist, we don't use grid
                pi_e_J=reshape(kron(pi_e_J,ones(N_a,1,'gpuArray'))',[N_a*N_j,1,N_e]); % Give it the size required for fastOLG value function
                % transpathoptions.pi_e_J_alt=permute(pi_e_J,[1,3,2]); % But is (j,z,z') for agent dist with fastOLG [note, this permute is off the previous one]
                if transpathoptions.epathtrivial==0
                    transpathoptions.pi_e_J_T=repelem(permute(transpathoptions.pi_e_J_T,[3,1,2,4]),N_a,1,1,1);  % We want it to be (a-j,1,e,t)
                    transpathoptions.pi_e_J_sim_T=zeros(N_a*(N_j-1)*N_z,N_e,T,'gpuArray');
                    for tt=1:T
                        temp=reshape(transpathoptions.pi_e_J_T(:,:,:,tt),[N_a*N_j,N_e]); % transpathoptions.fastOLG means pi_e_J is [N_a*N_j,1,N_e]
                        transpathoptions.pi_e_J_sim_T(:,:,tt)=kron(ones(N_z,1,'gpuArray'),gpuArray(temp(N_a+1:end,:)));
                    end
                end
            elseif gridpiboth==3 || gridpiboth==4 % For value fn, both e_gridvals_J and pi_e_J
                e_gridvals_J=permute(e_gridvals_J,[3,4,1,2]); % Give it the size required for CreateReturnFnMatrix_Case1_Disc_Par2_fastOLGe: (j,1,N_e,l_e)
                pi_e_J=reshape(kron(pi_e_J,ones(N_a,1,'gpuArray'))',[N_a*N_j,1,N_e]); % Give it the size required for fastOLG value function
                % transpathoptions.pi_e_J_alt=permute(pi_e_J,[1,3,2]); % But is (j,z,z') for agent dist with fastOLG [note, this permute is off the previous one]
                if transpathoptions.epathtrivial==0
                    temp=transpathoptions.e_grid_J_T;
                    transpathoptions=rmfield(transpathoptions,'e_grid_J_T');
                    transpathoptions.e_gridvals_J_T=zeros(N_e,l_e,N_j,T,'gpuArray');
                    for tt=1:T
                        for jj=1:N_j
                            e_gridvals_J(:,:,jj,tt)=CreateGridvals(n_e,temp(:,jj,tt),1);
                        end
                    end
                    transpathoptions.e_gridvals_J_T=permute(transpathoptions.e_gridvals_J_T,[3,5,1,2,4]); % from (e,j,t) to (j,e,t) [second dimension is singular, this is how I want it for fastOLG value fn where first dim is j, then second is z (which is not relevant to e)]
                    transpathoptions.pi_e_J_T=repelem(permute(transpathoptions.pi_e_J_T,[3,1,2,4]),N_a,1,1,1);  % We want it to be (a-j,1,e,t)
                    transpathoptions.pi_e_J_sim_T=zeros(N_a*(N_j-1)*N_z,N_e,T,'gpuArray');
                    for tt=1:T
                        temp=reshape(transpathoptions.pi_e_J_T(:,:,:,tt),[N_a*N_j,N_e]); % transpathoptions.fastOLG means pi_e_J is [N_a*N_j,1,N_e]
                        transpathoptions.pi_e_J_sim_T(:,:,tt)=kron(ones(N_z,1,'gpuArray'),gpuArray(temp(N_a+1:end,:)));
                    end
                end
            end
        else
            if gridpiboth==1 % for most FnsToEvaluate, we don't use pi_e
                e_gridvals_J=permute(e_gridvals_J,[3,1,2]); % Give it the size required for CreateReturnFnMatrix_Case1_Disc_Par2_fastOLG: (j,N_e,l_e)
                if transpathoptions.epathtrivial==0
                    temp=transpathoptions.e_grid_J_T;
                    transpathoptions=rmfield(transpathoptions,'e_grid_J_T');
                    transpathoptions.e_gridvals_J_T=zeros(N_e,l_e,N_j,T,'gpuArray');
                    for tt=1:T
                        for jj=1:N_j
                            e_gridvals_J(:,:,jj,tt)=CreateGridvals(n_e,temp(:,jj,tt),1);
                        end
                    end
                    transpathoptions.e_gridvals_J_T=permute(transpathoptions.e_gridvals_J_T,[3,1,2,4]); % from (e,j,t) to (j,e,t)
                end
            elseif gridpiboth==2 % For agent dist, we don't use grid
                pi_e_J=reshape(kron(pi_e_J,ones(N_a,1,'gpuArray'))',[N_a*N_j,N_e]); % Give it the size required for fastOLG value function
                if transpathoptions.epathtrivial==0
                    transpathoptions.pi_e_J_T=repelem(permute(transpathoptions.pi_e_J_T,[3,1,2,4]),N_a,1,1,1);  % We want it to be (a-j,e,t)
                    transpathoptions.pi_e_J_sim_T=zeros(N_a*(N_j-1)*N_z,N_e,T,'gpuArray');
                    for tt=1:T
                        temp=reshape(transpathoptions.pi_e_J_T(:,:,:,tt),[N_a*N_j,N_e]); % transpathoptions.fastOLG means pi_e_J is [N_a*N_j,N_e]
                        transpathoptions.pi_e_J_sim_T(:,:,tt)=kron(ones(N_z,1,'gpuArray'),gpuArray(temp(N_a+1:end,:)));
                    end
                end
            elseif gridpiboth==3 || gridpiboth==4 % For value fn, both e_gridvals_J and pi_e_J
                e_gridvals_J=permute(e_gridvals_J,[3,1,2]); % Give it the size required for CreateReturnFnMatrix_Case1_Disc_Par2_fastOLG: (j,N_e,l_e)
                pi_e_J=reshape(kron(pi_e_J,ones(N_a,1,'gpuArray'))',[N_a*N_j,N_e]); % Give it the size required for fastOLG value function
                if transpathoptions.epathtrivial==0
                    temp=transpathoptions.e_grid_J_T;
                    transpathoptions=rmfield(transpathoptions,'e_grid_J_T');
                    transpathoptions.e_gridvals_J_T=zeros(N_e,l_e,N_j,T,'gpuArray');
                    for tt=1:T
                        for jj=1:N_j
                            e_gridvals_J(:,:,jj,tt)=CreateGridvals(n_e,temp(:,jj,tt),1);
                        end
                    end
                    transpathoptions.e_gridvals_J_T=permute(transpathoptions.e_gridvals_J_T,[3,1,2,4]); % from (e,j,t) to (j,e,t)
                    transpathoptions.pi_e_J_T=repelem(permute(transpathoptions.pi_e_J_T,[3,1,2,4]),N_a,1,1,1);  % We want it to be (a-j,e,t)
                    transpathoptions.pi_e_J_sim_T=zeros(N_a*(N_j-1)*N_z,N_e,T,'gpuArray');
                    for tt=1:T
                        temp=reshape(transpathoptions.pi_e_J_T(:,:,:,tt),[N_a*N_j,N_e]); % transpathoptions.fastOLG means pi_e_J is [N_a*N_j,N_e]
                        transpathoptions.pi_e_J_sim_T(:,:,tt)=kron(ones(N_z,1,'gpuArray'),gpuArray(temp(N_a+1:end,:)));
                    end
                end
            end
        end
    end

    e_gridvals_J=gpuArray(e_gridvals_J);
    pi_e_J=gpuArray(pi_e_J);
    % e_gridvals_J is [N_e,l_e,N_j]   if transpathoptions.fastOLG=0
    %              is [N_j,1,N_e,l_e] if transpathoptions.fastOLG=1 & z
    %              is [N_j,N_e,l_e] if transpathoptions.fastOLG=1 & no z
    % pi_e_J is [N_e,N_j]             if transpathoptions.fastOLG=0 (e,j)
    % pi_e_J is [N_a*N_j,1,N_e]       if transpathoptions.fastOLG=1 & z (a-j,1,e)
    % pi_e_J is [N_a*N_j,N_e]       if transpathoptions.fastOLG=1 & no z (a-j,1,e)
    % pi_e_J and e_gridvals_J are both gpuArrays

    % vfoptions.e_grid_J=e_gridvals_J;
    % vfoptions.pi_e_J=pi_e_J;
    % simoptions.e_grid_J=e_gridvals_J;
    % simoptions.pi_e_J=pi_e_J;
end





%% Create ze_gridvals_J_fastOLG, which is used for AggVars. Will be z_gridvals_J_fastOLG or e_gridvals_J_OLG if only one of them is used
% gridpiboth=4: sometimes (trans path GE) we want both grid and transition probabilties, including pi_z_J_sim alternative transition probs
% gridpiboth=3: sometimes (value fn iter) we want both grid and transition probabilties
% gridpiboth=2: sometimes (agent dist)    we want just transition probabilties, including pi_z_J_sim alternative transition probs
% gridpiboth=1: sometimes (FnsToEvaluate) we want just grid
if gridpiboth==3 || gridpiboth==2
    ze_gridvals_J_fastOLG=[];
else
    if transpathoptions.zpathtrivial==0 || transpathoptions.epathtrivial==0
        transpathoptions.zepathtrivial=0;
    else
        transpathoptions.zepathtrivial=1;
    end

    if N_e==0 && N_z==0
        ze_gridvals_J_fastOLG=[];
    else
        l_ze=l_z+l_e;
        if l_z>0 && l_e>0
            N_ze=N_z*N_e;
            if transpathoptions.fastOLG==0
                ze_gridvals_J=[repmat(z_gridvals_J,N_e,1),repelem(e_gridvals_J,N_z,1)];
            elseif transpathoptions.fastOLG==1
                ze_gridvals_J=zeros(N_j,N_ze,l_ze,'gpuArray');
                ze_gridvals_J(:,:,1:l_z)=repmat(z_gridvals_J,1,N_e,1);
                ze_gridvals_J(:,:,l_z+1:end)=repmat(squeeze(e_gridvals_J),1,N_z,1);
            end
        end

        % Version that is used for model stats, good shape for calculating AggVars fast
        if transpathoptions.fastOLG==0
            if N_z>0 && N_e>0
                ze_gridvals_J_fastOLG=shiftdim(permute(ze_gridvals_J,[3,1,2]),-1); % [1,N_j,N_ze,l_ze] need this for fastOLG agent dist (it is not used for value fn, so can overwrite)
            elseif N_z>0
                ze_gridvals_J_fastOLG=shiftdim(permute(z_gridvals_J,[3,1,2]),-1); % [1,N_j,N_z,l_z] need this for fastOLG agent dist, but need the standard still for the value fn without fastOLG
            elseif N_e>0
                ze_gridvals_J_fastOLG=shiftdim(permute(e_gridvals_J,[3,1,2]),-1); % [1,N_j,N_e,l_e] need this for fastOLG agent dist, but need the standard still for the value fn without fastOLG
            end
        else
            if N_z>0 && N_e>0
                ze_gridvals_J_fastOLG=shiftdim(ze_gridvals_J,-1); % [1,N_j,N_ze,l_ze] need this for fastOLG agent dist (it is not used for value fn, so can overwrite)
            elseif N_z>0
                ze_gridvals_J_fastOLG=shiftdim(z_gridvals_J,-1); % [1,N_j,N_z,l_z] need this for fastOLG agent dist, but need the standard still for the value fn without fastOLG
            elseif N_e>0
                ze_gridvals_J_fastOLG=shiftdim(e_gridvals_J,-1); % [1,N_j,N_e,l_e] need this for fastOLG agent dist, but need the standard still for the value fn without fastOLG
            end
        end
    end

    % If ze_gridvals_J_fastOLG depends on t, set up transpathoptions.ze_gridvals_J_T_fastOLG
    if transpathoptions.zepathtrivial==0
        if l_z>0 && l_e>0
            N_ze=N_z*N_e;
            if transpathoptions.fastOLG==0
                ze_gridvals_J_T=[repmat(transpathoptions.z_gridvals_J_T,N_e,1,1),repelem(transpathoptions.e_gridvals_J_T,N_z,1,1)];
            elseif transpathoptions.fastOLG==1
                ze_gridvals_J_T=zeros(N_j,N_ze,l_ze,'gpuArray');
                ze_gridvals_J_T(:,:,1:l_z,:)=repmat(transpathoptions.z_gridvals_J,1,N_e,1);
                ze_gridvals_J_T(:,:,l_z+1:end,:)=repmat(squeeze(transpathoptions.e_gridvals_J),1,N_z,1);
            end
        end

        if transpathoptions.fastOLG==0
            if N_z>0 && N_e>0
                transpathoptions.ze_gridvals_J_T_fastOLG=shiftdim(permute(ze_gridvals_J_T,[3,1,2,4]),-1); % [1,N_j,N_ze,l_ze,T] need this for fastOLG agent dist (it is not used for value fn, so can overwrite)
            elseif N_z>0
                transpathoptions.ze_gridvals_J_T_fastOLG=shiftdim(permute(transpathoptions.z_gridvals_J_T,[3,1,2,4]),-1); % [1,N_j,N_z,l_z,T] need this for fastOLG agent dist, but need the standard still for the value fn without fastOLG
            elseif N_e>0
                transpathoptions.ze_gridvals_J_T_fastOLG=shiftdim(permute(transpathoptions.e_gridvals_J_T,[3,1,2,4]),-1); % [1,N_j,N_e,l_e,T] need this for fastOLG agent dist, but need the standard still for the value fn without fastOLG
            end
        else
            if N_z>0 && N_e>0
                transpathoptions.ze_gridvals_J_T_fastOLG=shiftdim(transpathoptions.ze_gridvals_J_T,-1); % [1,N_j,N_ze,l_ze,T] need this for fastOLG agent dist (it is not used for value fn, so can overwrite)
            elseif N_z>0
                transpathoptions.ze_gridvals_J_T_fastOLG=shiftdim(transpathoptions.z_gridvals_J_T,-1); % [1,N_j,N_z,l_z,T] need this for fastOLG agent dist, but need the standard still for the value fn without fastOLG
            elseif N_e>0
                transpathoptions.ze_gridvals_J_T_fastOLG=shiftdim(transpathoptions.e_gridvals_J_T,-1); % [1,N_j,N_e,l_e,T] need this for fastOLG agent dist, but need the standard still for the value fn without fastOLG
            end
        end

    end
end


end