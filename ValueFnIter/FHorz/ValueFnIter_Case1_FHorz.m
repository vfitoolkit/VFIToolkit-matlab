function [V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

V=nan;
Policy=nan;

%% Check which vfoptions have been used, set all others to defaults 
if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
    vfoptions.parallel=1+(gpuDeviceCount>0);
    if vfoptions.parallel==2
        vfoptions.returnmatrix=2; % On GPU, must use this option
    end
    if isfield(vfoptions,'returnmatrix')==0
        if isa(ReturnFn,'function_handle')==1
            vfoptions.returnmatrix=0;
        else
            vfoptions.returnmatrix=1;
        end
    end
    vfoptions.verbose=0;
    vfoptions.lowmemory=0;
    vfoptions.paroverz=1; % This is just a refinement of lowmemory=0
    vfoptions.incrementaltype=0; % (vector indicating endogenous state is an incremental endogenous state variable)
    vfoptions.polindorval=1;
    vfoptions.policy_forceintegertype=0;
    vfoptions.outputkron=0; % If 1 then leave output in Kron form
    vfoptions.exoticpreferences='None';
    vfoptions.dynasty=0;
    vfoptions.experienceasset=0;
    vfoptions.experienceassetu=0;
    vfoptions.riskyasset=0;
    vfoptions.residualasset=0;
    vfoptions.n_ambiguity=0;
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(vfoptions,'parallel')
        vfoptions.parallel=1+(gpuDeviceCount>0);
    end
    if vfoptions.parallel==2
        vfoptions.returnmatrix=2; % On GPU, must use this option
    end
    if ~isfield(vfoptions,'returnmatrix')
        if isa(ReturnFn,'function_handle')==1
            vfoptions.returnmatrix=0;
        else
            vfoptions.returnmatrix=1;
        end
    end
    if ~isfield(vfoptions,'lowmemory')
        vfoptions.lowmemory=0;
    end
    if ~isfield(vfoptions,'paroverz') % Only used when vfoptions.lowmemory=0
        vfoptions.paroverz=1;
    end
    if ~isfield(vfoptions,'verbose')
        vfoptions.verbose=0;
    end
    if isfield(vfoptions,'incrementaltype')==0
        vfoptions.incrementaltype=0; % (vector indicating endogenous state is an incremental endogenous state variable)
    end
    if ~isfield(vfoptions,'polindorval')
        vfoptions.polindorval=1;
    end
    if ~isfield(vfoptions,'policy_forceintegertype')
        vfoptions.policy_forceintegertype=0;
    end
    if isfield(vfoptions,'ExogShockFn')
        vfoptions.ExogShockFnParamNames=getAnonymousFnInputNames(vfoptions.ExogShockFn);
    end
    if isfield(vfoptions,'EiidShockFn')
        vfoptions.EiidShockFnParamNames=getAnonymousFnInputNames(vfoptions.EiidShockFn);
    end
    if ~isfield(vfoptions,'outputkron')
        vfoptions.outputkron=0; % If 1 then leave output in Kron form
    end
    if ~isfield(vfoptions,'exoticpreferences')
        vfoptions.exoticpreferences='None';
    end
    if ~isfield(vfoptions,'dynasty')
        vfoptions.dynasty=0;
    end
    if ~isfield(vfoptions,'experienceasset')
        vfoptions.experienceasset=0;
    end
    if ~isfield(vfoptions,'experienceassetu')
        vfoptions.experienceassetu=0;
    end
    if ~isfield(vfoptions,'riskyasset')
        vfoptions.riskyasset=0;
    end
    if ~isfield(vfoptions,'residualasset')
        vfoptions.residualasset=0;
    end
end

if isempty(n_d)
    error('If you have no d (decision) variables, set n_d=0;')
end
if isempty(n_z)
    error('If you have no z (exogenous markov) variables, set n_z=0;')
end
N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if ~all(size(d_grid)==[sum(n_d), 1])
    if ~isempty(n_d) % Make sure d is being used before complaining about size of d_grid
        if n_d~=0
            error('d_grid is not the correct shape (should be of size sum(n_d)-by-1)')
        end
    end
end
if ~all(size(a_grid)==[sum(n_a), 1])
    error('a_grid is not the correct shape (should be of size sum(n_a)-by-1; a (stacked) column vector)')
end
% Check z_grid inputs
if isa(z_grid,'function_handle')
    % okay
elseif ndims(z_grid)==2
    if ~all(size(z_grid)==[sum(n_z),1]) && ~all(size(z_grid)==[prod(n_z),length(n_z)]) && ~all(size(z_grid)==[n_z(1),length(n_z)]) && ~all(size(z_grid)==[sum(n_z),N_j])
        % all(size(z_grid)==[sum(n_z), 1]) is the grid as a stacked vector
        % all(size(z_grid)==[prod(n_z),length(n_z)]) is a joint-grid
        % all(size(z_grid)==[n_z(1),length(n_z)]) is a joint-grid
        % all(size(z_grid)==[sum(n_z), N_j]) is the grid as an age-dependent stacked vector
        if N_z>0
            error('z_grid is not the correct shape (typically should be of size sum(n_z)-by-1)')
        end
    end
elseif ndims(z_grid)==3
    if ~all(size(z_grid)==[prod(n_z),length(n_z),N_j]) && ~all(size(z_grid)==[n_z(1),length(n_z),N_j])
        % all(size(z_grid)==[prod(n_z),length(n_z),N_j]) is an age-dependent joint-grid
        % all(size(z_grid)==[n_z(1),length(n_z),N_j]) is an age-dependent joint-grid
        if N_z>0
            error('z_grid is not the correct shape (typically should be of size sum(n_z)-by-N_j)')
        end
    end
else
    error('z_grid is not the correct shape (typically should be of size sum(n_z)-by-1)')
end
% Check pi_z inputs
if isa(z_grid,'function_handle')
    % okay (dont need to check pi_z
elseif ndims(pi_z)==2
    if ~isequal(size(pi_z), [N_z, N_z])
        if N_z>0
            error('pi_z is not of size N_z-by-N_z')
        end 
    end
elseif ndims(pi_z)==3
    if ~isequal(size(pi_z), [N_z, N_z,N_j])
        if N_z>0
            error('pi_z is not of size N_z-by-N_z-by-N_j')
        end 
    end
end


if isfield(vfoptions,'n_e')
    if vfoptions.parallel<2
        error('Sorry but e (i.i.d) variables are not implemented for cpu, you will need a gpu to use them')
    end
    if ~isfield(vfoptions,'e_grid')
        error('When using vfoptions.n_e you must declare vfoptions.e_grid')
    elseif ~isfield(vfoptions,'pi_e')
        error('When using vfoptions.n_e you must declare vfoptions.pi_e')
    else
        % check size of e_grid and pi_e
        if isfield(vfoptions,'e_grid')
            if ndims(vfoptions.e_grid)==2
                if ~all(size(vfoptions.e_grid)==[sum(vfoptions.n_e), 1]) && ~all(size(vfoptions.e_grid)==[prod(vfoptions.n_e),length(vfoptions.n_e)]) && ~all(size(vfoptions.e_grid)==[sum(vfoptions.n_e), N_j])
                    error('vfoptions.e_grid is not the correct shape (should be of size sum(n_e)-by-1; or sum(n_e)-by-N_j or N_e-by-l_e or N_e-by-l_e-by_N_j )')
                end
            elseif ndims(vfoptions.e_grid)==3
                if ~all(size(vfoptions.e_grid)==[prod(vfoptions.n_e),length(vfoptions.n_e),N_j])
                    error('vfoptions.e_grid is not the correct shape (should be of size sum(n_e)-by-1; or sum(n_e)-by-N_j or N_e-by-l_e or N_e-by-l_e-by_N_j )')
                end
            else
                error('vfoptions.e_grid is not the correct shape (should be of size sum(n_e)-by-1; or sum(n_e)-by-N_j or N_e-by-l_e or N_e-by-l_e-by_N_j )')
            end
        end
        if isfield(vfoptions,'pi_e')
            if ~all(size(vfoptions.pi_e)==[prod(vfoptions.n_e),1]) && ~all(size(vfoptions.pi_e)==[prod(vfoptions.n_e),N_j])
                error('vfoptions.pi_e is not the correct shape (should be of size N_e-by-1 or N_e-by-N_j)')
            end
        end
    end
end

%% Exogenous shock grids

% % % When using a joint-grid, change n_z to the form I always use internally: first element is N_z, followed by a bunch of ones (that way prod(n_z) still gives N_z)
% % jointgrid=0;
% % if all(size(z_grid)==[prod(n_z),length(n_z)])
% %     jointgrid=1;
% %     n_z=[prod(n_z,ones(1,length(n_z)-1))];
% % elseif all(size(z_grid)==[n_z(1),length(n_z)])
% %     % already in this form
% %     jointgrid=1;
% % end

% NOTE: If vfoptions.parallel~=2 (so using cpu), then only simply stacked columns that do not depend on age are allowed for z_grid
if vfoptions.parallel==2
    % Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
    % Gradually rolling these out so that all the commands build off of these
    z_gridvals_J=zeros(prod(n_z),length(n_z),'gpuArray');
    pi_z_J=zeros(prod(n_z),prod(n_z),'gpuArray');
    if isfield(vfoptions,'ExogShockFn')
        if isfield(vfoptions,'ExogShockFnParamNames')
            for jj=1:N_j
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
                pi_z_J(:,:,jj)=gpuArray(pi_z);
                if all(size(z_grid)==[sum(n_z),1])
                    z_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(n_z,z_grid,1));
                else % already joint-grid
                    z_gridvals_J(:,:,jj)=gpuArray(z_grid,1);
                end
            end
        else
            for jj=1:N_j
                [z_grid,pi_z]=vfoptions.ExogShockFn(N_j);
                pi_z_J(:,:,jj)=gpuArray(pi_z);
                if all(size(z_grid)==[sum(n_z),1])
                    z_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(n_z,z_grid,1));
                else % already joint-grid
                    z_gridvals_J(:,:,jj)=gpuArray(z_grid,1);
                end
            end
        end
    elseif prod(n_z)==0 % no z
        z_gridvals_J=[];
    elseif ndims(z_grid)==3 % already an age-dependent joint-grid
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

    % If using e variable, do same for this
    if isfield(vfoptions,'n_e')
        if prod(vfoptions.n_e)==0
            vfoptions=rmfield(vfoptions,'n_e');
        else
            if isfield(vfoptions,'e_grid_J')
                error('No longer use vfoptions.e_grid_J, instead just put the age-dependent grid in vfoptions.e_grid (functionality of VFI Toolkit has changed to make it easier to use)')
            end
            if ~isfield(vfoptions,'e_grid') % && ~isfield(vfoptions,'e_grid_J')
                error('You are using an e (iid) variable, and so need to declare vfoptions.e_grid')
            elseif ~isfield(vfoptions,'pi_e')
                error('You are using an e (iid) variable, and so need to declare vfoptions.pi_e')
            end
            
            vfoptions.e_gridvals_J=zeros(prod(vfoptions.n_e),length(vfoptions.n_e),'gpuArray');
            vfoptions.pi_e_J=zeros(prod(vfoptions.n_e),prod(vfoptions.n_e),'gpuArray');

            if isfield(vfoptions,'EiidShockFn')
                if isfield(vfoptions,'EiidShockFnParamNames')
                    for jj=1:N_j
                        EiidShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.EiidShockFnParamNames,jj);
                        EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                        for ii=1:length(EiidShockFnParamsVec)
                            EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                        end
                        [vfoptions.e_grid,vfoptions.pi_e]=vfoptions.EiidShockFn(EiidShockFnParamsCell{:});
                        vfoptions.pi_e_J(:,jj)=gpuArray(vfoptions.pi_e);
                        if all(size(vfoptions.e_grid)==[sum(vfoptions.n_e),1])
                            vfoptions.e_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(vfoptions.n_e,vfoptions.e_grid,1));
                        else % already joint-grid
                            vfoptions.e_gridvals_J(:,:,jj)=gpuArray(vfoptions.e_grid,1);
                        end
                    end
                else
                    for jj=1:N_j
                        [vfoptions.e_grid,vfoptions.pi_e]=vfoptions.EiidShockFn(N_j);
                        vfoptions.pi_e_J(:,jj)=gpuArray(vfoptions.pi_e);
                        if all(size(vfoptions.e_grid)==[sum(vfoptions.n_e),1])
                            vfoptions.e_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(vfoptions.n_e,vfoptions.e_grid,1));
                        else % already joint-grid
                            vfoptions.e_gridvals_J(:,:,jj)=gpuArray(vfoptions.e_grid,1);
                        end
                    end
                end
            elseif ndims(vfoptions.e_grid)==3 % already an age-dependent joint-grid
                if all(size(vfoptions.e_grid)==[prod(vfoptions.n_e),length(vfoptions.n_e),N_j])
                    vfoptions.e_gridvals_J=vfoptions.e_grid;
                end
                vfoptions.pi_e_J=vfoptions.pi_e;
            elseif all(size(vfoptions.e_grid)==[sum(vfoptions.n_e),N_j]) % age-dependent stacked-grid
                for jj=1:N_j
                    vfoptions.e_gridvals_J(:,:,jj)=CreateGridvals(vfoptions.n_e,vfoptions.e_grid(:,jj),1);
                end
                vfoptions.pi_e_J=vfoptions.pi_e;
            elseif all(size(vfoptions.e_grid)==[prod(vfoptions.n_e),length(vfoptions.n_e)]) % joint grid
                vfoptions.e_gridvals_J=vfoptions.e_grid.*ones(1,1,N_j,'gpuArray');
                vfoptions.pi_e_J=vfoptions.pi_e.*ones(1,N_j,'gpuArray');
            elseif all(size(vfoptions.e_grid)==[sum(vfoptions.n_e),1]) % basic grid
                vfoptions.e_gridvals_J=CreateGridvals(vfoptions.n_e,vfoptions.e_grid,1).*ones(1,1,N_j,'gpuArray');
                vfoptions.pi_e_J=vfoptions.pi_e.*ones(1,N_j,'gpuArray');
            end
        end
    end
end


%% Implement new way of handling ReturnFn inputs
if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_aprime=l_a;
l_z=length(n_z);
if n_z(1)==0
    l_z=0;
end
if isfield(vfoptions,'SemiExoStateFn')
    l_z=length(vfoptions.n_semiz)+l_z;
end
l_e=0;
if isfield(vfoptions,'n_e')
    l_e=length(vfoptions.n_e);
end
if vfoptions.experienceasset>0
    % One of the endogenous states should only be counted once.
    l_aprime=l_aprime-vfoptions.experienceasset;
end
if vfoptions.experienceassetu>0
    % One of the endogenous states should only be counted once.
    l_aprime=l_aprime-vfoptions.experienceassetu;
end
if vfoptions.riskyasset>0
    % One of the endogenous states should only be counted once.
    l_aprime=l_aprime-vfoptions.riskyasset;
end
if vfoptions.residualasset>0
    % One of the endogenous states should only be counted once.
    l_aprime=l_aprime-vfoptions.residualasset;
end
% If no ReturnFnParamNames inputted, then figure it out from ReturnFn
if isempty(ReturnFnParamNames)
    temp=getAnonymousFnInputNames(ReturnFn);
    if length(temp)>(l_d+l_aprime+l_a+l_z+l_e) % This is largely pointless, the ReturnFn is always going to have some parameters
        ReturnFnParamNames={temp{l_d+l_aprime+l_a+l_z+l_e+1:end}}; % the first inputs will always be (d,aprime,a,z,e)
    else
        ReturnFnParamNames={};
    end
end
% [l_d,l_aprime,l_a,l_z,l_e]
% ReturnFnParamNames
% clear l_d l_a l_z l_e % These are all messed up so make sure they are not reused later

%% Implement new way of handling warm-glow of bequests (currently only used by Epstein-Zin preferences)
if isfield(vfoptions,'WarmGlowBequestsFn')
    temp=getAnonymousFnInputNames(vfoptions.WarmGlowBequestsFn);
    vfoptions.WarmGlowBequestsFnParamsNames={temp{l_a+1:end}}; % the first inputs will always be aprime
end
% clear l_d l_a l_z l_e % These are all messed up so make sure they are not reused later

%% 
if vfoptions.parallel==2 
   % If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
   pi_z=gpuArray(pi_z);
   d_grid=gpuArray(d_grid);
   a_grid=gpuArray(a_grid);
   z_grid=gpuArray(z_grid);
else
   % If using CPU make sure all the relevant inputs are CPU arrays (not standard arrays)
   % This may be completely unnecessary.
   pi_z=gather(pi_z);
   d_grid=gather(d_grid);
   a_grid=gather(a_grid);
   z_grid=gather(z_grid);
end

if vfoptions.verbose==1
    vfoptions
end

%% Deal with Exotic preferences if need to do that.
if strcmp(vfoptions.exoticpreferences,'None')
    % Just ignore and will then continue on.
elseif strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
    [V, Policy]=ValueFnIter_Case1_FHorz_QuasiHyperbolic(n_d,n_a,n_z,N_j,d_grid,a_grid,z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return
elseif strcmp(vfoptions.exoticpreferences,'EpsteinZin') && vfoptions.riskyasset==0 % deal with risky asset elsewhere
    [V, Policy]=ValueFnIter_Case1_FHorz_EpsteinZin(n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return
elseif strcmp(vfoptions.exoticpreferences,'GulPesendorfer')
    [V, Policy]=ValueFnIter_Case1_FHorz_GulPesendorfer(n_d,n_a,n_z,N_j,d_grid,a_grid,z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return
elseif strcmp(vfoptions.exoticpreferences,'AmbiguityAversion')
    [V, Policy]=ValueFnIter_Case1_FHorz_Ambiguity(n_d,n_a,n_z,N_j,d_grid,a_grid,z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return
end

%% Using both Experience Asset and Semi-Exogenous state
if vfoptions.experienceasset==1 && isfield(vfoptions,'SemiExoStateFn')
    % First, sort out splitting up the decision variables (other, semiexo, experienceasset)
    if length(n_d)>2
        n_d1=n_d(1:end-2);
    else
        n_d1=0;
    end
    n_d2=n_d(end-1); % n_d2 is the decision variable that influences the experience asset
    n_d3=n_d(end); % n_d3 is the decision variable that influences the transition probabilities of the semi-exogenous state
    d1_grid=d_grid(1:sum(n_d1));
    d2_grid=d_grid(sum(n_d1)+1:sum(n_d1)+sum(n_d2));
    d3_grid=d_grid(sum(n_d1)+sum(n_d2)+1:end);
    % Split endogenous assets into the standard ones and the experience asset
    if length(n_a)==1
        n_a1=0;
    else
        n_a1=n_a(1:end-1);
    end
    n_a2=n_a(end); % n_a2 is the experience asset
    a1_grid=a_grid(1:sum(n_a1));
    a2_grid=a_grid(sum(n_a1)+1:end);

    % Second, set up the semi-exogenous state
    if ~isfield(vfoptions,'n_semiz')
        error('When using vfoptions.SemiExoShockFn you must declare vfoptions.n_semiz')
    end
    if ~isfield(vfoptions,'semiz_grid')
        error('When using vfoptions.SemiExoShockFn you must declare vfoptions.semiz_grid')
    end
    % Create the transition matrix in terms of (d,zprime,z) for the semi-exogenous states for each age
    N_semiz=prod(vfoptions.n_semiz);
    l_semiz=length(vfoptions.n_semiz);
    temp=getAnonymousFnInputNames(vfoptions.SemiExoStateFn);
    if length(temp)>(1+l_semiz+l_semiz) % This is largely pointless, the SemiExoShockFn is always going to have some parameters
        SemiExoStateFnParamNames={temp{1+l_semiz+l_semiz+1:end}}; % the first inputs will always be (d,semizprime,semiz)
    else
        SemiExoStateFnParamNames={};
    end
    pi_semiz_J=zeros(N_semiz,N_semiz,n_d3,N_j,'gpuArray');
    for jj=1:N_j
        SemiExoStateFnParamValues=CreateVectorFromParams(Parameters,SemiExoStateFnParamNames,jj);
        pi_semiz_J(:,:,:,jj)=gpuArray(CreatePiSemiZ(n_d3,vfoptions.n_semiz,d3_grid,vfoptions.semiz_grid,vfoptions.SemiExoStateFn,SemiExoStateFnParamValues));
    end
    semiz_gridvals_J=gpuArray(CreateGridvals(vfoptions.n_semiz,vfoptions.semiz_grid,1).*ones(1,1,N_j));

    % Now just send it off
    [V,Policy]=ValueFnIter_Case1_FHorz_ExpAssetSemiExo(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,vfoptions.n_semiz, N_j, d1_grid , d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return

end

%% Deal with Experience Asset if need to do that
% experienceasset: aprime(d,a)
% experienceassetu: aprime(d,a,u)
% experienceassetz: aprime(d,a,z)
% experienceassetz: aprime(d,a,z,e)

if vfoptions.experienceasset==1 || vfoptions.experienceassetu==1
    % It is simply assumed that the experience asset is the last asset, and that the decision that influences it is the last decision.
    
    % Split endogenous assets into the standard ones and the experience asset
    if length(n_a)==1
        n_a1=0;
    else
        n_a1=n_a(1:end-1);
    end
    n_a2=n_a(end); % n_a2 is the experience asset
    a1_grid=a_grid(1:sum(n_a1));
    a2_grid=a_grid(sum(n_a1)+1:end);
    % Split decision variables into the standard ones and the one relevant to the experience asset
    if length(n_d)==1
        n_d1=0;
    else
        n_d1=n_d(1:end-1);
    end
    n_d2=n_d(end); % n_d2 is the decision variable that influences next period vale of the experience asset
    d1_grid=d_grid(1:sum(n_d1));
    d2_grid=d_grid(sum(n_d1)+1:end);

    % Now just send all this to the right value fn iteration command
    if vfoptions.experienceasset==1
        [V,Policy]=ValueFnIter_Case1_FHorz_ExpAsset(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid , d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    elseif vfoptions.experienceassetu==1
        [V,Policy]=ValueFnIter_Case1_FHorz_ExpAssetu(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid , d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    elseif vfoptions.experienceassetz==1
        % I want to implement this too :)
    elseif vfoptions.experienceassetze==1
        % I want to implement this too :)
    end
    return
end


%% Deal with risky asset if need to do that
if vfoptions.riskyasset==1
    % It is simply assumed that the experience asset is the last asset, and that all decisions influence it.

    % Split endogenous assets into the standard ones and the experience asset
    if length(n_a)==1
        n_a1=0;
    else
        n_a1=n_a(1:end-1);
    end
    n_a2=n_a(end); % n_a2 is the experience asset
    a1_grid=a_grid(1:sum(n_a1));
    a2_grid=a_grid(sum(n_a1)+1:end);

    % Check that aprimeFn is inputted
    if ~isfield(vfoptions,'aprimeFn')
        error('You have vfoptions.riskyasset=1, but have not setup vfoptions.aprimeFn')
    end    
    % Check that the u shocks are inputted
    if ~isfield(vfoptions,'n_u')
        error('You have vfoptions.riskyasset=1, but have not setup vfoptions.n_u')
    end
    if ~isfield(vfoptions,'u_grid')
        error('You have vfoptions.riskyasset=1, but have not setup vfoptions.u_grid')
    end
    if ~isfield(vfoptions,'pi_u') % && ~isfield(vfoptions,'pi_u_J')
        error('You have vfoptions.riskyasset=1, but have not setup vfoptions.pi_u')
    end

    % Now just send all this to the right value fn iteration command
    [V,Policy]=ValueFnIter_Case1_FHorz_RiskyAsset(n_d,n_a1,n_a2,n_z,vfoptions.n_u, N_j, d_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.u_grid, pi_z_J, vfoptions.pi_u, ReturnFn, vfoptions.aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return
end

%% Deal with residual asset if need to do that
if vfoptions.residualasset==1
    % Split endogenous assets into the standard ones and the experience asset
    if length(n_a)==1
        n_a1=0;
    else
        n_a1=n_a(1:end-1);
    end
    n_r=n_a(end); % n_a2 is the experience asset
    a1_grid=a_grid(1:sum(n_a1));
    r_grid=a_grid(sum(n_a1)+1:end);
    
    % Now just send all this to the right value fn iteration command
    [V,Policy]=ValueFnIter_Case1_FHorz_ResidAsset(n_d,n_a1,n_r,n_z, N_j, d_grid, a1_grid, r_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return
end

%% Deal with StateDependentVariables_z if need to do that.
if isfield(vfoptions,'StateDependentVariables_z')==1
    if vfoptions.verbose==1
        fprintf('StateDependentVariables_z option is being used \n')
    end
    
    if N_d==0
        [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_no_d_SDVz_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    else
        [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_SDVz_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    end
    
    %Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
    V=reshape(VKron,[n_a,n_z,N_j]);
    Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, n_z, N_j,vfoptions);
    
    % Sometimes numerical rounding errors (of the order of 10^(-16) can mean
    % that Policy is not integer valued. The following corrects this by converting to int64 and then
    % makes the output back into double as Matlab otherwise cannot use it in
    % any arithmetical expressions.
    if vfoptions.policy_forceintegertype==1
        fprintf('USING vfoptions to force integer... \n')
        % First, give some output on the size of any changes in Policy as a
        % result of turning the values into integers
        temp=max(max(max(abs(round(Policy)-Policy))));
        while ndims(temp)>1
            temp=max(temp);
        end
        fprintf('  CHECK: Maximum change when rounding values of Policy is %8.6f (if these are not of numerical rounding error size then something is going wrong) \n', temp)
        % Do the actual rounding to integers
        Policy=round(Policy);
        % Somewhat unrelated, but also do a double-check that Policy is now all positive integers
        temp=min(min(min(Policy)));
        while ndims(temp)>1
            temp=min(temp);
        end
        fprintf('  CHECK: Minimum value of Policy is %8.6f (if this is <=0 then something is wrong) \n', temp)
        %     Policy=uint64(Policy);
        %     Policy=double(Policy);
    end
    
    return
end

%% Deal with dynasty if need to do that.
if vfoptions.dynasty==1
    if vfoptions.verbose==1
        fprintf('dynasty option is being used \n')
    end
    if isfield(vfoptions,'tolerance')==0
        vfoptions.tolerance=10^(-9);
    end
    
    if N_d==0
        [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_nod_Dynasty_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    else
        [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_Dynasty_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    end
    
    %Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
    V=reshape(VKron,[n_a,n_z,N_j]);
    Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, n_z, N_j,vfoptions);
    
    % Sometimes numerical rounding errors (of the order of 10^(-16) can mean
    % that Policy is not integer valued. The following corrects this by converting to int64 and then
    % makes the output back into double as Matlab otherwise cannot use it in
    % any arithmetical expressions.
    if vfoptions.policy_forceintegertype==1
        fprintf('USING vfoptions to force integer... \n')
        % First, give some output on the size of any changes in Policy as a
        % result of turning the values into integers
        temp=max(max(max(abs(round(Policy)-Policy))));
        while ndims(temp)>1
            temp=max(temp);
        end
        fprintf('  CHECK: Maximum change when rounding values of Policy is %8.6f (if these are not of numerical rounding error size then something is going wrong) \n', temp)
        % Do the actual rounding to integers
        Policy=round(Policy);
        % Somewhat unrelated, but also do a double-check that Policy is now all positive integers
        temp=min(min(min(Policy)));
        while ndims(temp)>1
            temp=min(temp);
        end
        fprintf('  CHECK: Minimum value of Policy is %8.6f (if this is <=0 then something is wrong) \n', temp)
        %     Policy=uint64(Policy);
        %     Policy=double(Policy);
    end
    
    return
end

%% Semi-exogenous state
% The transition matrix of the exogenous shocks depends on the value of the 'last' decision variable(s).
if isfield(vfoptions,'SemiExoStateFn')
    if ~isfield(vfoptions,'n_semiz')
        error('When using vfoptions.SemiExoShockFn you must declare vfoptions.n_semiz')
    end
    if ~isfield(vfoptions,'semiz_grid')
        error('When using vfoptions.SemiExoShockFn you must declare vfoptions.semiz_grid')
    end
    if ~isfield(vfoptions,'numd_semiz')
        vfoptions.numd_semiz=1; % by default, only one decision variable influences the semi-exogenous state
    end
    if length(n_d)>vfoptions.numd_semiz
        n_d1=n_d(1:end-vfoptions.numd_semiz);
        d1_grid=d_grid(1:sum(n_d1));
    else
        n_d1=0; d1_grid=[];
    end
    n_d2=n_d(end-vfoptions.numd_semiz+1:end); % n_d2 is the decision variable that influences the transition probabilities of the semi-exogenous state
    l_d2=length(n_d2);
    d2_grid=d_grid(sum(n_d1)+1:end);
    % Create the transition matrix in terms of (d,zprime,z) for the semi-exogenous states for each age
    N_semiz=prod(vfoptions.n_semiz);
    l_semiz=length(vfoptions.n_semiz);
    temp=getAnonymousFnInputNames(vfoptions.SemiExoStateFn);
    if length(temp)>(l_semiz+l_semiz+l_d2) % This is largely pointless, the SemiExoShockFn is always going to have some parameters
        SemiExoStateFnParamNames={temp{l_semiz+l_semiz+l_d2+1:end}}; % the first inputs will always be (semiz,semizprime,d)
    else
        SemiExoStateFnParamNames={};
    end
    pi_semiz_J=zeros(N_semiz,N_semiz,prod(n_d2),N_j);
    for jj=1:N_j
        SemiExoStateFnParamValues=CreateVectorFromParams(Parameters,SemiExoStateFnParamNames,jj);
        pi_semiz_J(:,:,:,jj)=CreatePiSemiZ(n_d2,vfoptions.n_semiz,d2_grid,vfoptions.semiz_grid,vfoptions.SemiExoStateFn,SemiExoStateFnParamValues);
    end
    if ndims(vfoptions.semiz_grid)==2
        if all(size(vfoptions.semiz_grid)==[sum(vfoptions.n_semiz),1])
            semiz_gridvals_J=CreateGridvals(vfoptions.n_semiz,vfoptions.semiz_grid,1).*ones(1,1,N_j,'gpuArray');
        elseif all(size(vfoptions.semiz_grid)==[prod(vfoptions.n_semiz),length(vfoptions.n_semiz)])
            semiz_gridvals_J=vfoptions.semiz_grid.*ones(1,1,N_j,'gpuArray');
        end
    else % Already age-dependent
        if all(size(vfoptions.semiz_grid)==[sum(vfoptions.n_semiz),N_j])
            semiz_gridvals_J=zeros(prod(vfoptions.n_semiz),length(vfoptions.n_semiz),N_j,'gpuArray');
            for jj=1:N_j
                semiz_gridvals_J(:,:,jj)=CreateGridvals(vfoptions.n_semiz,vfoptions.semiz_grid(:,jj),1);
            end
        elseif all(size(vfoptions.semiz_grid)==[prod(vfoptions.n_semiz),length(vfoptions.n_semiz),N_j])
            semiz_gridvals_J=vfoptions.semiz_grid;
        end
    end
    % Now that we have pi_semiz_J we are ready to compute the value function.
    if vfoptions.parallel==2
        if n_d1==0
            if isfield(vfoptions,'n_e')
                error('Have not implemented semi-exogenous shocks without at least two decision variables (one of which is that which determines the semi-exog transitions)')
            else
                if N_z==0
                    [VKron, Policy3]=ValueFnIter_Case1_FHorz_SemiExo_nod1_noz_raw(n_d2,n_a,vfoptions.n_semiz, N_j, d2_grid, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    error('Have not implemented semi-exogenous shocks without at least two decision variables (one of which is that which determines the semi-exog transitions)')
                end
            end
        else
            if isfield(vfoptions,'n_e')
                if N_z==0
                    error('Have not implemented semi-exogenous shocks without at least one z variable (not counting the semi-exogenous one) but with an e variable [you could fake it adding a single-valued z with pi_z=1]')
                else
                    [VKron, Policy3]=ValueFnIter_Case1_FHorz_SemiExo_e_raw(n_d1,n_d2,n_a,n_z,vfoptions.n_semiz,  vfoptions.n_e, N_j, d1_grid, d2_grid, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else
                if N_z==0
                    [VKron, Policy3]=ValueFnIter_Case1_FHorz_SemiExo_noz_raw(n_d1,n_d2,n_a,vfoptions.n_semiz, N_j, d1_grid, d2_grid, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron, Policy3]=ValueFnIter_Case1_FHorz_SemiExo_raw(n_d1,n_d2,n_a,n_z,vfoptions.n_semiz, N_j, d1_grid, d2_grid, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
        end
    end
    
    %Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
    if vfoptions.outputkron==0
        if isfield(vfoptions,'n_e')
            if N_z==0
                V=reshape(VKron,[n_a,vfoptions.n_semiz, vfoptions.n_e,N_j]);
                Policy=UnKronPolicyIndexes_Case1_FHorz_semiz_e(Policy3, n_d1,n_d2, n_a, vfoptions.n_semiz, vfoptions.n_e, N_j, vfoptions);
            else
                V=reshape(VKron,[n_a,vfoptions.n_semiz,n_z,vfoptions.n_e,N_j]);
                Policy=UnKronPolicyIndexes_Case1_FHorz_semiz_e(Policy3, n_d1,n_d2, n_a, [vfoptions.n_semiz,n_z], vfoptions.n_e, N_j, vfoptions);
            end
        else
            if N_z==0
                V=reshape(VKron,[n_a,vfoptions.n_semiz,N_j]);
                Policy=UnKronPolicyIndexes_Case1_FHorz_semiz(Policy3, n_d1, n_d2, n_a, vfoptions.n_semiz, N_j, vfoptions);
            else
                V=reshape(VKron,[n_a,vfoptions.n_semiz,n_z,N_j]);
                Policy=UnKronPolicyIndexes_Case1_FHorz_semiz(Policy3, n_d1, n_d2, n_a, [vfoptions.n_semiz,n_z], N_j, vfoptions);
            end
        end
    else
        V=VKron;
        Policy=Policy3;
    end

    return
end

%% Detect if using incremental endogenous states and solve this using purediscretization, prior to the main purediscretization routines
if any(vfoptions.incrementaltype)
    % Incremental Endogenous States: aprime either equals a, or one grid point higher (unchanged on incremental increase)
    [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_Increment(n_d,n_a,n_z,d_grid,a_grid,z_gridvals_J,N_j,pi_z_J,ReturnFn,Parameters,ReturnFnParamNames,DiscountFactorParamNames,vfoptions);
    
    %Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
    if isfield(vfoptions,'n_e')
        if N_z==0
            V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
        else
            V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz_e(PolicyKron, n_d, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
        end
    else
        V=reshape(VKron,[n_a,n_z,N_j]);
        Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, n_z, N_j, vfoptions);
    end
    
    return
end


%% Just do the standard case
if vfoptions.parallel==2
    if N_d==0
        if isfield(vfoptions,'n_e')
            if N_z==0
                [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    else % N_d
        if isfield(vfoptions,'n_e')
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_noz_e_raw(n_d,n_a,  vfoptions.n_e, N_j, d_grid, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_e_raw(n_d,n_a,n_z,  vfoptions.n_e, N_j, d_grid, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_noz_raw(n_d,n_a, N_j, d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end
elseif vfoptions.parallel==1
    if N_d==0
        if N_z==0
            [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_nod_noz_Par1_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_nod_Par1_raw(n_a, n_z, N_j, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else % N_d
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_noz_Par1_raw(n_d,n_a, N_j, d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else 
            [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_Par1_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
elseif vfoptions.parallel==0
    if N_d==0
        [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_nod_Par0_raw(n_a, n_z, N_j, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    else
        [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_Par0_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    end
end


%% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if vfoptions.outputkron==0
    if isfield(vfoptions,'n_e')
        if N_z==0
            V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
        else
            V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz_e(PolicyKron, n_d, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
        end
    else
        if N_z==0
            V=reshape(VKron,[n_a,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz_noz(PolicyKron, n_d, n_a, N_j, vfoptions);
        else
            V=reshape(VKron,[n_a,n_z,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, n_z, N_j, vfoptions);
        end
    end
else
    V=VKron;
    Policy=PolicyKron;
end

% Sometimes numerical rounding errors (of the order of 10^(-16) can mean
% that Policy is not integer valued. The following corrects this by converting to int64 and then
% makes the output back into double as Matlab otherwise cannot use it in
% any arithmetical expressions.
if vfoptions.policy_forceintegertype==1
    fprintf('USING vfoptions to force integer... \n')
    % First, give some output on the size of any changes in Policy as a
    % result of turning the values into integers
    temp=max(max(max(abs(round(Policy)-Policy))));
    while ndims(temp)>1
        temp=max(temp);
    end
    fprintf('  CHECK: Maximum change when rounding values of Policy is %8.6f (if these are not of numerical rounding error size then something is going wrong) \n', temp)
    % Do the actual rounding to integers
    Policy=round(Policy);
    % Somewhat unrelated, but also do a double-check that Policy is now all positive integers
    temp=min(min(min(Policy)));
    while ndims(temp)>1
        temp=min(temp);
    end
    fprintf('  CHECK: Minimum value of Policy is %8.6f (if this is <=0 then something is wrong) \n', temp)
%     Policy=uint64(Policy);
%     Policy=double(Policy);
elseif vfoptions.policy_forceintegertype==2
    % Do the actual rounding to integers
    Policy=round(Policy);
end

end