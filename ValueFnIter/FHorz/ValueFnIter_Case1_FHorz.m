function varargout=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Typically, varargout=[V, Policy]


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
    vfoptions.divideandconquer=0; % =1 Use divide-and-conquer to exploit monotonicity
    vfoptions.gridinterplayer=0; % Interpolate between grid points (not yet implemented for most cases)
    vfoptions.polindorval=1;
    vfoptions.policy_forceintegertype=0;
    vfoptions.outputkron=0; % If 1 then leave output in Kron form
    vfoptions.incrementaltype=0; % (vector indicating endogenous state is an incremental endogenous state variable)
    vfoptions.exoticpreferences='None';
    vfoptions.dynasty=0;
    vfoptions.experienceasset=0;
    vfoptions.experienceassetu=0;
    vfoptions.riskyasset=0;
    vfoptions.residualasset=0;
    vfoptions.n_ambiguity=0;
    % When calling as a subcommand, the following is used internally
    vfoptions.alreadygridvals=0;
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
    if ~isfield(vfoptions,'divideandconquer')
        vfoptions.divideandconquer=0; % =1 Use divide-and-conquer to exploit monotonicity
    end
    if ~isfield(vfoptions,'gridinterplayer')
        vfoptions.gridinterplayer=0; % =1 Interpolate between grid points (not yet implemented for most cases)
    elseif vfoptions.gridinterplayer==1
        if ~isfield(vfoptions,'ngridinterp')
            error('When using vfoptions.gridinterplayer=1 you must set vfoptoins.ngridinterp (number of points to interpolate for aprime between each consecutive pair of points in a_grid)')
        end
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
    if ~isfield(vfoptions,'outputkron')
        vfoptions.outputkron=0; % If 1 then leave output in Kron form
    end
    if ~isfield(vfoptions,'policy_forceintegertype')
        vfoptions.policy_forceintegertype=0;
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
    if ~isfield(vfoptions,'alreadygridvals')
        % When calling as a subcommand, the following is used internally
        vfoptions.alreadygridvals=0;
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
if isa(z_grid,'function_handle') || isfield(vfoptions,'ExogShockFn')
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
if isa(z_grid,'function_handle') || isfield(vfoptions,'ExogShockFn')
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


if isfield(vfoptions,'n_e') && ~isfield(vfoptions,'EiidShockFn')
    if isfield(vfoptions,'e_grid_J')
        error('No longer use vfoptions.e_grid_J, instead just put the age-dependent grid in vfoptions.e_grid (functionality of VFI Toolkit has changed to make it easier to use)')
    elseif ~isfield(vfoptions,'e_grid') 
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

if vfoptions.parallel<2
    if isfield(vfoptions,'n_e')
        error('Sorry but e (i.i.d) variables are not implemented for cpu, you will need a gpu to use them')
    end
    if isfield(vfoptions,'SemiExoStateFn')
        error('Sorry but Semi-Exogenous states are not implemented for cpu, you will need a gpu to use them')
    end
    if ~vfoptions.divideandconquer==0
        error('Sorry but divideandconquer is not implemented for cpu, you will need a gpu to use this algorithm')
    end
    if ~strcmp(vfoptions.exoticpreferences,'None')
        error('Sorry but exoticpreferences are not implemented for cpu, you will need a gpu to use them')
    end
    if ~vfoptions.experienceasset==0
        error('Sorry but experienceasset are not implemented for cpu, you will need a gpu to use them')
    end
    if ~vfoptions.experienceassetu==0
        error('Sorry but experienceassetu are not implemented for cpu, you will need a gpu to use them')
    end
    if ~vfoptions.riskyasset==0
        error('Sorry but riskyasset are not implemented for cpu, you will need a gpu to use them')
    end
    if ~vfoptions.residualasset==0
        error('Sorry but residualasset are not implemented for cpu, you will need a gpu to use them')
    end
    if ~vfoptions.dynasty==0
        error('Sorry but dynasty are not implemented for cpu, you will need a gpu to use them')
    end
end


%% 
if vfoptions.parallel==2 
   % If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
   d_grid=gpuArray(d_grid);
   a_grid=gpuArray(a_grid);
   z_grid=gpuArray(z_grid);
   pi_z=gpuArray(pi_z);
else
   % If using CPU make sure all the relevant inputs are CPU arrays (not standard arrays)
   % This may be completely unnecessary, no-one with a GPU would be using CPU.
   d_grid=gather(d_grid);
   a_grid=gather(a_grid);
   z_grid=gather(z_grid);
   pi_z=gather(pi_z);
end


%% Exogenous shock gridvals and pi
if vfoptions.alreadygridvals==0
    % Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
    [z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3);
    % output: z_gridvals_J, pi_z_J, vfoptions.e_gridvals_J, vfoptions.pi_e_J
    %
    % size(z_gridvals_J)=[prod(n_z),length(n_z),N_j]
    % size(pi_z_J)=[prod(n_z),prod(n_z),N_j]
    % size(e_gridvals_J)=[prod(n_e),length(n_e),N_j]
    % size(pi_e_J)=[prod(n_e),N_j]
    % If no z, then z_gridvals_J=[] and pi_z_J=[]
    % If no e, then e_gridvals_J=[] and pi_e_J=[]
elseif vfoptions.alreadygridvals==1
    z_gridvals_J=z_grid;
    pi_z_J=pi_z;
end


%% Semi-exogenous shock gridvals and pi 
if isfield(vfoptions,'n_semiz')
    % Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
    vfoptions=SemiExogShockSetup_FHorz(n_d,N_j,d_grid,Parameters,vfoptions,2);
    % output: vfoptions.semiz_gridvals_J, vfoptions.pi_semiz_J
    % size(semiz_gridvals_J)=[prod(n_z),length(n_z),N_j]
    % size(pi_semiz_J)=[prod(n_semiz),prod(n_semiz),prod(n_dsemiz),N_j]
    % If no semiz, then vfoptions just does not contain these field
end

%% Implement new way of handling ReturnFn inputs
if isempty(ReturnFnParamNames)
    ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);
end
% Basic setup: the first inputs of ReturnFn will be (d,aprime,a,z,..) and everything after this is a parameter, so we get the names of all these parameters.
% But this changes if you have e, semiz, or just multiple d, and if you use riskyasset, expasset, etc.
% So figure out which setup we have, and get the relevant ReturnFnParamNames


%% Implement new way of handling warm-glow of bequests (currently only used by Epstein-Zin preferences)
if isfield(vfoptions,'WarmGlowBequestsFn')
    l_a=length(n_a);
    temp=getAnonymousFnInputNames(vfoptions.WarmGlowBequestsFn);
    vfoptions.WarmGlowBequestsFnParamsNames={temp{l_a+1:end}}; % the first inputs will always be aprime
end
% clear l_d l_a l_z l_e % These are all messed up so make sure they are not reused later


%% Print out vfoptions (if vfoptions.verbose=1)
if vfoptions.verbose==1
    vfoptions
end

%% Deal with Exotic preferences if need to do that.
if strcmp(vfoptions.exoticpreferences,'None')
    % Just ignore and will then continue on.
elseif strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
    [V, Policy,Valt]=ValueFnIter_Case1_FHorz_QuasiHyperbolic(n_d,n_a,n_z,N_j,d_grid,a_grid,z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V, Policy,Valt};
    return
elseif strcmp(vfoptions.exoticpreferences,'EpsteinZin') && vfoptions.riskyasset==0 % deal with risky asset elsewhere
    [V, Policy]=ValueFnIter_Case1_FHorz_EpsteinZin(n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V, Policy};
    return
elseif strcmp(vfoptions.exoticpreferences,'GulPesendorfer')
    [V, Policy]=ValueFnIter_Case1_FHorz_GulPesendorfer(n_d,n_a,n_z,N_j,d_grid,a_grid,z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V, Policy};
    return
elseif strcmp(vfoptions.exoticpreferences,'AmbiguityAversion')
    [V, Policy]=ValueFnIter_Case1_FHorz_Ambiguity(n_d,n_a,n_z,N_j,d_grid,a_grid,z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V, Policy};
    return
end


%% Deal with Experience Asset if need to do that
% experienceasset: aprime(d,a)
% experienceassetu: aprime(d,a,u)
% experienceassetz: aprime(d,a,z)
% experienceassetze: aprime(d,a,z,e)

if vfoptions.experienceasset==1 || vfoptions.experienceassetu==1
    % It is simply assumed that the experience asset is the last asset, and that the decision that influences it is the last decision.
    
    if isfield(vfoptions,'SemiExoStateFn')
        % Split decision variables (other, semiexo, experienceasset)
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

    else % no semiz
        % Split decision variables into the standard ones and the one relevant to the experience asset
        if length(n_d)==1
            n_d1=0;
        else
            n_d1=n_d(1:end-1);
        end
        n_d2=n_d(end); % n_d2 is the decision variable that influences next period vale of the experience asset
        d1_grid=d_grid(1:sum(n_d1));
        d2_grid=d_grid(sum(n_d1)+1:end);
        % Split endogenous assets into the standard ones and the experience asset
        if length(n_a)==1
            n_a1=0;
        else
            n_a1=n_a(1:end-1);
        end
        n_a2=n_a(end); % n_a2 is the experience asset
        a1_grid=a_grid(1:sum(n_a1));
        a2_grid=a_grid(sum(n_a1)+1:end);
    end

    % Now just send all this to the right value fn iteration command
    if vfoptions.experienceasset==1
        if isfield(vfoptions,'n_semiz')
            [V,Policy]=ValueFnIter_Case1_FHorz_ExpAssetSemiExo(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,vfoptions.n_semiz, N_j, d1_grid , d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.semiz_gridvals_J, pi_z_J, vfoptions.pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [V,Policy]=ValueFnIter_FHorz_ExpAsset(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid , d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    elseif vfoptions.experienceassetu==1
        [V,Policy]=ValueFnIter_Case1_FHorz_ExpAssetu(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid , d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    elseif vfoptions.experienceassetz==1
        % I want to implement this too :)
    elseif vfoptions.experienceassetze==1
        % I want to implement this too :)
    end

    varargout={V, Policy};
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
    if isfield(vfoptions,'n_semiz')
        if strcmp(vfoptions.exoticpreferences,'EpsteinZin')
            [V, Policy]=ValueFnIter_FHorz_EpsteinZin_RiskyAsset_semiz(n_d,n_a1,n_a2,vfoptions.n_semiz,n_z,vfoptions.n_u, N_j, d_grid, a1_grid, a2_grid, vfoptions.semiz_gridvals_J,z_gridvals_J, vfoptions.u_grid, vfoptions.pi_semiz_J, pi_z_J, vfoptions.pi_u, ReturnFn, vfoptions.aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [V, Policy]=ValueFnIter_FHorz_RiskyAsset_semiz(n_d,n_a1,n_a2,vfoptions.n_semiz,n_z,vfoptions.n_u, N_j, d_grid, a1_grid, a2_grid, vfoptions.semiz_gridvals_J,z_gridvals_J, vfoptions.u_grid, vfoptions.pi_semiz_J, pi_z_J, vfoptions.pi_u, ReturnFn, vfoptions.aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else
        if strcmp(vfoptions.exoticpreferences,'EpsteinZin')
            [V, Policy]=ValueFnIter_Case1_FHorz_EpsteinZin_RiskyAsset(n_d,n_a1,n_a2,n_z,vfoptions.n_u,N_j,d_grid,a1_grid, a2_grid, z_gridvals_J, vfoptions.u_grid, pi_z_J, vfoptions.pi_u, ReturnFn, vfoptions.aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [V,Policy]=ValueFnIter_Case1_FHorz_RiskyAsset(n_d,n_a1,n_a2,n_z,vfoptions.n_u, N_j, d_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.u_grid, pi_z_J, vfoptions.pi_u, ReturnFn, vfoptions.aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end

    varargout={V, Policy};
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
    varargout={V, Policy};
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

    varargout={V, Policy};
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

    varargout={V, Policy};
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
    
    varargout={V, Policy};
    return
end

%% Semi-exogenous state
% The transition matrix of the exogenous shocks depends on the value of the 'last' decision variable(s).
if isfield(vfoptions,'SemiExoStateFn')
    if length(n_d)>vfoptions.l_dsemiz
        n_d1=n_d(1:end-vfoptions.l_dsemiz);
        d1_grid=d_grid(1:sum(n_d1));
    else
        n_d1=0; d1_grid=[];
    end
    n_d2=n_d(end-vfoptions.l_dsemiz+1:end); % n_d2 is the decision variable that influences the transition probabilities of the semi-exogenous state
    d2_grid=d_grid(sum(n_d1)+1:end);
    
    % Now that we have pi_semiz_J we are ready to compute the value function.
    [V,Policy]=ValueFnIter_FHorz_SemiExo(n_d1,n_d2,n_a,vfoptions.n_semiz,n_z,N_j,d1_grid,d2_grid, a_grid, z_gridvals_J, vfoptions.semiz_gridvals_J, pi_z_J, vfoptions.pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V, Policy};
    return
end

%% Just do the standard case
if vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==1
    % Solve by doing Divide-and-Conquer, and then a grid interpolation layer
    [V,Policy]=ValueFnIter_FHorz_DC_GI(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V, Policy};
    return
elseif vfoptions.divideandconquer==1
    % Solve using Divide-and-Conquer algorithm
    [V,Policy]=ValueFnIter_FHorz_DC(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V, Policy};
    return
elseif vfoptions.gridinterplayer==1
    % Solve using grid interpolation layer
    [V,Policy]=ValueFnIter_FHorz_GI(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V, Policy};
    return
end

if vfoptions.parallel==2 % GPU
    if N_d==0
        if isfield(vfoptions,'n_e')
            if N_z==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron,PolicyKron]=ValueFnIter_FHorz_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron,PolicyKron]=ValueFnIter_FHorz_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    else % N_d
        if isfield(vfoptions,'n_e')
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_noz_e_raw(n_d,n_a,  vfoptions.n_e, N_j, d_grid, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_e_raw(n_d,n_a,n_z,  vfoptions.n_e, N_j, d_grid, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_noz_raw(n_d,n_a, N_j, d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end
elseif vfoptions.parallel==1 % parallel CPU
    if N_d==0
        if N_z==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_Par1_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_Par1_nod_raw(n_a, n_z, N_j, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else % N_d
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_Par1_noz_raw(n_d,n_a, N_j, d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_Par1_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
elseif vfoptions.parallel==0 % just one CPU
    if N_d==0
        [VKron,PolicyKron]=ValueFnIter_FHorz_Par0_nod_raw(n_a, n_z, N_j, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    else
        [VKron, PolicyKron]=ValueFnIter_FHorz_Par0_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
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
    % First, give some output on the size of any changes in Policy as a result of turning the values into integers
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
elseif vfoptions.policy_forceintegertype==2
    % Do the actual rounding to integers
    Policy=round(Policy);
end

varargout={V, Policy};

end
