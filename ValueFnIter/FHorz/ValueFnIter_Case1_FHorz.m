function varargout=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Typically, varargout=[V, Policy]

if isUnderlyingType(a_grid,'single')
    V=single(nan);
    Policy=single(nan);
else
    V=nan;
    Policy=nan;
end


%% Check which vfoptions have been used, set all others to defaults
if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    % If vfoptions is not given, just use all the defaults
    vfoptions.verbose=0; % =1 print out feedback on what is happening internally
    vfoptions.divideandconquer=0; % =1 Use divide-and-conquer to exploit monotonicity
    vfoptions.gridinterplayer=0; % Interpolate between grid points (not yet implemented for alternative preferences)
    vfoptions.lowmemory=0; % use more loops and less parallelization, reduce memory use but at the cost of slower runtimes
    % Alternative model setups
    vfoptions.incrementaltype=0; % (vector indicating endogenous state is an incremental endogenous state variable)
    vfoptions.exoticpreferences='None';
    vfoptions.dynasty=0;
    vfoptions.experienceasset=0;
    vfoptions.experienceassetu=0;
    vfoptions.experienceassete=0;
    vfoptions.experienceassetz=0;
    vfoptions.experienceassetze=0;
    vfoptions.riskyasset=0;
    vfoptions.residualasset=0;
    vfoptions.n_ambiguity=0;
    vfoptions.n_e=0;
    vfoptions.n_semiz=0;
    % Largely just for internal use only
    vfoptions.parallel=1+(gpuDeviceCount>0);
    % When calling as a subcommand, the following are used internally
    vfoptions.outputkron=0; % If 1 then leave output in Kron form
    vfoptions.alreadygridvals=0; % =1 when calling as a subcommand
    vfoptions.alreadygridvals_semiexo=0; % =1 when calling as a subcommand
else
    % Check vfoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(vfoptions,'verbose')
        vfoptions.verbose=0;
    end
    if ~isfield(vfoptions,'divideandconquer')
        vfoptions.divideandconquer=0; % =1 Use divide-and-conquer to exploit monotonicity
    end
    if ~isfield(vfoptions,'gridinterplayer')
        vfoptions.gridinterplayer=0; % =1 Interpolate between grid points (not yet implemented for most cases)
    elseif vfoptions.gridinterplayer==1
        if ~isfield(vfoptions,'ngridinterp')
            error('When using vfoptions.gridinterplayer=1 you must set vfoptions.ngridinterp (number of points to interpolate for aprime between each consecutive pair of points in a_grid)')
        end
    end
    if ~isfield(vfoptions,'lowmemory')
        vfoptions.lowmemory=0;
    end
    % Alternative model setups
    if ~isfield(vfoptions,'incrementaltype')
        vfoptions.incrementaltype=0; % (vector indicating endogenous state is an incremental endogenous state variable)
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
    if ~isfield(vfoptions,'experienceassete')
        vfoptions.experienceassete=0;
    end
    if ~isfield(vfoptions,'experienceassetz')
        vfoptions.experienceassetz=0;
    end
    if ~isfield(vfoptions,'experienceassetze')
        vfoptions.experienceassetze=0;
    end
    if ~isfield(vfoptions,'riskyasset')
        vfoptions.riskyasset=0;
    end
    if ~isfield(vfoptions,'residualasset')
        vfoptions.residualasset=0;
    end
    if ~isfield(vfoptions,'n_ambiguity')
        vfoptions.n_ambiguity=0;
    end
    if ~isfield(vfoptions,'n_e')
        vfoptions.n_e=0;
    end
    if ~isfield(vfoptions,'n_semiz')
        vfoptions.n_semiz=0;
    end
    % Largely just for internal use only
    if ~isfield(vfoptions,'parallel')
        vfoptions.parallel=1+(gpuDeviceCount>0);
    end
    % When calling as a subcommand, the following are used internally
    if ~isfield(vfoptions,'outputkron')
        vfoptions.outputkron=0; % If 1 then leave output in Kron form
    end
    if ~isfield(vfoptions,'alreadygridvals')
        vfoptions.alreadygridvals=0; % =1 when calling as a subcommand
    end
    if ~isfield(vfoptions,'alreadygridvals_semiexo')
        vfoptions.alreadygridvals_semiexo=0; % =1 when calling as a subcommand
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
N_e=prod(vfoptions.n_e);

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

%% z_grid/pi_z/e_grid/pi_e shape validation is performed inside ExogShockSetup_FHorz (called below).

if vfoptions.parallel<2
    if N_e>0
        error('Sorry but e (i.i.d) variables are not implemented for cpu, you will need a gpu to use them')
    end
    if prod(vfoptions.n_semiz)>0
        error('Sorry but Semi-Exogenous states are not implemented for cpu, you will need a gpu to use them')
    end
    if ~vfoptions.divideandconquer==0
        error('Sorry but divideandconquer is not implemented for cpu, you will need a gpu to use this algorithm')
    end
    if ~strcmp(vfoptions.exoticpreferences,'None')
        error('Sorry but exoticpreferences are not implemented for cpu, you will need a gpu to use them')
    end
    if ~vfoptions.experienceasset==0 || ~vfoptions.experienceassetu==0  || ~vfoptions.experienceassetz==0  || ~vfoptions.experienceassete==0  || ~vfoptions.experienceassetze==0
        error('Sorry but experience assets are not implemented for cpu, you will need a gpu to use them')
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
    if size(d_grid,2)==1
        d_gridvals=CreateGridvals(n_d,d_grid,1);
    else % already d_gridvals
        d_gridvals=d_grid;
    end
else
   % CPU can be used, but only for the basics. Is kept separate here so that the rest of the codes can just assume you have GPU and work with it.
   [V,Policy]=ValueFnIter_FHorz_CPU(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
   varargout={V,Policy};
   return
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
    %
    % Check pi_z_J: that all elements are positive, and that the rows sum to one
    if N_z>0
        if ~all(pi_z_J(:)>=0)
            warning('Some elements of pi_z_J, the exogenous markov transition probabilities matrix, are negative')
        end
        if isUnderlyingType(a_grid,'single')
            if ~all(all(squeeze(sum(pi_z_J,2))-1 <1e-6)) % sum of row must be within 1e-6 of 1
                warning('Some rows of pi_z_J, the exogenous markov transition probabilities matrix, do not sum to one')
            end
        else
            if ~all(all(squeeze(sum(pi_z_J,2))-1 <1e-13)) % sum of row must be within 1e-13 of 1
                warning('Some rows of pi_z_J, the exogenous markov transition probabilities matrix, do not sum to one')
            end
        end
    end
    if isfield(vfoptions,'pi_e_J')
        if ~all(vfoptions.pi_e_J(:)>=0)
            warning('Some elements of pi_e_J, the exogenous i.i.d. probabilities matrix, are negative')
        end
        if isUnderlyingType(a_grid,'single')
            if ~all(squeeze(sum(vfoptions.pi_e_J,1))-1 <1e-6) % sum of row must be within 1e-6 of 1
                warning('Some columns of pi_e_J, the exogenous i.i.d. probabilities matrix, do not sum to one')
            end
        else
            if ~all(squeeze(sum(vfoptions.pi_e_J,1))-1 <1e-13) % sum of row must be within 1e-13 of 1
                warning('Some columns of pi_e_J, the exogenous i.i.d. probabilities matrix, do not sum to one')
            end
        end
    end
elseif vfoptions.alreadygridvals==1
    z_gridvals_J=z_grid;
    pi_z_J=pi_z;
end


%% Semi-exogenous shock gridvals and pi
if vfoptions.alreadygridvals_semiexo==0
    if prod(vfoptions.n_semiz)>0
        % Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
        vfoptions=SemiExogShockSetup_FHorz(n_d,N_j,d_grid,Parameters,vfoptions,3);
        % output: vfoptions.semiz_gridvals_J, vfoptions.pi_semiz_J
        % size(semiz_gridvals_J)=[prod(n_z),length(n_z),N_j]
        % size(pi_semiz_J)=[prod(n_semiz),prod(n_semiz),prod(n_dsemiz),N_j]
        % If no semiz, then vfoptions just does not contain these field
    end
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
if vfoptions.verbose>=1
    vfoptions
end

%% Deal with Exotic preferences if need to do that.
if strcmp(vfoptions.exoticpreferences,'None')
    % Just ignore and will then continue on.
elseif strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic') && vfoptions.experienceasset==0 && vfoptions.experienceassetu==0 && vfoptions.experienceassetz==0 && vfoptions.experienceassete==0 && vfoptions.experienceassetze==0
    % QH composed with experienceasset variants is handled in the experience-asset
    % block below, so it can reuse the n_d / n_a splitting there.
    [V,Policy, Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolic(n_d,n_a,n_z,N_j,d_grid,a_grid,z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    if strcmp(vfoptions.quasi_hyperbolic,'Naive')
        varargout={V, Policy,Valt,Policyalt}; % Vtilde, Policytilde, V, Policy (the last two are the exponential discounter)
    elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
        varargout={V, Policy,Valt}; % Vhat, Policyhat, Vunderbar (the last is the exponential discounter V from the Policyhat)
    end
    return
elseif strcmp(vfoptions.exoticpreferences,'EpsteinZin') && vfoptions.riskyasset==0 % deal with risky asset elsewhere
    [V, Policy]=ValueFnIter_FHorz_EpsteinZin(n_d,n_a,n_z,N_j,d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V, Policy};
    return
elseif strcmp(vfoptions.exoticpreferences,'GulPesendorfer')
    [V, Policy]=ValueFnIter_FHorz_GulPesendorfer(n_d,n_a,n_z,N_j,d_gridvals,a_grid,z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V, Policy};
    return
elseif strcmp(vfoptions.exoticpreferences,'AmbiguityAversion')
    [V, Policy]=ValueFnIter_FHorz_Ambiguity(n_d,n_a,n_z,N_j,d_gridvals,a_grid,z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V, Policy};
    return
end


%% Deal with Experience Asset if need to do that
% experienceasset: aprime(d,a)
% experienceassetu: aprime(d,a,u)
% experienceassetz: aprime(d,a,z)
% experienceassete: aprime(d,a,e)
% experienceassetze: aprime(d,a,z,e)

if vfoptions.experienceasset==1 || vfoptions.experienceassetu==1 || vfoptions.experienceassetz==1 || vfoptions.experienceassete==1 || vfoptions.experienceassetze==1
    % It is simply assumed that the experience asset is the last asset, and that the decision that influences it is the last decision.
    % When using both semiexo and experience asset, the last decision variable influences semi-exo and the second last decision variable influences the experience asset

    if vfoptions.experienceasset==1
        if ~isfield(vfoptions,'l_dexperienceasset')
            vfoptions.l_dexperienceasset=1; % by default, only one decision variable influences the experienceasset
        end
    elseif vfoptions.experienceassetu==1
        if ~isfield(vfoptions,'l_dexperienceassetu')
            vfoptions.l_dexperienceassetu=1; % by default, only one decision variable influences the experienceassetu
        end
    elseif vfoptions.experienceassete==1
        if ~isfield(vfoptions,'l_dexperienceassete')
            vfoptions.l_dexperienceassete=1; % by default, only one decision variable influences the experienceassete
        end
    elseif vfoptions.experienceassetz==1
        if ~isfield(vfoptions,'l_dexperienceassetz')
            vfoptions.l_dexperienceassetz=1; % by default, only one decision variable influences the experienceassetz
        end
    elseif vfoptions.experienceassetze==1
        if ~isfield(vfoptions,'l_dexperienceassetze')
            vfoptions.l_dexperienceassetze=1; % by default, only one decision variable influences the experienceassetze
        end
    end

    if vfoptions.experienceasset==1
        vfoptions.l_d2=vfoptions.l_dexperienceasset;
    elseif vfoptions.experienceassetu==1
        vfoptions.l_d2=vfoptions.l_dexperienceassetu;
    elseif vfoptions.experienceassete==1
        vfoptions.l_d2=vfoptions.l_dexperienceassete;
    elseif vfoptions.experienceassetz==1
        vfoptions.l_d2=vfoptions.l_dexperienceassetz;
    elseif vfoptions.experienceassetze==1
        vfoptions.l_d2=vfoptions.l_dexperienceassetze;
    end

    if prod(vfoptions.n_semiz)>0
        if ~isfield(vfoptions,'l_dsemiz')
            vfoptions.l_dsemiz=1; % by default, only one decision variable influences the semi-exogenous state
        end

        % Split decision variables (other, semiexo, experienceasset)
        if length(n_d)>(vfoptions.l_d2+vfoptions.l_dsemiz)
            n_d1=n_d(1:end-vfoptions.l_d2-vfoptions.l_dsemiz);
        else
            n_d1=0;
        end
        n_d2=n_d(end-vfoptions.l_d2-vfoptions.l_dsemiz+1:end-vfoptions.l_dsemiz); % n_d2 is the decision variable that influences the experience asset
        n_d3=n_d(end-vfoptions.l_dsemiz+1:end); % n_d3 is the decision variable that influences the transition probabilities of the semi-exogenous state
        d1_grid=d_grid(1:sum(n_d1));
        d2_grid=d_grid(sum(n_d1)+1:sum(n_d1)+sum(n_d2));
        d3_grid=d_grid(sum(n_d1)+sum(n_d2)+1:end);
        % Split endogenous assets into the standard ones and the experience asset
        if isscalar(n_a)
            n_a1=0;
        else
            n_a1=n_a(1:end-1);
        end
        n_a2=n_a(end); % n_a2 is the experience asset
        a1_grid=a_grid(1:sum(n_a1));
        a2_grid=a_grid(sum(n_a1)+1:end);

    else % no semiz
        % Split decision variables into the standard ones and the one relevant to the experience asset
        if length(n_d)>vfoptions.l_d2
            n_d1=n_d(1:end-vfoptions.l_d2);
        else
            n_d1=0;
        end
        n_d2=n_d(end-vfoptions.l_d2+1:end); % n_d2 is the decision variable that influences next period vale of the experience asset
        d1_grid=d_grid(1:sum(n_d1));
        d2_grid=d_grid(sum(n_d1)+1:end);
        % Split endogenous assets into the standard ones and the experience asset
        if isscalar(n_a)
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
        if prod(vfoptions.n_semiz)>0
            [V,Policy]=ValueFnIter_FHorz_ExpAssetSemiExo(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,vfoptions.n_semiz, N_j, d1_grid , d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.semiz_gridvals_J, pi_z_J, vfoptions.pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [V,Policy]=ValueFnIter_FHorz_ExpAsset(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid , d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    elseif vfoptions.experienceassetu==1
        if prod(vfoptions.n_semiz)>0
            [V,Policy]=ValueFnIter_FHorz_ExpAssetuSemiExo(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,vfoptions.n_semiz, N_j, d1_grid , d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.semiz_gridvals_J, pi_z_J, vfoptions.pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [V,Policy]=ValueFnIter_FHorz_ExpAssetu(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid , d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    elseif vfoptions.experienceassete==1
        if prod(vfoptions.n_semiz)>0
            [V,Policy]=ValueFnIter_FHorz_ExpAsseteSemiExo(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,vfoptions.n_semiz, N_j, d1_grid , d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.semiz_gridvals_J, pi_z_J, vfoptions.pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [V,Policy]=ValueFnIter_FHorz_ExpAssete(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid , d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    elseif vfoptions.experienceassetz==1
        if prod(vfoptions.n_semiz)>0
            [V,Policy]=ValueFnIter_FHorz_ExpAssetzSemiExo(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,vfoptions.n_semiz, N_j, d1_grid , d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.semiz_gridvals_J, pi_z_J, vfoptions.pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        elseif strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
            [V,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetz(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid, d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            if strcmp(vfoptions.quasi_hyperbolic,'Naive')
                varargout={V, Policy, Valt, Policyalt};
            else
                varargout={V, Policy, Valt};
            end
            return
        else
            [V,Policy]=ValueFnIter_FHorz_ExpAssetz(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid , d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    elseif vfoptions.experienceassetze==1
        if prod(vfoptions.n_semiz)>0
            [V,Policy]=ValueFnIter_FHorz_ExpAssetzeSemiExo(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,vfoptions.n_semiz, N_j, d1_grid , d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.semiz_gridvals_J, pi_z_J, vfoptions.pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [V,Policy]=ValueFnIter_FHorz_ExpAssetze(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid , d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end

    varargout={V, Policy};
    return
end


%% Deal with risky asset if need to do that
if vfoptions.riskyasset==1
    % It is simply assumed that the risky asset is the last asset, and that all decisions influence it.

    % Split endogenous assets into the standard ones and the risky asset
    if isscalar(n_a)
        n_a1=0;
    else
        n_a1=n_a(1:end-1);
    end
    n_a2=n_a(end); % n_a2 is the risky asset
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
    if ~isfield(vfoptions,'refine_d')
        warning('Using vfoptions.riskyasset=1 without setting vfoptions.refine_d is outdated behaviour, it is strongly recommended you set vfoptions.refine_d')
    end

    % Now just send all this to the right value fn iteration command
    if prod(vfoptions.n_semiz)>0
        if strcmp(vfoptions.exoticpreferences,'EpsteinZin')
            [V, Policy]=ValueFnIter_FHorz_EpsteinZin_RiskyAsset_semiz(n_d,n_a1,n_a2,vfoptions.n_semiz,n_z,vfoptions.n_u, N_j, d_grid, a1_grid, a2_grid, vfoptions.semiz_gridvals_J,z_gridvals_J, vfoptions.u_grid, vfoptions.pi_semiz_J, pi_z_J, vfoptions.pi_u, ReturnFn, vfoptions.aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [V, Policy]=ValueFnIter_FHorz_RiskyAssetSemiExo(n_d,n_a1,n_a2,vfoptions.n_semiz,n_z,vfoptions.n_u, N_j, d_grid, a1_grid, a2_grid, vfoptions.semiz_gridvals_J,z_gridvals_J, vfoptions.u_grid, vfoptions.pi_semiz_J, pi_z_J, vfoptions.pi_u, ReturnFn, vfoptions.aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else
        if strcmp(vfoptions.exoticpreferences,'EpsteinZin')
            [V, Policy]=ValueFnIter_FHorz_EpsteinZin_RiskyAsset(n_d,n_a1,n_a2,n_z,vfoptions.n_u,N_j,d_grid,a1_grid, a2_grid, z_gridvals_J, vfoptions.u_grid, pi_z_J, vfoptions.pi_u, ReturnFn, vfoptions.aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [V,Policy]=ValueFnIter_FHorz_RiskyAsset(n_d,n_a1,n_a2,n_z,vfoptions.n_u, N_j, d_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.u_grid, pi_z_J, vfoptions.pi_u, ReturnFn, vfoptions.aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end

    varargout={V, Policy};
    return
end

%% Deal with residual asset if need to do that
if vfoptions.residualasset==1
    % Split endogenous assets into the standard ones and the residual asset
    if isscalar(n_a)
        n_a1=0;
    else
        n_a1=n_a(1:end-1);
    end
    n_r=n_a(end); % n_a2 is the residual asset
    a1_grid=a_grid(1:sum(n_a1));
    r_grid=a_grid(sum(n_a1)+1:end);

    % Now just send all this to the right value fn iteration command
    [V,Policy]=ValueFnIter_FHorz_ResidAsset(n_d,n_a1,n_r,n_z, N_j, d_grid, a1_grid, r_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V, Policy};
    return
end

%% Deal with StateDependentVariables_z if need to do that.
if isfield(vfoptions,'StateDependentVariables_z')==1
    if vfoptions.verbose==1
        fprintf('StateDependentVariables_z option is being used \n')
    end

    if N_d==0
        [VKron,PolicyKron]=ValueFnIter_FHorz_nod_SDVz_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        % Policy without d
        PolicyKron=shiftdim(PolicyKron,-1);
    else
        [VKron, PolicyKron]=ValueFnIter_FHorz_SDVz_raw(n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    end

    if vfoptions.outputkron==1
        varargout={VKron, PolicyKron};
        return
    end

    if N_d==0
        n_daprime=n_a;
    else
        n_daprime=[n_d,n_a];
    end

    V=reshape(VKron,[n_a,n_z,N_j]);
    Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_daprime,n_a,n_z,N_j,vfoptions);
    varargout={V, Policy};
    return
end

%% Deal with dynasty if need to do that.
if vfoptions.dynasty==1
    if vfoptions.verbose==1
        fprintf('dynasty option is being used \n')
    end
    if ~isfield(vfoptions,'tolerance')
        vfoptions.tolerance=10^(-9);
    end

    if N_d==0
        [VKron,PolicyKron]=ValueFnIter_FHorz_Dynasty_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        % Policy without d
        PolicyKron=shiftdim(PolicyKron,-1);
    else
        [VKron, PolicyKron]=ValueFnIter_FHorz_Dynasty_raw(n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    end

    if vfoptions.outputkron==1
        varargout={VKron, PolicyKron};
        return
    end

    if N_d==0
        n_daprime=n_a;
    else
        n_daprime=[n_d,n_a];
    end

    V=reshape(VKron,[n_a,n_z,N_j]);
    Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_daprime,n_a,n_z,N_j,vfoptions);
    varargout={V, Policy};
    return
end



%% Semi-exogenous state
% The transition matrix of the exogenous shocks depends on the value of the 'last' decision variable(s).
if prod(vfoptions.n_semiz)>0
    if length(n_d)>vfoptions.l_dsemiz
        n_d1=n_d(1:end-vfoptions.l_dsemiz);
        d1_grid=d_grid(1:sum(n_d1));
    else
        n_d1=0; d1_grid=[];
    end
    n_d2=n_d(end-vfoptions.l_dsemiz+1:end); % n_d2 is the decision variable that influences the transition probabilities of the semi-exogenous state
    d2_grid=d_grid(sum(n_d1)+1:end);

    d1_gridvals=CreateGridvals(n_d1,d1_grid,1);
    d2_gridvals=CreateGridvals(n_d2,d2_grid,1);

    % Now that we have pi_semiz_J we are ready to compute the value function.
    [V,Policy]=ValueFnIter_FHorz_SemiExo(n_d1,n_d2,n_a,vfoptions.n_semiz,n_z,N_j,d1_gridvals,d2_gridvals, a_grid, z_gridvals_J, vfoptions.semiz_gridvals_J, pi_z_J, vfoptions.pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V, Policy};
    return
end

%% Just do the standard case
if vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==1
    % Solve by doing Divide-and-Conquer, and then a grid interpolation layer
    [V,Policy]=ValueFnIter_FHorz_DC_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V, Policy};
    return
elseif vfoptions.divideandconquer==1
    % Solve using Divide-and-Conquer algorithm
    [V,Policy]=ValueFnIter_FHorz_DC(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V, Policy};
    return
elseif vfoptions.gridinterplayer==1
    % Solve using grid interpolation layer
    [V,Policy]=ValueFnIter_FHorz_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V, Policy};
    return
end

% vfoptions.parallel==2 % GPU
if N_e==0
    if N_z==0
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_noz_raw(n_d,n_a, N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_raw(n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
else % N_e
    if N_z==0
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_noz_e_raw(n_d,n_a,  vfoptions.n_e, N_j, d_gridvals, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_e_raw(n_d,n_a,n_z,  vfoptions.n_e, N_j, d_gridvals, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
end


%% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if vfoptions.outputkron==1
    varargout={VKron, PolicyKron};
    return
end

if N_d==0
    n_daprime=n_a;
else
    n_daprime=[n_d,n_a];
end

if N_e==0
    if N_z==0
        V=reshape(VKron,[n_a,N_j]);
        Policy=UnKronPolicyIndexes1_FHorz_noz(PolicyKron,n_daprime,n_a,N_j,vfoptions);
    else
        V=reshape(VKron,[n_a,n_z,N_j]);
        Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_daprime,n_a,n_z,N_j,vfoptions);
    end
else
    if N_z==0
        V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_daprime,n_a,vfoptions.n_e,N_j,vfoptions);  % Treat e as z (because no z)
    else
        V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron,n_daprime,n_a,n_z,vfoptions.n_e,N_j,vfoptions);
    end
end

varargout={V, Policy};

end
