function [VPath,PolicyPath]=ValueFnOnTransPath_InfHorz(PricePath, ParamPath, T, V_final, Policy_final, Parameters, n_d,n_a,n_z, d_grid,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, transpathoptions, vfoptions)
% transpathoptions, vfoptions and simoptions are optional inputs

if prod(n_z)>1 && all(size(d_grid)==[prod(n_z),prod(n_z)])
    error('Check input order: pi_z comes after z_grid') % Keep this error message until end of 2027, can remove after that
end

%% Check which transpathoptions have been used, set all others to defaults
if ~exist('transpathoptions','var')
    %If transpathoptions is not given, just use all the defaults
    transpathoptions.parallel=2;
    transpathoptions.verbose=0;
else
    %Check transpathoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(transpathoptions,'parallel')
        transpathoptions.parallel=2;
    end
    if ~isfield(transpathoptions,'verbose')
        transpathoptions.verbose=0;
    end
end

%% Check which vfoptions have been used, set all others to defaults
vfoptions.parallel=2; % GPU, has to be or transpath will already have thrown an error
if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
    vfoptions.verbose=0;
    vfoptions.lowmemory=0;
    % Model setup:
    vfoptions.exoticpreferences='None';
    vfoptions.experienceasset=0;
    % Algorithm to use:
    vfoptions.solnmethod='purediscretization'; % Currently this does nothing
    vfoptions.divideandconquer=0;
    vfoptions.gridinterplayer=0;
    vfoptions.localsearch=0;
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(vfoptions,'lowmemory')
        vfoptions.lowmemory=0;
    end
    if ~isfield(vfoptions,'verbose')
        vfoptions.verbose=0;
    end
    % Model setup:
    if ~isfield(vfoptions,'exoticpreferences')
        vfoptions.exoticpreferences='None';
    end
    if strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
        if ~isfield(vfoptions,'quasi_hyperbolic')
            vfoptions.quasi_hyperbolic='Naive'; % This is the default, alternative is 'Sophisticated'.
        elseif ~strcmp(vfoptions.quasi_hyperbolic,'Naive') && ~strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
            fprintf('ERROR: when using Quasi-Hyperbolic discounting vfoptions.quasi_hyperbolic must be either Naive or Sophisticated \n')
            dbstack
            return
        end
    end
    if ~isfield(vfoptions,'experienceasset')
        vfoptions.experienceasset=0;
    end
    % Algorithm to use:
    if ~isfield(vfoptions,'solnmethod')
        vfoptions.solnmethod='purediscretization'; % Currently this does nothing
    end
    if ~isfield(vfoptions,'divideandconquer')
        vfoptions.divideandconquer=0;
    end
    if ~isfield(vfoptions,'gridinterplayer')
        vfoptions.gridinterplayer=0;
    elseif vfoptions.gridinterplayer==1
        if ~isfield(vfoptions,'ngridinterp')
            error('When using vfoptions.gridinterplayer=1 you must set vfoptions.ngridinterp')
        end
    end
    if ~isfield(vfoptions,'localsearch')
        vfoptions.localsearch=0;
    end
    % LOCAL SEARCH IS IGNORED HERE AT vfoptions.localsearch=1, WHICH IS THE VALUE THE SOLVER USES.
    % This command produces the value and policy path that the welfare numbers, the agent
    % distribution and every downstream statistic are built from, and those should come from an
    % unrestricted search. In TransitionPath_InfHorz a restricted sweep is safe because a final
    % unrestricted sweep certifies the answer before it is accepted; a single backward pass has
    % nothing to certify against, so a restricted answer here would simply be trusted. The speed
    % on offer is one pass, against a solve that takes tens of them.
    % IT IS IGNORED RATHER THAN REFUSED because people reuse the vfoptions they gave the solver,
    % where localsearch=1 is exactly what they were told to set. Refusing would make every local
    % search user carry a second options struct and would buy nothing.
    % IT WARNS, unlike vfoptions.lowmemory which the restricted sweeps ignore in silence. Ignoring
    % lowmemory is the intended behaviour and costs nothing; here the user has asked for something
    % that would quietly degrade the numbers every statistic downstream rests on, so they are told.
    % localsearch=2 FORCES IT, for a caller who has seen the warning, accepts that this pass is
    % uncertified, and wants the speed on a big model -- and for the test bank, which has to be able
    % to exercise the restricted path through this command at all.
    if vfoptions.localsearch==1
        warning(['ValueFnOnTransPath_InfHorz: vfoptions.localsearch=1 is IGNORED here and the value ' ...
            'and policy path are computed with a full unrestricted search. This command has no ' ...
            'verification sweep to certify a restricted answer against, and its output is what the ' ...
            'welfare numbers and the agent distribution are built from. Set vfoptions.localsearch=2 ' ...
            'to force the restricted search anyway.'])
        vfoptions.localsearch=0;
    elseif vfoptions.localsearch==2
        vfoptions.localsearch=1; % forced, deliberately, by a caller who asked twice
    elseif vfoptions.localsearch>2
        error('vfoptions.localsearch must be 0, 1 (ignored here, with a warning) or 2 (forced)')
    end
    if vfoptions.localsearch==1
        if ~isfield(vfoptions,'nlocalsearch')
            error('When using vfoptions.localsearch=1 you must set vfoptions.nlocalsearch')
        end
        % nlocalsearch=0 would make the window a single grid point. Without the grid interpolation
        % layer that is a search over one aprime, and with it there is no second point to put the
        % ngridinterp points between, so the layer would do nothing at all. Neither is a solver.
        if vfoptions.nlocalsearch<1
            error('vfoptions.nlocalsearch must be at least 1')
        end
        % The experience asset branch below calls the ExpAsset single step directly, bypassing the
        % dispatcher, so without this the option would be silently ignored rather than refused
        if vfoptions.experienceasset>=1
            error('vfoptions.localsearch=1 is not implemented with an experience asset')
        end
        % The window slides at the grid ends to stay 2*nlocalsearch+1 points wide, so the grid
        % has to be at least that long
        if n_a(1)<2*vfoptions.nlocalsearch+1
            error('vfoptions.nlocalsearch is too large for the grid: need n_a>=2*nlocalsearch+1')
        end
    end
end

if vfoptions.divideandconquer==1
    if ~isfield(vfoptions,'level1n')
        if isscalar(n_a)
            vfoptions.level1n=floor(sqrt(n_a(1)));
            if n_a(1)<5
                error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
            end
        elseif length(n_a)>1
            vfoptions.level1n=[floor(sqrt(n_a(1))),n_a(2:end)]; % default DC2A: level1n(2)==n_a(2) triggers DC2A branch (everything after the first endogenous state is Kron'd together)
            if n_a(1)<5
                error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
            end
        end
        if vfoptions.verbose==1
            fprintf('Suggestion: When using vfoptions.divideandconquer it will be faster or slower if you set different values of vfoptions.level1n (for smaller models 7 or 9 is good, but for larger models something 15 or 21 can be better) \n')
        end
    end
    vfoptions.level1n(1)=min(vfoptions.level1n(1),n_a(1)); % Otherwise causes errors
end


%%
% Note: Internally PricePath is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'.
[PricePath,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec]=PricePathParamPath_StructToMatrix(PricePath,ParamPath,T);

%%
% The outputted VPath and PolicyPath are T-1 periods long (periods 0 (before the reforms are announced) & T are the initial and final values; they are not created by this command and instead can be used to provide double-checks of the output (the T-1 and the final should be identical if convergence has occurred).
% if n_d(1)==0
%     PolicyPath=zeros([length(n_d),n_a,n_z,N_j,T-1],'gpuArray'); %Periods 1 to T-1
% else
%     PolicyPath=zeros([length(n_d)+length(n_a),n_a,n_z,N_j,T-1],'gpuArray'); %Periods 1 to T-1
% end
% VPath=zeros([n_a,n_z,N_j,T-1],'gpuArray'); %Periods 1 to T-1

% PricePath is matrix of size T-by-'number of prices'
% ParamPath is matrix of size T-by-'number of parameters that change over path'

% Remark to self: No real need for T as input, as this is anyway the length of PricePathOld

% AgeWeightsParamNames are not actually needed as an input, but require
% them anyway to make it easier to 'copy-paste' input lists from other
% similar functions the user is likely to be using.

%% iid e shocks
% Read here, because ExogShockSetup_InfHorz_TPath below removes n_e, e_grid and pi_e from vfoptions
if ~isfield(vfoptions,'n_e')
    vfoptions.n_e=0;
end
n_e=vfoptions.n_e;
N_e=prod(n_e);
if N_e>0 && vfoptions.experienceasset>=1
    error('ValueFnOnTransPath_InfHorz: an experience asset together with an e variable is not yet implemented')
end

%% Semi-exogenous states
% The last vfoptions.l_dsemiz decision variables (d2) drive the semiz transitions pi_semiz(semiz,semiz',d2). Exogenous
% states are (semiz,z,e), semiz fastest; V and Policy carry the composite bothz=(semiz,z) with e separate.
if ~isfield(vfoptions,'n_semiz')
    vfoptions.n_semiz=0;
end
n_semiz=vfoptions.n_semiz;
N_semiz=prod(n_semiz);
if N_semiz>0
    if prod(n_d)==0
        error('ValueFnOnTransPath_InfHorz: semi-exogenous states (vfoptions.n_semiz) require a decision variable: the last vfoptions.l_dsemiz decision variables determine the semiz transitions')
    end
    if length(n_a)>1
        error('ValueFnOnTransPath_InfHorz: semi-exogenous states (vfoptions.n_semiz) only allow a single endogenous state (cannot have length(n_a)>1)')
    end
    if vfoptions.experienceasset>=1
        error('ValueFnOnTransPath_InfHorz: semi-exogenous states (vfoptions.n_semiz) are not yet implemented together with an experience asset')
    end
    if ~strcmp(vfoptions.exoticpreferences,'None')
        error('ValueFnOnTransPath_InfHorz: semi-exogenous states (vfoptions.n_semiz) are not yet implemented with vfoptions.exoticpreferences')
    end
    if vfoptions.localsearch~=0
        error('ValueFnOnTransPath_InfHorz: vfoptions.localsearch is not implemented with a semi-exogenous state')
    end
end

%% Implement new way of handling ReturnFn inputs
ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,0,vfoptions,Parameters);

%%
if size(d_grid,2)==1
    d_gridvals=CreateGridvals(n_d,gpuArray(d_grid),1);
else % already gridvals
    d_gridvals=gpuArray(d_grid);
end
a_grid=gpuArray(a_grid);
pi_z=gpuArray(pi_z);
PricePath=gpuArray(PricePath);


if transpathoptions.verbose>=1
    transpathoptions
end
if vfoptions.verbose==1
    vfoptions
end

N_d=prod(n_d);
N_z=prod(n_z);
N_a=prod(n_a);

%% Local search: the reference policy the window is centred on
% THE REFERENCE IS LEFT EMPTY AND THE DISPATCHER'S OWN DEFAULT IS USED. It used to be built here,
% as the current a index -- the window centred on staying put -- once, outside the loop over t, and
% handed unchanged to every period. That was the trap: it looked like a placeholder for something
% better while silently restricting every period's search to a window around the household's
% current wealth, and the third output recorded a reference that was never fed back.
% Reaching localsearch=1 here now means the caller passed localsearch=2, having been warned, so the
% dispatcher's stay-put default is what they get. Nothing in this command carries a reference from
% one period to the next; the reference ITERATION exists only in TransitionPath_InfHorz_shooting and
% AndersonAccelerationTPath, where a verification sweep certifies the result.
% The tempting design is to centre period t's window on period t+1's answer, which is defensible
% within one backward pass because prices move smoothly. It is still uncertified, so it is not done.
aprimeReferencePolicy=[];

% Internally the paths keep one exogenous dimension, (z,e) together, which is a singleton when there is
% neither. V is reshaped to what each single step expects inside the loop.
N_ze=max(N_z,1)*max(N_e,1);
if N_semiz>0
    N_ze=N_semiz*N_ze; % with semiz the one exogenous dimension is (semiz,z,e), semiz fastest
end
V_final=reshape(V_final,[N_a,N_ze]);

if transpathoptions.verbose==2
    DiscountFactorParamNames
    ReturnFnParamNames
    ParamPathNames
    PricePathNames
end

VKronPath=zeros(N_a,N_ze,T);
VKronPath(:,:,T)=V_final;


%% Set up exogenous shock processes
% Handles the time-invariant and the path-varying input shapes alike, and sets
% transpathoptions.zpathtrivial (=0 when z_gridvals/pi_z vary along the path, in which case the
% full paths are put in transpathoptions.z_gridvals_T and transpathoptions.pi_z_T and the plain
% z_gridvals/pi_z returned here are only the tt=1 placeholders).
% Same for e: transpathoptions.epathtrivial=0 puts the paths in transpathoptions.e_gridvals_T and
% transpathoptions.pi_e_T.
[z_gridvals, pi_z, ~, e_gridvals, pi_e, ~, ~, transpathoptions, vfoptions]=ExogShockSetup_InfHorz_TPath(n_z,z_grid,pi_z,Parameters,PricePathNames,ParamPath,ParamPathNames,ParamPathSizeVec,T,transpathoptions,vfoptions,3);
% Semi-exogenous states: semiz_gridvals [N_semiz,l_semiz], pi_semiz [N_semiz,N_semiz,N_dsemiz] (gpu); when
% transpathoptions.semizpathtrivial=0 the path is in transpathoptions.pi_semiz_T [N_semiz,N_semiz,N_dsemiz,T]
if N_semiz>0
    [semiz_gridvals, pi_semiz, transpathoptions, vfoptions]=SemiExogShockSetup_InfHorz_TPath(n_d,d_grid,Parameters,PricePathNames,ParamPath,ParamPathNames,ParamPathSizeVec,T,transpathoptions,vfoptions,3);
end

%%
if N_d==0
    l_daprime=length(n_a);
else
    l_daprime=length(n_d)+length(n_a);
end
if vfoptions.experienceasset>=1
    l_daprime=l_daprime-1;
end

PolicyIndexesPath=zeros(l_daprime+2*(vfoptions.gridinterplayer>0),N_a,N_ze,T,'gpuArray'); % Periods 1 to T-1
PolicyIndexesPath(:,:,:,T)=reshape(Policy_final, [size(Policy_final,1),N_a,N_ze]);

% NO aprimeReferencePath OUTPUT. It was returned for five days, read by nothing, and under the
% normal path -- localsearch=1, which this command ignores -- it would have been an array of zeros
% that looked like a reference. The reference a restricted sweep produces is only meaningful to a
% caller that feeds it back, and the only callers that do are TransitionPath_InfHorz_shooting and
% AndersonAccelerationTPath, which get it from TransitionPath_InfHorz_singlepathiter instead.


%%
if N_semiz>0
    % Go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
    N_bothz=N_semiz*max(N_z,1);
    Vnext=V_final;
    for ttr=1:T-1 %so t=T-i

        for kk=1:length(PricePathNames)
            Parameters.(PricePathNames{kk})=PricePath(T-ttr,kk);
        end
        for kk=1:length(ParamPathNames)
            Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,kk);
        end

        if N_z>0 && transpathoptions.zpathtrivial==0
            z_gridvals=transpathoptions.z_gridvals_T(:,:,T-ttr);
            pi_z=transpathoptions.pi_z_T(:,:,T-ttr);
        end
        % e: this period's grid, NEXT period's distribution
        if N_e>0 && transpathoptions.epathtrivial==0
            e_gridvals=transpathoptions.e_gridvals_T(:,:,T-ttr);
            pi_e=transpathoptions.pi_e_T(:,T-ttr+1);
        end
        % semiz: pi_semiz_T(:,:,:,t) is the transition from t to t+1 (the semiz grid is fixed over the path)
        if transpathoptions.semizpathtrivial==0
            pi_semiz=transpathoptions.pi_semiz_T(:,:,:,T-ttr);
        end

        if N_z>0 && N_e==0
            [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_SemiExo(reshape(Vnext,[N_a,N_bothz]),n_d,n_a,n_z,n_semiz,d_gridvals, a_grid, z_gridvals, semiz_gridvals, pi_z, pi_semiz, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        elseif N_z==0 && N_e==0
            [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_SemiExo_noz(reshape(Vnext,[N_a,N_bothz]),n_d,n_a,n_semiz,d_gridvals, a_grid, semiz_gridvals, pi_semiz, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        elseif N_z==0 && N_e>0
            [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_SemiExo_noz_e(reshape(Vnext,[N_a,N_bothz,N_e]),n_d,n_a,n_semiz,n_e,d_gridvals, a_grid, semiz_gridvals, e_gridvals, pi_semiz, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        elseif N_z>0 && N_e>0
            [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_SemiExo_e(reshape(Vnext,[N_a,N_bothz,N_e]),n_d,n_a,n_z,n_semiz,n_e,d_gridvals, a_grid, z_gridvals, semiz_gridvals, e_gridvals, pi_z, pi_semiz, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
        % The VKron input is next period value fn, the VKron output is this period. Policy is [l_d+l_a(+2),N_a,N_bothz(,N_e)]

        PolicyIndexesPath(:,:,:,T-ttr)=reshape(Policy,[size(Policy,1),N_a,N_ze]);

        V=reshape(V,[N_a,N_ze]);
        VKronPath(:,:,T-ttr)=V;
        Vnext=V;
    end
elseif vfoptions.experienceasset==0
    % Go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
    Vnext=V_final;
    for ttr=1:T-1 %so t=T-i

        for kk=1:length(PricePathNames)
            Parameters.(PricePathNames{kk})=PricePath(T-ttr,kk);
        end
        for kk=1:length(ParamPathNames)
            Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,kk);
        end

        if N_z>0 && transpathoptions.zpathtrivial==0
            z_gridvals=transpathoptions.z_gridvals_T(:,:,T-ttr);
            pi_z=transpathoptions.pi_z_T(:,:,T-ttr);
        end
        % e: this period's grid, NEXT period's distribution. pi_e_T(:,t) is the distribution of the e
        % realized IN period t (same convention as pi_e_J in FHorz), and V_t takes its expectation over e_{t+1}.
        if N_e>0 && transpathoptions.epathtrivial==0
            e_gridvals=transpathoptions.e_gridvals_T(:,:,T-ttr);
            pi_e=transpathoptions.pi_e_T(:,T-ttr+1);
        end

        if N_z>0 && N_e==0
            [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep(Vnext,n_d,n_a,n_z,d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions);
        elseif N_z==0 && N_e==0
            [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_noz(Vnext,n_d,n_a,d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        elseif N_z==0 && N_e>0
            [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_noz_e(reshape(Vnext,[N_a,N_e]),n_d,n_a,n_e,d_gridvals, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        elseif N_z>0 && N_e>0
            [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_e(reshape(Vnext,[N_a,N_z,N_e]),n_d,n_a,n_z,n_e,d_gridvals, a_grid, z_gridvals, e_gridvals, pi_z, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
        % The VKron input is next period value fn, the VKron output is this period. Policy is kept in the form where it is just a single-value in (d,a')

        PolicyIndexesPath(:,:,:,T-ttr)=reshape(Policy,[size(Policy,1),N_a,N_ze]);

        V=reshape(V,[N_a,N_ze]);
        VKronPath(:,:,T-ttr)=V;
        Vnext=V;
    end
else
    % Split decision variables into the standard ones and the one relevant to the experience asset
    if isscalar(n_d)
        n_d1=0;
    else
        n_d1=n_d(1:end-1);
    end
    n_d2=n_d(end); % n_d2 is the decision variable that influences next period vale of the experience asset
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

    if isfield(vfoptions,'aprimeFn')
        aprimeFn=vfoptions.aprimeFn;
    else
        error('To use an experience asset you must define vfoptions.aprimeFn')
    end

    % aprimeFnParamNames in same fashion
    l_d2=length(n_d2);
    l_a2=length(n_a2);
    temp=getAnonymousFnInputNames(aprimeFn);
    if length(temp)>(l_d2+l_a2)
        aprimeFnParamNames={temp{l_d2+l_a2+1:end}}; % the first inputs will always be (d2,a2)
    else
        aprimeFnParamNames={};
    end


    % Go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
    Vnext=V_final;
    for ttr=1:T-1 %so t=T-i

        for kk=1:length(PricePathNames)
            Parameters.(PricePathNames{kk})=PricePath(T-ttr,kk);
        end
        for kk=1:length(ParamPathNames)
            Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,kk);
        end

        if transpathoptions.zpathtrivial==0
            z_gridvals=transpathoptions.z_gridvals_T(:,:,T-ttr);
            pi_z=transpathoptions.pi_z_T(:,:,T-ttr);
        end

        [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_ExpAsset(Vnext,n_d1,n_d2,n_a1,n_a2,n_z, d1_grid, d2_grid, a1_grid, a2_grid, z_gridvals, pi_z, ReturnFn, vfoptions.aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        % The VKron input is next period value fn, the VKron output is this period. Policy is kept in the form where it is just a single-value in (d,a')

        PolicyIndexesPath(:,:,:,T-ttr)=Policy;

        VKronPath(:,:,T-ttr)=V;
        Vnext=V;
    end
end


%% Unkron to get into the shape for output
if N_semiz>0
    if N_z==0 && N_e==0
        VPath=reshape(VKronPath,[n_a,n_semiz,T]);
        PolicyPath=reshape(PolicyIndexesPath,[size(PolicyIndexesPath,1),n_a,n_semiz,T]);
    elseif N_z>0 && N_e==0
        VPath=reshape(VKronPath,[n_a,n_semiz,n_z,T]);
        PolicyPath=reshape(PolicyIndexesPath,[size(PolicyIndexesPath,1),n_a,n_semiz,n_z,T]);
    elseif N_z==0 && N_e>0
        VPath=reshape(VKronPath,[n_a,n_semiz,n_e,T]);
        PolicyPath=reshape(PolicyIndexesPath,[size(PolicyIndexesPath,1),n_a,n_semiz,n_e,T]);
    elseif N_z>0 && N_e>0
        VPath=reshape(VKronPath,[n_a,n_semiz,n_z,n_e,T]);
        PolicyPath=reshape(PolicyIndexesPath,[size(PolicyIndexesPath,1),n_a,n_semiz,n_z,n_e,T]);
    end
elseif N_z==0 && N_e==0
    VPath=reshape(VKronPath,[n_a,T]);
    PolicyPath=reshape(PolicyIndexesPath,[size(PolicyIndexesPath,1),n_a,T]);
elseif N_z>0 && N_e==0
    VPath=reshape(VKronPath,[n_a,n_z,T]);
    PolicyPath=reshape(PolicyIndexesPath,[size(PolicyIndexesPath,1),n_a,n_z,T]);
elseif N_z==0 && N_e>0
    VPath=reshape(VKronPath,[n_a,n_e,T]);
    PolicyPath=reshape(PolicyIndexesPath,[size(PolicyIndexesPath,1),n_a,n_e,T]);
elseif N_z>0 && N_e>0
    VPath=reshape(VKronPath,[n_a,n_z,n_e,T]);
    PolicyPath=reshape(PolicyIndexesPath,[size(PolicyIndexesPath,1),n_a,n_z,n_e,T]);
end


end
