function [RealizedPricePath, RealizedParamPath, PricePath, multirevealsummary]=MultipleRevealTransitionPath_Case1_FHorz_PType(PricePathShaper, ParamPath, T, StationaryDist_initial, jequaloneDist, n_d, n_a, n_z, N_j, Names_i, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, GeneralEqmEqns_Transition, Parameters, DiscountFactorParamNames, AgeWeightsParamNames, PTypeDistParamNames, transpathoptions, simoptions_path, vfoptions_path, heteroagentoptions, simoptions_finaleqm, vfoptions_finaleqm)

% This is essentially going to call each of HeteroAgentStationaryEqm_Case1_FHorz_PType and TransitionPath_Case1_FHorz_PType repeatedly
% It is mostly just handling a large amount of for loops for you.

% General idea:
% For-loop over rr, the different reveals
%    solve the final stationary eqm
%    solve the transition path for this reveal
%    get the agent dist when we are about the make the next reveal (and the
%    prices so we can use them as initial guess for next path)
% End

% Note: vfoptions_finaleqm and vfoptions_path can be used to set vfoptions differently 
% for the final eqm and for the transition path (same for simoptions)

%% Get N_i so can use it to check inputs and set things up
if iscell(Names_i)
    N_i=length(Names_i);
else
    N_i=Names_i;
end

%% A few checks on the inputs, figure out which periods we are revealing
revealperiodnames=fieldnames(ParamPath);
nReveals=length(revealperiodnames);
if ~any(strcmp(revealperiodnames,'t0001'))
    error('Multiple reveal transition paths: you must have a t0001 path in ParamPath')
end
revealperiods=zeros(nReveals,1);
for rr=1:nReveals
    currentrevealname=revealperiodnames{rr};
    try
        revealperiods(rr)=str2double(currentrevealname(2:end));
    catch
        error('Multiple reveal transition paths: a field in ParamPath is misnamed (it must be tXXXX, where the X are numbers; you have one of the XXXX not being a number)')
    end
    if ~strcmp(currentrevealname(1),'t')
        error('Multiple reveal transition paths: a field in ParamPath is misnamed (it must be tXXXX, where the X are numbers; you have one that does not start with t)')
    end
end
durationofreveal=revealperiods(2:end)-revealperiods(1:end-1)+1; % E.g., if you have t0001 and t0005, then this will be 5, which is how many periods you follow the current reveal before being surprised by the new reveal (including today)

%% Set default options
if ~isfield(transpathoptions,'usepreviouspathshapeasinitialguess')
    transpathoptions.usepreviouspathshapeasinitialguess=zeros(nReveals,1); % 1=reuse the solution from previous reveal
elseif length(transpathoptions.usepreviouspathshapeasinitialguess)~=(nReveals)
    error('transpathoptions.usepreviouspathshapeasinitialguess must be of length equal to the number of paths you are revealing minus one')
end
if transpathoptions.usepreviouspathshapeasinitialguess(1)~=0
    error('transpathoptions.usepreviouspathshapeasinitialguess(1), must be equal to zero (cannot reuse previous path shape when it is the first path)')
end

%% How many parameters are in ParamPath for a given reveal
ParamsOnPathNames=fieldnames(ParamPath.t0001);
nParamsOnPath=length(ParamsOnPathNames);
ParamsOnPath_ptypedependence=zeros(nParamsOnPath,N_i); % indicator for which depend on ptype
% Check that the ParamPath is length T for every reveal
for rr=1:nReveals
    for pp=1:nParamsOnPath
        if isstruct(ParamPath.(revealperiodnames{rr}).(ParamsOnPathNames{pp})) % depends on ptype
            tempNames_i=fieldnames(ParamPath.(revealperiodnames{rr}).(ParamsOnPathNames{pp}));
            for ii=1:N_i
                if any(strcmp(tempNames_i,Names_i{ii}))
                    ParamsOnPath_ptypedependence(pp,ii)=1;
                    if ~any(size(ParamPath.(revealperiodnames{rr}).(ParamsOnPathNames{pp}).(Names_i{ii}))==T)
                        fprintf('Problem with ParamPath is in reveal %s and parmeter %s for ptype %i \n', revealperiodnames{rr}, ParamsOnPathNames{pp}, Names_i{ii})
                        error('Something in ParamPath does not have the T periods (see previous line of output)')
                    end
                end
                % Make sure it is something-by-T
                if size(ParamPath.(revealperiodnames{rr}).(ParamsOnPathNames{pp}).(Names_i{ii}),1)==T
                    ParamPath.(revealperiodnames{rr}).(ParamsOnPathNames{pp}).(Names_i{ii})=ParamPath.(revealperiodnames{rr}).(ParamsOnPathNames{pp}).(Names_i{ii})'; % transpose
                end
            end
        elseif any(size(ParamPath.(revealperiodnames{rr}).(ParamsOnPathNames{pp}))==N_i) % vector that depends on ptype
            ParamsOnPath_ptypedependence(pp,:)=1;
            % Replace it with a structure using names, makes my life easier
            temp=ParamPath.(revealperiodnames{rr}).(ParamsOnPathNames{pp});
            ParamPath.(revealperiodnames{rr})=rmfield(ParamPath.(revealperiodnames{rr}),ParamsOnPathNames{pp});
            for ii=1:N_i
                if size(temp(ii),1)==T % Make sure it is something-by-T
                    ParamPath.(revealperiodnames{rr}).(ParamsOnPathNames{pp}).(Names_i{ii})=temp(ii)';
                else
                    ParamPath.(revealperiodnames{rr}).(ParamsOnPathNames{pp}).(Names_i{ii})=temp(ii);
                end
            end
        else
            if ~any(size(ParamPath.(revealperiodnames{rr}).(ParamsOnPathNames{pp}))==T)
                fprintf('Problem with ParamPath is in reveal %s and parmeter %s \n', revealperiodnames{rr}, ParamsOnPathNames{pp})
                error('Something in ParamPath does not have the T periods (see previous line of output)')
            end
            % Make sure it is something-by-T
            if size(ParamPath.(revealperiodnames{rr}).(ParamsOnPathNames{pp}),1)==T
                ParamPath.(revealperiodnames{rr}).(ParamsOnPathNames{pp})=ParamPath.(revealperiodnames{rr}).(ParamsOnPathNames{pp})'; % transpose
            end
        end
    end
end

%% And price paths
PricesOnPathNames=fieldnames(PricePathShaper);
if any(transpathoptions.usepreviouspathshapeasinitialguess==1)
    shaperdiffersbyreveal=1;
    if any(strcmp(PricesOnPathNames,'t0001'))
        PricesOnPathNames=fieldnames(PricePathShaper.t0001);
        nPricesOnPath=length(PricesOnPathNames);
        PricesOnPath_ptypedependence=zeros(nPricesOnPath,N_i); % indicator for which depend on ptype
        % Check the PricePathShaper is the right size
        for pp=1:nPricesOnPath
            for rr=1:nReveals
                if transpathoptions.usepreviouspathshapeasinitialguess(rr)==0
                    if isstruct(PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp})) % depends on ptype
                        tempNames_i=fieldnames(PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp}));
                        for ii=1:N_i
                            if any(strcmp(tempNames_i,Names_i{ii}))
                                PricesOnPath_ptypedependence(pp,ii)=1; % Note, this won't differ by reveal, so is just overwriting unnecessarily, but not big deal
                                if ~any(size(PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp}).(Names_i{ii}))==T)
                                    fprintf('Problem with PricePathShaper is in reveal %s and parmeter %s and ptype %s \n', revealperiodnames{rr}, PricesOnPathNames{pp},Names_i{ii})
                                    error('Something in PricePathShaper does not have the T periods (see previous line of output)')
                                end
                            end
                            % Make sure it is something-by-T
                            if size(PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp}).(Names_i{ii}),1)==T
                                PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp}).(Names_i{ii})=PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp}).(Names_i{ii})'; % transpose
                            end
                        end
                    elseif any(size(PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp}))==N_i) % vector that depends on ptype
                        PricesOnPath_ptypedependence(pp,:)=1; % Note, this won't differ by reveal, so is just overwriting unnecessarily, but not big deal
                        % Replace it with a structure using names, makes my life easier
                        temp=PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp});
                        PricePathShaper.(revealperiodnames{rr})=rmfield(PricePathShaper.(revealperiodnames{rr}),PricesOnPathNames{pp});
                        for ii=1:N_i
                            if size(temp(ii),1)==T % Make sure it is something-by-T
                                PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp}).(Names_i{ii})=temp(ii)';
                            else
                                PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp}).(Names_i{ii})=temp(ii);
                            end
                        end
                    else
                        if ~any(size(PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp}))==T)
                            fprintf('Problem with PricePathShaper is in reveal %s and parmeter %s \n', revealperiodnames{rr}, PricesOnPathNames{pp})
                            error('Something in PricePathShaper does not have the T periods (see previous line of output)')
                        end
                    end
                    % Make sure it is something-by-T
                    if size(PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp}),1)==T
                        PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp})=PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp})'; % transpose
                    end
                end
            end
        end
    else
        nPricesOnPath=length(PricesOnPathNames);
        PricesOnPath_ptypedependence=zeros(nPricesOnPath,N_i); % indicator for which depend on ptype
        for pp=1:nPricesOnPath
            if isstruct(PricePathShaper.(PricesOnPathNames{pp})) % depends on ptype
                tempNames_i=fieldnames(PricePathShaper.(PricesOnPathNames{pp}));
                for ii=1:N_i
                    if any(strcmp(tempNames_i,Names_i{ii}))
                        PricesOnPath_ptypedependence(pp,ii)=1; % Note, this won't differ by reveal, so is just overwriting unnecessarily, but not big deal
                        if ~any(size(PricePathShaper.(PricesOnPathNames{pp}).(Names_i{ii}))==T)
                            fprintf('Problem with PricePathShaper is in parmeter %s and ptype %s \n', PricesOnPathNames{pp}, Names_i{ii})
                            error('Something in PricePathShaper does not have the T periods (see previous line of output)')
                        end
                    end
                    % Make sure it is something-by-T
                    if size(PricePathShaper.(PricesOnPathNames{pp}).(Names_i{ii}),1)==T
                        PricePathShaper.(PricesOnPathNames{pp}).(Names_i{ii})=PricePathShaper.(PricesOnPathNames{pp}).(Names_i{ii})'; % transpose
                    end
                end
            elseif any(size(PricePathShaper.(PricesOnPathNames{pp}))==N_i) % vector that depends on ptype
                PricesOnPath_ptypedependence(pp,:)=1; % Note, this won't differ by reveal, so is just overwriting unnecessarily, but not big deal
                % Replace it with a structure using names, makes my life easier
                temp=PricePathShaper.(PricesOnPathNames{pp});
                PricePathShaper=rmfield(PricePathShaper,PricesOnPathNames{pp});
                for ii=1:N_i
                    if size(temp(ii),1)==T % Make sure it is something-by-T
                        PricePathShaper.(PricesOnPathNames{pp}).(Names_i{ii})=temp(ii)';
                    else
                        PricePathShaper.(PricesOnPathNames{pp}).(Names_i{ii})=temp(ii);
                    end
                end
            else
                if ~any(size(PricePathShaper.(PricesOnPathNames{pp}))==T)
                    fprintf('Problem with PricePathShaper is in parmeter %s \n', PricesOnPathNames{pp})
                    error('Something in PricePathShaper does not have the T periods (see previous line of output)')
                end
                % Make sure it is something-by-T
                if size(PricePathShaper.(PricesOnPathNames{pp}),1)==T
                    PricePathShaper.(PricesOnPathNames{pp})=PricePathShaper.(PricesOnPathNames{pp})'; % transpose
                end
            end
            % Set it up as for each period (as need this so can interact with transpathoptions.usepreviouspathshapeasinitialguess
            for rr=1:nReveals
                if transpathoptions.usepreviouspathshapeasinitialguess(rr)==0
                    PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp})=PricePathShaper.(PricesOnPathNames{pp});
                end
            end
            PricePathShaper=rmfield(PricePathShaper,PricesOnPathNames{pp});
        end
    end
    % Note: this PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp}) is only created
    % for reveals with transpathoptions.usepreviouspathshapeasinitialguess(rr)==0
    % for the other rr it gets updated based on the previous path.
elseif any(strcmp(PricesOnPathNames,'t0001'))
    shaperdiffersbyreveal=1;
    PricesOnPathNames=fieldnames(PricePathShaper.t0001);
    nPricesOnPath=length(PricesOnPathNames);
    PricesOnPath_ptypedependence=zeros(nPricesOnPath,N_i); % indicator for which depend on ptype
    % Check the PricePathShaper is the right size
    for pp=1:nPricesOnPath
        for rr=1:nReveals
            if isstruct(PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp})) % depends on ptype
                tempNames_i=fieldnames(PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp}));
                for ii=1:N_i
                    if any(strcmp(tempNames_i,Names_i{ii}))
                        PricesOnPath_ptypedependence(pp,ii)=1; % Note, this won't differ by reveal, so is just overwriting unnecessarily, but not big deal
                        if ~any(size(PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp}).(Names_i{ii}))==T)
                            fprintf('Problem with PricePathShaper is in reveal %s and parmeter %s and ptype %s \n', revealperiodnames{rr}, PricesOnPathNames{pp},Names_i{ii})
                            error('Something in PricePathShaper does not have the T periods (see previous line of output)')
                        end
                    end
                    % Make sure it is something-by-T
                    if size(PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp}).(Names_i{ii}),1)==T
                        PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp}).(Names_i{ii})=PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp}).(Names_i{ii})'; % transpose
                    end
                end
            elseif any(size(PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp}))==N_i) % vector that depends on ptype
                PricesOnPath_ptypedependence(pp,:)=1; % Note, this won't differ by reveal, so is just overwriting unnecessarily, but not big deal
                % Replace it with a structure using names, makes my life easier
                temp=PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp});
                PricePathShaper.(revealperiodnames{rr})=rmfield(PricePathShaper.(revealperiodnames{rr}),PricesOnPathNames{pp});
                for ii=1:N_i
                    if size(temp(ii),1)==T % Make sure it is something-by-T
                        PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp}).(Names_i{ii})=temp(ii)';
                    else
                        PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp}).(Names_i{ii})=temp(ii);
                    end
                end
            else
                if ~any(size(PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp}))==T)
                    fprintf('Problem with PricePathShaper is in reveal %s and parmeter %s \n', revealperiodnames{rr}, PricesOnPathNames{pp})
                    error('Something in PricePathShaper does not have the T periods (see previous line of output)')
                end
            end
            % Make sure it is something-by-T
            if size(PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp}),1)==T
                PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp})=PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp})'; % transpose
            end
        end
    end
else
    shaperdiffersbyreveal=0;
    nPricesOnPath=length(PricesOnPathNames);
    PricesOnPath_ptypedependence=zeros(nPricesOnPath,N_i); % indicator for which depend on ptype
    for pp=1:nPricesOnPath
        if isstruct(PricePathShaper.(PricesOnPathNames{pp})) % depends on ptype
            tempNames_i=fieldnames(PricePathShaper.(PricesOnPathNames{pp}));
            for ii=1:N_i
                if any(strcmp(tempNames_i,Names_i{ii}))
                    PricesOnPath_ptypedependence(pp,ii)=1; % Note, this won't differ by reveal, so is just overwriting unnecessarily, but not big deal
                    if ~any(size(PricePathShaper.(PricesOnPathNames{pp}).(Names_i{ii}))==T)
                        fprintf('Problem with PricePathShaper is in parmeter %s and ptype %s \n', PricesOnPathNames{pp}, Names_i{ii})
                        error('Something in PricePathShaper does not have the T periods (see previous line of output)')
                    end
                end
                % Make sure it is something-by-T
                if size(PricePathShaper.(PricesOnPathNames{pp}).(Names_i{ii}),1)==T
                    PricePathShaper.(PricesOnPathNames{pp}).(Names_i{ii})=PricePathShaper.(PricesOnPathNames{pp}).(Names_i{ii})'; % transpose
                end
            end
        elseif any(size(PricePathShaper.(PricesOnPathNames{pp}))==N_i) % vector that depends on ptype
            PricesOnPath_ptypedependence(pp,:)=1; % Note, this won't differ by reveal, so is just overwriting unnecessarily, but not big deal
            % Replace it with a structure using names, makes my life easier
            temp=PricePathShaper.(PricesOnPathNames{pp});
            PricePathShaper=rmfield(PricePathShaper,PricesOnPathNames{pp});
            for ii=1:N_i
                if size(temp(ii),1)==T % Make sure it is something-by-T
                    PricePathShaper.(PricesOnPathNames{pp}).(Names_i{ii})=temp(ii)';
                else
                    PricePathShaper.(PricesOnPathNames{pp}).(Names_i{ii})=temp(ii);
                end
            end
        else
            if ~any(size(PricePathShaper.(PricesOnPathNames{pp}))==T)
                fprintf('Problem with PricePathShaper is in parmeter %s \n', PricesOnPathNames{pp})
                error('Something in PricePathShaper does not have the T periods (see previous line of output)')
            end
            % Make sure it is something-by-T
            if size(PricePathShaper.(PricesOnPathNames{pp}),1)==T
                PricePathShaper.(PricesOnPathNames{pp})=PricePathShaper.(PricesOnPathNames{pp})'; % transpose
            end
        end
    end
end

% The stationary general eqm must be based on the same parameters as the transition paths
GEPriceParamNames=PricesOnPathNames;

% Get the t0001 p_eqm_initial by just using whatever is in the Parameters
for pp=1:nPricesOnPath
    p_eqm_initial.(PricesOnPathNames{pp})=Parameters.(PricesOnPathNames{pp}); % Even if they depend on ptype this is still correct
end


%% Loop over the reveals that are being done
for rr=1:nReveals
    %% For each reveal
    fprintf('Currently doing multiple reveal transition path: reveal %i of %i \n', rr, nReveals)

    %% Compute the final stationary eqm
    % First, just get the stuff that relates to the final stationary eqm
    for pp=1:nParamsOnPath
        if any(ParamsOnPath_ptypedependence(pp,:)==1) % depends on ptypes
            for ii=1:N_i
                if ParamsOnPath_ptypedependence(pp,ii)==1
                    temp=ParamPath.(revealperiodnames{rr}).(ParamsOnPathNames{pp}).(Names_i{ii});
                    Parameters.(ParamsOnPathNames{pp}).(Names_i{ii})=temp(:,end); % the final value of the ParamPath on this parameter
                end
            end
        else
            temp=ParamPath.(revealperiodnames{rr}).(ParamsOnPathNames{pp});
            Parameters.(ParamsOnPathNames{pp})=temp(:,end); % the final value of the ParamPath on this parameter
        end
    end
    % Now solve the stationary general eqm
    [p_eqm_final,~,GeneralEqmCondnValues]=HeteroAgentStationaryEqm_Case1_FHorz_PType(n_d, n_a, n_z, N_j,Names_i, 0, pi_z, d_grid, a_grid, z_grid, jequaloneDist, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, AgeWeightsParamNames, PTypeDistParamNames, GEPriceParamNames, heteroagentoptions, simoptions_finaleqm, vfoptions_finaleqm);
    % Store the final stationary eqm
    multirevealsummary.finaleqm.(revealperiodnames{rr}).p_eqm_final=p_eqm_final;
    multirevealsummary.finaleqm.(revealperiodnames{rr}).GeneralEqmCondnValues=GeneralEqmCondnValues;

    for pp=1:nPricesOnPath
        Parameters.(PricesOnPathNames{pp})=p_eqm_final.(PricesOnPathNames{pp});  % Even if they depend on ptype this is still correct
    end

    % Get V_final
    [V_final, Policy_final]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,N_j, Names_i, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions_finaleqm);

    %% Set up guess on PricePath0
    PricePath0_rr=struct();
    if shaperdiffersbyreveal==0
        for pp=1:nPricesOnPath
            if any(PricesOnPath_ptypedependence(pp,:)==1) % depends on ptypes
                for ii=1:N_i
                    if PricesOnPath_ptypedependence(pp,ii)==1
                        temp_pp1=p_eqm_initial.(PricesOnPathNames{pp}).(Names_i{ii});
                        temp_pp2=p_eqm_final.(PricesOnPathNames{pp}).(Names_i{ii});
                        PricePath0_rr.(PricesOnPathNames{pp}).(Names_i{ii})=temp_pp1 + (temp_pp2-temp_pp1)*PricePathShaper.(PricesOnPathNames{pp}).(Names_i{ii});
                    end
                end
            else
                temp_pp1=p_eqm_initial.(PricesOnPathNames{pp});
                temp_pp2=p_eqm_final.(PricesOnPathNames{pp});
                PricePath0_rr.(PricesOnPathNames{pp})=temp_pp1 + (temp_pp2-temp_pp1)*PricePathShaper.(PricesOnPathNames{pp});
            end
        end
    elseif shaperdiffersbyreveal==1
        for pp=1:nPricesOnPath
            if any(PricesOnPath_ptypedependence(pp,:)==1) % depends on ptypes
                for ii=1:N_i
                    if PricesOnPath_ptypedependence(pp,ii)==1
                        temp_pp1=p_eqm_initial.(PricesOnPathNames{pp}).(Names_i{ii});
                        temp_pp2=p_eqm_final.(PricesOnPathNames{pp}).(Names_i{ii});
                        PricePath0_rr.(PricesOnPathNames{pp}).(Names_i{ii})=temp_pp1 + (temp_pp2-temp_pp1)*PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp}).(Names_i{ii});
                    end
                end
            else
                temp_pp1=p_eqm_initial.(PricesOnPathNames{pp});
                temp_pp2=p_eqm_final.(PricesOnPathNames{pp});
                PricePath0_rr.(PricesOnPathNames{pp})=temp_pp1 + (temp_pp2-temp_pp1)*PricePathShaper.(revealperiodnames{rr}).(PricesOnPathNames{pp});
            end
        end
    end
    
    ParamPath_rr=ParamPath.(revealperiodnames{rr});


    % We already have StationaryDist_initial, but if one of our ParamPaths is on the age weights, we need to impose this onto StationaryDist_initial
    if any(strcmp(ParamsOnPathNames,AgeWeightsParamNames))
        for ii=1:N_i
            tempsize=size(StationaryDist_initial.(Names_i{ii}));
            StationaryDist_initial.(Names_i{ii})=reshape(StationaryDist_initial.(Names_i{ii}),[numel(StationaryDist_initial.(Names_i{ii}))/N_j,N_j]);
            StationaryDist_initial.(Names_i{ii})=StationaryDist_initial.(Names_i{ii})./sum(StationaryDist_initial.(Names_i{ii}),1); % remove current age weights
            temp=ParamPath.(revealperiodnames{rr}).(AgeWeightsParamNames{1});
            StationaryDist_initial.(Names_i{ii})=StationaryDist_initial.(Names_i{ii}).*temp(:,1)'; % Note: we already put current ParamPath reveal into Parameters
            StationaryDist_initial.(Names_i{ii})=reshape(StationaryDist_initial.(Names_i{ii}),tempsize);
        end
    end
    
    %% Compute the transition path
    PricePath_rr=TransitionPath_Case1_FHorz_PType(PricePath0_rr, ParamPath_rr, T, V_final, StationaryDist_initial, jequaloneDist, n_d, n_a, n_z, N_j, Names_i,d_grid,a_grid,z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqns_Transition, Parameters, DiscountFactorParamNames, AgeWeightsParamNames, PTypeDistParamNames, transpathoptions, simoptions_path, vfoptions_path);
    % Keep each of the price paths
    PricePath.(revealperiodnames{rr})=PricePath_rr;
    
    
    % %% Purely for debugging purposes
    % str=['temp',num2str(rr),'.mat'];
    % StationaryDist_initial_rr=StationaryDist_initial;
    % V_final_rr=V_final;
    % p_eqm_final_rr=p_eqm_final;
    % save(str, 'PricePath0_rr', 'PricePath_rr', 'ParamPath_rr', 'V_final_rr', 'StationaryDist_initial_rr','p_eqm_final_rr')

    %% Need the agent dist along the path so that we can move to the next reveal

    % You can calculate the value and policy functions for the transition path
    [VPath_rr,PolicyPath_rr]=ValueFnOnTransPath_Case1_FHorz_PType(PricePath_rr, ParamPath_rr, T, V_final, Policy_final, Parameters, n_d, n_a, n_z, N_j, Names_i, d_grid, a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, transpathoptions,vfoptions_path);
    
    % You can then use these to calculate the agent distribution for the transition path
    AgentDistPath_rr=AgentDistOnTransPath_Case1_FHorz_PType(StationaryDist_initial, jequaloneDist, PricePath_rr, ParamPath_rr, PolicyPath_rr, AgeWeightsParamNames,n_d,n_a,n_z,N_j,Names_i,pi_z,T, Parameters, transpathoptions, simoptions_path);

    % And then we can calculate AggVars for the path
    AggVarsPath_rr=EvalFnOnTransPath_AggVars_Case1_FHorz_PType(FnsToEvaluate, AgentDistPath_rr, PolicyPath_rr, PricePath_rr, ParamPath_rr, Parameters, T, n_d, n_a, n_z, N_j, Names_i, d_grid, a_grid,z_grid, transpathoptions, simoptions_path);
    
    % Store these, they might be of interest if user wants to see way too much info about each path :)
    multirevealsummary.VPath.(revealperiodnames{rr})=VPath_rr;
    multirevealsummary.PolicyPath.(revealperiodnames{rr})=PolicyPath_rr;
    multirevealsummary.AgentDistPath.(revealperiodnames{rr})=AgentDistPath_rr;
    multirevealsummary.AggVarsPath.(revealperiodnames{rr})=AggVarsPath_rr;

    %% Check if we need to use this reveal path for shaping initial guess for the next reveal
    if rr<nReveals
        if transpathoptions.usepreviouspathshapeasinitialguess(rr+1)==1
            % Set PricePathShaper for the next reveal based on the current transition path solution
            for pp=1:nPricesOnPath
                if any(PricesOnPath_ptypedependence(pp,:)==1)
                    for ii=1:N_i
                        if PricesOnPath_ptypedependence(pp,ii)==1
                            temp_pp1=p_eqm_initial.(PricesOnPathNames{pp}).(Names_i{ii});
                            temp_pp2=p_eqm_final.(PricesOnPathNames{pp}).(Names_i{ii});
                            PricePathShaper.(revealperiodnames{rr+1}).(PricesOnPathNames{pp}).(Names_i{ii})=(PricePath_rr.(PricesOnPathNames{pp}).(Names_i{ii})-temp_pp1)./(temp_pp2-temp_pp1);
                        end
                    end

                else
                    temp_pp1=p_eqm_initial.(PricesOnPathNames{pp});
                    temp_pp2=p_eqm_final.(PricesOnPathNames{pp});
                    PricePathShaper.(revealperiodnames{rr+1}).(PricesOnPathNames{pp})=(PricePath_rr.(PricesOnPathNames{pp})-temp_pp1)./(temp_pp2-temp_pp1);
                end
            end
        end
    end

    %% Update the StationaryDist_initial for use in the next transition
    if rr<nReveals
        StationaryDist_initial=struct();
        StationaryDist_initial.ptweights=AgentDistPath_rr.ptweights;
        sizesbyptype=struct(); % this is same for every reveal, buy just going to overwrite each time anyway
        for ii=1:N_i
            % Get agent dist in period durationofreveal(rr) of the current path
            sizesbyptype(ii).temp_agentdistsize=size(AgentDistPath_rr.(Names_i{ii}));
            AgentDistPath_rr_ii=reshape(AgentDistPath_rr.(Names_i{ii}),[prod(sizesbyptype(ii).temp_agentdistsize(1:end-1)),T]);
            StationaryDist_initial.(Names_i{ii})=AgentDistPath_rr_ii(:,durationofreveal(rr));
            StationaryDist_initial.(Names_i{ii})=reshape(StationaryDist_initial.(Names_i{ii}),sizesbyptype(ii).temp_agentdistsize(1:end-1));
        end

        % and update p_eqm_initial for the use in the next transition
        p_eqm_initial=struct();
        for pp=1:nPricesOnPath
            if any(PricesOnPath_ptypedependence(pp,:)==1)
                for ii=1:N_i
                    if PricesOnPath_ptypedependence(pp,ii)==1
                        temp_pp=PricePath_rr.(PricesOnPathNames{pp}).(Names_i{ii});
                        p_eqm_initial.(PricesOnPathNames{pp}).(Names_i{ii})=temp_pp(durationofreveal(rr));
                    end
                end
            else
                temp_pp=PricePath_rr.(PricesOnPathNames{pp});
                p_eqm_initial.(PricesOnPathNames{pp})=temp_pp(durationofreveal(rr));
            end
        end
    end

end


%% Create the realized paths for RealizedPricePath and RealizedParamPath from the reveals
% (I deliberately did not put this inside the same for loop so that it is easier to see how they are constructed)
% These will all be T periods from the final reveal period (including that period)
historylength=revealperiods(rr)+T-1;
for pp=1:nPricesOnPath
    if any(PricesOnPath_ptypedependence(pp,:)==1) % depends on ptype
        for ii=1:N_i
            if PricesOnPath_ptypedependence(pp,ii)==1
                if all(size(PricePath_rr.(PricesOnPathNames{pp}).(Names_i{ii}))==1)
                    RealizedPricePath.(PricesOnPathNames{pp}).(Names_i{ii})=zeros(1,historylength);
                else
                    RealizedPricePath.(PricesOnPathNames{pp}).(Names_i{ii})=zeros(size(PricePath_rr.(PricesOnPathNames{pp}).(Names_i{ii}),2),historylength);
                end
            end
        end
    else
        if all(size(PricePath_rr.(PricesOnPathNames{pp}))==1)
            RealizedPricePath.(PricesOnPathNames{pp})=zeros(1,historylength);
        else
            RealizedPricePath.(PricesOnPathNames{pp})=zeros(size(PricePath_rr.(PricesOnPathNames{pp}),2),historylength);
        end
    end
end
for pp=1:nParamsOnPath
    if any(ParamsOnPath_ptypedependence(pp,:)==1) % depends on ptype
        for ii=1:N_i
            if ParamsOnPath_ptypedependence(pp,ii)==1
                if all(size(ParamPath.(revealperiodnames{1}).(ParamsOnPathNames{pp}).(Names_i{ii}))==1)
                    RealizedParamPath.(ParamsOnPathNames{pp}).(Names_i{ii})=zeros(1,historylength);
                else
                    RealizedParamPath.(ParamsOnPathNames{pp}).(Names_i{ii})=zeros(size(ParamPath.(revealperiodnames{1}).(ParamsOnPathNames{pp}).(Names_i{ii}),1),historylength);
                end
            end
        end
    else
        if all(size(ParamPath.(revealperiodnames{1}).(ParamsOnPathNames{pp}))==1)
            RealizedParamPath.(ParamsOnPathNames{pp})=zeros(1,historylength);
        else
            RealizedParamPath.(ParamsOnPathNames{pp})=zeros(size(ParamPath.(revealperiodnames{1}).(ParamsOnPathNames{pp}),1),historylength);
        end
    end
end
for ii=1:N_i
    % Figure I may as well also create the historical paths for some other things the user might find useful
    sizesbyptype(ii).temp_vsize=size(VPath_rr.(Names_i{ii}));
    multirevealsummary.RealizedVPath.(Names_i{ii})=zeros([prod(sizesbyptype(ii).temp_vsize(1:end-1)),historylength]);
    sizesbyptype(ii).temp_policysize=size(PolicyPath_rr.(Names_i{ii}));
    multirevealsummary.RealizedPolicyPath.(Names_i{ii})=zeros([prod(sizesbyptype(ii).temp_policysize(1:end-1)),historylength]);
    %sizesbyptype(ii).temp_agentdistsize, we already got this above
    multirevealsummary.RealizedAgentDistPath.(Names_i{ii})=zeros([prod(sizesbyptype(ii).temp_agentdistsize(1:end-1)),historylength]);
end
AggVarNames=fieldnames(AggVarsPath_rr);
for aa=1:length(AggVarNames)
    multirevealsummary.RealizedAggVarsPath.(AggVarNames{aa}).Mean=zeros(1,historylength);
    whichptypeforthisAggVar=fieldnames(AggVarsPath_rr.(AggVarNames{aa}));
    for ii=1:N_i
        if any(strcmp(whichptypeforthisAggVar,Names_i{ii}))
            multirevealsummary.RealizedAggVarsPath.(AggVarNames{aa}).(Names_i{ii}).Mean=zeros(1,historylength);
        end
    end
end

% Modify to add in some extra elements so I can just use these in a loop
revealperiods=[revealperiods;revealperiods(end)+T];
durationofreveal2=[durationofreveal-1; T];


% Okay, created the objects, now fill them in
for rr=1:nReveals
    for pp=1:nPricesOnPath
        if any(PricesOnPath_ptypedependence(pp,:)==1) % depends on ptype
            for ii=1:N_i
                if PricesOnPath_ptypedependence(pp,ii)==1
                    temp=RealizedPricePath.(PricesOnPathNames{pp}).(Names_i{ii});
                    temp2=PricePath.(revealperiodnames{rr}).(PricesOnPathNames{pp}).(Names_i{ii});
                    if any(size(temp2)==1)
                        temp(revealperiods(rr):revealperiods(rr+1)-1)=temp2(1:durationofreveal2(rr));
                    else
                        temp(revealperiods(rr):revealperiods(rr+1)-1)=temp2(:,1:durationofreveal2(rr));
                    end
                    RealizedPricePath.(PricesOnPathNames{pp}).(Names_i{ii})=temp;
                end
            end
        else
            temp=RealizedPricePath.(PricesOnPathNames{pp});
            temp2=PricePath.(revealperiodnames{rr}).(PricesOnPathNames{pp});
            if any(size(temp2)==1)
                temp(revealperiods(rr):revealperiods(rr+1)-1)=temp2(1:durationofreveal2(rr));
            else
                temp(revealperiods(rr):revealperiods(rr+1)-1)=temp2(:,1:durationofreveal2(rr));
            end
            RealizedPricePath.(PricesOnPathNames{pp})=temp;
        end
    end
    
    for pp=1:nParamsOnPath
        if any(ParamsOnPath_ptypedependence(pp,:)==1) % depends on ptype
            for ii=1:N_i
                if ParamsOnPath_ptypedependence(pp,ii)==1
                    temp=RealizedParamPath.(ParamsOnPathNames{pp}).(Names_i{ii});
                    temp2=ParamPath.(revealperiodnames{rr}).(ParamsOnPathNames{pp}).(Names_i{ii});
                    temp(:,revealperiods(rr):revealperiods(rr+1)-1)=temp2(:,1:durationofreveal2(rr));
                    RealizedParamPath.(ParamsOnPathNames{pp}).(Names_i{ii})=temp;
                end
            end
        else
            temp=RealizedParamPath.(ParamsOnPathNames{pp});
            temp2=ParamPath.(revealperiodnames{rr}).(ParamsOnPathNames{pp});
            temp(:,revealperiods(rr):revealperiods(rr+1)-1)=temp2(:,1:durationofreveal2(rr));
            RealizedParamPath.(ParamsOnPathNames{pp})=temp;
        end
    end

    for ii=1:N_i
        temp=multirevealsummary.RealizedVPath.(Names_i{ii});
        temp2=multirevealsummary.VPath.(revealperiodnames{rr}).(Names_i{ii});
        temp2=reshape(temp2,[prod(sizesbyptype(ii).temp_vsize(1:end-1)),T]);
        temp(:,revealperiods(rr):revealperiods(rr+1)-1)=temp2(:,1:durationofreveal2(rr));
        multirevealsummary.RealizedVPath.(Names_i{ii})=temp;

        temp=multirevealsummary.RealizedPolicyPath.(Names_i{ii});
        temp2=multirevealsummary.PolicyPath.(revealperiodnames{rr}).(Names_i{ii});
        temp2=reshape(temp2,[prod(sizesbyptype(ii).temp_policysize(1:end-1)),T]);
        temp(:,revealperiods(rr):revealperiods(rr+1)-1)=temp2(:,1:durationofreveal2(rr));
        multirevealsummary.RealizedPolicyPath.(Names_i{ii})=temp;

        temp=multirevealsummary.RealizedAgentDistPath.(Names_i{ii});
        temp2=multirevealsummary.AgentDistPath.(revealperiodnames{rr}).(Names_i{ii});
        temp2=reshape(temp2,[prod(sizesbyptype(ii).temp_agentdistsize(1:end-1)),T]);
        temp(:,revealperiods(rr):revealperiods(rr+1)-1)=temp2(:,1:durationofreveal2(rr));
        multirevealsummary.RealizedAgentDistPath.(Names_i{ii})=temp;
    end

    for aa=1:length(AggVarNames)
        temp=multirevealsummary.RealizedAggVarsPath.(AggVarNames{aa}).Mean;
        temp2=multirevealsummary.AggVarsPath.(revealperiodnames{rr}).(AggVarNames{aa}).Mean;
        temp(revealperiods(rr):revealperiods(rr+1)-1)=temp2(1:durationofreveal2(rr));
        multirevealsummary.RealizedAggVarsPath.(AggVarNames{aa}).Mean=temp;
        for ii=1:N_i
            if any(strcmp(whichptypeforthisAggVar,Names_i{ii}))
                temp=multirevealsummary.RealizedAggVarsPath.(AggVarNames{aa}).(Names_i{ii}).Mean;
                temp2=multirevealsummary.AggVarsPath.(revealperiodnames{rr}).(AggVarNames{aa}).(Names_i{ii}).Mean;
                temp(revealperiods(rr):revealperiods(rr+1)-1)=temp2(1:durationofreveal2(rr));
                multirevealsummary.RealizedAggVarsPath.(AggVarNames{aa}).(Names_i{ii}).Mean=temp;
            end
        end
    end
end
% Reshape for output (get them out of kron from)
for ii=1:N_i
    multirevealsummary.RealizedVPath.(Names_i{ii})=reshape(multirevealsummary.RealizedVPath.(Names_i{ii}),[sizesbyptype(ii).temp_vsize(1:end-1),historylength]);
    multirevealsummary.RealizedPolicyPath.(Names_i{ii})=reshape(multirevealsummary.RealizedPolicyPath.(Names_i{ii}),[sizesbyptype(ii).temp_policysize(1:end-1),historylength]);
    multirevealsummary.RealizedAgentDistPath.(Names_i{ii})=reshape(multirevealsummary.RealizedAgentDistPath.(Names_i{ii}),[sizesbyptype(ii).temp_agentdistsize(1:end-1),historylength]);
end

% I feel like it just makes sense to keep copies of PricePath and
% PolicyPath in multirevealsummary, as that way everything is in there if
% you want it.
multirevealsummary.PricePath=PricePath;
multirevealsummary.ParamPath=ParamPath;
multirevealsummary.RealizedPricePath=RealizedPricePath;
multirevealsummary.RealizedParamPath=RealizedParamPath;



end