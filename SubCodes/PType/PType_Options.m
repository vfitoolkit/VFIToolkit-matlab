function options_temp=PType_Options(options,iistr)
    % Extracts the options for PType iistr from options
    options_temp=struct();
    OptionNames=fieldnames(options); % all the different parameters
    nFields=length(OptionNames);
    % First, check if using options.ptype001 approach
    if any(strcmp(OptionNames,iistr))
        options_temp=options.(iistr);
    else
        for ff=1:nFields
            % The only way your are allowed to set options for PTypes is by using names (structure), you are not allowed to do it as a
            % vector because it then becomes impossible to tell it from other reasons for using a vector.
            if ~isstruct(options.(OptionNames{ff}))
                options_temp.(OptionNames{ff})=options.(OptionNames{ff});
            else % It is a structure, using a simoptions.verbose.ptype001 approach
                if isfield(options.(OptionNames{ff}),iistr) % Check if the field is called iistr, if yes then it is a dependence on the current ptype
                    options_temp.(OptionNames{ff})=options.(OptionNames{ff}).(iistr);  % Get the option specific to this type
                end
            end
        end
    end
    % Note that options that are declared only for 'other' PTypes are not set, and so they will just follow the defaults.


    %% Set some required options
    if ~isfield(options_temp,'n_e')
        options_temp.n_e=0;
    end
    if ~isfield(options_temp,'n_semiz')
        options_temp.n_semiz=0;
    end
end
