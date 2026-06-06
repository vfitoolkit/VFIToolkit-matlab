function options_temp=PType_Options_2L(options,Names_i,tt)
    % Top-level options peeler for the two-level PType commands.
    % Like PType_Options, but at the top level. For each field of options:
    %   - not a struct: pass through verbatim (broadcast across top types)
    %   - struct whose field-set includes Names_i{tt}: take that branch
    %   - struct whose field-set does not include Names_i{tt}: pass through
    %     verbatim. This is the difference vs PType_Options, which silently
    %     drops fields that do not name the current PType. Here the field is
    %     assumed to be keyed by bottom-level (N_i) names instead, and the
    %     inner PType_Options call will resolve it.
    options_temp=struct();
    OptionNames=fieldnames(options);
    nFields=length(OptionNames);
    for ff=1:nFields
        if ~isstruct(options.(OptionNames{ff}))
            options_temp.(OptionNames{ff})=options.(OptionNames{ff});
        else
            if isfield(options.(OptionNames{ff}),Names_i{tt})
                options_temp.(OptionNames{ff})=options.(OptionNames{ff}).(Names_i{tt});
            else
                options_temp.(OptionNames{ff})=options.(OptionNames{ff});
            end
        end
    end

    %% Set some required options
    if ~isfield(options_temp,'n_e')
        options_temp.n_e=0;
    end
    if ~isfield(options_temp,'n_semiz')
        options_temp.n_semiz=0;
    end
end
