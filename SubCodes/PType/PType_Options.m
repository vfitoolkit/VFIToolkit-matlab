function options_temp=PType_Options(options,Names_i,ii)
    % Extracts the options for PType ii from options
    options_temp=struct();
    OptionNames=fieldnames(options); % all the different parameters
    nFields=length(OptionNames);
    for ff=1:nFields
        if ~isstruct(options.(OptionNames{ff}))
            % The only way your are allowed to set options for PTypes is by
            % using names (structure), you are not allowed to do it as a
            % vector because it then becomes impossible to tell it from
            % other reasons for using a vector.
            options_temp.(OptionNames{ff})=options.(OptionNames{ff});
%             if length(options.(OptionNames{ff}))==1 % Same option is set for all types
%                 options_temp.(OptionNames{ff})=options.(OptionNames{ff});
%             elseif length(options.(OptionNames{ff}))>1 % Get the option specific to this type
%                 temp=options.(OptionNames{ff});
%                 options_temp.(OptionNames{ff})=temp(ii);
%             end
        else % It is a structure
            if isfield(options.(OptionNames{ff}), Names_i{ii})
                options_temp.(OptionNames{ff})=options.(OptionNames{ff}).(Names_i{ii});  % Get the option specific to this type
%             else
                % Do nothing, this option is only relevant to other types
            end
        end
    end
    % Note that options that are declared only for 'other' PTypes are not set, and so they will just follow the defaults.
end
