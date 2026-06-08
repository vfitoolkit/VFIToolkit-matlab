function StationaryDist=StationaryDist_FHorz_PType2L(jequaloneDist,AgeWeightsParamNames,TopPTypeDistParamNames,PTypeDistParamNames,Policy,n_d,n_a,n_z,N_j,Names_i,N_i,pi_z,Parameters,simoptions)
% Two-level permanent type dispatcher (top level) for finite-horizon stationary distributions.
%
% Top level is named (Names_i, cell array) with structure-keyed dependence.
% Bottom level is numeric (N_i) and is handled by StationaryDist_Case1_FHorz_PType.
%
% TopPTypeDistParamNames is a cell array of parameter names that together
% specify the top-level PType weights. Each named parameter must be either:
%   - a numeric vector of length length(Names_i), or
%   - a struct keyed by Names_i with scalar entries.
% Top weights are the elementwise product of these vectors.
%
% PTypeDistParamNames is passed through to the inner dispatcher and is
% interpreted as before (per-bottom weights). Its values in Parameters may
% legitimately be top-keyed structs (peeled here) or bottom-keyed.
%
% Output:
%   StationaryDist.(topname)          inner StationaryDist struct (includes .ptweights for bottom)
%   StationaryDist.topptweights       length-N_topi vector of top weights

if ~iscell(Names_i)
    error('Names_i must be a cell array of top-level PType names for the two-level PType command.')
end
N_topi=length(Names_i);

if ~exist('simoptions','var')
    error('You must input simoptions; you can always set simoptions=struct().')
end

StationaryDist=struct();

for ii_top=1:N_topi
    iistr=Names_i{ii_top};

    % First set up simoptions
    simoptions_temp=PType_Options_2L(simoptions,iistr);
    if ~isfield(simoptions_temp,'verbose')
        simoptions_temp.verbose=0;
    end
    if ~isfield(simoptions_temp,'verboseparams')
        simoptions_temp.verboseparams=0;
    end
    if ~isfield(simoptions_temp,'ptypestorecpu')
        simoptions_temp.ptypestorecpu=0;
    end

    if simoptions_temp.verbose==1
        fprintf('Top-level permanent type: %i of %i (%s)\n',ii_top,N_topi,iistr)
    end

    % Go through everything which might be dependent on the top-level fixed
    % type. Struct-only at this level; anything else passes through to the
    % inner PType dispatcher.
    if isstruct(n_d)
        n_d_temp=n_d.(iistr);
    else
        n_d_temp=n_d;
    end
    if isstruct(n_a)
        n_a_temp=n_a.(iistr);
    else
        n_a_temp=n_a;
    end

    if isstruct(N_j)
        N_j_temp=N_j.(iistr);
    else
        N_j_temp=N_j;
    end
    if isstruct(N_i)
        N_i_temp=N_i.(iistr);
    else
        N_i_temp=N_i;
    end

    % Exogenous shocks
    [n_z_temp,~,pi_z_temp,simoptions_temp]=PType_setup_ExogShocks(ii_top,iistr,N_topi,n_z,[],pi_z,simoptions_temp,1);

    %%
    if isstruct(jequaloneDist) && isfield(jequaloneDist,iistr)
        jequaloneDist_temp=jequaloneDist.(iistr);
    else
        jequaloneDist_temp=jequaloneDist;
    end

    AgeWeightsParamNames_temp=AgeWeightsParamNames;
    if isstruct(AgeWeightsParamNames)
        names=fieldnames(AgeWeightsParamNames);
        for jj=1:length(names)
            if strcmp(names{jj},Names_i{ii_top})
                AgeWeightsParamNames_temp=AgeWeightsParamNames.(names{jj});
            end
        end
    end

    PTypeDistParamNames_temp=PTypeDistParamNames;
    if isstruct(PTypeDistParamNames)
        names=fieldnames(PTypeDistParamNames);
        for jj=1:length(names)
            if strcmp(names{jj},Names_i{ii_top})
                PTypeDistParamNames_temp=PTypeDistParamNames.(names{jj});
            end
        end
    end

    % Parameters are only allowed to depend on top-level PType through a structure keyed by Names_i.
    Parameters_temp=PType_setup_Parameters(ii_top,iistr,N_topi,Parameters,1);

    if simoptions_temp.verboseparams==1
        sprintf('Parameter values for the current top-level permanent type')
        Parameters_temp
    end

    StationaryDist_tt=StationaryDist_Case1_FHorz_PType(jequaloneDist_temp,AgeWeightsParamNames_temp,PTypeDistParamNames_temp,Policy.(iistr),n_d_temp,n_a_temp,n_z_temp,N_j_temp,N_i_temp,pi_z_temp,Parameters_temp,simoptions_temp);

    StationaryDist.(iistr)=StationaryDist_tt;

    clear StationaryDist_tt

end

%% Top-level weights
topptweights=ones(N_topi,1);
for kk=1:length(TopPTypeDistParamNames)
    val=Parameters.(TopPTypeDistParamNames{kk});
    if isstruct(val)
        v=zeros(N_topi,1);
        for ii_top=1:N_topi
            v(ii_top)=val.(Names_i{ii_top});
        end
        val=v;
    end
    topptweights=topptweights.*val(:);
end
StationaryDist.topptweights=topptweights;


end
