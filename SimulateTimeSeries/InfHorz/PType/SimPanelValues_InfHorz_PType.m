function SimPanelValues=SimPanelValues_InfHorz_PType(InitialDist,PTypeDistParamNames,Policy,FnsToEvaluate,Parameters,n_d,n_a,n_z,Names_i,d_grid,a_grid,z_grid,pi_z,simoptions)
% Simulates a panel of 'numbersims' agents of length 'simperiods', with permanent types, beginning from InitialDist. Calls
% SimPanelValues_InfHorz() for each permanent type. (The InfHorz counterpart of SimPanelValues_FHorz_Case1_PType.)
%
% InitialDist can either be the same for all permanent types (n_a-by-n_z, of mass one), or a structure with a field per type
% (each of mass one); e.g. the StationaryDist from StationaryDist_InfHorz_PType, field by field.
%
% The number of simulations of each type is floor(ptweight*numbersims), with the few left over given to the first types, so the
% panel is representative of the type masses; a type of zero mass has no simulations. The panel holds the simulations of the
% first type, then those of the second, and so on. (To know the type of each simulation, add a FnsToEvaluate that returns it.)
%
% Inputs follow ValueFnIter_InfHorz_PType: anything that depends on the permanent type is given as a structure with a field per
% type (Parameters, n_z, z_grid, pi_z, simoptions fields, FnsToEvaluate fields) or with one value per type where that form is
% accepted. Names_i is the cell of type names, or just the number of types.
%
% Output:
%   SimPanelValues.(fnname)   simperiods-by-numbersims (NaN for the simulations of a type to which the function is not relevant)

if iscell(Names_i)
    N_i=length(Names_i);
else
    N_i=Names_i; % It is the number of PTypes (which have not been given names)
    Names_i=cell(1,N_i);
    for ii=1:N_i
        if ii<10
            Names_i{ii}=['ptype00',num2str(ii)];
        elseif ii<100
            Names_i{ii}=['ptype0',num2str(ii)];
        elseif ii<1000
            Names_i{ii}=['ptype',num2str(ii)];
        end
    end
end

if ~isstruct(FnsToEvaluate)
    error('You can only use PType when FnsToEvaluate is a structure')
end
FnNames=fieldnames(FnsToEvaluate);
numFnsToEvaluate=length(FnNames);

if ~exist('simoptions','var')
    simoptions.numbersims=10^3;
    simoptions.simperiods=50;
    simoptions.verbose=0;
    simoptions.verboseparams=0;
else
    if ~isfield(simoptions,'numbersims')
        simoptions.numbersims=10^3;
    end
    if ~isfield(simoptions,'simperiods')
        simoptions.simperiods=50;
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions,'verboseparams')
        simoptions.verboseparams=0;
    end
end

%% How many simulations of each type: perfectly representative of the type masses
ptweights=gather(reshape(Parameters.(PTypeDistParamNames{1}),[N_i,1]));
PType_numbersims=floor(ptweights*simoptions.numbersims);
% floor means a few too few; give the extra to the first types [as SimPanelValues_FHorz_Case1_PType]
ExtraSims=simoptions.numbersims-sum(PType_numbersims);
PType_numbersims(1:ExtraSims)=PType_numbersims(1:ExtraSims)+1;

SimPanelValues=struct();
for ff=1:numFnsToEvaluate
    SimPanelValues.(FnNames{ff})=nan(simoptions.simperiods,simoptions.numbersims);
end

%%
for ii=1:N_i
    iistr=Names_i{ii};
    if PType_numbersims(ii)==0
        continue % no simulations of this type
    end

    % First set up simoptions
    simoptions_temp=PType_Options(simoptions,iistr); % Note: already check for existence of simoptions and created it if it was not inputted
    simoptions_temp.numbersims=PType_numbersims(ii);
    if simoptions_temp.verbose==1
        fprintf('Permanent type: %i of %i \n',ii, N_i)
    end
    Policy_temp=gpuArray(Policy.(iistr));

    %% Go through everything which might be dependent on fixed type (PType)
    [n_d_temp,n_a_temp,d_grid_temp,a_grid_temp]=PType_setup_da(iistr,n_d,n_a,d_grid,a_grid);
    % Exogenous shocks
    [n_z_temp,z_grid_temp,pi_z_temp,simoptions_temp]=PType_setup_ExogShocks(ii,iistr,N_i,n_z,z_grid,pi_z,simoptions_temp,3);
    % Parameters
    Parameters_temp=PType_setup_Parameters(ii,iistr,N_i,Parameters,3);
    if simoptions_temp.verboseparams==1
        fprintf('Parameter values for the current permanent type \n')
        Parameters_temp
    end

    %% InitialDist
    if isstruct(InitialDist)
        if ~isfield(InitialDist,iistr)
            error(['You must input an InitialDist for permanent type ', iistr])
        end
        InitialDist_temp=InitialDist.(iistr);
    else
        InitialDist_temp=InitialDist; % the same for every type (so every type must have the same grids)
    end
    if abs(sum(InitialDist_temp(:))-1)>10^(-9)
        error(['The InitialDist must be of mass one for each type (it is not for type ',iistr,')'])
    end

    if n_d_temp(1)==0
        l_d_temp=0;
    else
        l_d_temp=length(n_d_temp);
    end
    l_a_temp=length(n_a_temp);
    if prod(n_z_temp)==0
        l_z_temp=0;
    else
        l_z_temp=length(n_z_temp);
    end

    % Which of the FnsToEvaluate are relevant to this type (kept as a structure)
    [FnsToEvaluate_temp,~,~,FnsAndPTypeIndicator_ii]=PType_FnsToEvaluate(FnsToEvaluate,Names_i,ii,l_d_temp,l_a_temp,l_z_temp,0);
    if sum(FnsAndPTypeIndicator_ii)==0
        continue % none of the FnsToEvaluate are relevant to this type (its simulations stay NaN)
    end

    SimPanelValues_ii=SimPanelValues_InfHorz(InitialDist_temp,Policy_temp,FnsToEvaluate_temp,[],Parameters_temp,n_d_temp,n_a_temp,n_z_temp,d_grid_temp,a_grid_temp,z_grid_temp,pi_z_temp,simoptions_temp);

    % The columns of this type in the panel
    cols=sum(PType_numbersims(1:ii-1))+(1:PType_numbersims(ii));
    for ff=1:numFnsToEvaluate
        if FnsAndPTypeIndicator_ii(ff)==1
            SimPanelValues.(FnNames{ff})(:,cols)=gather(SimPanelValues_ii.(FnNames{ff}));
        end
    end
end

end
