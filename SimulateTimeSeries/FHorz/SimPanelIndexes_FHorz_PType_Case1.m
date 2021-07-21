function SimPanel=SimPanelIndexes_FHorz_PType_Case1(InitialDist,Policy,n_d,n_a,n_z,N_j,N_i,pi_z, simoptions)
% Simulates a panel based on PolicyIndexes of 'numbersims' agents of length
% 'simperiods' beginning from points in InitialDist.
%
% InitialDist can be inputed as over the finite time-horizon (j), or
% without a time-horizon in which case it is assumed to be an InitialDist
% for time j=1. (So InitialDist is either n_a-by-n_z-by-n_j-by-n_i, or n_a-by-n_z-by-n_i)
%
% Is hard-coded that the proportions of each PType will be exactly those in
% the InitialDist (no randomness in this dimension when constructing the
% PanelData sample).

% Code essentially just takes the PType inputs, and then passes them to the
% SimPanelIndexes_FHorz_Case1 command.

N_a=prod(n_a);
N_z=prod(n_z);

l_a=length(n_a);
l_z=length(n_z);

if exist('simoptions','var')==1 % check whether simoptions was inputted
    %Check simoptions for missing fields, if there are some fill them with the defaults
    if isfield(simoptions,'simperiods')==0
        simoptions.simperiods=N_j;
    end
    if isfield(simoptions,'numbersims')==0
        simoptions.numbersims=10^3;
    end
    if isfield(simoptions,'verbose')==0
        simoptions.verbose=0;
    end
else
    simoptions.numbersims=10^3;
    simoptions.simperiods=N_j;
    simoptions.verbose=0;
end

numelInitialDist=gather(numel(InitialDist)); % Use it multiple times so precalculate once here for speed
if N_a*N_z*N_i==numelInitialDist %Does not depend on N_j
    InitialDist=reshape(InitialDist,[N_a,N_z,N_i]);
    PType_mass=permute(sum(sum(InitialDist,1),2),[3,2,1]);
else % Depends on N_j
    InitialDist=reshape(InitialDist,[N_a,N_z,N_j,N_i]);
    PType_mass=permute(sum(sum(sum(InitialDist,1),2),3),[4,3,2,1]);
end
PType_numbersims=round(PType_mass*simoptions.numbersims);

SimPanel=nan(l_a+l_z+2,simoptions.simperiods,simoptions.numbersims); % (a,z,j)
for ii=1:N_i
    simoptions_ii=simoptions;
    if simoptions_ii.verbose==1
        sprintf('Fixed type: %i of %i',ii, N_i)
    end
    simoptions_ii.numbersims=PType_numbersims(ii);
    if isfield(simoptions,'ExogShockFn') % If this exists, so will ExogShockFnParamNames, but I still treat them seperate as makes the code easier to read
        if length(simoptions.ExogShockFn)==1
            if simoptions_ii.ExogShockFn==1
            end
        else
            if simoptions.ExogShockFn(ii)==1
                simoptions_ii.ExogShockFn=simoptions.ExogShockFn(ii);
            end
        end
    end
    if isfield(simoptions,'ExogShockFnParamNames')
        if length(simoptions.ExogShockFnParamNames)==1
            if simoptions.ExogShockFnParamNames==1
            end
        else
            if simoptions.ExogShockFnParamNames(ii)==1
                simoptions_ii.ExogShockFnParamNames=simoptions.ExogShockFnParamNames(ii);
            end
        end
    end
    
    % Go through everything which might be dependent on fixed type (PType)
    % [THIS could be better coded, 'names' are same for all these and just need to be found once outside of ii loop]
    pi_z_temp=pi_z;
    if isa(pi_z,'struct')
        names=fieldnames(pi_z);
        pi_z_temp=pi_z.(names{ii});
    end
    Policy_temp=Policy;
    if isa(Policy,'struct')
        names=fieldnames(Policy);
        Policy_temp=Policy.(names{ii});
    end
    if N_a*N_z*N_i==numelInitialDist %Does not depend on N_j
        InitialDist_temp=InitialDist(:,:,ii);
        InitialDist_temp=InitialDist_temp./(sum(sum(InitialDist_temp)));
    else % Depends on N_j
        InitialDist_temp=InitialDist(:,:,:,ii);
        InitialDist_temp=InitialDist_temp./(sum(sum(sum(InitialDist_temp))));
    end

    
    
    SimPanel_ii=gather(SimPanelIndexes_FHorz_Case1(InitialDist_temp,Policy_temp,n_d,n_a,n_z,N_j,pi_z_temp, simoptions_ii));
    if ii==1
        SimPanel(1:(l_a+l_z+1),:,1:sum(PType_numbersims(1:ii)))=SimPanel_ii;
        SimPanel(l_a+l_z+2,:,1:sum(PType_numbersims(1:ii)))=ii*ones(1,simoptions_ii.simperiods,PType_numbersims(ii));
    else
        SimPanel(1:(l_a+l_z+1),:,(1+sum(PType_numbersims(1:(ii-1)))):sum(PType_numbersims(1:ii)))=SimPanel_ii;
        SimPanel(l_a+l_z+2,:,(1+sum(PType_numbersims(1:(ii-1)))):sum(PType_numbersims(1:ii)))=ii*ones(1,simoptions_ii.simperiods,PType_numbersims(ii));
    end
end



