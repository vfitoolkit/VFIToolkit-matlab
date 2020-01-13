function SimPanelValues=SimPanelValues_Case1(InitialDist,Policy,FnsToEvaluate,FnsToEvaluateParamNames,Parameters,n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z, simoptions, EntryExitParamNames)
% Simulates a panel based on PolicyIndexes of 'numbersims' agents of length
% 'simperiods' beginning from randomly drawn InitialDist.
% SimPanelValues is a 3-dimensional matrix with first dimension being the
% number of 'variables' to be simulated, second dimension is FHorz, and
% third dimension is the number-of-simulations
%
% InitialDist can be inputed as over the finite time-horizon (j), or
% without a time-horizon in which case it is assumed to be an InitialDist
% for time j=1. (So InitialDist is either n_a-by-n_z-by-n_j, or n_a-by-n_z)

N_a=prod(n_a);
N_z=prod(n_z);
N_d=prod(n_d);

%% Check which simoptions have been declared, set all others to defaults 
if exist('simoptions','var')==1
    %Check simoptions for missing fields, if there are some fill them with the defaults
    if isfield(simoptions,'parallel')==0
        simoptions.parallel=2;
    end
    if isfield(simoptions,'verbose')==0
        simoptions.verbose=0;
    end
    if isfield(simoptions,'simperiods')==0
        simoptions.simperiods=50;
    end
    if isfield(simoptions,'numbersims')==0
        simoptions.numbersims=10^3;
    end
    if isfield(simoptions,'agententryandexit')==0
        simoptions.agententryandexit=0;
    end
    if isfield(simoptions,'endogenousexit')==0
        simoptions.endogenousexit=0; % Note: this will only be relevant if agententryandexit=1
    end
    if isfield(simoptions,'entryinpanel')==0
        simoptions.entryinpanel=1; % Note: this will only be relevant if agententryandexit=1
    end
    if isfield(simoptions,'exitinpanel')==0
        simoptions.exitinpanel=1; % Note: this will only be relevant if agententryandexit=1
    end    
else
    %If simoptions is not given, just use all the defaults
    simoptions.parallel=2;
    simoptions.verbose=0;
    simoptions.simperiods=50;
    simoptions.numbersims=10^3;
    simoptions.agententryandexit=0;
    simoptions.endogenousexit=0; % Note: this will only be relevant if agententryandexit=1
    simoptions.entryinpanel=1; % Note: this will only be relevant if agententryandexit=1
    simoptions.exitinpanel=1; % Note: this will only be relevant if agententryandexit=1
end

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

if simoptions.parallel~=2
    d_grid=gather(d_grid);
    a_grid=gather(a_grid);
    z_grid=gather(z_grid);
end

% NOTE: ESSENTIALLY ALL THE RUN TIME IS IN THIS COMMAND. WOULD BE GOOD TO OPTIMIZE/IMPROVE.
PolicyIndexesKron=KronPolicyIndexes_Case1(Policy, n_d, n_a, n_z);%,simoptions); % Create it here as want it both here and inside SimPanelIndexes_FHorz_Case1 (which will recognise that it is already in this form)

if simoptions.agententryandexit==1
    DistOfNewAgents=Parameters.(EntryExitParamNames.DistOfNewAgents{1});
    CondlProbOfSurvival=Parameters.(EntryExitParamNames.CondlProbOfSurvival{1});
    RelativeMassOfEntrants=InitialDist.mass/Parameters.(EntryExitParamNames.MassOfNewAgents{1});

    % Rather than create a whole new function for Entry, just deal with it
    % by making repeated use of SimPanelIndexes_Case1(). This could be sped
    % up with better use of precomputing certain objects, but is easy.
    
    % First, figure out how big the eventual panel will be.
    NumberOfNewAgentsPerPeriod=round(RelativeMassOfEntrants*simoptions.numbersims);
    if simoptions.entryinpanel==0 % Don't want entry in panel data simulation
        NumberOfNewAgentsPerPeriod=0;
    end
    TotalNumberSims=simoptions.numbersims+simoptions.simperiods*NumberOfNewAgentsPerPeriod;
    SimPanelIndexes=nan(l_a+l_z,simoptions.simperiods,TotalNumberSims); % (a,z)
    % Start with those based on the initial distribution
    SimPanelIndexes(:,:,1:simoptions.numbersims)=SimPanelIndexes_Case1(InitialDist.pdf,PolicyIndexesKron,n_d,n_a,n_z,pi_z, simoptions, CondlProbOfSurvival);
    % Now do those for the entrants each period
    numbersims=simoptions.numbersims; % Store this, so can restore it after following loop
    simperiods=simoptions.simperiods;% Store this, so can restore it after following loop
    simoptions.numbersims=NumberOfNewAgentsPerPeriod;
    for t=1:simperiods
        SimPanelIndexes(:,t:end,numbersims+1+NumberOfNewAgentsPerPeriod*(t-1):numbersims+NumberOfNewAgentsPerPeriod*t)=SimPanelIndexes_Case1(DistOfNewAgents,PolicyIndexesKron,n_d,n_a,n_z,pi_z, simoptions, CondlProbOfSurvival);
        simoptions.simperiods=simoptions.simperiods-1;
    end
    simoptions.numbersims=numbersims; % Restore.
    simoptions.simperiods=simperiods;% Retore.
else
    SimPanelIndexes=SimPanelIndexes_Case1(InitialDist,PolicyIndexesKron,n_d,n_a,n_z,pi_z, simoptions);
end

% Move everything to cpu for what remains.
d_grid=gather(d_grid);
a_grid=gather(a_grid);
z_grid=gather(z_grid);
PolicyIndexesKron=gather(PolicyIndexesKron);

SimPanelValues=zeros(length(FnsToEvaluate), simoptions.simperiods, simoptions.numbersims);

%% Precompute the gridvals vectors.
a_gridvals=CreateGridvals(n_a,a_grid,1); % 1 at end indicates output as matrices.
z_gridvals=CreateGridvals(n_z,z_grid,1); % 1 at end indicates output as matrices.

d_val=zeros(1,l_d);
aprime_val=zeros(1,l_a);
% a_val=zeros(1,l_a);
% z_val=zeros(1,l_z);

%% For sure the following could be made faster by parallelizing some stuff.
if simoptions.agententryandexit==0
    SimPanelValues_ii=nan(length(FnsToEvaluate),simoptions.simperiods);
    for ii=1:simoptions.numbersims
        SimPanel_ii=SimPanelIndexes(:,:,ii);
        for t=1:simoptions.simperiods
            a_sub=SimPanel_ii(1:l_a,t);
            a_ind=sub2ind_homemade(n_a,a_sub);
            
            a_val=a_gridvals(a_ind,:);
            
            z_sub=SimPanel_ii((l_a+1):(l_a+l_z),t);
            z_ind=sub2ind_homemade(n_z,z_sub);
            z_val=z_gridvals(z_ind,:);
            
            j_ind=SimPanel_ii(end,t);
            
            if l_d==0
                aprime_ind=PolicyIndexesKron(a_ind,z_ind);  % Given dependence on t I suspect precomputing this as aprime_gridvals and d_gridvals would not be worthwhile
                aprime_sub=ind2sub_homemade(n_a,aprime_ind);
            else
                temp=PolicyIndexesKron(:,a_ind,z_ind);
                d_ind=temp(1); aprime_ind=temp(2);
                d_sub=ind2sub_homemade(n_d,d_ind);
                aprime_sub=ind2sub_homemade(n_a,aprime_ind);
                for kk1=1:l_d
                    if kk1==1
                        d_val(kk1)=d_grid(d_sub(kk1));
                    else
                        d_val(kk1)=d_grid(d_sub(kk1)+sum(n_d(1:kk1-1)));
                    end
                end
            end
            for kk2=1:l_a
                if kk2==1
                    aprime_val(kk2)=a_grid(aprime_sub(kk2));
                else
                    aprime_val(kk2)=a_grid(aprime_sub(kk2)+sum(n_a(1:kk2-1)));
                end
            end
            
            if l_d==0
                for vv=1:length(FnsToEvaluate)
                    if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'SSvalueParamNames={}'
                        tempcell=num2cell([aprime_val,a_val,z_val]');
                    else
                        ValuesFnParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,j_ind);
                        tempcell=num2cell([aprime_val,a_val,z_val,ValuesFnParamsVec]');
                    end
                    SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(tempcell{:});
                end
            else
                for vv=1:length(FnsToEvaluate)
                    if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'SSvalueParamNames={}'
                        tempcell=num2cell([d_val,aprime_val,a_val,z_val]');
                    else
                        ValuesFnParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,j_ind);
                        tempcell=num2cell([d_val,aprime_val,a_val,z_val,ValuesFnParamsVec]');
                    end
                    SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(tempcell{:});
                end
            end
            SimPanelValues(:,:,ii)=SimPanelValues_ii;
        end
    end
elseif simoptions.agententryandexit==1 && simoptions.endogenousexit==0
    % Need to add check for nan relating to a_ind and z_ind around entry/exit
    for ii=1:simoptions.numbersims
        SimPanelValues_ii=nan(length(FnsToEvaluate),simoptions.simperiods); % Want nan when agents 'die/exit' before end of panel
        SimPanel_ii=SimPanelIndexes(:,:,ii);
        for t=1:simoptions.simperiods
            a_sub=SimPanel_ii(1:l_a,t);
            a_ind=sub2ind_homemade(n_a,a_sub);
            if ~isnan(a_ind)
                a_val=a_gridvals(a_ind,:);
                
                z_sub=SimPanel_ii((l_a+1):(l_a+l_z),t);
                z_ind=sub2ind_homemade(n_z,z_sub);
                z_val=z_gridvals(z_ind,:);
                
                j_ind=SimPanel_ii(end,t);
                
                if l_d==0
                    aprime_ind=PolicyIndexesKron(a_ind,z_ind);  % Given dependence on t I suspect precomputing this as aprime_gridvals and d_gridvals would not be worthwhile
                    aprime_sub=ind2sub_homemade(n_a,aprime_ind);
                else
                    temp=PolicyIndexesKron(:,a_ind,z_ind);
                    d_ind=temp(1); aprime_ind=temp(2);
                    d_sub=ind2sub_homemade(n_d,d_ind);
                    aprime_sub=ind2sub_homemade(n_a,aprime_ind);
                    for kk1=1:l_d
                        if kk1==1
                            d_val(kk1)=d_grid(d_sub(kk1));
                        else
                            d_val(kk1)=d_grid(d_sub(kk1)+sum(n_d(1:kk1-1)));
                        end
                    end
                end
                for kk2=1:l_a
                    if kk2==1
                        aprime_val(kk2)=a_grid(aprime_sub(kk2));
                    else
                        aprime_val(kk2)=a_grid(aprime_sub(kk2)+sum(n_a(1:kk2-1)));
                    end
                end
                
                if l_d==0
                    for vv=1:length(FnsToEvaluate)
                        if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'SSvalueParamNames={}'
                            tempcell=num2cell([aprime_val,a_val,z_val]');
                        else
                            ValuesFnParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,j_ind);
                            tempcell=num2cell([aprime_val,a_val,z_val,ValuesFnParamsVec]');
                        end
                        SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(tempcell{:});
                    end
                else
                    for vv=1:length(FnsToEvaluate)
                        if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'SSvalueParamNames={}'
                            tempcell=num2cell([d_val,aprime_val,a_val,z_val]');
                        else
                            ValuesFnParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,j_ind);
                            tempcell=num2cell([d_val,aprime_val,a_val,z_val,ValuesFnParamsVec]');
                        end
                        SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(tempcell{:});
                    end
                end
            end
            SimPanelValues(:,:,ii)=SimPanelValues_ii;
        end
    end
elseif simoptions.agententryandexit==1 && simoptions.endogenousexit==1
    % Need to add check for nan relating to a_ind and z_ind around entry/exit
    % Need to add check for zeros relating to aprime_ind endogenous exit
    for ii=1:simoptions.numbersims
        SimPanelValues_ii=nan(length(FnsToEvaluate),simoptions.simperiods); % Want nan when agents 'die/exit' before end of panel
        SimPanel_ii=SimPanelIndexes(:,:,ii);
        for t=1:simoptions.simperiods
            a_sub=SimPanel_ii(1:l_a,t);
            a_ind=sub2ind_homemade(n_a,a_sub);
            if ~isnan(a_ind)
                a_val=a_gridvals(a_ind,:);
                
                z_sub=SimPanel_ii((l_a+1):(l_a+l_z),t);
                z_ind=sub2ind_homemade(n_z,z_sub);
                z_val=z_gridvals(z_ind,:);
                
                j_ind=SimPanel_ii(end,t);
                
                if l_d==0
                    aprime_ind=PolicyIndexesKron(a_ind,z_ind);  % Given dependence on t I suspect precomputing this as aprime_gridvals and d_gridvals would not be worthwhile
                    aprime_sub=ind2sub_homemade(n_a,aprime_ind);
                else
                    temp=PolicyIndexesKron(:,a_ind,z_ind);
                    d_ind=temp(1); aprime_ind=temp(2);
                    d_sub=ind2sub_homemade(n_d,d_ind);
                    aprime_sub=ind2sub_homemade(n_a,aprime_ind);
                    if d_ind~=0 % Not exiting (that or using 'keeppolicy' option)
                        for kk1=1:l_d
                            if kk1==1
                                d_val(kk1)=d_grid(d_sub(kk1));
                            else
                                d_val(kk1)=d_grid(d_sub(kk1)+sum(n_d(1:kk1-1)));
                            end
                        end
                    end
                end
                if aprime_ind~=0 % Not exiting (that or using 'keeppolicy' option)
                    for kk2=1:l_a
                        if kk2==1
                            aprime_val(kk2)=a_grid(aprime_sub(kk2));
                        else
                            aprime_val(kk2)=a_grid(aprime_sub(kk2)+sum(n_a(1:kk2-1)));
                        end
                    end
                end
                
                if aprime_ind~=0 % Not exiting (that or using 'keeppolicy' option) % (can only get d_val=0 in same situation as aprime=0, so no need to check both of them)
                    if l_d==0
                        for vv=1:length(FnsToEvaluate)
                            if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'SSvalueParamNames={}'
                                tempcell=num2cell([aprime_val,a_val,z_val]');
                            else
                                ValuesFnParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,j_ind);
                                tempcell=num2cell([aprime_val,a_val,z_val,ValuesFnParamsVec]');
                            end
                            SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(tempcell{:});
                        end
                    else
                        for vv=1:length(FnsToEvaluate)
                            if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'SSvalueParamNames={}'
                                tempcell=num2cell([d_val,aprime_val,a_val,z_val]');
                            else
                                ValuesFnParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,j_ind);
                                tempcell=num2cell([d_val,aprime_val,a_val,z_val,ValuesFnParamsVec]');
                            end
                            SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(tempcell{:});
                        end
                    end
                end
            end
            SimPanelValues(:,:,ii)=SimPanelValues_ii;
        end
    end
end

end



