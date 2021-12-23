function SimLifeCycleProfiles=SimLifeCycleProfiles_FHorz_Case1(InitialDist,Policy,FnsToEvaluate,FnsToEvaluateParamNames,Parameters,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,pi_z, simoptions)
% Simulates a panel based on PolicyIndexes of 'numbersims' agents of length
% 'simperiods' beginning from randomly drawn InitialDist. Then based on
% this it computes life-cycle profiles for the 'FnsToEvalute' and reports
% mean, median, min, 19 intermediate ventiles, and max. (you can change from
% ventiles using simperiods.lifecyclepercentiles)
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
    if ~isfield(simoptions,'parallel')
        simoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions,'simperiods')
        simoptions.simperiods=N_j;
    end
    if ~isfield(simoptions,'numbersims')
        simoptions.numbersims=10^4; % Given that aim is to calculate ventiles of life-cycle profiles 10^4 seems appropriate
    end
    if ~isfield(simoptions,'lifecyclepercentiles')
        simoptions.lifecyclepercentiles=20; % by default gives ventiles
    end
    if isfield(simoptions,'ExogShockFn') % If using ExogShockFn then figure out the parameter names
        simoptions.ExogShockFnParamNames=getAnonymousFnInputNames(simoptions.ExogShockFn);
    end
else
    %If simoptions is not given, just use all the defaults
    simoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    simoptions.verbose=0;
    simoptions.simperiods=N_j;
    simoptions.numbersims=10^4; % Given that aim is to calculate ventiles of life-cycle profiles 10^4 seems appropriate
    simoptions.lifecyclepercentiles=20; % by default gives ventiles
end

if isempty(n_d)
    n_d=0;
    l_d=0;
elseif n_d(1)==0
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

PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z, N_j);%,simoptions); % This is actually being created inside SimPanelIndexes already

% if simoptions.parallel~=2
%     PolicyIndexesKron=gather(PolicyIndexesKron);
% end
% Move everything to cpu for what remains.
simoptions.parallel=1;
d_grid=gather(d_grid);
a_grid=gather(a_grid);
z_grid=gather(z_grid);
PolicyIndexesKron=gather(PolicyIndexesKron);
InitialDist=gather(InitialDist);
pi_z=gather(pi_z);

%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        if length(temp)>(l_d+l_a+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(AggVarNames{ff});
    end    
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end

%% Because we want life-cycle profiles we only use the part of InitialDist that is how agents appear 'at birth' (in j=1).
% (InitialDist is either n_a-by-n_z-by-n_j, or n_a-by-n_z)
if numel(InitialDist)==N_a*N_z % Has just been given for age j=1
    SimPanelIndexes=SimPanelIndexes_FHorz_Case1(InitialDist,PolicyIndexesKron,n_d,n_a,n_z,N_j,pi_z, simoptions);
else
    InitialDist=reshape(InitialDist,[N_a*N_z,N_j]);
    SimPanelIndexes=SimPanelIndexes_FHorz_Case1(InitialDist(:,1),PolicyIndexesKron,n_d,n_a,n_z,N_j,pi_z, simoptions); % Use only j=1: InitialDist(:,1)
end

%% Precompute the gridvals vectors.
z_gridvals=CreateGridvals(n_z,z_grid,1);
a_gridvals=CreateGridvals(n_a,a_grid,1);

eval('fieldexists_ExogShockFn=1;simoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;simoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')
eval('fieldexists_pi_z_J=1;simoptions.pi_z_J;','fieldexists_pi_z_J=0;')

if fieldexists_pi_z_J
    for jj=1:N_j
        fullgridvals(jj).z_gridvals=CreateGridvals(n_z,simoptions.z_grid_J(:,jj),1);
    end
elseif fieldexists_ExogShockFn==1
    for jj=1:N_j
        if fieldexists_ExogShockFnParamNames==1
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
            ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
            for kk=1:length(ExogShockFnParamsVec)
                ExogShockFnParamsCell(kk,1)={ExogShockFnParamsVec(kk)};
            end
            [z_grid,~]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
        else
            [z_grid,~]=simoptions.ExogShockFn(jj);
        end
        fullgridvals(jj).z_gridvals=CreateGridvals(n_z,z_grid,1);
    end
else
    for jj=1:N_j
        fullgridvals(jj).z_gridvals=z_gridvals;
    end
end

for jj=1:N_j
    PolicyIndexes=reshape(Policy,[size(Policy,1),N_a,N_z,N_j]);
    [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes(:,:,:,jj),n_d,n_a,n_a,n_z,d_grid,a_grid,1, 1);
    if l_d~=0
        fullgridvals(jj).d_gridvals=d_gridvals;
    end
    fullgridvals(jj).aprime_gridvals=aprime_gridvals;
end

%%
SimPanelValues=zeros(length(FnsToEvaluate), simoptions.simperiods, simoptions.numbersims);

%% For sure the following could be made faster by parallelizing some stuff. (It could likely be done on the gpu, currently just do single cpu and parallel cpu)
if simoptions.parallel==0
    SimPanelValues_ii=zeros(length(FnsToEvaluate),simoptions.simperiods);
    for ii=1:simoptions.numbersims
        SimPanel_ii=SimPanelIndexes(:,:,ii);
        for tt=1:simoptions.simperiods
            a_sub=SimPanel_ii(1:l_a,tt);
            a_ind=sub2ind_homemade(n_a,a_sub);
            
            z_sub=SimPanel_ii((l_a+1):(l_a+l_z),tt);
            z_ind=sub2ind_homemade(n_z,z_sub);
            
            j_ind=SimPanel_ii(end,tt);
            
            a_val=a_gridvals(a_ind,:);
            z_val=fullgridvals(j_ind).z_gridvals(z_ind,:);
            if l_d==0
                aprime_ind=PolicyIndexesKron(a_ind,z_ind,j_ind); % Given dependence on t I suspect precomputing this as aprime_gridvals and d_gridvals would not be worthwhile
                aprime_val=fullgridvals(j_ind).aprime_gridvals(aprime_ind,:);
            else
                temp=PolicyIndexesKron(:,a_ind,z_ind,j_ind);
                d_ind=temp(1); aprime_ind=temp(2);
                aprime_val=fullgridvals(j_ind).aprime_gridvals(aprime_ind,:);
                d_val=fullgridvals(j_ind).d_gridvals(d_ind,:);
            end
            
            if l_d==0
                for vv=1:length(FnsToEvaluate)
                    if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames={}'
                        tempcell=num2cell([aprime_val,a_val,z_val]');
                    else
                        ValuesFnParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,j_ind);
                        tempcell=num2cell([aprime_val,a_val,z_val,ValuesFnParamsVec]');
                    end
                    SimPanelValues_ii(vv,tt)=FnsToEvaluate{vv}(tempcell{:});
                end
            else
                for vv=1:length(FnsToEvaluate)
                    if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames={}'
                        tempcell=num2cell([d_val,aprime_val,a_val,z_val]');
                    else
                        ValuesFnParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,j_ind);
                        tempcell=num2cell([d_val,aprime_val,a_val,z_val,ValuesFnParamsVec]');
                    end
                    SimPanelValues_ii(vv,tt)=FnsToEvaluate{vv}(tempcell{:});
                end
            end
        end
        SimPanelValues(:,:,ii)=SimPanelValues_ii;
    end
else % simoptions.parallel==1
    parfor ii=1:simoptions.numbersims % This is only change from simoptions.parallel==0 case
        SimPanel_ii=SimPanelIndexes(:,:,ii);
        SimPanelValues_ii=zeros(length(FnsToEvaluate),simoptions.simperiods);
        for tt=1:simoptions.simperiods
            a_sub=SimPanel_ii(1:l_a,tt);
            a_ind=sub2ind_homemade(n_a,a_sub);
            
            z_sub=SimPanel_ii((l_a+1):(l_a+l_z),tt);
            z_ind=sub2ind_homemade(n_z,z_sub);
            
            j_ind=SimPanel_ii(end,tt);
            
            a_val=a_gridvals(a_ind,:);
            z_val=fullgridvals(j_ind).z_gridvals(z_ind,:);
            if l_d==0
                aprime_ind=PolicyIndexesKron(a_ind,z_ind,j_ind); % Given dependence on t I suspect precomputing this as aprime_gridvals and d_gridvals would not be worthwhile
                aprime_val=fullgridvals(j_ind).aprime_gridvals(aprime_ind,:);
            else
                temp=PolicyIndexesKron(:,a_ind,z_ind,j_ind);
                d_ind=temp(1); aprime_ind=temp(2);
                aprime_val=fullgridvals(j_ind).aprime_gridvals(aprime_ind,:);
                d_val=fullgridvals(j_ind).d_gridvals(d_ind,:);
            end
            
            if l_d==0
                for vv=1:length(FnsToEvaluate)
                    if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames={}'
                        tempcell=num2cell([aprime_val,a_val,z_val]');
                    else
                        ValuesFnParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,j_ind);
                        tempcell=num2cell([aprime_val,a_val,z_val,ValuesFnParamsVec]');
                    end
                    SimPanelValues_ii(vv,tt)=FnsToEvaluate{vv}(tempcell{:});
                end
            else
                for vv=1:length(FnsToEvaluate)
                    if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames={}'
                        tempcell=num2cell([d_val,aprime_val,a_val,z_val]');
                    else
                        ValuesFnParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,j_ind);
                        tempcell=num2cell([d_val,aprime_val,a_val,z_val,ValuesFnParamsVec]');
                    end
                    SimPanelValues_ii(vv,tt)=FnsToEvaluate{vv}(tempcell{:});
                end
            end
        end
        SimPanelValues(:,:,ii)=SimPanelValues_ii;
    end
end

% Now get the life-cycle profiles by looking at (age-conditional) cross-sections of this panel data.
if simoptions.lifecyclepercentiles>0
    SimLifeCycleProfiles=zeros(length(FnsToEvaluate), simoptions.simperiods, simoptions.lifecyclepercentiles+3); % Mean, median, min, 19 intermediate ventiles, and max.
    prctilegrid=(0:1/simoptions.lifecyclepercentiles:1)*100;
    for ii=1:length(FnsToEvaluate)
        for tt=1:simoptions.simperiods
            temp=SimPanelValues(ii,tt,:);
            SimLifeCycleProfiles(ii,tt,1)=mean(temp);
            SimLifeCycleProfiles(ii,tt,2)=median(temp);
            SimLifeCycleProfiles(ii,tt,3:end)=prctile(temp,prctilegrid);
        end
    end
else
    SimLifeCycleProfiles=zeros(length(FnsToEvaluate), simoptions.simperiods, 2); % Mean, median
    for ii=1:length(FnsToEvaluate)
        for tt=1:simoptions.simperiods
            temp=SimPanelValues(ii,tt,:);
            SimLifeCycleProfiles(ii,tt,1)=mean(temp);
            SimLifeCycleProfiles(ii,tt,2)=median(temp);
        end
    end
end


if FnsToEvaluateStruct==1
    % Change the output into a structure
    SimLifeCycleProfiles2=SimLifeCycleProfiles;
    clear SimLifeCycleProfiles
    SimLifeCycleProfiles=struct();
%     AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        SimLifeCycleProfiles.(AggVarNames{ff}).Mean=SimLifeCycleProfiles2(ff,:,1);
        SimLifeCycleProfiles.(AggVarNames{ff}).Median=SimLifeCycleProfiles2(ff,:,2);
        if simoptions.lifecyclepercentiles>0
            SimLifeCycleProfiles.(AggVarNames{ff}).prctile=SimLifeCycleProfiles2(ff,:,3:end);
        end
    end
end


end
