function SimPanelValues=SimPanelValues_FHorz_Case1(InitialDist,Policy,FnsToEvaluate,FnsToEvaluateParamNames,Parameters,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,pi_z, simoptions)
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
        simoptions.numbersims=10^3;
    end 
    if isfield(simoptions,'ExogShockFn') % If using ExogShockFn then figure out the parameter names
        simoptions.ExogShockFnParamNames=getAnonymousFnInputNames(simoptions.ExogShockFn);
    end
else
    %If simoptions is not given, just use all the defaults
    simoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    simoptions.verbose=0;
    simoptions.simperiods=N_j;
    simoptions.numbersims=10^3;
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

% NOTE: ESSENTIALLY ALL THE RUN TIME IS IN THIS COMMAND. WOULD BE GOOD TO OPTIMIZE/IMPROVE.
PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z, N_j);%,simoptions); % Create it here as want it both here and inside SimPanelIndexes_FHorz_Case1 (which will recognise that it is already in this form)

SimPanelIndexes=SimPanelIndexes_FHorz_Case1(InitialDist,PolicyIndexesKron,n_d,n_a,n_z,N_j,pi_z, simoptions);

% COMMENT: I COULD MAKE THINGS FASTER BY JUST DOING
% SimPanelIndexes TO CREATE KRON, AS I AM JUST HAVING TO CONVERT TO THIS
% LATER TO ASSIGN VALUES

% Move everything to cpu for what remains.
d_grid=gather(d_grid);
a_grid=gather(a_grid);
z_grid=gather(z_grid);
PolicyIndexesKron=gather(PolicyIndexesKron);

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


%%
SimPanelValues=zeros(length(FnsToEvaluate), simoptions.simperiods, simoptions.numbersims);

%% Precompute the gridvals vectors.
a_gridvals=CreateGridvals(n_a,a_grid,1); % 1 at end indicates output as matrices.

d_val=zeros(1,l_d);
aprime_val=zeros(1,l_a);

eval('fieldexists_ExogShockFn=1;simoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;simoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')
eval('fieldexists_pi_z_J=1;simoptions.pi_z_J;','fieldexists_pi_z_J=0;')

if fieldexists_pi_z_J
    for jj=1:N_j
        fullgridvals(jj).z_gridvals=CreateGridvals(n_z,simoptions.z_grid_J(:,:,jj),1);
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
    z_gridvals=CreateGridvals(n_z,z_grid,1); % 1 at end indicates output as matrices.
    for jj=1:N_j
        fullgridvals(jj).z_gridvals=z_gridvals;
    end
end

%%
SimPanelValues_ii=nan(length(FnsToEvaluate),simoptions.simperiods); % Want nan when agents 'die' (reach N_j) before end of panel

%% For sure the following could be made faster by improving how I do it
parfor ii=1:simoptions.numbersims
    SimPanel_ii=SimPanelIndexes(:,:,ii);
    t=0;
    j_ind=0;
    d_val=0; % This and following two lines are just to help matlab figure out how to parfor
    aprime_val=0;
    SimPanelValues_ii=zeros(length(FnsToEvaluate),simoptions.simperiods);
    while t<=simoptions.simperiods && j_ind<N_j % Once we pass N_j all entries are just nan; j_ind<N_j last round means at most j_ind<=N_j this round
        t=t+1;        
        j_ind=SimPanel_ii(end,t);

        a_sub=SimPanel_ii(1:l_a,t);
        a_ind=sub2ind_homemade(n_a,a_sub);
        a_val=a_gridvals(a_ind,:);
        
        z_sub=SimPanel_ii((l_a+1):(l_a+l_z),t);
        z_ind=sub2ind_homemade(n_z,z_sub);
        z_val=fullgridvals(j_ind).z_gridvals(z_ind,:);
        
        
        if l_d==0
            aprime_ind=PolicyIndexesKron(a_ind,z_ind,t);  % Given dependence on t I suspect precomputing this as aprime_gridvals and d_gridvals would not be worthwhile
            aprime_sub=ind2sub_homemade(n_a,aprime_ind);
        else
            temp=PolicyIndexesKron(:,a_ind,z_ind,t);
            d_ind=temp(1); 
            aprime_ind=temp(2);
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
                if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames={}'
                   tempcell=num2cell([aprime_val,a_val,z_val]');
                else
                   ValuesFnParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,j_ind);
                   tempcell=num2cell([aprime_val,a_val,z_val,ValuesFnParamsVec]');
                end
                SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(tempcell{:});
            end
        else
            for vv=1:length(FnsToEvaluate)
                if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames={}'
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

%% Implement new way of handling FnsToEvaluate: convert results
if FnsToEvaluateStruct==1
    if FnsToEvaluateStruct==1
    % Change the output into a structure
    SimPanelValues2=SimPanelValues;
    clear SimPanelValues
    SimPanelValues=struct();
%     AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        SimPanelValues.(AggVarNames{ff})=shiftdim(SimPanelValues2(ff,:,:),1);
    end
    end
end



end



