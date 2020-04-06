function SimPanelValues=SimPanelValues_TransPath_Case1(PricePath, ParamPath, T, V_final, AgentDist_initial, n_d, n_a, n_z, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, transpathoptions,simoptions)
% This code will work for all transition paths except those that involve at
% change in the transition matrix pi_z (can handle a change in pi_z, but
% only if it is a 'surprise', not anticipated changes) 

% PricePathOld is matrix of size T-by-'number of prices'
% ParamPath is matrix of size T-by-'number of parameters that change over path'

% Simulates a panel based on PolicyIndexes of 'numbersims' agents of length
% 'T' beginning from randomly drawn InitialDist.
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

% Internally PricePathOld is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'. 
PricePathNames=fieldnames(PricePath);
PricePathStruct=PricePath;
PricePath=zeros(T,length(PricePathNames));
for ii=1:length(PricePathNames)
    PricePath(:,ii)=PricePathStruct.(PricePathNames{ii});
end
ParamPathNames=fieldnames(ParamPath);
ParamPathStruct=ParamPath; 
ParamPath=zeros(T,length(ParamPathNames));
for ii=1:length(ParamPathNames)
    ParamPath(:,ii)=ParamPathStruct.(ParamPathNames{ii});
end


%% Check which transpathoptions and simoptions have been declared, set all others to defaults 
if exist('transpathoptions','var')==0
    disp('No transpathoptions given, using defaults')
    %If transpathoptions is not given, just use all the defaults
    transpathoptions.parallel=2;
    transpathoptions.lowmemory=0;
    transpathoptions.exoticpreferences=0;
    transpathoptions.verbose=0;
else
    %Check transpathoptions for missing fields, if there are some fill them with the defaults
    if isfield(transpathoptions,'parallel')==0
        transpathoptions.parallel=2;
    end
    if isfield(transpathoptions,'lowmemory')==0
        transpathoptions.lowmemory=0;
    end
    if isfield(transpathoptions,'exoticpreferences')==0
        transpathoptions.exoticpreferences=0;
    end
    if isfield(transpathoptions,'verbose')==0
        transpathoptions.verbose=0;
    end
end

if exist('simoptions','var')==1
    %Check simoptions for missing fields, if there are some fill them with the defaults
    if isfield(simoptions,'parallel')==0
        simoptions.parallel=2;
    end
    if isfield(simoptions,'verbose')==0
        simoptions.verbose=0;
    end
    if isfield(simoptions,'simperiods')==0
        simoptions.simperiods=T-1; % This can be made shorter, but not longer
    end
    if isfield(simoptions,'numbersims')==0
        simoptions.numbersims=10^3;
    end    
else
    %If simoptions is not given, just use all the defaults
    simoptions.parallel=2;
    simoptions.verbose=0;
    simoptions.simperiods=T-1;
    simoptions.numbersims=10^3;
end

if transpathoptions.exoticpreferences~=0
    disp('ERROR: Only transpathoptions.exoticpreferences==0 is supported by TransitionPath_Case1')
    dbstack
else
    if length(DiscountFactorParamNames)~=1
        disp('WARNING: DiscountFactorParamNames should be of length one')
        dbstack
    end
end
if transpathoptions.parallel~=2
    disp('ERROR: Only transpathoptions.parallel==2 is supported by TransitionPath_Case1')
else
    d_grid=gpuArray(d_grid); a_grid=gpuArray(a_grid); z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
    PricePath=gpuArray(PricePath);
end
unkronoptions.parallel=2;


if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);
l_p=size(PricePath,2);

if transpathoptions.parallel==2
    % Make sure things are on gpu where appropriate.
    if N_d>0
        d_grid=gather(d_grid);
    end
    a_grid=gather(a_grid);
    z_grid=gather(z_grid);
end
if simoptions.parallel~=2
    d_grid=gather(d_grid);
    a_grid=gather(a_grid);
    z_grid=gather(z_grid);
end

if N_d==0
    fprintf('SimPanelValues_TransPath_Case1 with no d variable has not yet been implemented: please email me if you want to be able to do this \n')
%     SimPanelValues=SimPanelValues_TransPath_Case1_nod(PricePath, PricePathNames, ParamPath, ParamPathNames, T, V_final, InitialDist, n_a, n_z, pi_z, a_grid,z_grid, ReturnFn, FnsToEvaluate, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, transpathoptions,simoptions);
    % NOTE: ACTUALLY, JUST NEED TO IMPLEMENT THE SimPanelIndexes FOR no_d,
    % THE LOWER PART OF THIS SCRIPT ALREADY ALLOWS FOR no d variable
    dbstack
    return
end

if transpathoptions.lowmemory==1
    % The lowmemory option is going to use gpu (but loop over z instead of parallelize) for value fn.
    [SimPanelIndexes,PolicyIndexesKron]=SimPanelIndexes_TransPath_Case1_lowmem(PricePath, PricePathNames, ParamPath, ParamPathNames, T, V_final, AgentDist_initial, n_d, n_a, n_z, pi_z, d_grid,a_grid,z_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, transpathoptions,simoptions);
else
    [SimPanelIndexes,PolicyIndexesKron]=SimPanelIndexes_TransPath_Case1(PricePath, PricePathNames, ParamPath, ParamPathNames, T, V_final, AgentDist_initial, n_d, n_a, n_z, pi_z, d_grid,a_grid,z_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, transpathoptions,simoptions);
end

% Move everything to cpu for what remains.
d_grid=gather(d_grid);
a_grid=gather(a_grid);
z_grid=gather(z_grid);
PolicyIndexesKron=gather(PolicyIndexesKron);
PricePath=gather(PricePath);
ParamPath=gather(ParamPath);

SimPanelValues=zeros(length(FnsToEvaluate), simoptions.simperiods, simoptions.numbersims);

%% Precompute the gridvals vectors.
z_gridvals=CreateGridvals(n_z,z_grid,1); % 1 is to create z_gridvals as matrix
a_gridvals=CreateGridvals(n_a,a_grid,1); % 1 is to create a_gridvals as matrix

d_val=zeros(1,l_d);
aprime_val=zeros(1,l_a);
a_val=zeros(1,l_a);
z_val=zeros(1,l_z);

%% The following is precomputed for speed (otherwise it would end up inside the for-loop over simoptions.numbersims)

FnsToEvaluateParamsVecStruct=struct(); %struct(length(FnsToEvaluate),simoptions.simperiods);
for vv=1:length(FnsToEvaluate)
    
    FnsToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names);
    for tt=1:simoptions.simperiods
        if ~isempty(FnsToEvaluateParamNames(vv).Names)  % If it isempty() then no need to do anything.

            IndexesForParamPathInFnsToEvaluateParamsVec=CreateParamVectorIndexes(FnsToEvaluateParamNames(vv).Names, ParamPathNames);
            IndexesForPricePathInFnsToEvaluateParamsVec=CreateParamVectorIndexes(FnsToEvaluateParamNames(vv).Names, PricePathNames);

            IndexesForFnsToEvaluateParamsInParamPath=CreateParamVectorIndexes(ParamPathNames, FnsToEvaluateParamNames(vv).Names);
            IndexesForFnsToEvaluateParamsInPricePath=CreateParamVectorIndexes(PricePathNames, FnsToEvaluateParamNames(vv).Names);

            if ~isnan(IndexesForPricePathInFnsToEvaluateParamsVec)
                FnsToEvaluateParamsVec(IndexesForPricePathInFnsToEvaluateParamsVec)=PricePath(tt,IndexesForFnsToEvaluateParamsInPricePath);
            end
            if ~isnan(IndexesForParamPathInFnsToEvaluateParamsVec)
                FnsToEvaluateParamsVec(IndexesForParamPathInFnsToEvaluateParamsVec)=ParamPath(tt,IndexesForFnsToEvaluateParamsInParamPath);
            end
            
        end
        FnsToEvaluateParamsVecStruct.(['vv',num2str(vv)]).(['tt',num2str(tt)])=FnsToEvaluateParamsVec;
    end
end

%%
SimPanelValues_ii=nan(length(FnsToEvaluate),simoptions.simperiods); % Want nan when agents 'die' (reach N_j) before end of panel
%% For sure the following could be made faster by parallelizing some stuff.

for ii=1:simoptions.numbersims
    SimPanel_ii=SimPanelIndexes(:,:,ii);
    for tt=1:simoptions.simperiods
        a_sub=SimPanel_ii(1:l_a,tt);
        a_ind=sub2ind_homemade(n_a,a_sub);
        a_val=a_gridvals(a_ind,:);
         
        z_sub=SimPanel_ii((l_a+1):(l_a+l_z),tt);
        z_ind=sub2ind_homemade(n_z,z_sub);
        z_val=z_gridvals(z_ind,:);
        
%         j_ind=SimPanel_ii(end,t);
        
        if l_d==0
            aprime_ind=PolicyIndexesKron(a_ind,z_ind,tt);  % Given dependence on t I suspect precomputing this as aprime_gridvals and d_gridvals would not be worthwhile
            aprime_sub=ind2sub_homemade(n_a,aprime_ind);
        else
            temp=PolicyIndexesKron(:,a_ind,z_ind,tt);
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
%                    FnsToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,j_ind);
                    FnsToEvaluateParamsVec=FnsToEvaluateParamsVecStruct.(['vv',num2str(vv)]).(['tt',num2str(tt)]);
                    tempcell=num2cell([aprime_val,a_val,z_val,FnsToEvaluateParamsVec]');
                end
                SimPanelValues_ii(vv,tt)=FnsToEvaluate{vv}(tempcell{:});
            end
        else
            for vv=1:length(FnsToEvaluate)
                if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'SSvalueParamNames={}'
                    tempcell=num2cell([d_val,aprime_val,a_val,z_val]');
                else
%                     FnsToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,j_ind);
                    FnsToEvaluateParamsVec=FnsToEvaluateParamsVecStruct.(['vv',num2str(vv)]).(['tt',num2str(tt)]);
                    tempcell=num2cell([d_val,aprime_val,a_val,z_val,FnsToEvaluateParamsVec]');
                end
                SimPanelValues_ii(vv,tt)=FnsToEvaluate{vv}(tempcell{:});
            end
        end
        
    end
    SimPanelValues(:,:,ii)=SimPanelValues_ii;
end


end



