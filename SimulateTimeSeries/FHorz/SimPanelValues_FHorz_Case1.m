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

%%
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

%%
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
    z_gridvals=CreateGridvals(n_z,z_grid,1); % 1 at end indicates output as matrices.
    for jj=1:N_j
        fullgridvals(jj).z_gridvals=z_gridvals;
    end
end

% If using n_e need to set that up too
if isfield(simoptions,'n_e')
    % Because of how FnsToEvaluate works I can just get the e variables and
    % then 'combine' them with z
    eval('fieldexists_EiidShockFn=1;simoptions.EiidShockFn;','fieldexists_EiidShockFn=0;')
    eval('fieldexists_EiidShockFnParamNames=1;simoptions.EiidShockFnParamNames;','fieldexists_EiidShockFnParamNames=0;')
    eval('fieldexists_pi_e_J=1;simoptions.pi_e_J;','fieldexists_pi_e_J=0;')
    
%     N_e=prod(simoptions.n_e);
    l_e=length(simoptions.n_e);
    
    if fieldexists_pi_e_J==1
        e_grid_J=simoptions.e_grid_J;
    elseif fieldexists_EiidShockFn==1
        e_grid_J=zeros(sum(simoptions.n_e),N_j);
        for jj=1:N_j
            if fieldexists_EiidShockFnParamNames==1
                EiidShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.EiidShockFnParamNames,jj);
                EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                for ii=1:length(EiidShockFnParamsVec)
                    EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                end
                [e_grid,~]=simoptions.EiidShockFn(EiidShockFnParamsCell{:});
            else
                [e_grid,~]=simoptions.EiidShockFn(jj);
            end
            e_grid_J(:,jj)=gather(e_grid);
        end
    else
        e_grid_J=repmat(simoptions.e_grid,1,N_j);
    end
    
    for jj=1:N_j
        fullgridvals(jj).e_gridvals=CreateGridvals(simoptions.n_e,e_grid_J(:,jj),1);
    end
end

%%
d_grid=gather(d_grid);
a_grid=gather(a_grid);

% NOTE: ESSENTIALLY ALL THE RUN TIME IS IN THIS COMMAND. WOULD BE GOOD TO OPTIMIZE/IMPROVE.
if isfield(simoptions,'n_e')
    PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z, N_j,simoptions.n_e); % Create it here as want it both here and inside SimPanelIndexes_FHorz_Case1 (which will recognise that it is already in this form)
else
    PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z, N_j); % Create it here as want it both here and inside SimPanelIndexes_FHorz_Case1 (which will recognise that it is already in this form)
end
simoptions.simpanelindexkron=1; % Keep the output as kron form as will want this later anyway for assigning the values
SimPanelIndexes=SimPanelIndexes_FHorz_Case1(InitialDist,PolicyIndexesKron,n_d,n_a,n_z,N_j,pi_z, simoptions);

PolicyIndexesKron=gather(PolicyIndexesKron);

% disp('HERE3')
% max(max(max(max(abs(PolicyIndexesKron-round(PolicyIndexesKron))))))
% min(min(min(min(isfinite(PolicyIndexesKron)))))
% min(min(min(min(abs(PolicyIndexesKron)))))
% min(min(min(min(PolicyIndexesKron))))
% % This seems fine to me

% 
% disp('HERE4')
% max(max(max(max(abs(SimPanelIndexes-round(SimPanelIndexes))))))
% min(min(min(min(isfinite(SimPanelIndexes)))))
% min(min(min(min(abs(SimPanelIndexes)))))
% min(min(min(min(SimPanelIndexes))))
% sum(sum(sum(sum(isnan(SimPanelIndexes)))))
% sum(sum(sum(sum(isfinite(SimPanelIndexes)))))
% numel(SimPanelIndexes)

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

if isfield(simoptions,'outputasstructure')
    if simoptions.outputasstructure==1
        FnsToEvaluateStruct=1;
        AggVarNames=simoptions.AggVarNames;
    elseif simoptions.outputasstructure==0
        FnsToEvaluateStruct=0;
    end
end

%%
SimPanelValues=zeros(length(FnsToEvaluate), simoptions.simperiods, simoptions.numbersims);

%% Precompute the gridvals vectors.
a_gridvals=CreateGridvals(n_a,a_grid,1); % 1 at end indicates output as matrices.

[dPolicy_gridvals, aprimePolicy_gridvals]=CreateGridvals_PolicyKron(PolicyIndexesKron,n_d,n_a,n_a,n_z,d_grid,a_grid,1, 1);

%%
simperiods=simoptions.simperiods; % Helps the parfor reduce overhead
N_a=prod(n_a);

%% For sure the following could be made faster by improving how I do it
parfor ii=1:simoptions.numbersims
    SimPanelIndexes_ii=SimPanelIndexes(:,:,ii);
    t=0;
    SimPanelValues_ii=zeros(length(FnsToEvaluate),simperiods);
    j_ind=1; % Note, this is just to satify the 'while' constraint, it will be overwritten before being used for anything
    
    while t<=simperiods && j_ind<N_j % Once we pass N_j all entries are just nan; j_ind<N_j last round means at most j_ind<=N_j this round
        t=t+1;
        
        a_ind=SimPanelIndexes_ii(1,t);
        z_ind=SimPanelIndexes_ii(2,t);
        j_ind=SimPanelIndexes_ii(3,t);
        
        if ~isnan(z_ind) % The simulations sometimes include nan values, so I use this to skip those ones
        
            az_ind=a_ind+N_a*(z_ind-1);
            
            a_val=a_gridvals(a_ind,:);
            z_val=fullgridvals(j_ind).z_gridvals(z_ind,:);
            
            %         j_ind=SimPanelIndexes_ii(end,t);
            %
            %         a_sub=SimPanelIndexes_ii(1:l_a,t);
            %         a_ind=sub2ind_homemade(n_a,a_sub);
            %         a_val=a_gridvals(a_ind,:);
            %
            %         z_sub=SimPanelIndexes_ii((l_a+1):(l_a+l_z),t);
            %         z_ind=sub2ind_homemade(n_z,z_sub);
            %         z_val=fullgridvals(j_ind).z_gridvals(z_ind,:);
            %
            %
            %         if l_d==0
            %             aprime_ind=PolicyIndexesKron(a_ind,z_ind,t);  % Given dependence on t I suspect precomputing this as aprime_gridvals and d_gridvals would not be worthwhile
            %             aprime_sub=ind2sub_homemade(n_a,aprime_ind);
            %         else
            %             temp=PolicyIndexesKron(:,a_ind,z_ind,t);
            %             d_ind=temp(1);
            %             aprime_ind=temp(2);
            %             d_sub=ind2sub_homemade(n_d,d_ind);
            %             aprime_sub=ind2sub_homemade(n_a,aprime_ind);
            %             for kk1=1:l_d
            %                 if kk1==1
            %                     d_val(kk1)=d_grid(d_sub(kk1));
            %                 else
            %                     d_val(kk1)=d_grid(d_sub(kk1)+sum(n_d(1:kk1-1)));
            %                 end
            %             end
            %         end
            %         for kk2=1:l_a
            %             if kk2==1
            %                 aprime_val(kk2)=a_grid(aprime_sub(kk2));
            %             else
            %                 aprime_val(kk2)=a_grid(aprime_sub(kk2)+sum(n_a(1:kk2-1)));
            %             end
            %         end
            
            if l_d==0
                aprime_val=aprimePolicy_gridvals(az_ind,:);
                
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
                d_val=dPolicy_gridvals(az_ind,:);
                aprime_val=aprimePolicy_gridvals(az_ind,:);
                
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
    end
    SimPanelValues(:,:,ii)=SimPanelValues_ii;
end

%% Implement new way of handling FnsToEvaluate: convert results
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



