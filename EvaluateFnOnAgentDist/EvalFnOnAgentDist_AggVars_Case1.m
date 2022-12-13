function AggVars=EvalFnOnAgentDist_AggVars_Case1(StationaryDist, PolicyIndexes, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel, simoptions, EntryExitParamNames, PolicyWhenExiting)
% Evaluates the aggregate value (weighted sum/integral) for each element of FnsToEvaluate
%
% Parallel, simoptions, EntryExitParamNames and PolicyWhenExiting are
% optional inputs, later two only needed when using endogenous entry and
% endogenous exit.

%%
if exist('Parallel','var')==0
    Parallel=1+(gpuDeviceCount>0);
elseif isempty(Parallel)
    Parallel=1+(gpuDeviceCount>0);
end
simoptions.parallel=Parallel;

if ~exist('simoptions', 'var')
    simoptions=struct();
end

if n_d(1)==0
    l_d=0;
    N_d=0;
else
    l_d=length(n_d);
    N_d=prod(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

N_a=prod(n_a);
N_z=prod(n_z);

%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluate_copy=FnsToEvaluate; % keep a copy in case needed for conditional restrictions
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    for ff=1:length(FnsToEvalNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(FnsToEvalNames{ff}));
        if length(temp)>(l_d+l_a+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(FnsToEvalNames{ff});
    end    
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end
if isfield(simoptions,'outputasstructure')
    if simoptions.outputasstructure==1
        FnsToEvaluateStruct=1;
        FnsToEvalNames=simoptions.AggVarNames;
    elseif simoptions.outputasstructure==0
        FnsToEvaluateStruct=0;
    end
end

%% Check for functions that use value function and send these off to a subversion called EvalFnOnAgentDist_AggVars_Case1_withV()
if isfield(simoptions,'eval_valuefn')
    AggVarsExtra=struct();
    for ff=1:length(FnsToEvalNames)
        if ~isempty(FnsToEvaluateParamNames(ff).Names)
            if strcmp(FnsToEvaluateParamNames(ff).Names{1},simoptions.eval_valuefnname{1})
                % This function to evaluate has value function as an input
                % Send it to subfunction, and remove it from FnsToEvaluate
                tempFnsToEvaluateParamNames.Names={};
                if length(FnsToEvaluateParamNames(ff).Names)>1
                    tempFnsToEvaluateParamNames(ff).Names=FnsToEvaluateParamNames(ff).Names{2:end};
                end
                AggVarsExtra=EvalFnOnAgentDist_AggVars_Case1_withV(simoptions.eval_valuefn,StationaryDist, PolicyIndexes, {FnsToEvaluate{ff}}, {FnsToEvalNames{ff}}, Parameters, tempFnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel, simoptions);
                FnsToEvaluate2=FnsToEvaluate; FnsToEvaluateParamNames2=FnsToEvaluateParamNames;
                clear FnsToEvaluateParamNames
                FnsToEvaluate={}; % Note: there may be no other FnsToEvaluate
                for gg=1:length(FnsToEvalNames)
                    if gg<ff
                        FnsToEvaluateParamNames(gg).Names=FnsToEvaluateParamNames2(gg).Names;
                        FnsToEvaluate{gg}=FnsToEvaluate2{gg};
                    elseif gg>ff
                        FnsToEvaluateParamNames(gg-1).Names=FnsToEvaluateParamNames2(gg).Names;
                        FnsToEvaluate{gg-1}=FnsToEvaluate2{gg};
                    end
                end
            end
        end
    end
    if isempty(FnsToEvaluate) % There are no FnsToEvaluate that do not have a dependence on the value function
        AggVars=AggVarsExtra;
        return
    end
end

%%
if isfield(simoptions,'statedependentparams')
    n_SDP=length(simoptions.statedependentparams.names);
    sdp=zeros(length(FnsToEvaluate),length(simoptions.statedependentparams.names));
    for ii=1:length(FnsToEvaluate)
        for jj=1:n_SDP
            for kk=1:length(FnsToEvaluateParamNames(ii).Names)
                if strcmp(simoptions.statedependentparams.names{jj},FnsToEvaluateParamNames(ii).Names{kk})
                    sdp(ii,jj)=1;
                    % Remove the statedependentparams from FnsToEvaluateParamNames
                    FnsToEvaluateParamNames(ii).Names=setdiff(FnsToEvaluateParamNames(ii).Names,simoptions.statedependentparams.names{jj});
                    % Set up the SDP variables
                end
            end
        end
    end
    if N_d>1
        n_full=[n_d,n_a,n_a,n_z];
    else
        n_full=[n_a,n_a,n_z];
    end
    
    % First state dependent parameter, get into form needed for the valuefn
    SDP1=Params.(simoptions.statedependentparams.names{1});
    SDP1_dims=simoptions.statedependentparams.dimensions.(simoptions.statedependentparams.names{1});
    %     simoptions.statedependentparams.dimensions.kmax=[3,4,5,6,7]; % The d,a & z variables (in VFI toolkit notation)
    temp=ones(1,l_d+l_a+l_a+l_z);
    for jj=1:max(SDP1_dims)
        [v,ind]=max(SDP1_dims==jj);
        if v==1
            temp(jj)=n_full(ind);
        end
    end
    if isscalar(SDP1)
        SDP1=SDP1*ones(temp);
    else
        SDP1=reshape(SDP1,temp);
    end
    if n_SDP>=2
        % Second state dependent parameter, get into form needed for the valuefn
        SDP2=Params.(simoptions.statedependentparams.names{2});
        SDP2_dims=simoptions.statedependentparams.dimensions.(simoptions.statedependentparams.names{2});
        temp=ones(1,l_d+l_a+l_a+l_z);
        for jj=1:max(SDP2_dims)
            [v,ind]=max(SDP2_dims==jj);
            if v==1
                temp(jj)=n_full(ind);
            end
        end
        if isscalar(SDP2)
            SDP2=SDP2*ones(temp);
        else
            SDP2=reshape(SDP2,temp);
        end
    end
    if n_SDP>=3
        % Third state dependent parameter, get into form needed for the valuefn
        SDP3=Params.(simoptions.statedependentparams.names{3});
        SDP3_dims=simoptions.statedependentparams.dimensions.(simoptions.statedependentparams.names{3});
        temp=ones(1,l_d+l_a+l_a+l_z);
        for jj=1:max(SDP3_dims)
            [v,ind]=max(SDP3_dims==jj);
            if v==1
                temp(jj)=n_full(ind);
            end
        end
        if isscalar(SDP3)
            SDP3=SDP3*ones(temp);
        else
            SDP3=reshape(SDP3,temp);
        end
    end
    
    % Currently SDP1 is on (n_d,n_aprime,n_a,n_z). It will be better
    % for EvalFnOnAgentDist_Grid_Case1_SDP if this is reduced to just
    % (n_a,n_z) using the Policy function.
    if l_d==0
        PolicyIndexes_sdp=reshape(PolicyIndexes(l_a,N_a,N_z));
        PolicyIndexes_sdp=permute(PolicyIndexes_sdp,[2,3,1]);
        if l_a==1
            aprime_ind=PolicyIndexes_sdp(:,:,1);
        elseif l_a==2
            aprime_ind=PolicyIndexes_sdp(:,:,1)+n_a(1)*(PolicyIndexes_sdp(:,:,2)-1);
        elseif l_a==3
            aprime_ind=PolicyIndexes_sdp(:,:,1)+n_a(1)*(PolicyIndexes_sdp(:,:,2)-1)+prod(n_a(1:2))*(PolicyIndexes_sdp(:,:,3)-1);
        elseif l_a==4
            aprime_ind=PolicyIndexes_sdp(:,:,1)+n_a(1)*(PolicyIndexes_sdp(:,:,2)-1)+prod(n_a(1:2))*(PolicyIndexes_sdp(:,:,3)-1)+prod(n_a(1:3))*(PolicyIndexes_sdp(:,:,4)-1);
        end
        aprime_ind=reshape(aprime_ind,[N_a*N_z,1]);
        a_ind=reshape((1:1:N_a)'*ones(1,N_z),[N_a*N_z,1]);
        z_ind=reshape(ones(N_a,1)*1:1:N_z,[N_a*N_z,1]);
        aprimeaz_ind=aprime_ind+N_a*(a_ind-1)+N_a*N_a*(z_ind-1);
        SDP1=SDP1(aprimeaz_ind);
        if n_SDP>=2
            SDP2=SDP2(aprimeaz_ind);
        end
        if n_SDP>=3
            SDP3=SDP3(aprimeaz_ind);
        end
    else
        PolicyIndexes_sdp=reshape(PolicyIndexes(l_d+l_a,N_a,N_z));
        PolicyIndexes_sdp=permute(PolicyIndexes_sdp,[2,3,1]);
        if l_d==1
            d_ind=PolicyIndexes_sdp(:,:,1);
        elseif l_d==2
            d_ind=PolicyIndexes_sdp(:,:,1)+n_d(1)*(PolicyIndexes_sdp(:,:,2)-1);
        elseif l_d==3
            d_ind=PolicyIndexes_sdp(:,:,1)+n_d(1)*(PolicyIndexes_sdp(:,:,2)-1)+prod(n_d(1:2))*(PolicyIndexes_sdp(:,:,3)-1);
        elseif l_d==4
            d_ind=PolicyIndexes_sdp(:,:,1)+n_d(1)*(PolicyIndexes_sdp(:,:,2)-1)+prod(n_d(1:2))*(PolicyIndexes_sdp(:,:,3)-1)+prod(n_d(1:3))*(PolicyIndexes_sdp(:,:,4)-1);
        end
        if l_a==1
            aprime_ind=PolicyIndexes_sdp(:,:,l_d+1);
        elseif l_a==2
            aprime_ind=PolicyIndexes_sdp(:,:,l_d+1)+n_a(1)*(PolicyIndexes_sdp(:,:,l_d+2)-1);
        elseif l_a==3
            aprime_ind=PolicyIndexes_sdp(:,:,l_d+1)+n_a(1)*(PolicyIndexes_sdp(:,:,l_d+2)-1)+prod(n_a(1:2))*(PolicyIndexes_sdp(:,:,l_d+3)-1);
        elseif l_a==4
            aprime_ind=PolicyIndexes_sdp(:,:,l_d+1)+n_a(1)*(PolicyIndexes_sdp(:,:,l_d+2)-1)+prod(n_a(1:2))*(PolicyIndexes_sdp(:,:,l_d+3)-1)+prod(n_a(1:3))*(PolicyIndexes_sdp(:,:,l_d+4)-1);
        end
        d_ind=reshape(d_ind,[N_a*N_z,1]);
        aprime_ind=reshape(aprime_ind,[N_a*N_z,1]);
        a_ind=reshape((1:1:N_a)'*ones(1,N_z),[N_a*N_z,1]);
        z_ind=reshape(ones(N_a,1)*1:1:N_z,[N_a*N_z,1]);
        daprimeaz_ind=d_ind+N_d*aprime_ind+N_d*N_a*(a_ind-1)+N_d*N_a*N_a*(z_ind-1);
        SDP1=SDP1(daprimeaz_ind);
        if n_SDP>=2
            SDP2=SDP2(daprimeaz_ind);
        end
        if n_SDP>=3
            SDP3=SDP3(daprimeaz_ind);
        end
    end
    
    
    if n_SDP>3
        fprintf('WARNING: currently only three state dependent parameters are allowed. If you have a need for more please email robertdkirkby@gmail.com and let me know (I can easily implement more if needed) \n')
        dbstack
        return
    end
end

%% Deal with Entry and/or Exit if approprate
if isstruct(StationaryDist)
    % Note: if you want the agent mass of the stationary distribution you have to call it 'agentmass'
    if ~isfield(simoptions,'endogenousexit')
        simoptions.endogenousexit=0;
    end
    if simoptions.endogenousexit~=2
        AggVars=EvalFnOnAgentDist_AggVars_Case1_Mass(StationaryDist.pdf,StationaryDist.mass, PolicyIndexes, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, EntryExitParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel, simoptions);
    elseif simoptions.endogenousexit==2
        exitprobabilities=CreateVectorFromParams(Parameters, simoptions.exitprobabilities);
        exitprobs=[1-sum(exitprobabilities),exitprobabilities];
        AggVars=EvalFnOnAgentDist_AggVars_Case1_Mass_MixExit(StationaryDist.pdf,StationaryDist.mass, PolicyIndexes, PolicyWhenExiting, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, EntryExitParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel, exitprobs);
    end
    
    if FnsToEvaluateStruct==1
        % Change the output into a structure
        AggVars2=AggVars;
        clear AggVars
        AggVars=struct();
        %     AggVarNames=fieldnames(FnsToEvaluate);
        for ff=1:length(FnsToEvalNames)
            AggVars.(FnsToEvalNames{ff}).Aggregate=AggVars2(ff);
        end
    end
    
    return
end


%%
if Parallel==2 || Parallel==4
    Parallel=2;
    StationaryDist=gpuArray(StationaryDist);
    PolicyIndexes=gpuArray(PolicyIndexes);
    n_d=gpuArray(n_d);
    n_a=gpuArray(n_a);
    n_z=gpuArray(n_z);
    d_grid=gpuArray(d_grid);
    a_grid=gpuArray(a_grid);
    z_grid=gpuArray(z_grid);
    
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);

    AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');
    
    PolicyValues=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid);
    permuteindexes=[1+(1:1:(l_a+l_z)),1];    
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]
    
    for i=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(i).Names)  % check for 'FnsToEvaluateParamNames={}'
            FnToEvaluateParamsVec=[];
        else
            FnToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names));
        end
        if exist('sdp','var') % Use state dependent parameters
            if n_SDP==1
                Values=EvalFnOnAgentDist_Grid_Case1_SDP(FnsToEvaluate{i}, FnToEvaluateParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel,SDP1);
            elseif n_SDP==2
                Values=EvalFnOnAgentDist_Grid_Case1_SDP(FnsToEvaluate{i}, FnToEvaluateParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel,SDP1,SDP2);                
            elseif n_SDP==3
                Values=EvalFnOnAgentDist_Grid_Case1_SDP(FnsToEvaluate{i}, FnToEvaluateParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel,SDP1,SDP2,SDP3);
            end
        else
            Values=EvalFnOnAgentDist_Grid_Case1(FnsToEvaluate{i}, FnToEvaluateParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel);
        end
        Values=reshape(Values,[N_a*N_z,1]);
        % When evaluating value function (which may sometimes give -Inf
        % values) on StationaryDistVec (which at those points will be
        % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
        temp=Values.*StationaryDistVec;
        AggVars(i)=sum(temp(~isnan(temp)));
    end
    
else
    
    [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes,n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
    a_gridvals=CreateGridvals(n_a,a_grid,2);
    z_gridvals=CreateGridvals(n_z,z_grid,2);
    
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);
    
    StationaryDistVec=gather(StationaryDistVec);
    
    AggVars=zeros(length(FnsToEvaluate),1);
    
    if l_d>0
        
        for i=1:length(FnsToEvaluate)
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(i).Names) % check for 'SSvalueParamNames={}'
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{i}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                end
                % When evaluating value function (which may sometimes give -Inf
                % values) on StationaryDistVec (which at those points will be
                % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
                temp=Values.*StationaryDistVec;
                AggVars(i)=sum(temp(~isnan(temp)));
            else
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names));
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{i}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                end
                % When evaluating value function (which may sometimes give -Inf
                % values) on StationaryDistVec (which at those points will be
                % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
                temp=Values.*StationaryDistVec;
                AggVars(i)=sum(temp(~isnan(temp)));
            end
        end
    
    else %l_d=0
        
        for i=1:length(FnsToEvaluate)
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(i).Names) % check for 'SSvalueParamNames={}'
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{i}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                end
                % When evaluating value function (which may sometimes give -Inf
                % values) on StationaryDistVec (which at those points will be
                % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
                temp=Values.*StationaryDistVec;
                AggVars(i)=sum(temp(~isnan(temp)));
            else
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names));
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{i}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                end
                % When evaluating value function (which may sometimes give -Inf
                % values) on StationaryDistVec (which at those points will be
                % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
                temp=Values.*StationaryDistVec;
                AggVars(i)=sum(temp(~isnan(temp)));
            end
        end
    end
    
end


%%
if FnsToEvaluateStruct==1
    % Change the output into a structure
    AggVars2=AggVars;
    clear AggVars
    AggVars=struct();
%     AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(FnsToEvalNames)
        AggVars.(FnsToEvalNames{ff}).Mean=AggVars2(ff);
    end
end

if isfield(simoptions,'eval_valuefn')
    % There are some thing in AggVarsExtra that depended on the value
    % function, put them into AggVars
    extraNames=fieldnames(AggVarsExtra);
    for ff=1:length(extraNames)
        AggVars.(extraNames{ff})=AggVarsExtra.(extraNames{ff});
    end
end


%% If there are any conditional restrictions then send these off to be done
% Evaluate AggVars, but conditional on the restriction being non-zero.
%
% Code works by evaluating the the restriction and imposing this on the
% distribution (and renormalizing it) and then just sending this off to
% EvalFnOnAgendDist_AggVars_Case1() again. Some of the results are then
% modified so that there is both, e.g., 'mean' and 'total'.
if isfield(simoptions,'conditionalrestrictions')
    % First couple of lines get the conditional restrictions and convert
    % them to a names and cell
    CondlRestnFnNames=fieldnames(simoptions.conditionalrestrictions);
    for ff=1:length(CondlRestnFnNames)
        temp=getAnonymousFnInputNames(simoptions.conditionalrestrictions.(CondlRestnFnNames{ff}));
        if length(temp)>(l_d+l_a+l_a+l_z)
            CondlRestnFnParamNames(ff).Names={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            CondlRestnFnParamNames(ff).Names={};
        end
        CondlRestnFns{ff}=simoptions.conditionalrestrictions.(CondlRestnFnNames{ff});
    end
    simoptions=rmfield(simoptions,'conditionalrestrictions'); % Have to delete this before resend it to EvalFnOnAgentDist_AllStats_Case1()
    
    % Note that some things have already been created above, so we don't need
    % to recreate them to evaluated the restrictions.
    
    if simoptions.parallel==2
        % Evaluate the conditinal restrictions
        for kk=1:length(CondlRestnFnNames)
            % Includes check for cases in which no parameters are actually required
            if isempty(CondlRestnFnParamNames(kk).Names) % check for '={}'
                CondlRestnFnParamsVec=[];
            else
                CondlRestnFnParamsVec=CreateVectorFromParams(Parameters,CondlRestnFnParamNames(kk).Names);
            end
            
            Values=EvalFnOnAgentDist_Grid_Case1(CondlRestnFns{kk}, CondlRestnFnParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,simoptions.parallel);
            Values=reshape(Values,[N_a*N_z,1]);

            RestrictedStationaryDistVec=StationaryDistVec;
            RestrictedStationaryDistVec(Values==0)=0; % Drop all those that don't meet the restriction
            restrictedsamplemass=sum(RestrictedStationaryDistVec);
            RestrictedStationaryDistVec=RestrictedStationaryDistVec/restrictedsamplemass; % Normalize to mass one

            if restrictedsamplemass==0
                warning('One of the conditional restrictions evaluates to a zero mass')
                fprintf(['Specifically, the restriction called ',CondlRestnFnNames{kk},' has a restricted sample that is of zero mass \n'])
                AggVars.(CondlRestnFnNames{kk}).RestrictedSampleMass=restrictedsamplemass; % Just return this and hopefully it is clear to the user
            else
                AggVars.(CondlRestnFnNames{kk})=EvalFnOnAgentDist_AggVars_Case1(RestrictedStationaryDistVec, PolicyIndexes, FnsToEvaluate_copy, Parameters, [], n_d, n_a, n_z, d_grid, a_grid, z_grid, [], simoptions);
                
                % Create some renormalizations where relevant (just the mean)
                for ii=1:length(FnsToEvaluate) %Note FnsToEvaluate alread created above
                    AggVars.(CondlRestnFnNames{kk}).(FnsToEvalNames{ii}).Total=restrictedsamplemass*AggVars.(CondlRestnFnNames{kk}).(FnsToEvalNames{ii}).Mean;
                end
                AggVars.(CondlRestnFnNames{kk}).RestrictedSampleMass=restrictedsamplemass; % Seems likely this would be something user might want
            end
        end
    else % simoptions.parallel~=2
        for kk=1:length(FnsToEvaluate)
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(kk).Names) % check for 'FnsToEvaluateParamNames={}'
                Values=zeros(N_a*N_z,1);
                if l_d==0
                    for ii=1:N_a*N_z
                        j1=rem(ii-1,N_a)+1;
                        j2=ceil(ii/N_a);
                        Values(ii)=FnsToEvaluate{kk}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                    end
                else % l_d>0
                    for ii=1:N_a*N_z
                        j1=rem(ii-1,N_a)+1;
                        j2=ceil(ii/N_a);
                        Values(ii)=FnsToEvaluate{kk}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                    end
                end
            else
                Values=zeros(N_a*N_z,1);
                if l_d==0
                    FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(kk).Names));
                    Values=zeros(N_a*N_z,1);
                    for ii=1:N_a*N_z
                        j1=rem(ii-1,N_a)+1;
                        j2=ceil(ii/N_a);
                        Values(ii)=FnsToEvaluate{kk}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                    end
                else % l_d>0
                    FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(kk).Names));
                    for ii=1:N_a*N_z
                        j1=rem(ii-1,N_a)+1;
                        j2=ceil(ii/N_a);
                        Values(ii)=FnsToEvaluate{kk}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                    end
                end
            end
            
            RestrictedStationaryDistVec=StationaryDistVec;
            RestrictedStationaryDistVec(Values==0)=0; % Drop all those that don't meet the restriction
            restrictedsamplemass=sum(RestrictedStationaryDistVec);
            RestrictedStationaryDistVec=RestrictedStationaryDistVec/restrictedsamplemass; % Normalize to mass one

            if restrictedsamplemass==0
                warning('One of the conditional restrictions evaluates to a zero mass')
                fprintf(['Specifically, the restriction called ',CondlRestnFnNames{kk},' has a restricted sample that is of zero mass \n'])
                AggVars.(CondlRestnFnNames{kk}).RestrictedSampleMass=restrictedsamplemass; % Just return this and hopefully it is clear to the user
            else
                AggVars.(CondlRestnFnNames{kk})=EvalFnOnAgentDist_AggVars_Case1(RestrictedStationaryDistVec, PolicyIndexes, FnsToEvaluate_copy, Parameters, [], n_d, n_a, n_z, d_grid, a_grid, z_grid, [], simoptions);
                
                % Create some renormalizations where relevant (just the mean)
                for ii=1:length(FnsToEvaluate) %Note FnsToEvaluate alread created above
                    AggVars.(CondlRestnFnNames{kk}).(FnsToEvalNames{ii}).Total=restrictedsamplemass*AggVars.(CondlRestnFnNames{kk}).(FnsToEvalNames{ii}).Mean;
                end
                AggVars.(CondlRestnFnNames{kk}).RestrictedSampleMass=restrictedsamplemass; % Seems likely this would be something user might want
            end
        end
    end
end



end
