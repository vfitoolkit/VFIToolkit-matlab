function AggVars=EvalFnOnAgentDist_AggVars_Case1(StationaryDist, PolicyIndexes, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel, simoptions, EntryExitParamNames, PolicyWhenExiting)
% Evaluates the aggregate value (weighted sum/integral) for each element of FnsToEvaluate
%
% Parallel, simoptions and EntryExitParamNames are optional inputs, only needed when using endogenous entry

if exist('Parallel','var')==0
    Parallel=1+(gpuDeviceCount>0);
elseif isempty(Parallel)
    Parallel=1+(gpuDeviceCount>0);
end

if exist('simoptions', 'var')
    if isfield(simoptions,'statedependentparams')
        n_SDP=length(simoptions.statedependentparams.names);
        sdp=zeros(length(FnsToEvaluate),length(simoptions.statedependentparams.names));
        for ii=1:length(FnsToEvaluate)
            for jj=1:n_SDP
                for kk=1:length(FnsToEvaluateParamNames(ii).Names)
                    if strcmp(simoptions.statedependentparams.names{jj},FnsToEvaluateParamNames(ii).Names{kk})
                        sdp(ii,jj)=1;
                        % Remove the statedependentparams from ReturnFnParamNames
                        FnsToEvaluateParamNames(ii).Names=setdiff(FnsToEvaluateParamNames(ii).Names,vfoptions.statedependentparams.names{jj});
                        % Set up the SDP variables
                    end
                end
            end
        end
        if N_d>1
            l_d=length(n_d);
            n_full=[n_d,n_a,n_a,n_z];
        else
            l_d=0;
            n_full=[n_a,n_a,n_z];
        end
        l_a=length(n_a);
        l_z=length(n_z);
        
        % First state dependent parameter, get into form needed for the valuefn
        SDP1=Params.(vfoptions.statedependentparams.names{1});
        SDP1_dims=vfoptions.statedependentparams.dimensions.(vfoptions.statedependentparams.names{1});
        %     vfoptions.statedependentparams.dimensions.kmax=[3,4,5,6,7]; % The d,a & z variables (in VFI toolkit notation)
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
            SDP2=Params.(vfoptions.statedependentparams.names{2});
            SDP2_dims=vfoptions.statedependentparams.dimensions.(vfoptions.statedependentparams.names{2});
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
            SDP3=Params.(vfoptions.statedependentparams.names{3});
            SDP3_dims=vfoptions.statedependentparams.dimensions.(vfoptions.statedependentparams.names{3});
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
end

if isstruct(StationaryDist)
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
    return
end

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
    
    % l_d not needed with Parallel=2 implementation
    l_a=length(n_a);
    l_z=length(n_z);
    
    N_a=prod(n_a);
    N_z=prod(n_z);
    
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);

    AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');
    
    PolicyValues=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid, Parallel);
    permuteindexes=[1+(1:1:(l_a+l_z)),1];    
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]
    
    for i=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(i).Names)  % check for 'SSvalueParamNames={}'
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
    if n_d(1)==0
        l_d=0;
    else
        l_d=length(n_d);
    end
    
    N_a=prod(n_a);
    N_z=prod(n_z);
    
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


end
