function TimeSeriesKron=TimeSeries_Case1_raw(TimeSeriesIndexesKron, PolicyIndexes, Parameters, FnsToEvaluate,FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid)

if prod(n_d)==0
    l_d=0;
else
    l_d=length(n_d);
end

N_a=prod(n_a);

numFnsToEvaluate=length(FnsToEvaluate);

T=length(TimeSeriesIndexesKron(1,:));

TimeSeriesKron=zeros(numFnsToEvaluate,T);

[d_gridvals, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes,n_d,n_a,n_a,n_z,d_grid,a_grid,1, 1);
a_gridvals=CreateGridvals(n_a,a_grid,1);
z_gridvals=CreateGridvals(n_z,z_grid,1);

FnUsesParams=ones(numFnsToEvaluate,1);
for ff=1:numFnsToEvaluate
    FullParamsCell.(['a',num2str(ff)])=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names));
    if isempty(FullParamsCell.(['a',num2str(ff)]))
        FnUsesParams(ff)=0;
    end
end


%%
if l_d==0
    parfor t=1:T
        
        TimeSeriesKron_t=zeros(numFnsToEvaluate,1);
        TimeSeriesIndexesKron_t=TimeSeriesIndexesKron(:,t);
        
        a_ind=TimeSeriesIndexesKron_t(1);
        a_val=a_gridvals(a_ind,:);
        
        z_ind=TimeSeriesIndexesKron_t(2);
        z_val=z_gridvals(z_ind,:);
        
        az_ind=a_ind+N_a*(z_ind-1);
        aprime_val=aprime_gridvals(az_ind,:);
        
        for ff=1:numFnsToEvaluate
            ParamsCell=FullParamsCell.(['a',num2str(ff)]);
            if FnUsesParams(ff)==1
                TimeSeriesKron_t(ff)=FnsToEvaluate{ff}(aprime_val,a_val,z_val,ParamsCell{:});
            else
                TimeSeriesKron_t(ff)=FnsToEvaluate{ff}(aprime_val,a_val,z_val);
            end
        end
       
        TimeSeriesKron(:,t)=TimeSeriesKron_t;
        
    end

else % l_d>0
    parfor t=1:T
        
        TimeSeriesKron_t=zeros(numFnsToEvaluate,1);
        TimeSeriesIndexesKron_t=TimeSeriesIndexesKron(:,t);
        
        a_ind=TimeSeriesIndexesKron_t(1);
        a_val=a_gridvals(a_ind,:);
        
        z_ind=TimeSeriesIndexesKron_t(2);
        z_val=z_gridvals(z_ind,:);
        
        az_ind=a_ind+N_a*(z_ind-1);
        aprime_val=aprime_gridvals(az_ind,:);
        d_val=d_gridvals(az_ind,:);
        
        for ff=1:numFnsToEvaluate
            ParamsCell=FullParamsCell.(['a',num2str(ff)]);
            if FnUsesParams(ff)==1
                TimeSeriesKron_t(ff)=FnsToEvaluate{ff}(d_val,aprime_val,a_val,z_val,ParamsCell{:});
            else
                TimeSeriesKron_t(ff)=FnsToEvaluate{ff}(d_val,aprime_val,a_val,z_val);
            end
        end
        
        TimeSeriesKron(:,t)=TimeSeriesKron_t;
        
    end    
end

end