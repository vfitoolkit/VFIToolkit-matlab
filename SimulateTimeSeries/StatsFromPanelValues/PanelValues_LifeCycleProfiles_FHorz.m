function AgeConditionalStats=PanelValues_LifeCycleProfiles_FHorz(PanelValues,N_j,simoptions)

if ~exist('simoptions','var')
    simoptions.whichstats=ones(7,1); % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes 
    simoptions.agegroupings=1:1:N_j; % by default does each period seperately, can be used to say, calculate gini for age bins
    simoptions.nquantiles=20; % by default gives ventiles
    simoptions.npoints=100; % number of points for lorenz curve
    simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
else
    if ~isfield(simoptions,'whichstats')
        simoptions.whichstats=ones(7,1); % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes
    end
    if ~isfield(simoptions,'agegroupings')
        simoptions.agegroupings=1:1:N_j; % by default does each period seperately, can be used to say, calculate gini for age bins
    end
    if ~isfield(simoptions,'nquantiles')
        simoptions.nquantiles=20; % by default gives ventiles
    end
    if ~isfield(simoptions,'npoints')
        simoptions.npoints=100; % number of points for lorenz curve
    end
    if ~isfield(simoptions,'tolerance')
        simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    end
end

PanelVariableNames=fieldnames(PanelValues);


% They are not sorted
presorted=0;

% Preallocate various things for the stats (as many will have jj as a dimension)
% Stats to calculate and store in AgeConditionalStats.(PanelVariableNames{ff})
for ff=1:length(PanelVariableNames)
    if simoptions.whichstats(1)==1
        AgeConditionalStats.(PanelVariableNames{ff}).Mean=nan(1,length(simoptions.agegroupings),'gpuArray');
    end
    if simoptions.whichstats(2)==1
        AgeConditionalStats.(PanelVariableNames{ff}).Median=nan(1,length(simoptions.agegroupings),'gpuArray');
    end
    if simoptions.whichstats(3)==1
        AgeConditionalStats.(PanelVariableNames{ff}).Variance=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(PanelVariableNames{ff}).StdDeviation=nan(1,length(simoptions.agegroupings),'gpuArray');
    end
    if simoptions.whichstats(4)==1
        AgeConditionalStats.(PanelVariableNames{ff}).Gini=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(PanelVariableNames{ff}).LorenzCurve=nan(simoptions.npoints,length(simoptions.agegroupings),'gpuArray');
    end
    if simoptions.whichstats(5)==1
        AgeConditionalStats.(PanelVariableNames{ff}).Minimum=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(PanelVariableNames{ff}).Maximum=nan(1,length(simoptions.agegroupings),'gpuArray');
    end
    if simoptions.whichstats(6)==1
        AgeConditionalStats.(PanelVariableNames{ff}).QuantileCutoffs=nan(simoptions.nquantiles+1,length(simoptions.agegroupings),'gpuArray'); % Includes the min and max values
        AgeConditionalStats.(PanelVariableNames{ff}).QuantileMeans=nan(simoptions.nquantiles,length(simoptions.agegroupings),'gpuArray');
    end
    if simoptions.whichstats(7)==1
        AgeConditionalStats.(PanelVariableNames{ff}).MoreInequality.Top1share=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(PanelVariableNames{ff}).MoreInequality.Top5share=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(PanelVariableNames{ff}).MoreInequality.Top10share=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(PanelVariableNames{ff}).MoreInequality.Bottom50share=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(PanelVariableNames{ff}).MoreInequality.Percentile50th=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(PanelVariableNames{ff}).MoreInequality.Percentile90th=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(PanelVariableNames{ff}).MoreInequality.Percentile95th=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(PanelVariableNames{ff}).MoreInequality.Percentile99th=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(PanelVariableNames{ff}).QuantileCutoffs=nan(simoptions.nquantiles+1,length(simoptions.agegroupings),'gpuArray'); % Includes the min and max values
        AgeConditionalStats.(PanelVariableNames{ff}).QuantileMeans=nan(simoptions.nquantiles,length(simoptions.agegroupings),'gpuArray');
    end
end


%%
for ff=1:length(PanelVariableNames)
    values_ff=PanelValues.(PanelVariableNames{ff});

    for jj=1:N_j
        % Every observation has equal weight
        Weights_jj=ones(size(values_ff(jj,:)));
        Weights_jj=Weights_jj/sum(Weights_jj);

        temp=StatsFromWeightedGrid(values_ff(jj,:),Weights_jj,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance, presorted, simoptions.whichstats);

        if simoptions.whichstats(1)==1
            AgeConditionalStats.(PanelVariableNames{ff}).Mean(jj)=temp.Mean;
        end
        if simoptions.whichstats(2)==1
            AgeConditionalStats.(PanelVariableNames{ff}).Median(jj)=temp.Median;
        end
        if simoptions.whichstats(3)==1
            AgeConditionalStats.(PanelVariableNames{ff}).Variance(jj)=temp.Variance;
            AgeConditionalStats.(PanelVariableNames{ff}).StdDeviation(jj)=temp.StdDeviation;
        end
        if simoptions.whichstats(4)==1
            AgeConditionalStats.(PanelVariableNames{ff}).Gini(jj)=temp.Gini;
            AgeConditionalStats.(PanelVariableNames{ff}).LorenzCurve(:,jj)=temp.LorenzCurve;
        end
        if simoptions.whichstats(5)==1
            AgeConditionalStats.(PanelVariableNames{ff}).Minimum(jj)=temp.Minimum;
            AgeConditionalStats.(PanelVariableNames{ff}).Maximum(jj)=temp.Maximum;
        end
        if simoptions.whichstats(6)==1
            AgeConditionalStats.(PanelVariableNames{ff}).QuantileCutoffs(:,jj)=temp.QuantileCutoffs; % Includes the min and max values
            AgeConditionalStats.(PanelVariableNames{ff}).QuantileMeans(:,jj)=temp.QuantileMeans;
        end
        if simoptions.whichstats(7)==1
            AgeConditionalStats.(PanelVariableNames{ff}).MoreInequality.Top1share(jj)=temp.MoreInequality.Top1share;
            AgeConditionalStats.(PanelVariableNames{ff}).MoreInequality.Top5share(jj)=temp.MoreInequality.Top5share;
            AgeConditionalStats.(PanelVariableNames{ff}).MoreInequality.Top10share(jj)=temp.MoreInequality.Top10share;
            AgeConditionalStats.(PanelVariableNames{ff}).MoreInequality.Bottom50share(jj)=temp.MoreInequality.Bottom50share;
            AgeConditionalStats.(PanelVariableNames{ff}).MoreInequality.Percentile50th(jj)=temp.MoreInequality.Percentile50th;
            AgeConditionalStats.(PanelVariableNames{ff}).MoreInequality.Percentile90th(jj)=temp.MoreInequality.Percentile90th;
            AgeConditionalStats.(PanelVariableNames{ff}).MoreInequality.Percentile95th(jj)=temp.MoreInequality.Percentile95th;
            AgeConditionalStats.(PanelVariableNames{ff}).MoreInequality.Percentile99th(jj)=temp.MoreInequality.Percentile99th;
            AgeConditionalStats.(PanelVariableNames{ff}).QuantileCutoffs(:,jj)=temp.QuantileCutoffs; % Includes the min and max values
            AgeConditionalStats.(PanelVariableNames{ff}).QuantileMeans(:,jj)=temp.QuantileMeans;
        end

    end

end




end