function AllStats=StatsFromWeightedGrid(Values,Weights,npoints,nquantiles,tolerance, presorted, whichstats)
% Inputs: Values is a grid of values, Weights is a grid of corresponding weights
% Weights is assumed to be of mass 1 (so you will get wrong answers if it is not)

% Output takes following form
AllStats=struct();
% AllStats.Mean=nan(1,1);
% AllStats.Median=nan(1,1);
% AllStats.RatioMeanToMedian=nan(1,1);
% AllStats.Variance=nan(1,1);
% AllStats.StdDeviation=nan(1,1);
% AllStats.LorenzCurve=nan(npoints,1);
% AllStats.Gini=nan(1,1);
% AllStats.QuantileCutoffs=nan(nquantiles+1,1); % Includes the min and max values
% AllStats.QuantileMeans=nan(nquantiles,1);

if ~exist('presorted','var')
    presorted=0; % Optional input when you know that values and weights are already sorted (and zero weighted points eliminated) [and are column vectors]
end

if ~exist('whichstats','var')
    whichstats=ones(1,7); % by default, compute all stats
    % zero values in this optional input are used to skip some stats and thereby cut runtimes
    % 1st element: mean
    % 2nd element: median
    % 3rd element: std dev and variance
    % 4th element: lorenz curve and gini coefficient
    % 5th element: min/max
    % 6th element: quantiles
    % 7th element: More Inequality
    % Note: RatioMeanToMedian is computed whenever both mean and median are
    %
    % whichstats(4)=3 gives just the gini coefficient (no lorenz curve).
    % whichstats(4)=2 and whichstats(6)=2 used to switch to a faster but more memory intensive version; there
    % is now just one version (which is both the fast one and the low memory one), so 2 is treated the same as 1.
end

if whichstats(6)>=1 && nquantiles==1
    error('Not allowed to set simoptions.nquantiles=1 (you anyway have this as the median, set higher or set equal zero to disable')
end

%%
if presorted==0
    % Do I want to add unique() here???? No, you should unique before passing when appropriate (too much run time to do when unnecessary)
    Values=reshape(Values,[numel(Values),1]);
    Weights=reshape(Weights,[numel(Weights),1]);

    % Eliminate all the zero-weights from these (trivial increase in runtime, but makes it easier to spot when there is no variance)
    temp=logical(Weights==0);
    Weights=Weights(~temp);
    Values=Values(~temp);
    % [The sort is done below, and only if one of the stats needs it]
elseif presorted==2
    % sorted, but might sill contain some zero weights
    % Eliminate all the zero-weights from these (trivial increase in runtime, but makes it easier to spot when there is no variance)
    temp=logical(Weights==0);
    Weights=Weights(~temp);
    Values=Values(~temp);
end
% (presorted==1 needs nothing doing)

%% If there are no points with positive weight (e.g., an age at which a cohort has not yet entered the model), all the stats are NaN
if isempty(Values)
    if whichstats(1)==1
        AllStats.Mean=NaN;
    end
    if whichstats(2)==1
        AllStats.Median=NaN;
        if whichstats(1)==1
            AllStats.RatioMeanToMedian=NaN;
        end
    end
    if whichstats(3)==1
        AllStats.Variance=NaN;
        AllStats.StdDeviation=NaN;
    end
    if whichstats(4)>=1
        AllStats.Gini=NaN;
        if whichstats(4)<3
            AllStats.LorenzCurve=nan(npoints,1);
        end
    end
    if whichstats(5)==1
        AllStats.Minimum=NaN;
        AllStats.Maximum=NaN;
    end
    if whichstats(6)>=1
        AllStats.QuantileCutoffs=nan(nquantiles+1,1);
        AllStats.QuantileMeans=nan(nquantiles,1);
    end
    if whichstats(7)==1
        AllStats.MoreInequality.Top1share=NaN;
        AllStats.MoreInequality.Top5share=NaN;
        AllStats.MoreInequality.Top10share=NaN;
        AllStats.MoreInequality.Bottom50share=NaN;
        AllStats.MoreInequality.Percentile50th=NaN;
        AllStats.MoreInequality.Percentile90th=NaN;
        AllStats.MoreInequality.Percentile95th=NaN;
        AllStats.MoreInequality.Percentile99th=NaN;
    end
    return
end

%% Now the stats themselves
% Mean and variance do not need the values to be sorted, so do the mean before sorting
if whichstats(1)==1
    % Calculate the 'age conditional' mean
    AllStats.Mean=sum(Values.*Weights);
end

%% Sort, if any of the stats need it (everything except mean and variance does)
needsort=(whichstats(2)==1 || any(whichstats(4:7)>=1));
if needsort
    if presorted==0
        [SortedValues,SortedValues_index] = sort(Values);
        SortedWeights = Weights(SortedValues_index);
    else
        SortedValues=Values;
        SortedWeights=Weights;
    end
    WeightedSortedValues=SortedValues.*SortedWeights;
    CumSumSortedWeights=cumsum(SortedWeights);
    % All the mass is on the first point (cumsum is nondecreasing, so this says every later weight adds nothing; weights of magnitude, e.g. 1e-26, can do this)
    skipcheck=(CumSumSortedWeights(1)>=CumSumSortedWeights(end));
    allsamevalue=(SortedValues(1)==SortedValues(end) || skipcheck);
elseif whichstats(3)==1
    allsamevalue=gather(min(Values)==max(Values)); % without the sort, this is how to spot that there is no variance
else
    allsamevalue=false; % only the mean, which is already done
end

%% Deal with case where all the values are just the same anyway
if allsamevalue
    % The current FnsToEvaluate takes only one value, so nothing but the mean and median make sense
    % OR
    % Due to numerical rounding, it has multiple values but only one has any meaning as all the mass is in one place (weights of magnitude,
    % e.g. 1e-26 can turn into cumulative weights with zero difference between them)
    if whichstats(2)==1
        AllStats.Median=SortedValues(1);
        if whichstats(1)==1
            AllStats.RatioMeanToMedian=AllStats.Mean/AllStats.Median;
        end
    end
    if whichstats(3)==1
        AllStats.Variance=0;
        AllStats.StdDeviation=0;
    end
    if whichstats(4)==1 || whichstats(4)==2
        AllStats.LorenzCurve=(1/npoints:1/npoints:1)';
        AllStats.Gini=0;
    elseif whichstats(4)==3
        AllStats.Gini=0;
    end
    if whichstats(5)==1
        AllStats.Maximum=SortedValues(1);
        AllStats.Minimum=SortedValues(1);
    end
    if whichstats(6)>=1
        AllStats.QuantileCutoffs=SortedValues(1)*ones(nquantiles+1,1);
        AllStats.QuantileMeans=SortedValues(1)*ones(nquantiles,1);
    end
    if whichstats(7)==1
        AllStats.MoreInequality.Top1share=0.01;
        AllStats.MoreInequality.Top5share=0.05;
        AllStats.MoreInequality.Top10share=0.1;
        AllStats.MoreInequality.Bottom50share=0.5;
        AllStats.MoreInequality.Percentile50th=SortedValues(1);
        AllStats.MoreInequality.Percentile90th=SortedValues(1);
        AllStats.MoreInequality.Percentile95th=SortedValues(1);
        AllStats.MoreInequality.Percentile99th=SortedValues(1);
    end
else
    if needsort
        anynegative=gather(WeightedSortedValues(1)<0); % Lorenz curve, gini and top/bottom shares cannot be calculated when some values are negative (gather once, rather than at every if-statement)
    else
        anynegative=false; % not needed (only mean and variance)
    end

    if whichstats(3)==1
        % Calculate the 'age conditional' variance
        if whichstats(1)==1
            MeanForVariance=AllStats.Mean;
        else
            MeanForVariance=sum(Values.*Weights); % variance needs the mean even when whichstats(1)=0
        end
        AllStats.Variance=sum(((Values-MeanForVariance).^2).*Weights); % Weighted square of (values - mean)
        if AllStats.Variance<0 && AllStats.Variance>-10^(-6) % overwrite what is likely just numerical error
            AllStats.Variance=0;
        end
        AllStats.StdDeviation=sqrt(AllStats.Variance);
    end

    %% Median, min/max, quantile cutoffs, the percentiles, and the points of the Lorenz curve are all 'the first index with CumSumSortedWeights>=p', for a list of p
    % Because CumSumSortedWeights is sorted, we can find them all at once: the number of points with cumulative mass strictly below p, plus one
    % (this is one pass, rather than one pass per p; there are over 100 p with the default npoints and nquantiles)
    % pvec is: [median/p50; min; max; p99; p95; p90; quantile cutoffs; lorenz curve points]
    needquantiles=(whichstats(6)>=1 && nquantiles>0);
    needlorenz=((whichstats(4)==1 || whichstats(4)==2) && npoints>0 && ~anynegative);
    needsearch=(whichstats(2)==1 || whichstats(5)==1 || whichstats(6)>=1 || whichstats(7)==1 || needlorenz);
    if needquantiles
        quantilecvec=(1/nquantiles:1/nquantiles:1-1/nquantiles)';
    else
        quantilecvec=zeros(0,1);
    end
    if needlorenz
        llvec=1/npoints:1/npoints:1;
        llvec=llvec(1:end-1)'; % the last point of the lorenz curve is just 1
    else
        llvec=zeros(0,1);
    end
    quantileindexes=6+(1:numel(quantilecvec))';
    lorenzindexes=6+numel(quantilecvec)+(1:numel(llvec))';

    % The Lorenz curve (not yet normalised by the total) at each p is needed for the Lorenz curve, the quantile means and the top/bottom shares
    needlorenzatp=(needlorenz || needquantiles || (whichstats(7)==1 && ~anynegative));
    if (whichstats(4)>=1 && ~anynegative) || needlorenzatp
        CumSumSortedWeightedValues=cumsum(WeightedSortedValues);
    end

    if needsearch
        pvec=[0.5; tolerance; 1-tolerance; 0.99; 0.95; 0.90; quantilecvec; llvec];
        [pvec_unique,~,uniqueindex]=unique(pvec);
        cutcounts=histcounts(CumSumSortedWeights,[-Inf; pvec_unique; Inf]); % cutcounts(k) is the number of points with p_{k-1}<=CumSumSortedWeights<p_k
        cutind=min(cumsum(cutcounts(1:end-1)')+1,numel(CumSumSortedWeights)); % first index with CumSumSortedWeights>=p (min() deals with p above the total mass)
        cutind=cutind(uniqueindex);
        CutValues=SortedValues(cutind);

        if needlorenzatp
            % Mass wholly below the cut, plus the part of the point that straddles it
            prevind=max(cutind-1,1);
            hasprev=(cutind>1);
            LorenzAtP=hasprev.*CumSumSortedWeightedValues(prevind)+(pvec-hasprev.*CumSumSortedWeights(prevind)).*CutValues;
        end
    end

    if whichstats(2)==1
        % Calculate the 'age conditional' median
        % Median is the smallest value with cumulative mass >=0.5 (the quantile function at 0.5; same convention as the percentiles and quantile cutoffs below)
        % [If the cumulative mass hits exactly 0.5 at some value, then anything between that value and the next is a median, and this picks the lower one]
        AllStats.Median=CutValues(1);
        if whichstats(1)==1
            AllStats.RatioMeanToMedian=AllStats.Mean/AllStats.Median;
        end
    end

    if whichstats(4)>=1
        if anynegative
            if whichstats(4)<3
                AllStats.LorenzCurve=nan(npoints,1);
                AllStats.LorenzCurveComment={'Lorenz curve cannot be calculated as some values are negative'};
            end
            AllStats.Gini=nan;
            AllStats.GiniComment={'Gini cannot be calculated as some values are negative'};
        else
            if whichstats(4)<3 && npoints==0
                AllStats.LorenzCurve=nan(npoints,1); % npoints=0 means no Lorenz curve (the Gini does not need it)
            elseif whichstats(4)<3
                % Lorenz curve, normalized so that they are fractions of the total
                AllStats.LorenzCurve=[LorenzAtP(lorenzindexes); CumSumSortedWeightedValues(end)]/CumSumSortedWeightedValues(end);
            end

            % Gini coefficient
            % Gini=(1/(S*W))*sum_i w_i*y_i*(2*F_{i-1}+w_i-W), where F_{i-1} is the mass strictly below point i, S=sum(w.*y), W=sum(w)
            % This is the weighted version of the sorted closed form Gini=2*sum(i*y_i)/(n*sum(y))-(n+1)/n (which is the case w_i=1/n; see QuantEcon.py PR 937)
            % It is exact for a discrete distribution (equals the trapezoid area under the piecewise-linear Lorenz curve), and W means it does not need weights of mass 1
            % [2*F_{i-1}+w_i is computed as 2*F_i-w_i, so it reuses CumSumSortedWeights]
            AllStats.Gini=sum(WeightedSortedValues.*(2*CumSumSortedWeights-SortedWeights-CumSumSortedWeights(end)))/(CumSumSortedWeightedValues(end)*CumSumSortedWeights(end));
        end
    end

    if whichstats(5)==1 || whichstats(6)>=1 % note: anyway need min/max for quantile cutoffs
        % Min value is the first with cumulative mass >=tolerance, max value is the first with cumulative mass >=1-tolerance
        AllStats.Maximum=CutValues(3);
        AllStats.Minimum=CutValues(2);
    end
    if needquantiles
        % Quantile cutoffs (ventiles by default)
        AllStats.QuantileCutoffs=[CutValues(2); CutValues(quantileindexes); CutValues(3)];
        % Quantile means (ventiles by default): the mass of values in each quantile is the difference of the Lorenz curve (not normalised) at the two cutoffs
        AllStats.QuantileMeans=diff([0; LorenzAtP(quantileindexes); CumSumSortedWeightedValues(end)])*nquantiles; % Note: *nquantiles is really /(1/nquantiles), it is dividing by the mass of the quantile
    end

    if whichstats(7)==1
        % Top X shares are the Lorenz curve evaluated at 0.99, 0.95, 0.90 and 0.5, computed exactly from the distribution (so they do not depend on npoints and do not need whichstats(4))
        if anynegative
            AllStats.MoreInequality.Top1share=NaN;
            AllStats.MoreInequality.Top5share=NaN;
            AllStats.MoreInequality.Top10share=NaN;
            AllStats.MoreInequality.Bottom50share=NaN;
            AllStats.MoreInequality.SharesComment={'Top/bottom shares cannot be calculated as some values are negative'};
        else
            AllStats.MoreInequality.Top1share=1-LorenzAtP(4)/CumSumSortedWeightedValues(end);
            AllStats.MoreInequality.Top5share=1-LorenzAtP(5)/CumSumSortedWeightedValues(end);
            AllStats.MoreInequality.Top10share=1-LorenzAtP(6)/CumSumSortedWeightedValues(end);
            AllStats.MoreInequality.Bottom50share=LorenzAtP(1)/CumSumSortedWeightedValues(end);
        end
        % Now some cutoffs
        AllStats.MoreInequality.Percentile50th=CutValues(1); % same as the median
        AllStats.MoreInequality.Percentile90th=CutValues(6);
        AllStats.MoreInequality.Percentile95th=CutValues(5);
        AllStats.MoreInequality.Percentile99th=CutValues(4);
    end
end


end
