function AllStats=StatsFromWeightedGrid(Values,Weights,npoints,nquantiles,tolerance, presorted, whichstats)
% Inputs: Values is a grid of values, Weights is a grid of corresponding weights
% Weights is assumed to be of mass 1 (so you will get wrong answers if it is not)

% 1. Establish the active precision
single_1 = cast(1, 'like', Values);
single_0 = cast(0, 'like', Values);
tol_eps  = eps(single_1) * 100; % Dynamic tolerance based on single/double

AllStats=struct();

if ~exist('presorted','var')
    presorted=0;
end
if ~exist('whichstats','var')
    whichstats=ones(1,7);
end

%%
if presorted==0
    Values=reshape(Values,[numel(Values),1]);
    Weights=reshape(Weights,[numel(Weights),1]);

    temp=logical(Weights==0);
    Weights=Weights(~temp);
    Values=Values(~temp);

    [SortedValues,SortedValues_index] = sort(Values);
    SortedWeights = Weights(SortedValues_index);
elseif presorted==1
    SortedValues=Values;
    SortedWeights=Weights;
elseif presorted==2
    temp=logical(Weights==0);
    Weights=Weights(~temp);
    Values=Values(~temp);

    SortedValues=Values;
    SortedWeights=Weights;
end

%% If there are no points with positive weight (e.g., an age at which a cohort has not yet entered the model), all the stats are NaN
if isempty(SortedValues)
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

WeightedSortedValues=SortedValues.*SortedWeights;

if any(whichstats(4:7)>=1) || whichstats(2)==1
    CumSumSortedWeights=cumsum(SortedWeights);
    % Prevent singlefp drift from failing the check
    skipcheck = all(abs(CumSumSortedWeights - single_1) < tol_eps);
else
    skipcheck=0;
end

%% Now the stats themselves
if whichstats(1)==1
    AllStats.Mean=sum(WeightedSortedValues);
end

if whichstats(2)==1
    [~,index_median]=min(abs(CumSumSortedWeights - cast(0.5, 'like', Values)));
    AllStats.Median=SortedValues(index_median);

    if whichstats(1)==1
        AllStats.RatioMeanToMedian=AllStats.Mean/AllStats.Median;
    end
end

%% Deal with case where all the values are just the same anyway
if SortedValues(1)==SortedValues(end) || skipcheck
    if whichstats(3)==1
        AllStats.Variance=single_0;
        AllStats.StdDeviation=single_0;
    end
    if whichstats(4)==1 || whichstats(4)==2
        AllStats.LorenzCurve=cast((1/npoints:1/npoints:1)', 'like', Values);
        AllStats.Gini=single_0;
    elseif whichstats(4)==3
        AllStats.Gini=single_0;
    end
    if whichstats(5)==1
        AllStats.Maximum=SortedValues(1);
        AllStats.Minimum=SortedValues(1);
    end
    if whichstats(6)>=1
        AllStats.QuantileCutoffs=nan(nquantiles+1,1, 'like', Values);
        AllStats.QuantileMeans=SortedValues(1)*ones(nquantiles,1, 'like', Values);
    end
    if whichstats(7)==1
        AllStats.MoreInequality.Top1share=cast(0.01, 'like', Values);
        AllStats.MoreInequality.Top5share=cast(0.05, 'like', Values);
        AllStats.MoreInequality.Top10share=cast(0.1, 'like', Values);
        AllStats.MoreInequality.Bottom50share=cast(0.5, 'like', Values);
        AllStats.MoreInequality.Percentile50th=SortedValues(1);
        AllStats.MoreInequality.Percentile90th=SortedValues(1);
        AllStats.MoreInequality.Percentile95th=SortedValues(1);
        AllStats.MoreInequality.Percentile99th=SortedValues(1);
    end
else
    if whichstats(3)==1
        AllStats.Variance=sum(((Values-AllStats.Mean).^2).*Weights);
        % Relaxed precision check for singlefp catastrophic cancellation
        if AllStats.Variance<0 && AllStats.Variance > -cast(1e-4, 'like', Values)
            AllStats.Variance=single_0;
        end
        AllStats.StdDeviation=sqrt(AllStats.Variance);
    end

    if (whichstats(4)>=1 && npoints>0 && ~(WeightedSortedValues(1)<0)) || whichstats(6)==2
        CumSumSortedWeightedValues=cumsum(WeightedSortedValues);
    end

    if whichstats(4)>=1
        if npoints>0
            if WeightedSortedValues(1)<0
                if whichstats(4)<3
                    AllStats.LorenzCurve=nan(npoints,1, 'like', Values);
                    AllStats.LorenzCurveComment={'Lorenz curve cannot be calculated as some values are negative'};
                end
                AllStats.Gini=cast(nan, 'like', Values);
                AllStats.GiniComment={'Gini cannot be calculated as some values are negative'};
            else
                if whichstats(4)<3
                    LorenzCurve=zeros(npoints,1, 'like', Values);
                    llvec=cast(1/npoints:1/npoints:1, 'like', Values);

                    if whichstats(4)==1
                        for ll=1:npoints-1
                            [~,lorenzcind]=max(CumSumSortedWeights >= llvec(ll));
                            if lorenzcind==1
                                LorenzCurve(ll)=llvec(ll)*SortedValues(lorenzcind);
                            else
                                LorenzCurve(ll)=CumSumSortedWeightedValues(lorenzcind-1)+(llvec(ll)-CumSumSortedWeights(lorenzcind-1))*SortedValues(lorenzcind);
                            end
                        end
                        LorenzCurve(npoints)=CumSumSortedWeightedValues(end);
                    elseif whichstats(4)==2
                        [CumSumSortedWeights2,u1index,~]=unique(CumSumSortedWeights);
                        if isscalar(CumSumSortedWeights2)
                            LorenzCurve=cast(1/npoints:1/npoints:1, 'like', Values);
                        else
                            temp=interp1(CumSumSortedWeights2,CumSumSortedWeightedValues(u1index),llvec(1:end-1));
                            LorenzCurve(1:end-1)=temp;
                            temp2=sum(isnan(temp));
                            % Replaced hardcoded 1e-15 with dynamic eps check
                            if abs(LorenzCurve(temp2+1)-CumSumSortedWeightedValues(1)) < tol_eps
                                temp2=temp2+1;
                            end
                            LorenzCurve(1:temp2)=(CumSumSortedWeightedValues(1) - SortedValues(1)*(CumSumSortedWeights(1)-temp2/npoints)) .*cast((1:1:temp2)/temp2, 'like', Values);
                            LorenzCurve(npoints)=CumSumSortedWeightedValues(end);
                        end
                    end
                    SumWeightedValues=sum(WeightedSortedValues);
                    AllStats.LorenzCurve=LorenzCurve/SumWeightedValues;
                end

                CumSumWeightedSortedValues=cumsum(WeightedSortedValues);
                CumSumWeightedSortedValues=CumSumWeightedSortedValues/CumSumWeightedSortedValues(end);
                AllStats.Gini=sum(CumSumWeightedSortedValues(2:end).*CumSumSortedWeights(1:end-1)- CumSumWeightedSortedValues(1:end-1).*CumSumSortedWeights(2:end));
            end
        end
    end

    if whichstats(5)==1 || whichstats(6)>=1
        % --- THE SINGLE-FP BULLETPROOFING GUARDS ---
        tolerance_c = cast(tolerance, 'like', Values);

        % Min value
        tempindex=find(CumSumSortedWeights>=tolerance_c,1,'first');
        if isempty(tempindex); tempindex = 1; end % Guard against drift
        minvalue=SortedValues(tempindex);

        % Max value
        tempindex=find(CumSumSortedWeights>=(single_1-tolerance_c),1,'first');
        if isempty(tempindex); tempindex = length(SortedValues); end % Guard against drift
        maxvalue=SortedValues(tempindex);

        AllStats.Maximum=maxvalue;
        AllStats.Minimum=minvalue;
    end

    if whichstats(6)>=1
        if nquantiles==1
            error('Not allowed to set simoptions.nquantiles=1 (you anyway have this as the median, set higher or set equal zero to disable')
        end
        if whichstats(6)==1
            if nquantiles>0
                QuantileMeans=zeros(nquantiles,1, 'like', Values);
                quantilecutoffindexes=zeros(nquantiles-1,1);
                quantilecvec=cast(1/nquantiles:1/nquantiles:1-1/nquantiles, 'like', Values);
                for quantilecind=1:nquantiles-1
                    [~,quantilecutoffindexes_quantilec]=max(CumSumSortedWeights >= quantilecvec(quantilecind));
                    quantilecutoffindexes(quantilecind)=quantilecutoffindexes_quantilec;
                end
                AllStats.QuantileCutoffs=[minvalue; SortedValues(quantilecutoffindexes); maxvalue];
                QuantileMeans(1)=sum(WeightedSortedValues(1:quantilecutoffindexes(1))) - SortedValues(quantilecutoffindexes(1))*(CumSumSortedWeights(quantilecutoffindexes(1))-cast(1/nquantiles, 'like', Values));
                for ll=2:nquantiles-1
                    if quantilecutoffindexes(ll-1)==quantilecutoffindexes(ll)
                        QuantileMeans(ll)=SortedValues(quantilecutoffindexes(ll))/cast(nquantiles, 'like', Values);
                    else
                        QuantileMeans(ll)=sum(WeightedSortedValues(quantilecutoffindexes(ll-1)+1:quantilecutoffindexes(ll))) - SortedValues(quantilecutoffindexes(ll))*(CumSumSortedWeights(quantilecutoffindexes(ll))-cast(ll/nquantiles, 'like', Values))  + SortedValues(quantilecutoffindexes(ll-1))*(CumSumSortedWeights(quantilecutoffindexes(ll-1))-cast((ll-1)/nquantiles, 'like', Values));
                    end
                end
                QuantileMeans(nquantiles)=sum(WeightedSortedValues(quantilecutoffindexes(nquantiles-1)+1:end)) + SortedValues(quantilecutoffindexes(nquantiles-1))*(CumSumSortedWeights(quantilecutoffindexes(nquantiles-1))-cast((nquantiles-1)/nquantiles, 'like', Values));
                AllStats.QuantileMeans=QuantileMeans*cast(nquantiles, 'like', Values);
            end
        elseif whichstats(6)==2
            if nquantiles>0
                [~,quantilecutoffindexes]=max(CumSumSortedWeights >= cast(1/nquantiles:1/nquantiles:1-1/nquantiles, 'like', Values));
                AllStats.QuantileCutoffs=[minvalue; SortedValues(quantilecutoffindexes); maxvalue];
                quantilecutoffindexes_lower=[1; quantilecutoffindexes'];
                quantilecutoffindexes_upper=[quantilecutoffindexes'; numel(WeightedSortedValues)];

                term1=CumSumSortedWeightedValues(quantilecutoffindexes_upper)-CumSumSortedWeightedValues(quantilecutoffindexes_lower);
                term2=SortedValues(quantilecutoffindexes_upper).*(CumSumSortedWeights(quantilecutoffindexes_upper)-cast((1:1:nquantiles)'/nquantiles, 'like', Values));
                term3=SortedValues(quantilecutoffindexes_lower).*(CumSumSortedWeights(quantilecutoffindexes_lower)-cast((0:1:nquantiles-1)'/nquantiles, 'like', Values));
                QuantileMeans=term1-term2+term3;

                temp=logical(quantilecutoffindexes_lower==quantilecutoffindexes_upper);
                QuantileMeans(temp)=SortedValues(quantilecutoffindexes_upper(temp))/cast(nquantiles, 'like', Values);
                AllStats.QuantileMeans=QuantileMeans*cast(nquantiles, 'like', Values);
            end
        end
    end

    if whichstats(7)==1
        if ~any(whichstats(4)==[1,2])
            error('whichstats(7)=1 can only be used with whichstats(4)=1 or 2 (Lorenz Curve forms basis for some of the stats in whichstats(7))')
        end

        Top1cutpoint=round(0.99*npoints);
        Top5cutpoint=round(0.95*npoints);
        Top10cutpoint=round(0.90*npoints);
        Top50cutpoint=round(0.50*npoints);

        AllStats.MoreInequality.Top1share=single_1-AllStats.LorenzCurve(Top1cutpoint);
        AllStats.MoreInequality.Top5share=single_1-AllStats.LorenzCurve(Top5cutpoint);
        AllStats.MoreInequality.Top10share=single_1-AllStats.LorenzCurve(Top10cutpoint);
        AllStats.MoreInequality.Bottom50share=AllStats.LorenzCurve(Top50cutpoint);

        AllStats.MoreInequality.Percentile50th=AllStats.Median;

        % Bulletproof Percentiles
        index_p90=find(CumSumSortedWeights>=cast(0.90, 'like', Values),1,'first');
        if isempty(index_p90); index_p90 = length(SortedValues); end
        AllStats.MoreInequality.Percentile90th=SortedValues(index_p90);

        index_p95=find(CumSumSortedWeights>=cast(0.95, 'like', Values),1,'first');
        if isempty(index_p95); index_p95 = length(SortedValues); end
        AllStats.MoreInequality.Percentile95th=SortedValues(index_p95);

        index_p99=find(CumSumSortedWeights>=cast(0.99, 'like', Values),1,'first');
        if isempty(index_p99); index_p99 = length(SortedValues); end
        AllStats.MoreInequality.Percentile99th=SortedValues(index_p99);
    end
end


end