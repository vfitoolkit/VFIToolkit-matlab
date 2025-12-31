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
    % For 4th and 6th elements, setting whichstats(4)=2 and ichstats(6)=2
    % switches to a faster but more memory intensive version.
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

    %% Sorted weighted values
    [SortedValues,SortedValues_index] = sort(Values);
    SortedWeights = Weights(SortedValues_index);
elseif presorted==1
    SortedValues=Values;
    SortedWeights=Weights;
elseif presorted==2
    % sorted and unique, but might sill contain some zero weights
    % Eliminate all the zero-weights from these (trivial increase in runtime, but makes it easier to spot when there is no variance)
    temp=logical(Weights==0);
    Weights=Weights(~temp);
    Values=Values(~temp);    
    SortedValues=Values;
    SortedWeights=Weights;
end
WeightedSortedValues=SortedValues.*SortedWeights;
if any(whichstats(4:7)>=1) || whichstats(2)==1
    CumSumSortedWeights=cumsum(SortedWeights);  % not needed if only want mean, median and std dev (& variance)
    skipcheck=all(CumSumSortedWeights==1);
else
    skipcheck=0;
end

%% Now the stats themselves
if whichstats(1)==1
    % Calculate the 'age conditional' mean
    AllStats.Mean=sum(WeightedSortedValues);
end
if whichstats(2)==1
    % Calculate the 'age conditional' median
    [~,index_median]=min(abs(CumSumSortedWeights-0.5));
    AllStats.Median=SortedValues(index_median); % The max is just to deal with 'corner' case where there is only one element in SortedWeightedValues
    if whichstats(1)==1
        AllStats.RatioMeanToMedian=AllStats.Mean/AllStats.Median;
    end
end

%% Deal with case where all the values are just the same anyway
if SortedValues(1)==SortedValues(end) || skipcheck
    % The current FnsToEvaluate takes only one value, so nothing but the mean and median make sense
    % OR
    % Due to numerical rounding, it has multiple values but only one has any meaning as all the mass is in one place (weights of magnitude,
    % e.g. 1e-26 can turn into cumulative weights with zero difference between them)
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
        AllStats.QuantileCutoffs=nan(nquantiles+1,1);
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
    if whichstats(3)==1
        % Calculate the 'age conditional' variance
        AllStats.Variance=sum(((Values-AllStats.Mean).^2).*Weights); % Weighted square of (values - mean)
        if AllStats.Variance<0 && AllStats.Variance>-10^(-6) % overwrite what is likely just numerical error
            AllStats.Variance=0;
        end
        AllStats.StdDeviation=sqrt(AllStats.Variance);
    end

    if (whichstats(4)>=1 && npoints>0 && ~(WeightedSortedValues(1)<0)) || whichstats(6)==2
        % precompute so don't duplicate; precompute needed if
        % WeightedSortedValues contains NaNs
        CumSumSortedWeightedValues=cumsum(WeightedSortedValues);
    end

    if whichstats(4)>=1
        % Lorenz curve
        if npoints>0
            if WeightedSortedValues(1)<0
                if whichstats(4)<3
                    AllStats.LorenzCurve=nan(npoints,1);
                    AllStats.LorenzCurveComment={'Lorenz curve cannot be calculated as some values are negative'};
                end
                AllStats.Gini=nan;
                AllStats.GiniComment={'Gini cannot be calculated as some values are negative'};
            else
                % CumSumSortedWeightedValues=cumsum(WeightedSortedValues); % precomputed

                if whichstats(4)<3
                    % Calculate the Lorenz curve
                    % (note, we already eliminated the zero mass points, and dealt with case that the remaining grid is just one point)
                    LorenzCurve=zeros(npoints,1);
                    llvec=1/npoints:1/npoints:1;
                    if whichstats(4)==1
                        for ll=1:npoints-1 % Note: because there are npoints points in lorenz curve, avoiding a loop here can be prohibitive in terms of memory use
                            [~,lorenzcind]=max(CumSumSortedWeights >= llvec(ll));
                            if lorenzcind==1
                                LorenzCurve(ll)=llvec(ll)*SortedValues(lorenzcind);
                            else
                                LorenzCurve(ll)=CumSumSortedWeightedValues(lorenzcind-1)+(llvec(ll)-CumSumSortedWeights(lorenzcind-1))*SortedValues(lorenzcind);
                            end
                        end
                        LorenzCurve(npoints)=CumSumSortedWeightedValues(end);
                    elseif whichstats(4)==2 % faster option, but can run out of memory
                        % Even thought the weights themselves are non-zero, you can still get that two consecutive elements of CumSumSortedWeights are the same (happened when the weight was 1e-26)
                        [CumSumSortedWeights2,u1index,~]=unique(CumSumSortedWeights);
                        % Sometimes this will become a single value, so need to check for this again
                        if length(CumSumSortedWeights2)==1
                            LorenzCurve=1/npoints:1/npoints:1;
                        else
                            temp=interp1(CumSumSortedWeights2,CumSumSortedWeightedValues(u1index),llvec(1:end-1));
                            LorenzCurve(1:end-1)=temp;
                            % Because of how interp1() works, it will put NaN at the bottom of the curve if there is a bunch of mass at first value
                            temp2=sum(isnan(temp));
                            if abs(LorenzCurve(temp2+1)-CumSumSortedWeightedValues(1))<1e-15
                                temp2=temp2+1;
                            end
                            LorenzCurve(1:temp2)=(CumSumSortedWeightedValues(1) - SortedValues(1)*(CumSumSortedWeights(1)-temp2/npoints)) .*((1:1:temp2)/temp2);
                            % Finished cleaning up the isnan()
                            LorenzCurve(npoints)=CumSumSortedWeightedValues(end);
                        end
                    end
                    % Now normalize the curve so that they are fractions of the total.
                    SumWeightedValues=sum(WeightedSortedValues);
                    AllStats.LorenzCurve=LorenzCurve/SumWeightedValues;
                end

                % Gini coefficient
                CumSumWeightedSortedValues=cumsum(WeightedSortedValues);
                CumSumWeightedSortedValues=CumSumWeightedSortedValues/CumSumWeightedSortedValues(end);
                AllStats.Gini=sum(CumSumWeightedSortedValues(2:end).*CumSumSortedWeights(1:end-1)- CumSumWeightedSortedValues(1:end-1).*CumSumSortedWeights(2:end));

                % % Calculate Gini coefficient (commented out is old version which was calculated from Lorenz Curve)
                % % Use the Gini=A/(A+B)=2*A formulation for Gini coefficent (see wikipedia).
                % A=(1/npoints:1/npoints:1)-AllStats.LorenzCurve'; % 'Height' between 45-degree line and Lorenz curve
                % A(logical(abs(A)<10^(-12)))=0; % Sometimes, get -10^(-15) due to numerical error, replace them with zero
                % A=sum(A)/npoints; % Note: 1/npoints is the 'width'. Area A is 'height times width' of gap from 45 degree line at each point on lorenz curve, summed up
                % % A=sum((1:1:npoints)/npoints-reshape(AllStats.LorenzCurve,[1,npoints]))/npoints;
                % AllStats.Gini=2*A;
            end
        end
    end
    
    if whichstats(5)==1 || whichstats(6)>=1 % note: anyway need min/max for quantile cutoffs
        % Min value
        tempindex=find(CumSumSortedWeights>=tolerance,1,'first');
        minvalue=SortedValues(tempindex);
        % Max value
        tempindex=find(CumSumSortedWeights>=(1-tolerance),1,'first');
        maxvalue=SortedValues(tempindex);
        % Create min and max as dedicated entries
        AllStats.Maximum=maxvalue;
        AllStats.Minimum=minvalue;
    end
    if whichstats(6)>=1
        if nquantiles==1
            error('Not allowed to set simoptions.nquantiles=1 (you anyway have this as the median, set higher or set equal zero to disable')
        end
        % Calculate the quantile means (ventiles by default)
        % Calculate the quantile cutoffs (ventiles by default)
        if whichstats(6)==1
            if nquantiles>0
                QuantileMeans=zeros(nquantiles,1);
                quantilecutoffindexes=zeros(nquantiles-1,1);
                quantilecvec=1/nquantiles:1/nquantiles:1-1/nquantiles;
                for quantilecind=1:nquantiles-1 % Note: because there are nquantiles points in quantiles, avoiding a loop here can be prohibitive in terms of memory use
                    [~,quantilecutoffindexes_quantilec]=max(CumSumSortedWeights >= quantilecvec(quantilecind));
                    quantilecutoffindexes(quantilecind)=quantilecutoffindexes_quantilec;
                end
                AllStats.QuantileCutoffs=[minvalue; SortedValues(quantilecutoffindexes); maxvalue];
                QuantileMeans(1)=sum(WeightedSortedValues(1:quantilecutoffindexes(1))) - SortedValues(quantilecutoffindexes(1))*(CumSumSortedWeights(quantilecutoffindexes(1))-1/nquantiles);
                for ll=2:nquantiles-1
                    if quantilecutoffindexes(ll-1)==quantilecutoffindexes(ll)
                        QuantileMeans(ll)=SortedValues(quantilecutoffindexes(ll))/nquantiles; % Note: need to /nquantiles, because later I *nquantiles
                    else
                        QuantileMeans(ll)=sum(WeightedSortedValues(quantilecutoffindexes(ll-1)+1:quantilecutoffindexes(ll))) - SortedValues(quantilecutoffindexes(ll))*(CumSumSortedWeights(quantilecutoffindexes(ll))-ll/nquantiles)  + SortedValues(quantilecutoffindexes(ll-1))*(CumSumSortedWeights(quantilecutoffindexes(ll-1))-(ll-1)/nquantiles);
                    end
                end
                QuantileMeans(nquantiles)=sum(WeightedSortedValues(quantilecutoffindexes(nquantiles-1)+1:end)) + SortedValues(quantilecutoffindexes(nquantiles-1))*(CumSumSortedWeights(quantilecutoffindexes(nquantiles-1))-(nquantiles-1)/nquantiles);
                AllStats.QuantileMeans=QuantileMeans*nquantiles; % Note: *nquantiles is really /(1/nquantiles), it is dividing by the mass of the quantile
            end
        elseif whichstats(6)==2 % Vectorizes so faster, but uses more memory (can cause out of memory errors if you have large nquantiles, hence it is not the default)
            if nquantiles>0
                % QuantileMeans=zeros(nquantiles,1,'gpuArray');
                [~,quantilecutoffindexes]=max(CumSumSortedWeights >= 1/nquantiles:1/nquantiles:1-1/nquantiles);
                AllStats.QuantileCutoffs=[minvalue; SortedValues(quantilecutoffindexes); maxvalue];
                
                quantilecutoffindexes_lower=[1; quantilecutoffindexes'];
                quantilecutoffindexes_upper=[quantilecutoffindexes'; numel(WeightedSortedValues)];
                
                % CumSumSortedWeightedValues=cumsum(WeightedSortedValues); % precomputed
                term1=CumSumSortedWeightedValues(quantilecutoffindexes_upper)-CumSumSortedWeightedValues(quantilecutoffindexes_lower);
                term2=SortedValues(quantilecutoffindexes_upper).*(CumSumSortedWeights(quantilecutoffindexes_upper)-(1:1:nquantiles)'/nquantiles);
                term3=SortedValues(quantilecutoffindexes_lower).*(CumSumSortedWeights(quantilecutoffindexes_lower)-(0:1:nquantiles-1)'/nquantiles);
                QuantileMeans=term1-term2+term3;
                
                % This formula only works when the cutoff indexes are different, so when they are not, do some overwriting
                temp=logical(quantilecutoffindexes_lower==quantilecutoffindexes_upper);
                QuantileMeans(temp)=SortedValues(quantilecutoffindexes_upper(temp))/nquantiles;  % Note: need to /nquantiles, because later I *nquantiles
                AllStats.QuantileMeans=QuantileMeans*nquantiles; % Note: *nquantiles is really /(1/nquantiles), it is dividing by the mass of the quantile
            end
        end

    end
    
    if whichstats(7)==1
        if ~any(whichstats(4)==[1,2])
            error('whichstats(7)=1 can only be used with whichstats(4)=1 or 2 (Lorenz Curve forms basis for some of the stats in whichstats(7))')
        end
        % Top X share indexes (npoints will be number of points in Lorenz Curve)
        Top1cutpoint=round(0.99*npoints);
        Top5cutpoint=round(0.95*npoints);
        Top10cutpoint=round(0.90*npoints);
        Top50cutpoint=round(0.50*npoints);
        AllStats.MoreInequality.Top1share=1-AllStats.LorenzCurve(Top1cutpoint);
        AllStats.MoreInequality.Top5share=1-AllStats.LorenzCurve(Top5cutpoint);
        AllStats.MoreInequality.Top10share=1-AllStats.LorenzCurve(Top10cutpoint);
        AllStats.MoreInequality.Bottom50share=AllStats.LorenzCurve(Top50cutpoint);
        % Now some cutoffs
        AllStats.MoreInequality.Percentile50th=AllStats.Median; % just a duplicate for convenience
        index_p90=find(CumSumSortedWeights>=0.90,1,'first');
        AllStats.MoreInequality.Percentile90th=SortedValues(index_p90);
        index_p95=find(CumSumSortedWeights>=0.95,1,'first');
        AllStats.MoreInequality.Percentile95th=SortedValues(index_p95);
        index_p99=find(CumSumSortedWeights>=0.99,1,'first');
        AllStats.MoreInequality.Percentile99th=SortedValues(index_p99);
    end
end


%% Comment: To find, e.g. the median, we can either do
% medianindex=find(CumSumSortedWeights>=0.50,1,'first');
% Or
% [~,medianindex]=max(CumSumSortedWeights>=0.50)
% I ran a bunch of tests and both take essentially the same amount of time
% (on average find() was slower, but in some runs it was faster, on average difference was something like 10%, so not worth worrying which is used)






end


