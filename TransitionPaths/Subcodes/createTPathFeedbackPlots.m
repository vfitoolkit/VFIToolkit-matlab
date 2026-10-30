function createTPathFeedbackPlots(PricePathNames,AggVarNames,GEeqnNames,PricePathOld,AggVarsPath,GEcondnPath,transpathoptions,PricePathSizeVec)
% Creates plots of the current transition path iteration. Control what is plotted with transpathoptions
%
% PricePathSizeVec (optional) is 2-by-nPrices, the first and last column of each price in PricePathOld. A
% price can have several columns (one per ptype, or one per age for an age-dependent price), and all of
% them are drawn in that price's subplot. Without it every price is taken to be one column.
% GEeqnNames has one name per column of GEcondnPath (callers with general eqm conditions by ptype pass
% the expanded names).
if ~exist('PricePathSizeVec','var')
    PricePathSizeVec=[1:length(PricePathNames); 1:length(PricePathNames)];
end

if transpathoptions.graphpricepath==1 || transpathoptions.graphaggvarspath==1 || transpathoptions.graphGEcondns==1
    disp(' ') % For some reason without this Matlab just sometimes doesn't bother updating the graphs
end
% Note: the disp() above is only done when something is actually going to be plotted. It is a
% hack to make Matlab flush the graphics queue, so it serves no purpose when no graphs are drawn,
% and it was otherwise putting one line per transition path iteration into the diary.

if transpathoptions.graphpricepath==1
    % Do a graph of the PricePaths
    if length(PricePathNames)>12
        ncolumns=4;
    elseif length(PricePathNames)>6
        ncolumns=3;
    else
        ncolumns=2;
    end
    nrows=ceil(length(PricePathNames)/ncolumns);
    figure(1)
    for pp=1:length(PricePathNames)
        subplot(nrows,ncolumns,pp); plot(PricePathOld(:,PricePathSizeVec(1,pp):PricePathSizeVec(2,pp)))
        title(PricePathNames{pp})
    end
    sgtitle('Current Price Path');
end
if transpathoptions.graphaggvarspath==1
    % Do a graph of the AggVars
    if length(AggVarNames)>12
        ncolumns=4;
    elseif length(AggVarNames)>6
        ncolumns=3;
    else
        ncolumns=2;
    end
    nrows=ceil(length(AggVarNames)/ncolumns);
    figure(2)
    if isstruct(AggVarsPath)
        for pp=1:length(AggVarNames)
            subplot(nrows,ncolumns,pp); plot(AggVarsPath.(AggVarNames{pp}).Mean)
            title(AggVarNames{pp})
        end
    else
        for pp=1:length(AggVarNames)
            subplot(nrows,ncolumns,pp); plot(AggVarsPath(:,pp))
            title(AggVarNames{pp})
        end
    end
    sgtitle('Current Agg Vars Path');
end
if transpathoptions.graphGEcondns==1
    % Assumes transpathoptions.GEnewprice==3
    % Do an additional graph, this one of the general eqm conditions
    if length(GEeqnNames)>12
        ncolumns=4;
    elseif length(GEeqnNames)>6
        ncolumns=3;
    else
        ncolumns=2;
    end
    nrows=ceil(length(GEeqnNames)/ncolumns);
    figure(3)
    for pp=1:length(GEeqnNames)
        subplot(nrows,ncolumns,pp); plot(GEcondnPath(:,pp))
        title(GEeqnNames{pp})
    end
    sgtitle('Current General Eqm Condns Path');
end


end