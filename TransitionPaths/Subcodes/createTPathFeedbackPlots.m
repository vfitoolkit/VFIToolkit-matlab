function createTPathFeedbackPlots(PricePathNames,AggVarNames,GEeqnNames,PricePathOld,AggVarsPath,GEcondnPath,transpathoptions)
% Creates plots of the current transition path iteration. Control what is plotted with transpathoptions

disp(' ') % For some reason without this Matlab just sometimes doesn't bother updating the graphs

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
    fig1=figure(1);
    for pp=1:length(PricePathNames)
        subplot(nrows,ncolumns,pp); plot(PricePathOld(:,pp))
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
    fig2=figure(2);
    for pp=1:length(AggVarNames)
        subplot(nrows,ncolumns,pp); plot(AggVarsPath(:,pp))
        title(AggVarNames{pp})
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
    fig3=figure(3);
    for pp=1:length(GEeqnNames)
        subplot(nrows,ncolumns,pp); plot(GEcondnPath(:,pp))
        title(GEeqnNames{pp})
    end
    sgtitle('Current General Eqm Condns Path');
end


end