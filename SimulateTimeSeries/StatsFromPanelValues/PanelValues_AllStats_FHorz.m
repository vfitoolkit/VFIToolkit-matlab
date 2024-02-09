function AllStats=PanelValues_AllStats_FHorz(PanelValues,simoptions)

if ~exist('simoptions','var')
    simoptions.nquantiles=20; % by default gives ventiles
    simoptions.npoints=100; % number of points for lorenz curve
    simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    simoptions.whichstats=ones(7,1); % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes 
else
    if ~isfield(simoptions,'whichstats')
        simoptions.whichstats=ones(7,1); % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes
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

% Every observation has equal weight
Weights=ones(size(PanelValues.(PanelVariableNames{1})));
Weights=Weights/sum(Weights);
% They are not sorted
presorted=0;

for ff=1:length(PanelVariableNames)
    AllStats.(PanelVariableNames{ff})=StatsFromWeightedGrid(PanelValues.(PanelVariableNames{ff}),Weights,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance, presorted, simoptions.whichstats);
end


end