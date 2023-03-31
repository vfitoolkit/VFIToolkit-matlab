function LorenzCurve=LorenzCurve_FromProbDist(ProbabilityMassDistn, Values, npoints, Parallel)
% Inputs
% ProbabilityMassDistn should be the weights associated with each of the states/units, e.g., [0.1,0.5,0.4]
% Values should contain the values associated with each of the states/units, e.g., [3,2,7]
%
% Optional: npoints, the number of points used for the Lorenz curve
%           Parallel, 0 or 1 is cpu, 2 is gpu
if ~exist('npoints','var')
    npoints=100; % 100 points by default
end
if ~exist('Parallel','var')
    Parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
end

ProbabilityMassDistn=reshape(ProbabilityMassDistn,[numel(ProbabilityMassDistn),1]);
ProbabilityMassDistn=ProbabilityMassDistn/sum(ProbabilityMassDistn); % Make sure it sums to one.

Values=reshape(Values,[numel(Values),1]);
[SortedValues,SortedValues_index] = sort(Values);

SortedWeights = ProbabilityMassDistn(SortedValues_index);
CumSumSortedWeights=cumsum(SortedWeights);

WeightedValues=Values.*ProbabilityMassDistn;
SortedWeightedValues=WeightedValues(SortedValues_index);

LorenzCurve=LorenzCurve_subfunction_PreSorted(SortedWeightedValues,CumSumSortedWeights,npoints,Parallel);

end