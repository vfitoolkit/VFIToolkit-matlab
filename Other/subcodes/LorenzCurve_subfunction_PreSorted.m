function LorenzCurve=LorenzCurve_subfunction_PreSorted(SortedWeightedValues,CumSumSortedWeights,npoints,Parallel)

% I have considered making this an option to be set but for now is just hardcoded.
% Order to round to avoid numerical error
options.ordertoroundtoavoidnumericalerror=9;

% Calculate the 'age conditional' lorenz curve
%We now want to use interpolation, but this won't work unless all values in CumSumSortedWeights are distinct. So we now remove
%any duplicates (ie. points of zero probability mass/density). We then have to remove the corresponding points of SortedValues. Since we
%are just looking for 100 points to make up our cdf I round all variables to 9 decimal points before checking for uniqueness (Do
%this because otherwise rounding in the ~12th decimal place was causing problems with vector not being sorted as strictly increasing.
CumSumSortedWeights_temp=round(CumSumSortedWeights*10^options.ordertoroundtoavoidnumericalerror);
% Comment:
% round(CumSumSortedWeights,options.ordertoroundtoavoidnumericalerror);
% Works on cpu, but not implemented for GPU as of R2021a (likely better way to do this once Matlab implements it)
[~,UniqueIndex] = unique(CumSumSortedWeights_temp,'first');
CumSumSortedStationaryDistVec_NoDuplicates=CumSumSortedWeights(sort(UniqueIndex));
SortedWeightedValues_NoDuplicates=SortedWeightedValues(sort(UniqueIndex));
CumSumSortedWeightedValues_NoDuplicates=cumsum(SortedWeightedValues_NoDuplicates);
if Parallel==2
    InverseCDF_xgrid=gpuArray(1/npoints:1/npoints:1);
else
    InverseCDF_xgrid=1/npoints:1/npoints:1;
end

if numel(CumSumSortedStationaryDistVec_NoDuplicates)==1 % Have to treat the case of Lorenz curve for perfect equality (so just one unique element) seperately as otherwise causes interp1 to error
    if Parallel==2
        InverseCDF_SSvalues=CumSumSortedStationaryDistVec_NoDuplicates*ones(npoints,1,'gpuArray'); %0:(1/npoints):1;
    else
        InverseCDF_SSvalues=CumSumSortedStationaryDistVec_NoDuplicates*ones(npoints,1); %0:(1/npoints):1;        
    end
else
    InverseCDF_SSvalues=interp1(CumSumSortedStationaryDistVec_NoDuplicates,CumSumSortedWeightedValues_NoDuplicates, InverseCDF_xgrid);
end
% interp1 cannot work for the point of InverseCDF_xgrid=1 (gives NaN). Since we have already sorted 
% and removed duplicates this will just be the last point so we can just grab it directly.
InverseCDF_SSvalues(npoints)=CumSumSortedWeightedValues_NoDuplicates(end);
% interp1 may have similar problems at the bottom of the cdf
ll=1; %use ll to figure how many points with this problem
while InverseCDF_xgrid(ll)<CumSumSortedStationaryDistVec_NoDuplicates(1)
    ll=ll+1;
end
for jj=1:ll-1 %divide evenly through these states (they are all identical)
    InverseCDF_SSvalues(jj)=(jj/ll)*InverseCDF_SSvalues(ll);
end
LorenzCurve=(InverseCDF_SSvalues./CumSumSortedWeightedValues_NoDuplicates(end))';

end