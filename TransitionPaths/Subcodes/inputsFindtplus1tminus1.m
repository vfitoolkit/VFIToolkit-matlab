function [tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tminus1paramNames,tplus1pricePathkk]=inputsFindtplus1tminus1(FnsToEvaluate,GeneralEqmEqns,PricePathNames,ParamPathNames)
% Subscript that is used to determine if there are any '_tminus1' or
% '_tplus1' variables used as inputs to FnsToEvaluate or GeneralEqmEqns
% (Used as part of transition path codes)
% Look for the _tplus1 in PircePath, and for _tminus1 in AggVars, PricePath, and ParamPath

% Check for using _tminus1 (for price or AggVars) or _tplus1
FnInputNames={};
ninputs=0;
% Get all the input names for FnsToEvaluate
AggVarNames=fieldnames(FnsToEvaluate);
for ff=1:length(AggVarNames)
    temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
    tempninputs=length(temp);
    FnInputNames={FnInputNames{:},temp{:}}; % Note, this will include the (d,aprime,a,z), but that is irrelevant to our current purposes
    ninputs=ninputs+tempninputs;
end
% Get all the input names for GeneralEqmEqns
GEeqnNames=fieldnames(GeneralEqmEqns);
for ff=1:length(GEeqnNames)
    temp=getAnonymousFnInputNames(GeneralEqmEqns.(GEeqnNames{ff}));
    tempninputs=length(temp);
    FnInputNames={FnInputNames{:},temp{:}}; % Note, this will include the (d,aprime,a,z), but that is irrelevant to our current purposes
    ninputs=ninputs+tempninputs;
end
% Now, get rid of all the duplicates
FnInputNames=unique(FnInputNames);

% Find any which are _tplus1 or _tminus1
tplus1Names={};
ntplus1=0;
tminus1Names={};
ntminus1=0;
for ii=1:length(FnInputNames)
    if length(FnInputNames{ii})>7
        temp=FnInputNames{ii};
        if strcmp(temp(end-6:end),'_tplus1')
            ntplus1=ntplus1+1;
            tplus1Names{ntplus1}=temp(1:end-7); % Drop the _tminus1 from name
        end
        if length(FnInputNames{ii})>8
            if strcmp(temp(end-7:end),'_tminus1')
                ntminus1=ntminus1+1;
                tminus1Names{ntminus1}=temp(1:end-8); % Drop the _tminus1 from name
            end
        end
    end
end

tplus1UsedAsPriceOrAggVar=zeros(ntplus1,1); % I replace the zeros with ones as I find them
tminus1UsedAsPriceOrAggVar=zeros(ntminus1,1); % I replace the zeros with ones as I find them
tplus1priceNames={}; ntplus1prices=0;
tminus1priceNames={}; ntminus1prices=0;
tminus1paramNames={}; ntminus1params=0;
tminus1AggVarsNames={}; ntminus1AggVars=0;
tplus1pricePathkk=[];

% Check that they are prices, otherwise error
% Note: the for-loop over PricePathNames has to be inside the loop over
% ntplus1 because we need to follow the ordering in ntplus1 when creating
% tplus1priceNames and tplus1pricePathkk
for ii=1:ntplus1
    for kk=1:length(PricePathNames)
        if strcmp(tplus1Names{ii},PricePathNames{kk})
            ntplus1prices=ntplus1prices+1;
            tplus1priceNames{ntplus1prices}=tplus1Names{ii};
            tplus1UsedAsPriceOrAggVar(ii)=1;
            tplus1pricePathkk(ii)=kk;
        end
    end
end
for ii=1:ntminus1
    for kk=1:length(PricePathNames)
        if strcmp(tminus1Names{ii},PricePathNames{kk})
            ntminus1prices=ntminus1prices+1;
            tminus1priceNames{ntminus1prices}=tminus1Names{ii};
            tminus1UsedAsPriceOrAggVar(ii)=1;
        end
    end
end
for ii=1:ntminus1
    for kk=1:length(ParamPathNames)
        if strcmp(tminus1Names{ii},ParamPathNames{kk})
            ntminus1params=ntminus1params+1;
            tminus1paramNames{ntminus1params}=tminus1Names{ii};
            tminus1UsedAsPriceOrAggVar(ii)=1;
        end
    end
end
for ii=1:ntminus1
    for kk=1:length(AggVarNames)
        if strcmp(tminus1Names{ii},AggVarNames{kk})
            ntminus1AggVars=ntminus1AggVars+1;
            tminus1AggVarsNames{ntminus1AggVars}=tminus1Names{ii};
            tminus1UsedAsPriceOrAggVar(ii)=1;
        end
    end
end
% Check that they have all been use, otherwise error
if prod(tplus1UsedAsPriceOrAggVar)==0
    fprintf('ERROR: FnsToEvaluate or GeneralEqmEqns are trying to use a _tplus1 input that is NOT a price \n')
    dbstack
    return
elseif prod(tminus1UsedAsPriceOrAggVar)==0
    fprintf('ERROR: FnsToEvaluate or GeneralEqmEqns are trying to use a _tminus1 input that NEITHER a price NOR an an aggregate (determined by FnsToEvaluate) \n')
    dbstack
    return
end


end
