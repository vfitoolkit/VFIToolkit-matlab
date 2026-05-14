function [tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tminus1paramNames,tplus1pricePathkk,...
          use_tplus1price,use_tminus1price,use_tminus1params,use_tminus1AggVars]=...
          inputsFindtplus1tminus1(FnsToEvaluate,GeneralEqmEqns,PricePathNames,ParamPathNames,Names_i,transpathoptions)
% Subscript that is used to determine if there are any '_tminus1' or
% '_tplus1' variables used as inputs to FnsToEvaluate or GeneralEqmEqns
% (Used as part of transition path codes)
% Look for the _tplus1 in PricePath, and for _tminus1 in AggVars, PricePath, and ParamPath
%
% Names_i and transpathoptions are required inputs.
%  - Pass Names_i={} when permanent types are not relevant.
%  - Pass simoptions in place of transpathoptions when calling from
%    EvalFnOnTransPath_* commands (those store initialvalues on simoptions
%    rather than transpathoptions).

% Check for using _tminus1 (for price or AggVars) or _tplus1, honoring PType
FnInputNames={}; % FnNames belonging to no PType
PTypeFnInputNames=struct(); % Built as we find them
ninputs=0;
% Get all the input names for FnsToEvaluate
AggVarNames=fieldnames(FnsToEvaluate);
PTypeAggVarNames=struct();
for ff=1:length(AggVarNames)
    if ~isempty(Names_i)
        temp={};
        for ii=1:length(Names_i)
            if isfield(FnsToEvaluate.(AggVarNames{ff}), Names_i{ii})
                if ~isfield(PTypeAggVarNames, Names_i{ii})
                    PTypeAggVarNames.(Names_i{ii})={};
                    PTypeFnInputNames.(Names_i{ii})={};
                end
                PTypeAggVarNames.(Names_i{ii})=[PTypeAggVarNames.(Names_i{ii});AggVarNames{ff}];
                temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}).(Names_i{ii}));
                PTypeFnInputNames.(Names_i{ii})=[PTypeFnInputNames.(Names_i{ii});temp]; % Note, this will include the (d,aprime,a,z), but that is irrelevant to our current purposes
                break
            end
        end
        if isempty(temp)
            error(['Could not find', AggVarNames{ff}, 'with PType in {', Names_i, '}'])
        end
    else
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        FnInputNames={FnInputNames{:},temp{:}}; % Note, this will include the (d,aprime,a,z), but that is irrelevant to our current purposes
    end
    tempninputs=length(temp);
    ninputs=ninputs+tempninputs;
end
for ii=1:length(Names_i) % If Names_i is empty this loop is not executed
    name_matches=ismember(AggVarNames, PTypeAggVarNames.(Names_i{ii}));
    AggVarNames(name_matches)=[];
end

% Get all the input names for GeneralEqmEqns
GEeqnNames=fieldnames(GeneralEqmEqns);
for gg=1:length(GEeqnNames)
    temp=getAnonymousFnInputNames(GeneralEqmEqns.(GEeqnNames{gg}));
    tempninputs=length(temp);
    FnInputNames={FnInputNames{:},temp{:}}; % Note, this will include the (d,aprime,a,z), but that is irrelevant to our current purposes
    ninputs=ninputs+tempninputs;
end
% Now, get rid of all the duplicates
FnInputNames=unique(FnInputNames);
for ii=1:length(Names_i) % If Names_i is empty this loop is not executed
    PTypeFnInputNames.(Names_i{ii})=unique(PTypeFnInputNames.(Names_i{ii}));
end

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
for ii=1:length(Names_i)
    PType_tminus1AggVarsNames.(Names_i{ii})={};
    PType_ntminus1AggVars.(Names_i{ii})=0;
end
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
if isempty(Names_i)
    for ii=1:ntminus1
        for kk=1:length(AggVarNames)
            if strcmp(tminus1Names{ii},AggVarNames{kk})
                ntminus1AggVars=ntminus1AggVars+1;
                tminus1AggVarsNames{ntminus1AggVars}=tminus1Names{ii};
                tminus1UsedAsPriceOrAggVar(ii)=1;
            end
        end
    end
    % Return the array of tminus1AggVarsNames
elseif ntminus1>0
    for ii=1:ntminus1
        for nn=1:length(Names_i)
            AggVarNames_nn=PTypeAggVarNames.(Names_i{nn});
            for kk=1:length(AggVarNames_nn)
                if strcmp(tminus1Names{ii},AggVarNames_nn{kk})
                    PType_ntminus1AggVars.(Names_i{nn})=PType_ntminus1AggVars.(Names_i{nn})+1;
                    PType_tminus1AggVarsNames.(Names_i{nn}){PType_ntminus1AggVars.(Names_i{nn})}=tminus1Names{ii};
                    tminus1UsedAsPriceOrAggVar(ii)=1;
                end
            end
        end
    end
    % It is easier to leave empty values than remove/retest the fields
    % for ii=1:length(Names_i)
    %     if isempty(PType_tminus1AggVarsNames.(Names_i{ii}))
    %         PType_tminus1AggVarsNames=rmfield(PType_tminus1AggVarsNames, Names_i{ii});
    %     end
    % end
    % Return the PType structure of the arrays of tminus1AggVarsNames
    tminus1AggVarsNames=PType_tminus1AggVarsNames;
end
% Check that they have all been used, otherwise error
if prod(tplus1UsedAsPriceOrAggVar)==0
    error('FnsToEvaluate or GeneralEqmEqns are trying to use a _tplus1 input that is NOT a price')
elseif prod(tminus1UsedAsPriceOrAggVar)==0
    error('FnsToEvaluate or GeneralEqmEqns are trying to use a _tminus1 input that NEITHER a price NOR an an aggregate (determined by FnsToEvaluate)')
end

%% Set the use_* flags, and (where relevant) check that initialvalues are present in transpathoptions
use_tplus1price=0;
if ~isempty(tplus1priceNames)
    use_tplus1price=1;
end
use_tminus1price=0;
if ~isempty(tminus1priceNames)
    use_tminus1price=1;
    for ii=1:length(tminus1priceNames)
        if ~isfield(transpathoptions.initialvalues,tminus1priceNames{ii})
            error('Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1priceNames{ii})
        end
    end
end
use_tminus1params=0;
if ~isempty(tminus1paramNames)
    use_tminus1params=1;
    for ii=1:length(tminus1paramNames)
        if ~isfield(transpathoptions.initialvalues,tminus1paramNames{ii})
            error('Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1paramNames{ii})
        end
    end
end
use_tminus1AggVars=0;
if ~isempty(tminus1AggVarsNames)
    use_tminus1AggVars=1;
    % For PType callers, tminus1AggVarsNames is a struct keyed by Names_i —
    % the initialvalues check is left to the caller in that case.
    if iscell(tminus1AggVarsNames)
        for ii=1:length(tminus1AggVarsNames)
            if ~isfield(transpathoptions.initialvalues,tminus1AggVarsNames{ii})
                error('Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1AggVarsNames{ii})
            end
        end
    end
end
% Note: I used this approach (rather than just creating _tplus1 and _tminus1 for everything) as it will be same computation.

if isfield(transpathoptions,'verbose') && transpathoptions.verbose==2
    use_tplus1price
    use_tminus1price
    use_tminus1params
    use_tminus1AggVars
    % tplus1pricePathkk
end

end
