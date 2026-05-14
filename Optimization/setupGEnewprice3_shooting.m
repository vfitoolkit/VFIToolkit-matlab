function options=setupGEnewprice3_shooting(options,GeneralEqmEqns,PriceParamNames,N_i,PricePathSizeVec)
% Can be used for stationary general eqm, or for general eqm transition paths
% For stationary eqm: 
%    input GEPriceParamNames for PriceParamNames
%    input heteroagentoptions for options
% For stationary eqm: 
%    input PricePathNames for PriceParamNames
%    input transpathoptions for options
%
% N_i is an optional input, only needed for models with permanent types
% PricePathSizeVec is an optional input, only needed for models with permanent types

% Note: order of rows in the output options.GEnewprice3.howtoupdate will
% follow the order of PriceParamNames, as it will be used to update the
% prices.

GEeqnNames=fieldnames(GeneralEqmEqns);
nGeneralEqmEqns=length(GEeqnNames);

%% Set up GEnewprice==3 (if relevant)
fminalgo5=0;
if isfield(options,'fminalgo5') % in stationary general eqm, fminalgo5 is shooting
    fminalgo5=1;
    % options.GEnewprice=3;
    options.GEnewprice3=options.fminalgo5;
elseif options.GEnewprice~=3
    return % Not being used
end
if ~isfield(options,'oldpathweight')
    options.oldpathweight=0; % Not actually used for anything
end

options.weightscheme=0;

if ~isfield(options,'updateaccuracycutoff')
    options.updateaccuracycutoff=0; % No cut-off (only changes in the price larger in magnitude that this will be made (can be set to, e.g., 10^(-6) to help avoid changes at overly high precision))
end

if ~isstruct(GeneralEqmEqns)
    error('Cannot call this unless GeneralEqmEqns is struct')
end

if size(options.GEnewprice3.howtoupdate,2)~=4
    error('options.GEnewprice3.howtoupdate should have 4 columns: GECondnName, Price name, add, factor')
end
if size(options.GEnewprice3.howtoupdate,1)~=nGeneralEqmEqns
    error('options.GEnewprice3.howtoupdate should have ones row for each general eqm eqn')
end

%%
if ~isfield(options,'GEptype') % For models without permanent type

    % Need to make sure that order of rows in options.GEnewprice3.howtoupdate
    % Is same as order of PriceParamNames
    % I do this by just reordering rows of options.GEnewprice3.howtoupdate
    temp=options.GEnewprice3.howtoupdate;
    for pp=1:length(PriceParamNames)
        for jj=1:size(temp,1)
            if strcmp(temp{jj,1},PriceParamNames{pp}) % Names match
                options.GEnewprice3.howtoupdate{pp,1}=temp{jj,1}; % general eqm eqn name
                options.GEnewprice3.howtoupdate{pp,2}=temp{jj,2}; % general eqm price name
                options.GEnewprice3.howtoupdate{pp,3}=temp{jj,3}; % add(/subtract)
                options.GEnewprice3.howtoupdate{pp,4}=temp{jj,4}; % factor
            end
        end
    end

    options.GEnewprice3.add=[options.GEnewprice3.howtoupdate{:,3}];
    options.GEnewprice3.factor=[options.GEnewprice3.howtoupdate{:,4}];
    options.GEnewprice3.keepold=ones(size(options.GEnewprice3.factor));
    tempweight=options.oldpathweight;
    options.oldpathweight=zeros(size(options.GEnewprice3.factor));
    for ii=1:length(options.GEnewprice3.factor)
        if options.GEnewprice3.factor(ii)==Inf
            options.GEnewprice3.factor(ii)=1;
            options.GEnewprice3.keepold(ii)=0;
            options.oldpathweight(ii)=tempweight;
        end
    end
    if ~(size(options.GEnewprice3.howtoupdate,1)==nGeneralEqmEqns) % Note: inputs were already tested
        error('options.GEnewprice3.howtoupdate does not fit with GeneralEqmEqns (number of rows is different to the number of GeneralEqmEqns fields) \n')
    end
    % Create 'permute' which is used to reorder the vector of GEcondn values to have same order as the PriceParamNames
    % Note: prices=GEcond(permute)
    options.GEnewprice3.permute=zeros(size(options.GEnewprice3.howtoupdate,1),1);
    for pp=1:length(PriceParamNames) % number of rows is the number of prices (and number of GE conditions)
        for gg=1:length(GEeqnNames)
            if strcmp(options.GEnewprice3.howtoupdate{pp,1},GEeqnNames{gg})
                options.GEnewprice3.permute(pp)=gg;
            end
        end
    end

else
    %% Model with permanent type and using options.GEptype
    nGeneralEqmEqns_acrossptypes=sum(options.GEptype==0)+N_i*sum(options.GEptype==1);

    % Before starting, make sure that GE that depend on ptype match up with PricePaths that depend on ptype
    options.Priceptype=zeros(length(PriceParamNames),1);
    for gg=1:nGeneralEqmEqns
        if options.GEptype(gg)==1
            for gg2=1:size(options.GEnewprice3.howtoupdate,1)
                if strcmp(options.GEnewprice3.howtoupdate{gg2,1},GEeqnNames{gg})
                    pricename_gg=options.GEnewprice3.howtoupdate{gg2,2};
                end
            end
            for pp=1:length(PriceParamNames)
                if strcmp(pricename_gg,PriceParamNames{pp})
                    if PricePathSizeVec(2,pp)-PricePathSizeVec(1,pp)+1~=N_i
                        fprintf('Following error relates to GE condition %s and to price %s \n',GEeqnNames{gg},PriceParamNames{pp})
                        error('You declared a GE condition to depend on permenent type, but the price that relates to it (in options.GEnewprice3.howtoupdate) does not depend on ptype')
                    end
                    options.Priceptype(pp)=1; % this price depends on ptype
                end
            end
        end
    end

    % Need to make sure that order of rows in options.GEnewprice3.howtoupdate
    % Is same as order of fields in PriceParamNames
    % I do this by just reordering rows of options.GEnewprice3.howtoupdate
    temp=options.GEnewprice3.howtoupdate;
    pp_index=zeros(1,length(PriceParamNames)+(N_i-1)*sum(options.Priceptype==1));
    pp_c=0;
    for pp=1:length(PriceParamNames)
        for jj=1:size(temp,1)
            if strcmp(temp{jj,2},PriceParamNames{pp}) % Names match
                for ii=1:(1+options.Priceptype(pp)*(N_i-1)) % Note: 1 or N_i, depending on options.GEptype(pp)
                    pp_c=pp_c+1;
                    options.GEnewprice3.howtoupdate{pp_c,1}=temp{jj,1};
                    options.GEnewprice3.howtoupdate{pp_c,2}=temp{jj,2};
                    options.GEnewprice3.howtoupdate{pp_c,3}=temp{jj,3};
                    options.GEnewprice3.howtoupdate{pp_c,4}=temp{jj,4};
                    pp_index(pp_c)=pp;
                end
            end
        end
    end
    % Note: options.GEnewprice3.howtoupdate will have extra repeated rows whenever options.GEptype(gg)=1
    
    options.GEnewprice3.add=[options.GEnewprice3.howtoupdate{:,3}];
    options.GEnewprice3.factor=[options.GEnewprice3.howtoupdate{:,4}];
    options.GEnewprice3.keepold=ones(size(options.GEnewprice3.factor));
    tempweight=options.oldpathweight;
    options.oldpathweight=zeros(size(options.GEnewprice3.factor));
    for ii=1:length(options.GEnewprice3.factor)
        if options.GEnewprice3.factor(ii)==Inf
            options.GEnewprice3.factor(ii)=1;
            options.GEnewprice3.keepold(ii)=0;
            options.oldpathweight(ii)=tempweight;
        end
    end
    if ~(size(options.GEnewprice3.howtoupdate,1)==nGeneralEqmEqns_acrossptypes)
        error('options.GEnewprice3.howtoupdate does not fit with GeneralEqmEqns (different number of conditions) \n')
    end

    % Create 'permute' which is used to reorder the vector of GEcondn values to have same order as the PriceParamNames
    % Note: prices=GEcond(permute)
    options.GEnewprice3.permute=zeros(length(PriceParamNames)+(N_i-1)*sum(options.Priceptype==1),1);
    gg_c=1;
    for gg=1:nGeneralEqmEqns
        % find the price name attached to this GE eqn
        for jj=1:size(options.GEnewprice3.howtoupdate,1)
            if strcmp(options.GEnewprice3.howtoupdate{jj,1},GEeqnNames{gg})
                pricename_gg = options.GEnewprice3.howtoupdate{jj,2};
                break % avoid that for ptype eqns it re-overwrites pricename_gg N_i times with the same value
            end
        end
        % find which pp that price is
        for pp=1:length(PriceParamNames)
            if strcmp(PriceParamNames{pp},pricename_gg)
                if options.GEptype(gg)==0
                    options.GEnewprice3.permute(PricePathSizeVec(1,pp)) = gg_c;
                    gg_c=gg_c+1;
                else
                    options.GEnewprice3.permute(PricePathSizeVec(1,pp):PricePathSizeVec(2,pp)) = gg_c + (0:1:N_i-1);
                    gg_c=gg_c+N_i;
                end
            end
        end
    end

    % Note: if some GE conditions are done conditional on ptype then
    % options.GEnewprice3.howtoupdate
    % options.GEnewprice3.permute
    % reflect this.
end


%% If doing stationary general eqm, call output fminalgo5 instead of GEnewprice3
if fminalgo5==1
    options.fminalgo5=options.GEnewprice3;
    options=rmfield(options,'GEnewprice3');
    options=rmfield(options,'oldpathweight');
end



end