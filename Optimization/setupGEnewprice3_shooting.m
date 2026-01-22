function options=setupGEnewprice3_shooting(options,GeneralEqmEqns,GEPriceParamNames,N_i,PricePathSizeVec)
% options can be heteroagentoptions or transpathoptions
% transpathoptions: GEPriceParamNames will be PricePathNames
% N_i is an optional input, only needed for models with permanent types
% PricePathSizeVec is an optional input, only needed for models with permanent types

GEeqnNames=fieldnames(GeneralEqmEqns);
nGeneralEqmEqns=length(GEeqnNames);

%% Set up GEnewprice==3 (if relevant)
fminalgo5=0;
if isfield(options,'fminalgo5') % in stationary general eqm, fminalgo5 is shooting
    fminalgo5=1;
    % options.GEnewprice=3;
    options.GEnewprice3=options.fminalgo5;
    options.oldpathweight=0; % Not actually used for anything
elseif options.GEnewprice~=3
    return % Not being used
end

options.weightscheme=0;

if ~isfield(options,'updateaccuracycutoff')
    options.updateaccuracycutoff=0; % No cut-off (only changes in the price larger in magnitude that this will be made (can be set to, e.g., 10^(-6) to help avoid changes at overly high precision))
end

if ~isstruct(GeneralEqmEqns)
    error('Cannot call this unless GeneralEqmEqns is struct')
end

%%
if ~isfield(options,'GEptype') % For models without permanent type

    % Need to make sure that order of rows in options.GEnewprice3.howtoupdate
    % Is same as order of fields in GeneralEqmEqns
    % I do this by just reordering rows of options.GEnewprice3.howtoupdate
    temp=options.GEnewprice3.howtoupdate;
    GEeqnNames=fieldnames(GeneralEqmEqns);
    for ii=1:length(GEeqnNames)
        for jj=1:size(temp,1)
            if strcmp(temp{jj,1},GEeqnNames{ii}) % Names match
                options.GEnewprice3.howtoupdate{ii,1}=temp{jj,1};
                options.GEnewprice3.howtoupdate{ii,2}=temp{jj,2};
                options.GEnewprice3.howtoupdate{ii,3}=temp{jj,3};
                options.GEnewprice3.howtoupdate{ii,4}=temp{jj,4};
            end
        end
    end

    options.GEnewprice3.add=[options.GEnewprice3.howtoupdate{:,3}];
    options.GEnewprice3.factor=[options.GEnewprice3.howtoupdate{:,4}];
    options.GEnewprice3.keepold=ones(size(options.GEnewprice3.factor));
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
    if size(options.GEnewprice3.howtoupdate,1)==nGeneralEqmEqns % Note: inputs were already testted
        % do nothing, this is how things should be
    else
        error('options.GEnewprice3.howtoupdate does not fit with GeneralEqmEqns (number of rows is different to the number of GeneralEqmEqns fields) \n')
    end
    options.GEnewprice3.permute=zeros(size(options.GEnewprice3.howtoupdate,1),1);
    for ii=1:size(options.GEnewprice3.howtoupdate,1) % number of rows is the number of prices (and number of GE conditions)
        for jj=1:length(GEPriceParamNames)
            if strcmp(options.GEnewprice3.howtoupdate{ii,2},GEPriceParamNames{jj})
                options.GEnewprice3.permute(ii)=jj;
            end
        end
    end

else
    %% Model with permanent type: allow for options.GEptype
    nGeneralEqmEqns_acrossptypes=sum(options.GEptype==0)+N_i*sum(options.GEptype==1);
    
    % Before starting, make sure that GE that depend on ptype match up with PricePaths that depend on ptype
    for gg=1:nGeneralEqmEqns
        if options.GEptype(gg)==1
            for gg2=1:size(options.GEnewprice3.howtoupdate,1)
                if strcmp(options.GEnewprice3.howtoupdate{gg2,1},GEPriceParamNames{gg})
                    pricename_gg=options.GEnewprice3.howtoupdate{gg2,2};
                end
            end
            for pp=1:length(GEPriceParamNames)
                if strcmp(pricename_gg,GEPriceParamNames{pp})
                    if PricePathSizeVec(2,pp)-PricePathSizeVec(1,pp)+1~=N_i
                        fprintf('Following error relates to GE condition %s and to price %s \n',GEeqnNames{gg},GEPriceParamNames{pp})
                        error('You declared a GE condition to depend on permenent type, but the price that relates to it (in options.GEnewprice3.howtoupdate) does not depend on ptype')
                    end
                end
            end
        end
    end
    
    % Need to make sure that order of rows in options.GEnewprice3.howtoupdate
    % Is same as order of fields in GeneralEqmEqns
    % I do this by just reordering rows of options.GEnewprice3.howtoupdate
    temp=options.GEnewprice3.howtoupdate;
    % GEeqnNames=fieldnames(GeneralEqmEqns);
    gg_index=zeros(1,nGeneralEqmEqns+(N_i-1)*sum(options.GEptype==1));
    gg_c=0;
    for gg=1:nGeneralEqmEqns
        for jj=1:size(temp,1)
            if strcmp(temp{jj,1},GEeqnNames{gg}) % Names match
                for ii=1:(1+options.GEptype(gg)*(N_i-1)) % Note: 1 or N_i, depending on options.GEptype(gg)
                    gg_c=gg_c+1;
                    options.GEnewprice3.howtoupdate{gg_c,1}=temp{jj,1};
                    options.GEnewprice3.howtoupdate{gg_c,2}=temp{jj,2};
                    options.GEnewprice3.howtoupdate{gg_c,3}=temp{jj,3};
                    options.GEnewprice3.howtoupdate{gg_c,4}=temp{jj,4};
                    gg_index(gg_c)=gg;
                end
            end
        end
    end
    % Note: options.GEnewprice3.howtoupdate will have extra repeated rows whenever options.GEptype(gg)=1

    options.GEnewprice3.add=[options.GEnewprice3.howtoupdate{:,3}];
    options.GEnewprice3.factor=[options.GEnewprice3.howtoupdate{:,4}];
    options.GEnewprice3.keepold=ones(size(options.GEnewprice3.factor));
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
    if size(options.GEnewprice3.howtoupdate,1)==nGeneralEqmEqns_acrossptypes
        % do nothing, this is how things should be
    else
        error('options.GEnewprice3.howtoupdate does not fit with GeneralEqmEqns (different number of conditions) \n')
    end
    options.GEnewprice3.permute=zeros(size(options.GEnewprice3.howtoupdate,1),1);

    gg_c=1;
    for gg=1:nGeneralEqmEqns % number of GE conditions
        if options.GEptype(gg)==0
            for pp=1:length(GEPriceParamNames)
                if strcmp(options.GEnewprice3.howtoupdate{gg_c,2},GEPriceParamNames{pp}) % Take advantage that options.GEnewprice3.howtoupdate has been set to same order as GEeqnNames
                    options.GEnewprice3.permute(PricePathSizeVec(1,pp))=gg_c;
                end
            end
            gg_c=gg_c+1;
        elseif options.GEptype(gg)==1
            for pp=1:length(GEPriceParamNames)
                if strcmp(options.GEnewprice3.howtoupdate{gg_c,2},GEPriceParamNames{pp}) % Take advantage that options.GEnewprice3.howtoupdate has been set to same order as GEeqnNames
                    for ii=PricePathSizeVec(1,pp):PricePathSizeVec(2,pp)
                        options.GEnewprice3.permute(PricePathSizeVec(1,pp):PricePathSizeVec(2,pp))=gg_c+(0:1:N_i-1);
                    end
                end
            end
            gg_c=gg_c+N_i;
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