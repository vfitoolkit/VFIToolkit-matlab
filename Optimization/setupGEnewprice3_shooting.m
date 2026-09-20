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
shootingfield=''; % the field holding howtoupdate when it is not GEnewprice3: 'fminalgo5' or 'fminalgo9' (stationary eqm), or 'GEnewprice2' (transition path)
if isfield(options,'fminalgo') && options.fminalgo==5 % fminalgo5 is stationary shooting
    shootingfield='fminalgo5';
    options.GEnewprice3=options.fminalgo5;
elseif isfield(options,'fminalgo') && options.fminalgo==9 % fminalgo9 (Anderson accel.) accelerates the same shooting map
    shootingfield='fminalgo9';
    options.GEnewprice3=options.fminalgo9;
elseif isfield(options,'GEnewprice') && options.GEnewprice==2 && isfield(options,'GEnewprice2') % transition path Anderson accel., again accelerating the same shooting map. The isfield() is so that the families where GEnewprice=2 is not implemented yet still fall through to the return below, and give their own 'not implemented' message rather than one about a missing field
    shootingfield='GEnewprice2';
    options.GEnewprice3=options.GEnewprice2;
elseif options.GEnewprice~=3
    return % Not being used
end

oldpathweightexisted=isfield(options,'oldpathweight'); % so that the clean up at the end only removes it if it was put there by the line below
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

% Additional factors: options.GEnewprice3.additionalfactor=[f_add,t1_add,t2_add] ramps the update
% factors over the shooting iterations. The factor in howtoupdate column 4 is used as it is on
% iterations 1 to t1_add, then goes linearly from that to f_add times it over iterations t1_add to
% t2_add, and stays at f_add times it from t2_add on. Given as a single [f_add,t1_add,t2_add] row
% used for every price, or with one row for each row of howtoupdate.
% Rolled into howtoupdate as columns 5, 6 and 7 here, while the rows are still in the order the user
% gave them, so that the reordering (and, for ptype, the row expansion) below carries them along
% with the factor they belong to.
if ~isfield(options.GEnewprice3,'additionalfactor')
    options.GEnewprice3.additionalfactor=[1,1,2]; % the no-op: f_add=1 leaves the factor alone
end
if size(options.GEnewprice3.additionalfactor,2)~=3
    error('options.GEnewprice3.additionalfactor should have three columns: [f_add,t1_add,t2_add]')
end
if size(options.GEnewprice3.additionalfactor,1)==1
    options.GEnewprice3.additionalfactor=repmat(options.GEnewprice3.additionalfactor,nGeneralEqmEqns,1);
end
if size(options.GEnewprice3.additionalfactor,1)~=nGeneralEqmEqns
    error('options.GEnewprice3.additionalfactor should have one row, or one row for each general eqm eqn')
end
if any(~isfinite(options.GEnewprice3.additionalfactor),'all')
    error('options.GEnewprice3.additionalfactor must be finite')
end
if any(options.GEnewprice3.additionalfactor(:,2)<1) || any(mod(options.GEnewprice3.additionalfactor(:,2),1)~=0)
    error('options.GEnewprice3.additionalfactor: t1_add (the second column) must be an integer of at least 1, as the shooting iterations are numbered from 1')
end
if any(options.GEnewprice3.additionalfactor(:,3)<=options.GEnewprice3.additionalfactor(:,2)) || any(mod(options.GEnewprice3.additionalfactor(:,3),1)~=0)
    error('options.GEnewprice3.additionalfactor: t2_add (the third column) must be an integer strictly greater than t1_add, as the ramp runs over iterations t1_add to t2_add')
end
% Anderson acceleration accelerates a FIXED point map p<-G(p), and a factor that changes from one
% iteration to the next makes G time-varying, so the acceleration would no longer be solving the
% problem it assumes. f_add=1 is the identity, so only an actual ramp is rejected here.
if (strcmp(shootingfield,'fminalgo9') || strcmp(shootingfield,'GEnewprice2')) && any(options.GEnewprice3.additionalfactor(:,1)~=1)
    error('%s.additionalfactor: the additional-factor ramp cannot be used with Anderson acceleration, as Anderson builds its step from the history of iterates and so needs the same fixed-point map at every iteration',shootingfield)
end
options.GEnewprice3.howtoupdate(:,5)=num2cell(options.GEnewprice3.additionalfactor(:,1)); % f_add
options.GEnewprice3.howtoupdate(:,6)=num2cell(options.GEnewprice3.additionalfactor(:,2)); % t1_add
options.GEnewprice3.howtoupdate(:,7)=num2cell(options.GEnewprice3.additionalfactor(:,3)); % t2_add

%%
if ~isfield(options,'GEptype') % For models without permanent type

    % Need to make sure that order of rows in options.GEnewprice3.howtoupdate
    % Is same as order of PriceParamNames
    % I do this by just reordering rows of options.GEnewprice3.howtoupdate
    temp=options.GEnewprice3.howtoupdate;
    rowfound=zeros(length(PriceParamNames),1);
    for pp=1:length(PriceParamNames)
        for jj=1:size(temp,1)
            if strcmp(temp{jj,2},PriceParamNames{pp}) % Names match: column 2 of howtoupdate is the price name, as in the GEptype branch below
                options.GEnewprice3.howtoupdate{pp,1}=temp{jj,1}; % general eqm eqn name
                options.GEnewprice3.howtoupdate{pp,2}=temp{jj,2}; % general eqm price name
                options.GEnewprice3.howtoupdate{pp,3}=temp{jj,3}; % add(/subtract)
                options.GEnewprice3.howtoupdate{pp,4}=temp{jj,4}; % factor
                options.GEnewprice3.howtoupdate{pp,5}=temp{jj,5}; % f_add
                options.GEnewprice3.howtoupdate{pp,6}=temp{jj,6}; % t1_add
                options.GEnewprice3.howtoupdate{pp,7}=temp{jj,7}; % t2_add
                rowfound(pp)=1;
            end
        end
    end
    if any(rowfound==0)
        error('options.GEnewprice3.howtoupdate: no row gives an update rule for the price %s',PriceParamNames{find(rowfound==0,1)})
    end

    options.GEnewprice3.add=[options.GEnewprice3.howtoupdate{:,3}];
    options.GEnewprice3.factor=[options.GEnewprice3.howtoupdate{:,4}];
    options.GEnewprice3.f_add=[options.GEnewprice3.howtoupdate{:,5}];
    options.GEnewprice3.t1_add=[options.GEnewprice3.howtoupdate{:,6}];
    options.GEnewprice3.t2_add=[options.GEnewprice3.howtoupdate{:,7}];
    options.GEnewprice3.keepold=ones(size(options.GEnewprice3.factor));
    tempweight=options.oldpathweight;
    options.oldpathweight=zeros(size(options.GEnewprice3.factor));
    for ii=1:length(options.GEnewprice3.factor)
        if options.GEnewprice3.factor(ii)==Inf
            if options.GEnewprice3.f_add(ii)~=1
                error('options.GEnewprice3.howtoupdate row %i: factor (column 4) is Inf, so the price is replaced outright rather than stepped toward, and there is no step for f_add (column 5) to scale. Leave f_add at 1 on this row.',ii)
            end
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
                    options.GEnewprice3.howtoupdate{pp_c,5}=temp{jj,5};
                    options.GEnewprice3.howtoupdate{pp_c,6}=temp{jj,6};
                    options.GEnewprice3.howtoupdate{pp_c,7}=temp{jj,7};
                    pp_index(pp_c)=pp;
                end
            end
        end
    end
    % Note: options.GEnewprice3.howtoupdate will have extra repeated rows whenever options.GEptype(gg)=1
    
    options.GEnewprice3.add=[options.GEnewprice3.howtoupdate{:,3}];
    options.GEnewprice3.factor=[options.GEnewprice3.howtoupdate{:,4}];
    options.GEnewprice3.f_add=[options.GEnewprice3.howtoupdate{:,5}];
    options.GEnewprice3.t1_add=[options.GEnewprice3.howtoupdate{:,6}];
    options.GEnewprice3.t2_add=[options.GEnewprice3.howtoupdate{:,7}];
    options.GEnewprice3.keepold=ones(size(options.GEnewprice3.factor));
    tempweight=options.oldpathweight;
    options.oldpathweight=zeros(size(options.GEnewprice3.factor));
    for ii=1:length(options.GEnewprice3.factor)
        if options.GEnewprice3.factor(ii)==Inf
            if options.GEnewprice3.f_add(ii)~=1
                error('options.GEnewprice3.howtoupdate row %i: factor (column 4) is Inf, so the price is replaced outright rather than stepped toward, and there is no step for f_add (column 5) to scale. Leave f_add at 1 on this row.',ii)
            end
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


%% If howtoupdate came from somewhere other than GEnewprice3, write the output back into that field
% (fminalgo5 or fminalgo9 for stationary general eqm, GEnewprice2 for the transition path)
if ~isempty(shootingfield)
    options.(shootingfield)=options.GEnewprice3;
    options=rmfield(options,'GEnewprice3');
    if oldpathweightexisted==0
        options=rmfield(options,'oldpathweight'); % it is not used for anything, and for stationary general eqm it has no business being in heteroagentoptions
    end
end



end