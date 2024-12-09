function [p_eqm,p_eqm_index,GeneralEqmConditions]=HeteroAgentStationaryEqm_Case1_FHorz(jequaloneDist,AgeWeightParamNames, n_d, n_a, n_z, N_j, n_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions)
% If n_p=0 then will use fminsearch to find the general equilibrium (find
% price vector that corresponds to GeneralEqmConditions=0). By setting n_p to
% nonzero it is assumed you want to use a grid on prices, which must then
% be passed in using heteroagentoptions.p_grid

% N_d=prod(n_d);
% N_a=prod(n_a);
% N_z=prod(n_z);
N_p=prod(n_p);
if isempty(n_p)
    N_p=0;
end

l_p=length(GEPriceParamNames); % Otherwise get problem when not using p_grid
%l_p=length(n_p);

p_eqm_vec=nan; p_eqm_index=nan; GeneralEqmConditions=nan;

%% Check 'double fminalgo'
if exist('heteroagentoptions','var')
    if isfield(heteroagentoptions,'fminalgo')
        if length(heteroagentoptions.fminalgo)>1
            temp=heteroagentoptions.fminalgo;
            heteroagentoptions.fminalgo=heteroagentoptions.fminalgo(1);
            p_eqm_previous=HeteroAgentStationaryEqm_Case1_FHorz(jequaloneDist,AgeWeightParamNames, n_d, n_a, n_z, N_j, n_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions);
            for pp=1:length(GEPriceParamNames)
                Parameters.(GEPriceParamNames{pp})=p_eqm_previous.(GEPriceParamNames{pp});
            end
            heteroagentoptions.fminalgo=temp(2:end);
        end
    end
end

%% Check which options have been used, set all others to defaults 
if exist('vfoptions','var')==0
    vfoptions.parallel=1+(gpuDeviceCount>0);
    %If vfoptions is not given, just use all the defaults
    %Note that the defaults will be set when we call 'ValueFnIter...'
    %commands and the like, so no need to set them here except for a few.
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if isfield(vfoptions,'parallel')==0
        vfoptions.parallel=1+(gpuDeviceCount>0);
    end
end

if ~exist('simoptions','var')
    simoptions=struct(); % create a 'placeholder' simoptions that can be passed to subcodes
end

if ~exist('heteroagentoptions','var')
    heteroagentoptions.multiGEcriterion=1;
    heteroagentoptions.multiGEweights=ones(1,length(fieldnames(GeneralEqmEqns)));
    heteroagentoptions.toleranceGEprices=10^(-4); % Accuracy of general eqm prices
    heteroagentoptions.toleranceGEcondns=10^(-4); % Accuracy of general eqm eqns
    heteroagentoptions.fminalgo=1; % use fminsearch
    heteroagentoptions.verbose=0;
    heteroagentoptions.maxiter=1000;
    % heteroagentoptions.outputGEform=0; % For internal use only
    heteroagentoptions.outputGEstruct=1; % output GE conditions as a structure (=2 will output as a vector)
    heteroagentoptions.outputgather=1; % output GE conditions on CPU [some optimization routines only work on CPU, some can handle GPU]
else
    if ~isfield(heteroagentoptions,'multiGEcriterion')
        heteroagentoptions.multiGEcriterion=1;
    end
    if ~isfield(heteroagentoptions,'multiGEweights')
        heteroagentoptions.multiGEweights=ones(1,length(GeneralEqmEqns));
    end
    if N_p~=0
        if ~isfield(heteroagentoptions,'pgrid')
            error('You have set n_p to a non-zero value, but not declared heteroagentoptions.pgrid')
        end
    end
    if ~isfield(heteroagentoptions,'toleranceGEprices')
        heteroagentoptions.toleranceGEprices=10^(-4); % Accuracy of general eqm prices
    end
    if ~isfield(heteroagentoptions,'toleranceGEcondns')
        heteroagentoptions.toleranceGEcondns=10^(-4); % Accuracy of general eqm prices
    end
    if ~isfield(heteroagentoptions,'verbose')
        heteroagentoptions.verbose=0;
    end
    if ~isfield(heteroagentoptions,'fminalgo')
        heteroagentoptions.fminalgo=1; % use fminsearch
    end
    if ~isfield(heteroagentoptions,'maxiter')
        heteroagentoptions.maxiter=1000; % use fminsearch
    end
    % heteroagentoptions.outputGEform=0; % For internal use only
    if ~isfield(heteroagentoptions,'outputGEstruct')
        heteroagentoptions.outputGEstruct=1; % output GE conditions as a structure (=2 will output as a vector)
    end
    if ~isfield(heteroagentoptions,'outputgather')
        heteroagentoptions.outputgather=1; % output GE conditions on CPU [some optimization routines only work on CPU, some can handle GPU]
    end
end

if heteroagentoptions.fminalgo==0
    heteroagentoptions.outputGEform=1;
elseif heteroagentoptions.fminalgo==5
    if isfield(heteroagentoptions,'toleranceGEprices_percent')==0
        heteroagentoptions.toleranceGEprices_percent=10^(-3); % one-tenth of one percent
    end
    heteroagentoptions.outputGEform=1; % Need to output GE condns as a vector when using fminalgo=5
    heteroagentoptions.outputgather=0; % leave GE condns vector on GPU
elseif heteroagentoptions.fminalgo==7
    heteroagentoptions.outputGEform=1; % Need to output GE condns as a vector when using fminalgo=7
else
    heteroagentoptions.outputGEform=0;
end


%% Change to FnsToEvaluate as cell so that it is not being recomputed all the time
if isstruct(FnsToEvaluate)
    if n_d(1)==0
        l_d=0;
    else
        l_d=length(n_d);
    end
    if n_z(1)==0
        l_z=0;
    else
        l_z=length(n_z);
    end
    if isfield(simoptions,'n_e')
        if simoptions.n_e(1)==0
            l_e=0;
        else
            l_e=length(simoptions.n_e);
        end
    else
        l_e=0;
    end
    l_a=length(n_a);
    
    AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        if length(temp)>(l_d+l_a+l_a+l_z+l_e)
            FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+l_z+l_e+1:end}}; % the first inputs will always be (d,aprime,a)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(AggVarNames{ff});
    end
    FnsToEvaluate=FnsToEvaluate2;
    % Change FnsToEvaluate out of structure form, but want to still create AggVars as a structure
    simoptions.outputasstructure=1;
    simoptions.AggVarNames=AggVarNames;
else
    % Do nothing
end

%%
if N_p~=0
    [p_eqm_vec,p_eqm_index,GeneralEqmConditions]=HeteroAgentStationaryEqm_Case1_FHorz_pgrid(jequaloneDist,AgeWeightParamNames, n_d, n_a, n_z, N_j, n_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, heteroagentoptions, simoptions, vfoptions);
    return
end

%% If using fminalgo=5, then need some further setup

if heteroagentoptions.fminalgo==5
    heteroagentoptions.weightscheme=0; % Don't do any weightscheme, is already taken care of by GEnewprice=3
    
    if isstruct(GeneralEqmEqns) 
        % Need to make sure that order of rows in transpathoptions.GEnewprice3.howtoupdate
        % Is same as order of fields in GeneralEqmEqns
        % I do this by just reordering rows of transpathoptions.GEnewprice3.howtoupdate
        temp=heteroagentoptions.fminalgo5.howtoupdate;
        GEeqnNames=fieldnames(GeneralEqmEqns);
        for tt=1:length(GEeqnNames)
            for jj=1:size(temp,1)
                if strcmp(temp{jj,1},GEeqnNames{tt}) % Names match
                    heteroagentoptions.fminalgo5.howtoupdate{tt,1}=temp{jj,1};
                    heteroagentoptions.fminalgo5.howtoupdate{tt,2}=temp{jj,2};
                    heteroagentoptions.fminalgo5.howtoupdate{tt,3}=temp{jj,3};
                    heteroagentoptions.fminalgo5.howtoupdate{tt,4}=temp{jj,4};
                end
            end
        end
        nGeneralEqmEqns=length(GEeqnNames);
    else
        nGeneralEqmEqns=length(GeneralEqmEqns);
    end
    heteroagentoptions.fminalgo5.add=[heteroagentoptions.fminalgo5.howtoupdate{:,3}];
    heteroagentoptions.fminalgo5.factor=[heteroagentoptions.fminalgo5.howtoupdate{:,4}];
    heteroagentoptions.fminalgo5.keepold=ones(size(heteroagentoptions.fminalgo5.factor));
    
    if size(heteroagentoptions.fminalgo5.howtoupdate,1)==nGeneralEqmEqns && nGeneralEqmEqns==length(GEPriceParamNames)
        % do nothing, this is how things should be
    else
        fprintf('ERROR: heteroagentoptions.fminalgo5..howtoupdate does not fit with GeneralEqmEqns (different number of conditions/prices) \n')
    end
    heteroagentoptions.fminalgo5.permute=zeros(size(heteroagentoptions.fminalgo5.howtoupdate,1),1);
    for tt=1:size(heteroagentoptions.fminalgo5.howtoupdate,1) % number of rows is the number of prices (and number of GE conditions)
        for jj=1:length(GEPriceParamNames)
            if strcmp(heteroagentoptions.fminalgo5.howtoupdate{tt,2},GEPriceParamNames{jj})
                heteroagentoptions.fminalgo5.permute(tt)=jj;
            end
        end
    end
    if isfield(heteroagentoptions,'updateaccuracycutoff')==0
        heteroagentoptions.updateaccuracycutoff=0; % No cut-off (only changes in the price larger in magnitude that this will be made (can be set to, e.g., 10^(-6) to help avoid changes at overly high precision))
    end
end


%%
if heteroagentoptions.maxiter>0 % Can use heteroagentoptions.maxiter=0 to just evaluate the current general eqm eqns
    
    %% Otherwise, use fminsearch to find the general equilibrium
    if heteroagentoptions.fminalgo~=3 && heteroagentoptions.fminalgo~=8
        GeneralEqmConditionsFnOpt=@(p) HeteroAgentStationaryEqm_Case1_FHorz_subfn(p, jequaloneDist,AgeWeightParamNames, n_d, n_a, n_z, N_j, l_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, heteroagentoptions, simoptions, vfoptions);
    elseif heteroagentoptions.fminalgo==3
        heteroagentoptions.outputGEform=1; % vector
        GeneralEqmConditionsFnOpt=@(p) HeteroAgentStationaryEqm_Case1_FHorz_subfn(p, jequaloneDist,AgeWeightParamNames, n_d, n_a, n_z, N_j, l_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, heteroagentoptions, simoptions, vfoptions);
    elseif heteroagentoptions.fminalgo==8
        heteroagentoptions.outputGEform=1; % vector
        weightsbackup=heteroagentoptions.multiGEweights;
        heteroagentoptions.multiGEweights=sqrt(heteroagentoptions.multiGEweights); % To use a weighting matrix in lsqnonlin(), we work with the square-roots of the weights
        GeneralEqmConditionsFnOpt=@(p) HeteroAgentStationaryEqm_Case1_FHorz_subfn(p, jequaloneDist,AgeWeightParamNames, n_d, n_a, n_z, N_j, l_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, heteroagentoptions, simoptions, vfoptions);
        heteroagentoptions.multiGEweights=weightsbackup; % change it back now that we have set up CalibrateLifeCycleModel_objectivefn()
    end
    
    p0=nan(length(GEPriceParamNames),1);
    for ii=1:length(GEPriceParamNames)
        p0(ii)=Parameters.(GEPriceParamNames{ii});
    end

    % Choosing algorithm for the optimization problem
    % https://au.mathworks.com/help/optim/ug/choosing-the-algorithm.html#bscj42s
    minoptions = optimset('TolX',heteroagentoptions.toleranceGEprices,'TolFun',heteroagentoptions.toleranceGEcondns,'MaxFunEvals',heteroagentoptions.maxiter,'MaxIter',heteroagentoptions.maxiter);
    if heteroagentoptions.fminalgo==0 % fzero, is based on root-finding so it needs just the vector of GEcondns, not the sum-of-squares (it is not a minimization routine)
        [p_eqm_vec,GeneralEqmConditions]=fzero(GeneralEqmConditionsFnOpt,p0,minoptions);
    elseif heteroagentoptions.fminalgo==1
        [p_eqm_vec,GeneralEqmConditions]=fminsearch(GeneralEqmConditionsFnOpt,p0,minoptions);
    elseif heteroagentoptions.fminalgo==2
        % Use the optimization toolbox so as to take advantage of automatic differentiation
        z=optimvar('z',length(p0));
        optimfun=fcn2optimexpr(GeneralEqmConditionsFnOpt, z);
        prob = optimproblem("Objective",optimfun);
        z0.z=p0;
        [sol,GeneralEqmConditions]=solve(prob,z0);
        p_eqm_vec=sol.z;
        % Note, doesn't really work as automatic differentiation is only for
        % supported functions, and the objective here is not a supported function
    elseif heteroagentoptions.fminalgo==3
        goal=zeros(length(p0),1);
        weight=ones(length(p0),1); % I already implement weights via heteroagentoptions
        [p_eqm_vec,GeneralEqmConditionsVec] = fgoalattain(GeneralEqmConditionsFnOpt,p0,goal,weight);
        GeneralEqmConditions=sum(abs(GeneralEqmConditionsVec));
    elseif heteroagentoptions.fminalgo==4 % CMA-ES algorithm (Covariance-Matrix adaptation - Evolutionary Stategy)
        % https://en.wikipedia.org/wiki/CMA-ES
        % https://cma-es.github.io/
        % Code is cmaes.m from: https://cma-es.github.io/cmaes_sourcecode_page.html#matlab
        if ~isfield(heteroagentoptions,'insigma')
            % insigma: initial coordinate wise standard deviation(s)
            heteroagentoptions.insigma=0.3*abs(p0)+0.1*(p0==0); % Set standard deviation to 30% of the initial parameter value itself (cannot input zero, so add 0.1 to any zeros)
        end
        if ~isfield(heteroagentoptions,'inopts')
            % inopts: options struct, see defopts below
            heteroagentoptions.inopts=[];
        end
        if isfield(heteroagentoptions,'toleranceGEcondns')
            heteroagentoptions.inopts.StopFitness=heteroagentoptions.toleranceGEcondns;
        end
        % varargin (unused): arguments passed to objective function
        if heteroagentoptions.verbose==1
            disp('VFI Toolkit is using the CMA-ES algorithm, consider giving a cite to: Hansen, N. and S. Kern (2004). Evaluating the CMA Evolution Strategy on Multimodal Test Functions' )
        end
    	% This is a minor edit of cmaes, because I want to use 'GeneralEqmConditionsFnOpt' as a function_handle, but the original cmaes code only allows for 'GeneralEqmConditionsFnOpt' as a string
        [p_eqm_vec,GeneralEqmConditions,counteval,stopflag,out,bestever] = cmaes_vfitoolkit(GeneralEqmConditionsFnOpt,p0,heteroagentoptions.insigma,heteroagentoptions.inopts); % ,varargin);
    elseif heteroagentoptions.fminalgo==5
        % Update based on rules in heteroagentoptions.fminalgo5.howtoupdate
        GeneralEqmEqnsNames=fieldnames(GeneralEqmEqns);
        GeneralEqmConditions=Inf;
        % Get initial prices, p
        p=nan(1,length(GEPriceParamNames));
        for ii=1:length(GEPriceParamNames)
            p(ii)=Parameters.(GEPriceParamNames{ii});
        end
        % Given current prices solve the model to get the general equilibrium conditions as a structure
        p_percentchange=Inf;
        while any(p_percentchange>heteroagentoptions.toleranceGEprices_percent) % GeneralEqmConditions>heteroagentoptions.toleranceGEcondns

            p_i=GeneralEqmConditionsFnOpt(p); % using heteroagentoptions.outputGEform=1, so this is a vector (note the transpose)

            GeneralEqmConditionsVec=p_i; % Need later to look at convergence

            % Update prices based on GEstruct following the howtoupdate rules
            p_i=p_i(heteroagentoptions.fminalgo5.permute); % Rearrange GeneralEqmEqns into the order of the relevant prices
            I_makescutoff=(abs(p_i)>heteroagentoptions.updateaccuracycutoff);
            p_i=I_makescutoff.*p_i;

            p_new=(p.*heteroagentoptions.fminalgo5.keepold)+heteroagentoptions.fminalgo5.add.*heteroagentoptions.fminalgo5.factor.*p_i-(1-heteroagentoptions.fminalgo5.add).*heteroagentoptions.fminalgo5.factor.*p_i;

            % Calculate GeneralEqmConditions which measures convergence
            if heteroagentoptions.multiGEcriterion==0
                GeneralEqmConditions=sum(abs(heteroagentoptions.multiGEweights.*GeneralEqmConditionsVec));
            elseif heteroagentoptions.multiGEcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market
                GeneralEqmConditions=sqrt(sum(heteroagentoptions.multiGEweights.*(GeneralEqmConditionsVec.^2)));
            end

            % Put new prices into Parameters
            for ii=1:length(GEPriceParamNames)
                Parameters.(GEPriceParamNames{ii})=p_new(ii);
            end

            % fprintf('Current iteration \n')
            % p_percentchange
            % p_new
            % p
            % p_i

            p_percentchange=max(abs(p_new-p)./abs(p));
            p_percentchange(p==0)=abs(p_new(p==0)); %-p(p==0)); but this is just zero anyway
            % Update p for next iteration
            p=p_new;
        end
        p_eqm_vec=p_new; % Need to put it in p_eqm_vec so that it can be used to create the final output
    elseif heteroagentoptions.fminalgo==6
        if ~isfield(heteroagentoptions,'lb') || ~isfield(heteroagentoptions,'ub')
            error('When using constrained optimization (heteroagentoptions.fminalgo=6) you must set the lower and upper bounds of the GE price parameters using heteroagentoptions.lb and heteroagentoptions.ub')
        end
        [p_eqm_vec,GeneralEqmConditions]=fmincon(GeneralEqmConditionsFnOpt,p0,[],[],[],[],heteroagentoptions.lb,heteroagentoptions.ub,[],minoptions);
    elseif heteroagentoptions.fminalgo==7 % Matlab fsolve()
        heteroagentoptions.multiGEcriterion=0;
        [p_eqm_vec,GeneralEqmConditions]=fsolve(GeneralEqmConditionsFnOpt,p0,minoptions);
    elseif heteroagentoptions.fminalgo==8 % Matlab lsqnonlin()
        minoptions = optimoptions('lsqnonlin','FiniteDifferenceStepSize',1e-2,'TolX',heteroagentoptions.toleranceGEprices,'TolFun',heteroagentoptions.toleranceGEcondns,'MaxFunEvals',heteroagentoptions.maxiter,'MaxIter',heteroagentoptions.maxiter);
        [p_eqm_vec,GeneralEqmConditions]=lsqnonlin(GeneralEqmConditionsFnOpt,p0,[],[],[],[],[],[],[],minoptions);
    end


    p_eqm_index=nan; % If not using p_grid then this is irrelevant/useless

    for ii=1:length(GEPriceParamNames)
        p_eqm.(GEPriceParamNames{ii})=p_eqm_vec(ii);
    end

    % vargout=[p_eqm,p_eqm_index,GeneralEqmConditions];
    % if heteroagentoptions.fminalgo==3
    %     vargout=[p_eqm,GeneralEqmConditions,counteval,stopflag,out,bestever];
    % end
 
%%
elseif heteroagentoptions.maxiter==0 % Can use heteroagentoptions.maxiter=0 to just evaluate the current general eqm eqns
    % Just use the prices that are currently in Params
    p_eqm=nan; % So user cannot misuse
    for ii=1:length(GEPriceParamNames)
        p_eqm_vec(ii)=Parameters.(GEPriceParamNames{ii});
    end
end

%% Add an evaluation of the general eqm eqns so that these can be output as a vectors/structure rather than just the sum of squares
if heteroagentoptions.outputGEstruct==1
    heteroagentoptions.outputGEform=2; % output as struct
elseif heteroagentoptions.outputGEstruct==2
    heteroagentoptions.outputGEform=1; % output as vector
end

if heteroagentoptions.outputGEstruct==1 || heteroagentoptions.outputGEstruct==2
    % Run once more to get the general eqm eqns in a nice form for output
    GeneralEqmConditionsFnOpt=@(p) HeteroAgentStationaryEqm_Case1_FHorz_subfn(p, jequaloneDist,AgeWeightParamNames, n_d, n_a, n_z, N_j, l_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, heteroagentoptions, simoptions, vfoptions); % update based on new heteragentoptions
    GeneralEqmConditions=GeneralEqmConditionsFnOpt(p_eqm_vec);
end





end