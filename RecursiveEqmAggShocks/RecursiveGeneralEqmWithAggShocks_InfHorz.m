function GeneralizedTransitionFn=RecursiveGeneralEqmWithAggShocks_InfHorz(T,n_d,n_a,n_z,n_S,d_grid,a_grid,z_grid,S_grid,pi_z,pi_S,ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, GEPriceParamNames, AggShockNames, recursiveeqmoptions,vfoptions,simoptions, heteroagentoptions)
% Solve aggregate shock models using the matched-expecations path algorithm of Hanbaek Lee

if ~isfield(recursiveeqmoptions,'divideT')
    recursiveeqmoptions.divideT=1; % Number of pieces to divide T into while solving value fn
end
if ~isfield(recursiveeqmoptions,'tolerance')
    recursiveeqmoptions.tolerance=10^(-4); % Accuracy of general eqm eqns
end
if ~isfield(recursiveeqmoptions,'verbose')
    recursiveeqmoptions.verbose=1;
end
if ~isfield(recursiveeqmoptions,'maxiter')
    recursiveeqmoptions.maxiter=1000; % maximum iterations of optimization routine
end
if ~isfield(recursiveeqmoptions,'burnin')
    recursiveeqmoptions.burnin=100; % burnin on the aggregate shock
end
if ~isfield(recursiveeqmoptions,'graphpricepath')
    recursiveeqmoptions.graphpricepath=0;
end
if ~isfield(recursiveeqmoptions,'graphaggvarspath')
    recursiveeqmoptions.graphaggvarspath=0;
end
if ~isfield(recursiveeqmoptions,'graphGEcondns')
    recursiveeqmoptions.graphGEcondns=0;
end
if ~isfield(recursiveeqmoptions,'historyofpricepath')
    recursiveeqmoptions.historyofpricepath=0;
end
% Note: recursiveeqmoptions.heteroagentoptions can be used to set heteroagentoptions for the initial guess

% vfoptions and simoptions are required inputs
if ~isfield(vfoptions,'gridinterplayer')
    vfoptions.gridinterplayer=0; 
end
if ~isfield(vfoptions,'divideandconquer')
    vfoptions.divideandconquer=1;
end
if vfoptions.divideandconquer==1
    if ~isfield(vfoptions,'level1n')
        vfoptions.level1n=ceil(sqrt(n_a));
    end
end
if ~isfield(vfoptions,'exoticpreferences')
    vfoptions.exoticpreferences='None';
end
if ~isfield(vfoptions,'lowmemory')
    vfoptions.lowmemory=0;
end
vfoptions.EVpre=1;
vfoptions.outputkron=1;
vfoptions.policy_forceintegertype=0;
if ~isfield(simoptions,'gridinterplayer')
    simoptions.gridinterplayer=0; 
end
simoptions.outputkron=1;

GeneralizedTransitionFn=struct();

%%
if recursiveeqmoptions.verbose>=1
    fprintf('VFI Toolkit uses the Matched-Expectations Path of Hanbaek Lee to solve models with Aggregate Shocks, please cite his paper if you use this in your publication: Global Nonlinear Solutions in Sequence Space and the Generalized Transition Function')
end

%% Just treat burnin+T as T, and then remove the burnin while cleaning up at the end
T=recursiveeqmoptions.burnin+T;

l_S=length(n_S);

%%
N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_S=prod(n_S);

% Move things to GPU
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
z_grid=gpuArray(z_grid);
S_grid=gpuArray(S_grid);
pi_z=gpuArray(pi_z);
pi_S=gpuArray(pi_S);

% Switch S_grid to joint-grid (if it not already)
if size(S_grid,2)==1
    S_gridvals=CreateGridvals(n_S,S_grid,1);
else
    S_gridvals=S_grid;
end
clear S_grid % make sure I don't accidently use it later

%% Implement new way of handling ReturnFn inputs
if length(AggShockNames)~=l_S
    error('length(AggShockNames) must be the same as length(n_S). They disagree on the number of aggregate shocks.')
end

for SS_c=1:l_S
    Parameters.(AggShockNames{SS_c})=0; % Just a placeholder while we set up ReturnFnParamNames
end
ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,0,vfoptions,Parameters);
for SS_c=1:l_S
    Parameters=rmfield(Parameters,AggShockNames{SS_c});% To make sure I don't accidently use them
end

%% Change to FnsToEvaluate as cell so that it is not being recomputed all the time
if N_d==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
if N_z==0
    l_z=0;
else
    l_z=length(n_z);
end
l_e=0; % Not yet implemented for InfHorz

l_daprime=l_d+l_a;

AggVarNames=fieldnames(FnsToEvaluate);
FnsToEvaluateCell=cell(1,length(AggVarNames));
for ff=1:length(AggVarNames)
    temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
    if length(temp)>(l_daprime+l_a+l_z+l_e) % Note: S is not counted here, as that is handled via Parameters
        FnsToEvaluateParamNames(ff).Names={temp{l_daprime+l_a+l_z+l_e+1:end}}; % the first inputs will always be (d,aprime,a,z)
    else
        FnsToEvaluateParamNames(ff).Names={};
    end
    FnsToEvaluateCell{ff}=FnsToEvaluate.(AggVarNames{ff});
end
% Change FnsToEvaluate out of structure form, but want to still create AggVars as a structure
simoptions.outputasstructure=1;

%% GE eqns, switch from structure to cell setup
GEeqnNames=fieldnames(GeneralEqmEqns);
nGeneralEqmEqns=length(GEeqnNames);

GeneralEqmEqnsCell=cell(1,nGeneralEqmEqns);
for gg=1:nGeneralEqmEqns
    temp=getAnonymousFnInputNames(GeneralEqmEqns.(GEeqnNames{gg}));
    GeneralEqmEqnParamNames(gg).Names=temp;
    GeneralEqmEqnsCell{gg}=GeneralEqmEqns.(GEeqnNames{gg});
end
% Now: 
%  GeneralEqmEqns is still the structure
%  GeneralEqmEqnsCell is cell
%  GeneralEqmEqnParamNames(ff).Names contains the names


%% Set up the path: 
% setup PricePath as the general eqm prices
% setup ParamPath as the aggregate shock path
% setup pi_z_T as the idiosyncratic transition probs, since they may depend on aggregate shocks
% setup z_gridvals_T as the idiosyncratic shock grids, since they may depend on aggregate shocks

% I do not set up PricePath here, because it will be based on an initial guess made later.
cumsum_pi_S=cumsum(pi_S,2);

% setup ParamPath as the aggregate shock path
AggShocksPathIndexesMatrix=zeros(l_S,T);
ss_ind=1+floor(rand(1,1)*N_S); % pick random initial S
for bb=1:recursiveeqmoptions.burnin
    [~,ss_ind]=max(cumsum_pi_S(ss_ind,:)>rand(1,1));
end
ss_ind_T=zeros(T,1); % one spare at end
pi_Sprime_T=zeros(1,T,1,N_S,'gpuArray');
for tt=1:T
    [~,ss_ind]=max(cumsum_pi_S(ss_ind,:)>rand(1,1));
    
    ss_ind_T(tt)=ss_ind;
    % Set up AggShocksPath
    ss_sub=ind2sub_homemade(n_S,ss_ind);
    for SS_c=1:length(n_S)
        AggShocksPathIndexesMatrix(SS_c,tt)=ss_sub(SS_c);
    end
    % We also need to keep the pi_S row given S, so that it can be used later to create expectations
    pi_Sprime_T(1,tt,1,:)=shiftdim(pi_S(ss_ind,:),-2);
end
% We actually just store the agg shock path in Params in terms of solving
for SS_c=1:length(n_S)
    AggShocksIndexesPath.(AggShockNames{SS_c})=AggShocksPathIndexesMatrix(SS_c,:);
    AggShocksPath.(AggShockNames{SS_c})=S_gridvals(AggShocksPathIndexesMatrix(SS_c,:),SS_c);
    Parameters.(AggShockNames{SS_c})=AggShocksPath.(AggShockNames{SS_c});
end


%% Matching setup
recursiveeqmoptions.matchingsetup=1; % Match based on Sprime and Distance
if ndims(pi_z)==4 % joint transition of z with S
    recursiveeqmoptions.matchingsetup=2; % Match based on (S,Sprime) and Distance
end

if recursiveeqmoptions.matchingsetup==1
    % To be able to speed up the code, we create a record of which t is which S
    % For a given S, the row of SSmask_T indicates all the periods with the same S
    SSmask_T=zeros(N_S,T,'gpuArray');
    SSprimemask_T=zeros(1,T,N_S,'gpuArray'); % For given SS_c and SSprime_c, it will give a vector of length T, with 1s where you get that combo (SS_c,SSprime_c) at tt
    for tt=1:T
        SSmask_T(ss_ind_T(tt),tt)=1;
        if tt<T
            SSprimemask_T(1,tt,ss_ind_T(tt+1))=1;
        end
    end
    SSmask_T=logical(SSmask_T);
    SSprimemask_T=logical(SSprimemask_T); % Note: for last period this is zero by construction (as cannot see periods after)
    % % notSSmask_Tplus1(SSprime_c,:) should have zeros when next-period is SSprime_c, otherwise has 1 (plus we will put 1 in period T and in burnin periods)
    % notSSmask_Tplus1=[(~SSmask_T(:,2:end)),logical(ones(N_S,1,'gpuArray'))]; % next period, 2:end, and then for final period just 'always not' with the ones()

    % LAZY ASS VERSION (I should after a few iterations just throw away the burnin, as it is a lot of things to calculate when you are not using them)
    % Just use Mask to rule out the whole burnin of the value fns
    SSmask_T(:,1:recursiveeqmoptions.burnin)=0; % remove these from consideration
    SSprimemask_T(1,1:recursiveeqmoptions.burnin,:)=0; % remove these from consideration
    % notSSmask_Tplus1(:,1:recursiveeqmoptions.burnin)=1; % remove these from consideration

    SSprimemask_T_indexes=repmat(gpuArray(1:1:T),1,1,N_S); % Have to initially have same size as SSprimemask_T, for the next line
    SSprimemask_T_indexes(~SSprimemask_T)=nan;
    SSprimemask_T_indexes=reshape(SSprimemask_T_indexes,[T,N_S]);
    SSprimemask_T_indexes=sort(SSprimemask_T_indexes,1); % Note: sort() moves all the Nan to the end
    % SSprimemask_T_indexes keeps the time period indexes that correspond to the ones in SSprimemask_T

elseif recursiveeqmoptions.matchingsetup==2
    % To be able to speed up the code, we create a record of which t is which S
    % For a given S, the row of SSmask_T indicates all the periods with the same S
    % We also want to find where there is same S today, with different Sprimes next period.
    SSmask_T=zeros(N_S,T,'gpuArray');
    SSprimemask_T=zeros(1,T,N_S,N_S,'gpuArray'); % For given SS_c and SSprime_c, it will give a vector of length T, with 1s where you get that combo (SS_c,SSprime_c) at tt
    for tt=1:T
        SSmask_T(ss_ind_T(tt),tt)=1;
        if tt<T
            SSprimemask_T(1,tt,ss_ind_T(tt),ss_ind_T(tt+1))=1;
        end
    end
    SSmask_T=logical(SSmask_T);
    SSprimemask_T=logical(SSprimemask_T); % Note: for last period this is zero by construction (as cannot see periods after)
    % % notSSmask_Tplus1(SSprime_c,:) should have zeros when next-period is SSprime_c, otherwise has 1 (plus we will put 1 in period T and in burnin periods)
    % notSSmask_Tplus1=[(~SSmask_T(:,2:end)),logical(ones(N_S,1,'gpuArray'))]; % next period, 2:end, and then for final period just 'always not' with the ones()
    
    % LAZY ASS VERSION (I should after a few iterations just throw away the burnin, as it is a lot of things to calculate when you are not using them)
    % Just use Mask to rule out the whole burnin of the value fns
    SSmask_T(:,1:recursiveeqmoptions.burnin)=0; % remove these from consideration
    SSprimemask_T(1,1:recursiveeqmoptions.burnin,:,:)=0; % remove these from consideration
    % notSSmask_Tplus1(:,1:recursiveeqmoptions.burnin)=1; % remove these from consideration

    SSprimemask_T_indexes=repmat(gpuArray(1:1:T),1,1,N_S,N_S); % Have to initially have same size as SSprimemask_T, for the next line
    SSprimemask_T_indexes(~SSprimemask_T)=nan;
    SSprimemask_T_indexes=reshape(SSprimemask_T_indexes,[T,N_S,N_S]);
    SSprimemask_T_indexes=sort(SSprimemask_T_indexes,1); % Note: sort() moves all the Nan to the end
    % SSprimemask_T_indexes keeps the time period indexes that correspond to the ones in SSprimemask_T
end

if recursiveeqmoptions.verbose==2
    fprintf('Number of instances of each S (multidimensional index) \n')
    sum(SSmask_T,2)
    figure(10)
    plot(-recursiveeqmoptions.burnin+1:1:T-recursiveeqmoptions.burnin,ss_ind_T)
    title('Time series for S index in the Generalized Transition Fn (shaded area is burnin)')
    hold on
    patch([-recursiveeqmoptions.burnin+1 0 0 -recursiveeqmoptions.burnin+1],[1 1 2 2],'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    hold off
    xlim([-recursiveeqmoptions.burnin+1,T-recursiveeqmoptions.burnin])
end

%% Setup path for idiosyncratic shocks
% setup pi_z_T as the idiosyncratic transition probs, since they may depend on aggregate shocks
% Need to do different things depending on if idiosyncratic shocks depend on aggregate shocks
if ndims(pi_z)==2 % Does not depend on S
    pi_z_T=repmat(gpuArray(pi_z),1,1,T);
elseif ndims(pi_z)==3 % Depends on current S
    pi_z_T=zeros(N_z,N_z,T,'gpuArray');
    for tt=1:T
        pi_z_T(:,:,tt)=pi_z(:,:,ss_ind_T(tt));
    end
elseif ndims(pi_z)==4 % joint transition of z with S
    pi_z_T=zeros(N_z,N_z,T,'gpuArray');
    for tt=1:T-1
        pi_z_T(:,:,tt)=pi_z(:,:,ss_ind_T(tt),ss_ind_T(tt+1));
    end
    % T
    [~,ss_ind]=max(pi_S(ss_ind_T(T),:)>rand(1,1)); %t+1
    pi_z_T(:,:,T)=pi_z(:,:,ss_ind_T(T),ss_ind);
end
% pi_z_T is currently (z,z',t) [When using options.fastOLG=1 it will get converted later to (t,z',z)]
% We want pi_z_T_sim to map (t,z)-to-z', specifically [(T-1)*N_z,N_z]
pi_z_T_sim=gather(reshape(permute(pi_z_T(:,:,1:T-1),[3,1,2]),[(T-1)*N_z,N_z]));
% Now extend it to map (t,z)-to-(t+1,z') [omits t=T and t+1=1]
II1=repmat(1:1:(T-1)*N_z,1,N_z); % index for (t,z)
II2=repmat(1:1:(T-1),1,N_z*N_z)+repelem((T-1)*(0:1:N_z-1),1,N_z*(T-1)); % index for (t,z')
pi_z_T_sim=sparse(II1,II2,pi_z_T_sim,(T-1)*N_z,(T-1)*N_z);
% pi_z_T needs to be (t,zprime,z) for fastOLG
pi_z_T_fastOLG=permute(pi_z_T,[3,2,1]); % pi_z_T_fastOLG is [T,N_zprime,N_z]
% setup z_gridvals_T
% NOT REALLY DONE YET, JUST GOING TO ASSUME INPUT WAS A BASIC z_grid WITH NO DEPENDENCE ON S
z_gridvals_T=repmat(shiftdim(CreateGridvals(n_z,gpuArray(z_grid),1),1),T,1,1); % [T,N_z,l_z] for fastOLG
% For the fastOLG evaluation of AggVars we need
z_gridvals_T_fastOLG=shiftdim(z_gridvals_T,-1); % [1,T,N_z,l_z] need this for fastOLG agent dist, but need the standard still for the value fn without fastOLG
% Keep some of this stuff for the output
GeneralizedTransitionFn.OtherStuff.pi_z_T=pi_z_T(:,:,recursiveeqmoptions.burnin+1:end);
GeneralizedTransitionFn.OtherStuff.z_gridvals_T=z_gridvals_T(recursiveeqmoptions.burnin+1:end,:,:);


%% Things for the initial guess

% Use the mean of S based on stationary dist of S as the value of S here
% (which makes sense with the way I treat S as idiosyncratic shock while doing the initial guess)
statdist_S=ones(N_S,1)/N_S;
for ii=1:1e3
    statdist_S=pi_S'*statdist_S;
end
initialguessobjects.Svalue=zeros(length(n_S),1);
for SS_c=1:length(n_S)
    initialguessobjects.Svalue(SS_c)=sum(S_gridvals(:,SS_c).*statdist_S);
end

% optional input
if isfield(recursiveeqmoptions,'heteroagentoptions')
    initialguessobjects.heteroagentoptions=recursiveeqmoptions.heteroagentoptions;
else
    initialguessobjects.heteroagentoptions.verbose=0;
end

initialguessobjects.methodforguess=1;
% =1: replace S with E[S]
% =2: treat S as idiosyncratic shock (this is probably a better idea?, but the initial guess becomes a memory bottleneck, which seems a bit silly)

if initialguessobjects.methodforguess==1 % Replace S with E[S]
    % Put things for the initial guess into a structure
    if ndims(pi_z)==2 % Does not depend on S
        initialguessobjects.pi_z=pi_z; % note, reverse order
    elseif ndims(pi_z)==3 % Depends on current S
        initialguessobjects.pi_z=sum(pi_z.*shiftdim(statdist_S,-2),3); % Average across S based on stationary dist of S
    elseif ndims(pi_z)==4 % joint transition of z with S
        temp=sum(pi_z.*shiftdim(pi_S,-2),4);
        temp=sum(temp.*shiftdim(statdist_S,-2),3);
        initialguessobjects.pi_z=reshape(temp,[N_z,N_z]);
    end

    % NOT REALLY DONE YET, JUST GOING TO ASSUME INPUT WAS A BASIC z_grid WITH NO DEPENDENCE ON S
    initialguessobjects.z_gridvals=CreateGridvals(n_z,z_grid,1);

elseif initialguessobjects.methodforguess==2 % treat S as idiosyncratic shock
    % Put things for the initial guess into a structure
    initialguessobjects.n_zS=[n_z,n_S];
    if ndims(pi_z)==2 % Does not depend on S
        initialguessobjects.pi_zS=kron(pi_S,pi_z); % note, reverse order
    elseif ndims(pi_z)==3 % Depends on current S
        initialguessobjects.pi_zS=repmat(reshape(permute(pi_z,[1,3,2]),[N_z*N_S,N_z]),1,N_S).*repelem(pi_S,N_z,N_z);
    elseif ndims(pi_z)==4 % joint transition of z with S
        initialguessobjects.pi_zS=reshape(permute(pi_z,[1,3,2,4]),[N_z*N_S,N_z*N_S]).*repelem(pi_S,N_z,N_z);
    end
    % NOT REALLY DONE YET, JUST GOING TO ASSUME INPUT WAS A BASIC z_grid WITH NO DEPENDENCE ON S
    initialguessobjects.zS_gridvals=[repmat(CreateGridvals(n_z,z_grid,1),N_S,1),repelem(S_gridvals,N_z,1)];
end

%% If you are dividing up the T dimension, so some setup for that
if recursiveeqmoptions.divideT>1
    t1vec=floor(1:T/recursiveeqmoptions.divideT:T)';
    recursiveeqmoptions.divideTindexes=[t1vec,[t1vec(2:end)-1;T]]; % (t1,t2), index for first and last t for each divideT-section
end

%% Set up the shooting algorithm
recursiveeqmoptions=setupGEnewprice3_shooting(recursiveeqmoptions,GeneralEqmEqns,GEPriceParamNames);

%% Now solve the Matched-expectations path
if N_d==0
    [PricePath,GEcondnPath,VPath,PolicyPath,AgentDistPath,DistMatches]=MatchedExpectationsPath_InfHorz_shooting_nod(AggShocksPath, AggShockNames, T, SSmask_T, SSprimemask_T,SSprimemask_T_indexes,ss_ind_T, n_a, n_z, n_S, l_a, l_z, a_grid,z_gridvals_T,z_gridvals_T_fastOLG, pi_Sprime_T, pi_z_T_fastOLG, pi_z_T_sim, ReturnFn, FnsToEvaluate, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GEPriceParamNames, GEeqnNames, GeneralEqmEqns, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, ReturnFnParamNames, initialguessobjects, vfoptions, simoptions,recursiveeqmoptions);
else
    [PricePath,GEcondnPath,VPath,PolicyPath,AgentDistPath,DistMatches]=MatchedExpectationsPath_InfHorz_shooting(AggShocksPath, AggShockNames, T, SSmask_T, SSprimemask_T,SSprimemask_T_indexes,ss_ind_T, n_d, n_a, n_z, n_S, l_d, l_a, l_z, d_grid, a_grid,z_gridvals_T,z_gridvals_T_fastOLG, pi_Sprime_T, pi_z_T_fastOLG, pi_z_T_sim, ReturnFn, FnsToEvaluate, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GEPriceParamNames, GEeqnNames, GeneralEqmEqns, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, ReturnFnParamNames, initialguessobjects, vfoptions, simoptions,recursiveeqmoptions);
end

%% Need to reshape these for output, permute for fastOLG, and store in GeneralizedTransitionFn
GeneralizedTransitionFn.PricePath=PricePath;
GeneralizedTransitionFn.GEcondnPath=GEcondnPath;
GeneralizedTransitionFn.VPath=reshape(permute(reshape(VPath,[N_a,T,N_z]),[1,3,2]),[n_a,n_z,T]);
GeneralizedTransitionFn.PolicyPath=reshape(permute(reshape(PolicyPath,[size(PolicyPath,1),N_a,T,N_z]),[1,2,4,3]),[size(PolicyPath,1),n_a,n_z,T]);
GeneralizedTransitionFn.AgentDistPath=reshape(permute(reshape(AgentDistPath,[N_a,T,N_z]),[1,3,2]),[n_a,n_z,T]);

%% Remove burnin from the outputs
for gg=1:length(GEeqnNames)
    temp=GeneralizedTransitionFn.GEcondnPath.(GEeqnNames{gg});
    GeneralizedTransitionFn.GEcondnPath.(GEeqnNames{gg})=temp(recursiveeqmoptions.burnin+1:end)';
end
for pp=1:length(GEPriceParamNames)
    temp=GeneralizedTransitionFn.PricePath.(GEPriceParamNames{pp});
    GeneralizedTransitionFn.PricePath.(GEPriceParamNames{pp})=temp(recursiveeqmoptions.burnin+1:end)';
end

GeneralizedTransitionFn.VPath=GeneralizedTransitionFn.VPath(:,:,recursiveeqmoptions.burnin+1:end);
GeneralizedTransitionFn.PolicyPath=GeneralizedTransitionFn.PolicyPath(:,:,:,recursiveeqmoptions.burnin+1:end);
GeneralizedTransitionFn.AgentDistPath=GeneralizedTransitionFn.AgentDistPath(:,:,recursiveeqmoptions.burnin+1:end);

%% Clean up the other outputs
% Use AggShockNames to describe the shock path
GeneralizedTransitionFn.OtherStuff.DistMatches=DistMatches; % reports the index and distance for the expectation matches
GeneralizedTransitionFn.AggShocksPath=AggShocksPath;
for aa=1:length(AggShockNames)
    temp=GeneralizedTransitionFn.AggShocksPath.(AggShockNames{aa});
    GeneralizedTransitionFn.AggShocksPath.(AggShockNames{aa})=temp(recursiveeqmoptions.burnin+1:end)';
    temp=AggShocksIndexesPath.(AggShockNames{aa});
    GeneralizedTransitionFn.OtherStuff.AggShocksIndexesPath.(AggShockNames{aa})=temp(recursiveeqmoptions.burnin+1:end); % other commands find it useful to have the index rather than the value for the aggregate shocks
end

end