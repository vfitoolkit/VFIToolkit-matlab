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
    recursiveeqmoptions.burnin=30; % burnin on the aggregate shock
    % Note: This burnin is used to avoid the ends of the generalized transition path during the matching expectations step.
    % Burnin at both ends.
    % Burnin this many periods at the beginning to eliminate influence of t=1 distribution
    % Burnin this many periods at the end to eliminate influence of t=T expected value fn
end
if ~isfield(recursiveeqmoptions,'burnin_simS')
    % separate burnin, this is just for the simulation of the S path, not used when iterating the path
    recursiveeqmoptions.burnin_simS=50; 
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
if ~isfield(recursiveeqmoptions,'initialguessmethod')
    recursiveeqmoptions.initialguessmethod=1; % how MEP_CreateInitialGuess builds the initial guess for the path: =1 replace S with E[S] (flat guess), =2 treat S as idiosyncratic shock, =3 SSJ (solve stationary eqm as for =1, then use sequence-space Jacobian to build a linearized guess)
end
if ~isfield(recursiveeqmoptions,'SSJmethod')
    recursiveeqmoptions.SSJmethod=1; % Only relevant when initialguessmethod==3. =1 fake-news algorithm (efficient; the intended production method), =2 brute-force oracle (slow, O(T^2), kept permanently as a validation reference)
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
        if isscalar(n_a)
            vfoptions.level1n=floor(sqrt(n_a(1)));
        elseif length(n_a)==2
            vfoptions.level1n=[floor(sqrt(n_a(1))),n_a(2)]; % default DC2A: level1n(2)==n_a(2) triggers DC2A branch
        end
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
if ~isfield(simoptions,'gridinterplayer')
    simoptions.gridinterplayer=0;
end
simoptions.outputkron=1;

GeneralizedTransitionFn=struct();

%%
if recursiveeqmoptions.verbose>=1
    fprintf(['VFI Toolkit uses the Matched-Expectations Path of Hanbaek Lee to solve models with Aggregate Shocks  \n' ...
        'Please cite his paper if you use this in your publication: Global Nonlinear Solutions in Sequence Space and the Generalized Transition Function  \n'])
end

%% Just treat burnin+T+burnin as T, and then remove the burnin while cleaning up at the end
T=recursiveeqmoptions.burnin+T+recursiveeqmoptions.burnin;
% Note: Burnin at both ends.
% Burnin this many periods at the beginning to eliminate influence of t=1 distribution
% Burnin this many periods at the end to eliminate influence of t=T expected value fn

l_S=length(n_S);

%%
% Move things to GPU
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
z_grid=gpuArray(z_grid);
S_grid=gpuArray(S_grid);
pi_z=gpuArray(pi_z);
pi_S=gpuArray(pi_S);

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_S=prod(n_S);

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

l_aprime=length(n_a);
l_daprime=l_d+l_aprime;

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
for bb=1:recursiveeqmoptions.burnin_simS
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


%% Setup path for idiosyncratic shocks: pi_z_T and z_gridvals_T
% Three possibilities for the dependence of idiosyncratic shocks on aggregate shocks
% pizdependS=0: idiosyncratic shock transition probabilities are independent of aggregate shocks
% pizdependS=1: idiosyncratic shock transition probabilities depend on S
% pizdependS=2: idiosyncratic shock transition probabilities depend on S and Sprime

% setup pi_z_T as the idiosyncratic transition probs, since they may depend on aggregate shocks
% Need to do different things depending on if idiosyncratic shocks depend on aggregate shocks
if ndims(pi_z)==2 % Does not depend on S
    pizdependS=0;
    pi_z_T=repmat(gpuArray(pi_z),1,1,T);
elseif ndims(pi_z)==3 % Depends on current S
    pizdependS=1;
    % pi_z_T=zeros(N_z,N_z,T,'gpuArray');
    pi_z_T=pi_z(:,:,ss_ind_T);
    % % TEST: TO DELETE LATER: I need to test the vectorized version on previous line works and then I can delete the following version with loop
    % pi_z_T2=zeros(N_z,N_z,T,'gpuArray');
    % for tt=1:T
    %     pi_z_T2(:,:,tt)=pi_z(:,:,ss_ind_T(tt));
    % end
    % if max(abs(pi_z_T-pi_z_T2))>1e-12
    %     error('Wrong vectorization (if you see this error, let me know on forum)')
    % end
elseif ndims(pi_z)==4 % joint transition of z with S
    pizdependS=2;
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

% setup z_gridvals_T: [T,N_z,l_z]
if ndims(z_grid)==2
    if all(size(z_grid)==[prod(n_z),length(l_z)]) % Joint-grid, no dependence on S
        zgriddependS=0; % Needed for creating the initial guess later
        z_gridvals=z_grid; % Needed for creating the initial guess later
        z_gridvals_T=repmat(shiftdim(z_gridvals,-1),T,1,1); % [T,N_z,l_z] for fastOLG
    elseif all(size(z_grid)==[sum(n_z),1]) % Stacked-column grid, no dependence on S
        zgriddependS=0; % Needed for creating the initial guess later
        if l_z>=1
            z1_gridvals=CreateGridvals(n_z(1),gpuArray(z_grid(1:n_z(1))),1);
            z_gridvals=z1_gridvals;
            if l_z>=2
                z2_gridvals=CreateGridvals(n_z(2),gpuArray(z_grid(n_z(1)+1:n_z(1)+n_z(2))),1);
                z_gridvals=[repmat(z_gridvals,n_z(2),1),repelem(z2_gridvals,n_z(1))];
                if l_z>=3
                    z3_gridvals=CreateGridvals(n_z(3),gpuArray(z_grid(sum(n_z(1:2))+1:sum(n_z(1:2))+n_z(3))),1);
                    z_gridvals=[repmat(z_gridvals,n_z(3),1),repelem(z3_gridvals,sum(n_z(1:2)),1)];
                    if l_z>=4
                        z4_gridvals=CreateGridvals(n_z(3),gpuArray(z_grid(sum(n_z(1:2))+1:sum(n_z(1:2))+n_z(3))),1);
                        z_gridvals=[repmat(z_gridvals,n_z(3),1),repelem(z4_gridvals,sum(n_z(1:2)),1)];
                        if l_z>=5
                            z5_gridvals=CreateGridvals(n_z(3),gpuArray(z_grid(sum(n_z(1:2))+1:sum(n_z(1:2))+n_z(3))),1);
                            z_gridvals=[repmat(z_gridvals,n_z(3),1),repelem(z5_gridvals,sum(n_z(1:2)),1)];
                        end
                    end
                end
            end
        end
        z_gridvals_T=repmat(shiftdim(z_gridvals,-1),T,1,1); % [T,N_z,l_z] for fastOLG
    elseif all(size(z_grid)==[sum(n_z),prod(n_S)]) || all(size(z_grid)==[sum(n_z),1,prod(n_S)]) % Stacked-column grid, depends on S
        z_grid=reshape(z_grid,[sum(n_z),prod(n_S)]);
        zgriddependS=1; % Needed for creating the initial guess later
        z_gridvals_S=zeros(N_z,l_z,N_S);
        for ss=1:N_S
            z_grid_ss=z_grid(:,ss);
            if l_z>=1
                z1_gridvals=CreateGridvals(n_z(1),gpuArray(z_grid_ss(1:n_z(1))),1);
                z_gridvals=z1_gridvals;
                if l_z>=2
                    z2_gridvals=CreateGridvals(n_z(2),gpuArray(z_grid_ss(n_z(1)+1:n_z(1)+n_z(2))),1);
                    z_gridvals=[repmat(z_gridvals,n_z(2),1),repelem(z2_gridvals,n_z(1))];
                    if l_z>=3
                        z3_gridvals=CreateGridvals(n_z(3),gpuArray(z_grid_ss(sum(n_z(1:2))+1:sum(n_z(1:2))+n_z(3))),1);
                        z_gridvals=[repmat(z_gridvals,n_z(3),1),repelem(z3_gridvals,sum(n_z(1:2)),1)];
                        if l_z>=4
                            z4_gridvals=CreateGridvals(n_z(3),gpuArray(z_grid_ss(sum(n_z(1:2))+1:sum(n_z(1:2))+n_z(3))),1);
                            z_gridvals=[repmat(z_gridvals,n_z(3),1),repelem(z4_gridvals,sum(n_z(1:2)),1)];
                            if l_z>=5
                                z5_gridvals=CreateGridvals(n_z(3),gpuArray(z_grid_ss(sum(n_z(1:2))+1:sum(n_z(1:2))+n_z(3))),1);
                                z_gridvals=[repmat(z_gridvals,n_z(3),1),repelem(z5_gridvals,sum(n_z(1:2)),1)];
                            end
                        end
                    end
                end
            end
            z_gridvals_S(:,:,ss)=z_gridvals;
        end
        z_gridvals_T=z_gridvals_S(:,:,ss_ind_T); % [N_z,l_z,T]
        z_gridvals_T=permute(z_gridvals_T,[3,1,2]); % [T,N_z,l_z] for fastOLG
    end
elseif ndims(z_grid)==3
    if all(size(z_grid)==[prod(n_z),length(l_z),S]) % Joint-grid, depends on S
        zgriddependS=1; % Needed for creating the initial guess later
        z_gridvals_S=z_grid; % Needed for creating the initial guess later
        z_gridvals_T=z_gridvals_S(:,:,ss_ind_T);
        z_gridvals_T=permute(z_gridvals_T,[3,1,2]); % [T,N_z,l_z] for fastOLG
    else
        error('size(z_grid) does not fit any accepted pattern')
    end
else
    error('size(z_grid) does not fit any accepted pattern')
end
% Check that I coded this right
if l_z==1
    if ~all(size(z_gridvals_T)==[T,N_z])
        error('The z_gridvals_T setup has a problem, please let me know on forum so I can fix')
    end
elseif ~all(size(z_gridvals_T)==[T,N_z,l_z])
    error('The z_gridvals_T setup has a problem, please let me know on forum so I can fix')
end

% For the fastOLG evaluation of AggVars we need
z_gridvals_T_fastOLG=shiftdim(z_gridvals_T,-1); % [1,T,N_z,l_z] need this for fastOLG agent dist, but need the standard still for the value fn without fastOLG
% Keep some of this stuff for the output
GeneralizedTransitionFn.OtherStuff.pi_z_T=pi_z_T(:,:,recursiveeqmoptions.burnin+1:end-recursiveeqmoptions.burnin+1);
GeneralizedTransitionFn.OtherStuff.z_gridvals_T=z_gridvals_T(recursiveeqmoptions.burnin+1:end-recursiveeqmoptions.burnin+1,:,:);

if recursiveeqmoptions.verbose>=2
    fprintf('\n Interpretation of shock setup: \n')
    if zgriddependS==0
        fprintf('Idiosyncratic shocks, z, grid is independent of aggregate S \n')
    elseif zgriddependS==1
        fprintf('Idiosyncratic shocks, z, grid depends on aggregate S \n')
    end
    if pizdependS==0
        fprintf('Idiosyncratic shocks, z, transition probabilities are independent of S \n')
    elseif pizdependS==1
        fprintf('Idiosyncratic shocks, z, transition probabilities depend on S \n')
    elseif pizdependS==2
        fprintf('Idiosyncratic shocks, z, transition probabilities depend on S and Sprime \n')
    end
    fprintf(' \n')
end



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

initialguessobjects.methodforguess=recursiveeqmoptions.initialguessmethod; % =1 is the default (set above)
% =1: replace S with E[S]
% =2: treat S as idiosyncratic shock (this is probably a better idea?, but the initial guess becomes a memory bottleneck, which seems a bit silly)
% =3: replace S with E[S] to solve the stationary eqm (as for =1), then use SSJ to build a linearized initial guess for the path

if initialguessobjects.methodforguess==1 || initialguessobjects.methodforguess==3 % Replace S with E[S] (methodforguess==3 SSJ uses this same stationary setup)
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

    % We aleady build z_gridvals
    if zgriddependS==0
        initialguessobjects.z_gridvals=z_gridvals;
    elseif zgriddependS==1
        initialguessobjects.z_gridvals=sum(z_gridvals_S.*shiftdim(statdist_S,-2),3); % Bit weird, but will do for now. Not obvious what a better choice would be.
    end

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

    % We aleady build z_gridvals
    if zgriddependS==0
        initialguessobjects.zS_gridvals=[repmat(z_gridvals,N_S,1),repelem(S_gridvals,N_z,1)];
    elseif zgriddependS==1
        z_gridvals=sum(z_gridvals_S.*shiftdim(statdist_S,-2)); % Bit weird, but will do for now. Not obvious what a better choice would be;
        initialguessobjects.z_gridvals=[repmat(z_gridvals,N_S,1),repelem(S_gridvals,N_z,1)];
    end
end

% zgriddependS
% size(z_gridvals_S)
% size(shiftdim(statdist_S,-2))
% size(initialguessobjects.pi_z)
% size(initialguessobjects.z_gridvals)
% error('stop')


%% Matching Expectations
% Set some options about exactly how the matching will be done.

% First, we normally match on Sprime, but sometimes need to do something more
recursiveeqmoptions.matchingsetup=1; % Match based on Sprime and Distance
if ndims(pi_z)==4 % joint transition of z with S
    recursiveeqmoptions.matchingsetup=2; % Match based on (S,Sprime) and Distance
end

% How to measure the distance:
recursiveeqmoptions.matchE_distmeasure=2; % How to measure the distance between agent distributions
% =1 is Kolmogorov-Smirnoff distance, not yet implemented
% =2 is Mean of (next period) Endogenous and (this period) Exogenous States

% How to handle the idiosyncratic exogenous states:
recursiveeqmoptions.matching_IdiosyncraticExogenousStates=1;
% When pi_z is independent of S and Sprime, we can omit exogenous states from the matching process entirely
if pizdependS==0 % pizdependS=0: idiosyncratic shock transition probabilities are independent of aggregate shocks
    % When matching, we can just ignore the idiosyncratic exogenous states as they will anyway be identical in every period.
    recursiveeqmoptions.matching_IdiosyncraticExogenousStates=0;
end
% When idiosyncratic states are being determined in General Eqm, WE WILL NEED THE CURRENTLY UNUSED
% recursiveeqmoptions.matching_IdiosyncraticExogenousStates=2;

% How many 'nearest' agent distributions to use when constructing expectations
recursiveeqmoptions.matchE_nnearest=1;
% Idea is that instead of just using the single best match (=1) we can,
% e.g., instead use the average of the best three matches (=3).
if recursiveeqmoptions.matchE_nnearest>1
    error('Cannot yet use recursiveeqmoptions.matchE_nnearest>1 to match more than one period')
end

% Convenient to just keep a bunch of things needed to match expectations together in a structure
matchexpectations=struct();

%% Matching Setup 1/3
% Build masks for the values of Sprime that allow us to speed up the computation of the expectations

% if recursiveeqmoptions.matchingsetup==1 % Match based on Sprime and Distance of t+1 [the baseline]

% To be able to speed up the code, we create a record of which t is which S
% For a given S, the row of SSmask_T indicates all the periods with the same S
SSmask_T=zeros(N_S,T,'gpuArray'); % For SS_c, it gives you a vector of length T, with 1s in the time periods tt where you get that SS_c
SSprimemask_T=zeros(1,T,N_S,'gpuArray'); % For SSprime_c, it gives you a vector of length T, with 1s in the time periods tt where you get that SSprime_c
for tt=1:T
    SSmask_T(ss_ind_T(tt),tt)=1;
    if tt<T
        SSprimemask_T(1,tt,ss_ind_T(tt+1))=1;
    end
end
% Use Mask to rule out the burnin of the value fns from consideration
SSmask_T(:,1:recursiveeqmoptions.burnin)=0; % remove these from consideration
SSmask_T(:,end-recursiveeqmoptions.burnin+1:end)=0; % remove these from consideration
SSprimemask_T(1,1:recursiveeqmoptions.burnin,:)=0; % remove these from consideration
SSprimemask_T(1,end-recursiveeqmoptions.burnin+1:end,:)=0; % remove these from consideration
% Convert to logical
SSmask_T=logical(SSmask_T);
SSprimemask_T=logical(SSprimemask_T); % Note: for last period this is zero by construction (as cannot see periods after)


for SSprime_c=1:N_S % Loop over the possible next-period aggregate shocks
    Sstr=['S',num2str(SSprime_c),'distancemodifier'];
    DistancesModifier=Inf*ones(T-1,T,'gpuArray');
    % tt1: this period, omits period T
    % tt2: potential next period
    % =Inf is essentially ruling everything out
    % Now we go through and rule things in
    % DistancesModifier(:,tt2)=1 when Sprime_tt2=Sprime_c
    for tt2=1:T
        if SSprimemask_T(1,tt2,SSprime_c)
            DistancesModifier(:,tt2)=1;
        end
    end
    % DistancesModifier(tt1,tt2)=0 when tt2=tt1+1 && Sprime_c is the actual Sprime_tt2
    for tt1=1:T-1
        if ss_ind_T(tt1+1)==SSprime_c
            DistancesModifier(tt1,tt1+1)=0 ;
        end
    end
    matchexpectations.(Sstr)=DistancesModifier;
end
    
if recursiveeqmoptions.matchingsetup==2
    % We additionally need that S matches as well
    for SSprime_c=1:N_S % Loop over the possible next-period aggregate shocks
        Sstr=['S',num2str(SSprime_c),'distancemodifier'];
        DistancesModifier=matchexpectations.(Sstr);

        % When S does not match, switch the distance to Inf
        for tt1=1:T-1
            for tt2=2:T
                if ss_ind_T(tt1)==ss_ind_T(tt2-1) % S matches the period before the expectations term comes from
                    % This is fine
                else % But if they don't match, cannot use this one
                    DistancesModifier(tt1,tt2)=Inf;
                end
            end
        end

        matchexpectations.(Sstr)=DistancesModifier;
    end
end


if recursiveeqmoptions.verbose==2
    fprintf('Number of instances of each S (multidimensional index) \n')
    sum(SSmask_T,2)'

    % Create a time-series plot of the aggregate shock path
    figure(1)
    plot(-recursiveeqmoptions.burnin+(1:1:T),ss_ind_T)
    title('Time series for S index in the Generalized Transition Fn (shaded area is burnin; omited from matching)')
    hold on
    patch([-recursiveeqmoptions.burnin+1 0 0 -recursiveeqmoptions.burnin+1],[1 1 2 2],'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    patch([0 T-recursiveeqmoptions.burnin+1 T-recursiveeqmoptions.burnin+1 0],[1 1 2 2],'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    hold off
    xlim([-recursiveeqmoptions.burnin+1,T-recursiveeqmoptions.burnin])
end


%% Matching Setup 2/3
% Build some FnsToEvaluate that are used to calculate the distances
if recursiveeqmoptions.matchE_distmeasure==2
    % Create FnsToEvaluate the are the endogenous state and exogenous state
    nfinput = l_aprime+l_a+l_z; % Variable number of inputs (this is the nod code)
    inputNames = compose('x%d', 1:nfinput); % Create {'x1', 'x2', ..., 'x5'}
    argStr = strjoin(inputNames, ','); % Create 'x1,x2,x3,x4,x5'
    % Endogenous states
    if l_a>=1
        FnsToEvaluate.EndoState1=str2func(['@(', argStr, ') ', 'x',num2str(l_aprime+1)]);
        if l_a>=2
            FnsToEvaluate.EndoState2=str2func(['@(', argStr, ') ', 'x',num2str(l_aprime+2)]);
            if l_a>=3
                FnsToEvaluate.EndoState3=str2func(['@(', argStr, ') ', 'x',num2str(l_aprime+3)]);
                if l_a>=4
                    error('Have not implemented expectations matching for lenght(n_a)>3')
                end
            end
        end
    end
    % Exogenous states
    if recursiveeqmoptions.matching_IdiosyncraticExogenousStates==2 % We only need these if the idiosyncratic exogenous shocks are being determined in General Eqm
        FnsToEvaluate_Exo=struct();
        if l_z>=1
            FnsToEvaluate_Exo.ExoState1=str2func(['@(', argStr, ') ', 'x',num2str(l_aprime+l_a+1)]);
            if l_z>=2
                FnsToEvaluate_Exo.ExoState2=str2func(['@(', argStr, ') ', 'x',num2str(l_aprime+l_a+2)]);
                if l_z>=3
                    FnsToEvaluate_Exo.ExoState3=str2func(['@(', argStr, ') ', 'x',num2str(l_aprime+l_a+3)]);
                    if l_z>=4
                        FnsToEvaluate_Exo.ExoState4=str2func(['@(', argStr, ') ', 'x',num2str(l_aprime+l_a+4)]);
                        if l_z>=5
                            FnsToEvaluate_Exo.ExoState5=str2func(['@(', argStr, ') ', 'x',num2str(l_aprime+l_a+5)]);
                            if L_z>=6
                                error('Have not implemented expectations matching for lenght(n_z)>5')
                            end
                        end
                    end
                end
            end
        end
    end
end

%% Matching Setup 3/3
% Precompute the matching distances for the idiosyncratic exogenous states as these are constant across path iterations
% recursiveeqmoptions.matching_IdiosyncraticExogenousStates
%  =0 we ignore the idiosyncratic exogenous states, omitting them from the  matching entirely
%  =1 we can precompute the distances for the idiosyncratic exogenous states as they do not differ across iteration of the path
%  =2 idiosyncratic exogenous states change across iterations of the path and so the distances must be computed at each iteration
%
% Note: zgriddependS does not change this behaviour in any way.

if recursiveeqmoptions.matching_IdiosyncraticExogenousStates==0
    % When matching, we can just ignore the idiosyncratic exogenous states as they will anyway be identical in every period.
    Distances_ExoState=[]; % placeholder, this will be ignored
elseif recursiveeqmoptions.matching_IdiosyncraticExogenousStates==1
    % We already have pi_z_T and z_gridvals_T
    % From these we can precompute the agent distribution over idiosyncratic exogenous shocks in every period, 
    % and thus the matching distances for the Idiosyncratic Exogenous States

    % First, calculate the AggPath on the ExoStates
    % We know that the period 1 of path will be from the initial guess
    zdist=ones(N_z,1,'gpuArray')/N_z;
    pi_z_transpose=initialguessobjects.pi_z';
    zdistlag=zeros(N_z,1,'gpuArray');
    while max(abs(zdist-zdistlag))>1e-9
        zdistlag=zdist;
        zdist=pi_z_transpose*zdist;
    end
    % Now create the whole zdist path
    zdistpath=zeros(N_z,T,'gpuArray');
    zdistpath(:,1)=zdist;
    for tt=2:T
       zdistpath(:,tt)=pi_z_T(:,:,tt)'*zdistpath(:,tt-1); % pi_z_T is (z,z',t)
    end
    % Now use the grids together with distribution to calculate the AggVarsPath
    % [Use AggVarsPath.ExoState1.Mean nomenclature so it is obvious how it relates to other parts of code when doing endogenous states, etc.]
    % z_gridvals_T is [T,N_z,l_z]
    z_gridvals_T_temp=permute(z_gridvals_T,[2,1,3]); % [N_z,T,l_z]
    if l_z>=1
        AggVarsPath.ExoState1.Mean=sum(zdistpath.*z_gridvals_T_temp(:,:,1),1);
        if l_z>=2
            AggVarsPath.ExoState2.Mean=sum(zdistpath.*z_gridvals_T_temp(:,:,2),1);
            if l_z>=3
                AggVarsPath.ExoState3.Mean=sum(zdistpath.*z_gridvals_T_temp(:,:,3),1);
                if l_z>=4
                    AggVarsPath.ExoState4.Mean=sum(zdistpath.*z_gridvals_T_temp(:,:,4),1);
                    if l_z>=5
                        AggVarsPath.ExoState5.Mean=sum(zdistpath.*z_gridvals_T_temp(:,:,5),1);
                    end
                end
            end
        end
    end
    % Now the distances
    % Exogenous states are determined in general eqm, so have to recompute matching distances every iteration of thte path
    if l_z>=1
        Distances_ExoState1=100*abs(AggVarsPath.ExoState1.Mean(2:T)'-AggVarsPath.ExoState1.Mean)./AggVarsPath.ExoState1.Mean(2:T)'; % percentage difference
        if l_z>=2
            Distances_ExoState2=100*abs(AggVarsPath.ExoState1.Mean(2:T)'-AggVarsPath.ExoState1.Mean)./AggVarsPath.ExoState1.Mean(2:T)'; % percentage difference
            if l_z>=3
                Distances_ExoState3=100*abs(AggVarsPath.ExoState1.Mean(2:T)'-AggVarsPath.ExoState1.Mean)./AggVarsPath.ExoState1.Mean(2:T)'; % percentage difference
                if l_z>=4
                    Distances_ExoState4=100*abs(AggVarsPath.ExoState1.Mean(2:T)'-AggVarsPath.ExoState1.Mean)./AggVarsPath.ExoState1.Mean(2:T)'; % percentage difference
                    if l_z>=5
                        Distances_ExoState5=100*abs(AggVarsPath.ExoState1.Mean(2:T)'-AggVarsPath.ExoState1.Mean)./AggVarsPath.ExoState1.Mean(2:T)'; % percentage difference
                        Distances_ExoState=(Distances_ExoState1+Distances_ExoState2+Distances_ExoState3+Distances_ExoState4+Distances_ExoState5)/5;
                    else
                        Distances_ExoState=(Distances_ExoState1+Distances_ExoState2+Distances_ExoState3+Distances_ExoState4)/4;
                    end
                else
                    Distances_ExoState=(Distances_ExoState1+Distances_ExoState2+Distances_ExoState3)/3;
                end
            else
                Distances_ExoState=(Distances_ExoState1+Distances_ExoState2)/2;
            end
        else
            Distances_ExoState=Distances_ExoState1;
        end
    end

    % How exactly we will use this distance depends slightly on the case
    if pizdependS==1 % pizdependS=1: idiosyncratic shock transition probabilities depend on S
        % We match based on getting the same value for exogenous shocks in 't+1'
        Distances_ExoState=Distances_ExoState1;
    elseif pizdependS==2 % pizdependS=2: idiosyncratic shock transition probabilities depend on S and Sprime
        % We match based on getting the same value for exogenous shocks in 't'
        % The dependence on Sprime means the actual next period exogenous shocks will be different for each Sprime
        Distances_ExoState=[0,Distances_ExoState1(1:T-1)];
    end
elseif recursiveeqmoptions.matching_IdiosyncraticExogenousStates==2
    temp=fieldnames(FnsToEvaluate_Exo);
    for ff=1:length(temp)
        FnsToEvaluate.(temp{ff})=FnsToEvaluate_Exo.(temp{ff});
    end
    Distances_ExoState=[]; % placeholder, this will be ignored
end

% Put a bunch of the things that are used for matching expectations into a structure for convenience
matchexpectations.Distances_ExoState=Distances_ExoState;


%% Change to FnsToEvaluate as cell so that it is not being recomputed all the time
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


%% If you are dividing up the T dimension, so some setup for that
if recursiveeqmoptions.divideT>1
    t1vec=floor(1:T/recursiveeqmoptions.divideT:T)';
    recursiveeqmoptions.divideTindexes=[t1vec,[t1vec(2:end)-1;T]]; % (t1,t2), index for first and last t for each divideT-section
end

%% Set up the shooting algorithm
recursiveeqmoptions=setupGEnewprice3_shooting(recursiveeqmoptions,GeneralEqmEqns,GEPriceParamNames);


if recursiveeqmoptions.verbose>=2
    fprintf('The recursiveeqmoptions are: \n')
    recursiveeqmoptions
    fprintf('Done setup, now moving on to intial guess. \n')
    fprintf(' \n')
end

%% Now solve the Matched-expectations path
if N_d==0
    [PricePath,GEcondnPath,VPath,PolicyPath,AgentDistPath,DistMatches]=MatchedExpectationsPath_InfHorz_shooting_nod(AggShocksPath, AggShockNames, T, ss_ind_T, n_a, n_z, n_S, l_aprime, l_a, l_z, a_grid,z_gridvals_T,z_gridvals_T_fastOLG, pi_Sprime_T, pi_z_T_fastOLG, pi_z_T_sim, ReturnFn, FnsToEvaluate, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GEPriceParamNames, GEeqnNames, GeneralEqmEqns, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, ReturnFnParamNames, initialguessobjects, matchexpectations, vfoptions, simoptions,recursiveeqmoptions);
else
    [PricePath,GEcondnPath,VPath,PolicyPath,AgentDistPath,DistMatches]=MatchedExpectationsPath_InfHorz_shooting(AggShocksPath, AggShockNames, T, ss_ind_T, n_d, n_a, n_z, n_S, l_d, l_aprime, l_a, l_z, d_grid, a_grid,z_gridvals_T,z_gridvals_T_fastOLG, pi_Sprime_T, pi_z_T_fastOLG, pi_z_T_sim, ReturnFn, FnsToEvaluate, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GEPriceParamNames, GEeqnNames, GeneralEqmEqns, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, ReturnFnParamNames, initialguessobjects, vfoptions, matchexpectations, simoptions,recursiveeqmoptions);
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