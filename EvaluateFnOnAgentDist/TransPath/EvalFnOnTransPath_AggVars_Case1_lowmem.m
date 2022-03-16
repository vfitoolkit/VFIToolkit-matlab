function AggVarsPath=EvalFnOnTransPath_AggVars_Case1_lowmem(FnsToEvaluate, FnsToEvaluateParamNames,PricePath,PricePathNames,PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, Parameters, T, V_final, AgentDist_initial, n_d, n_a, n_z, pi_z, d_grid, a_grid,z_grid, DiscountFactorParamNames, ReturnFn, ReturnFnParamNames, simoptions, transpathoptions)
% AggVarsPath is T periods long (periods 0 (before the reforms are announced) & T are the initial and final values).

% It is hardcoded that simoptions.outputasstructure=1, but need to set to zero for subfunction
simoptions.outputasstructure=0;

%% Check if using _tminus1 and/or _tplus1 variables.
if isstruct(FnsToEvaluate) && isstruct(GeneralEqmEqns)
    [tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tplus1pricePathkk]=inputsFindtplus1tminus1(FnsToEvaluate,GeneralEqmEqns,PricePathNames);
    tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tplus1pricePathkk
else
    tplus1priceNames=[];
    tminus1priceNames=[];
    tminus1AggVarsNames=[];
    tplus1pricePathkk=[];
end

use_tplus1price=0;
if length(tplus1priceNames)>0
    use_tplus1price=1;
end
use_tminus1price=0;
if length(tminus1priceNames)>0
    use_tminus1price=1;
    for tt=1:length(tminus1priceNames)
        if ~isfield(transpathoptions.initialvalues,tminus1priceNames{tt})
            fprintf('ERROR: Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1priceNames{tt})
            dbstack
            break
        end
    end
end
use_tminus1AggVars=0;
if length(tminus1AggVarsNames)>0
    use_tminus1AggVars=1;
    for tt=1:length(tminus1AggVarsNames)
        if ~isfield(transpathoptions.initialvalues,tminus1AggVarsNames{tt})
            fprintf('ERROR: Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1AggVarsNames{tt})
            dbstack
            break
        end
    end
end
% Note: I used this approach (rather than just creating _tplus1 and _tminus1 for everything) as it will be same computation.

use_tminus1price
use_tminus1AggVars

%%
N_d=prod(n_d);
N_z=prod(n_z);
N_a=prod(n_a);
l_z=length(n_z);

V_final=reshape(V_final,[N_a,N_z]);
AgentDist_initial=reshape(AgentDist_initial,[N_a*N_z,1]);
V=zeros(size(V_final),'gpuArray');
Policy=zeros(N_a,N_z,'gpuArray');

unkronoptions.parallel=2;

beta=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames)); % It is possible but unusual with infinite horizon that there is more than one discount factor and that these should be multiplied together
IndexesForPathParamsInDiscountFactor=CreateParamVectorIndexes(DiscountFactorParamNames, ParamPathNames);
ReturnFnParamsVec=gpuArray(CreateVectorFromParams(Parameters, ReturnFnParamNames));
[IndexesForPricePathInReturnFnParams, IndexesPricePathUsedInReturnFn]=CreateParamVectorIndexes(ReturnFnParamNames, PricePathNames);
[IndexesForPathParamsInReturnFnParams, IndexesParamPathUsedInReturnFn]=CreateParamVectorIndexes(ReturnFnParamNames, ParamPathNames);

z_gridvals=CreateGridvals(n_z,z_grid,1); % 1 is to create z_gridvals as matrix

PolicyIndexesPath=zeros(N_a,N_z,T-1,'gpuArray'); %Periods 1 to T-1
%First, go from T-1 to 1 calculating the Value function and Optimal
%policy function at each step. Since we won't need to keep the value
%functions for anything later we just store the next period one in
%Vnext, and the current period one to be calculated in V
Vnext=V_final;
for tt=0:T-1 %so tt=T-ttr
    
    if ~isnan(IndexesForPathParamsInDiscountFactor)
        beta(IndexesForPathParamsInDiscountFactor)=ParamPath(T-ttr,:); % This step could be moved outside all the loops
    end
    if ~isnan(IndexesForPricePathInReturnFnParams)
        ReturnFnParamsVec(IndexesForPricePathInReturnFnParams)=PricePath(T-ttr,IndexesPricePathUsedInReturnFn);
    end
    if ~isnan(IndexesForPathParamsInReturnFnParams)
        ReturnFnParamsVec(IndexesForPathParamsInReturnFnParams)=ParamPath(T-ttr,IndexesParamPathUsedInReturnFn); % This step could be moved outside all the loops by using BigReturnFnParamsVec idea
    end
    
    for z_c=1:N_z
        ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, ones(l_z,1), d_grid, a_grid, z_gridvals(z_c,:),ReturnFnParamsVec);

        %Calc the condl expectation term (except beta), which depends on z but
        %not on control variables
        EV_z=Vnext.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);
        
        entireEV_z=kron(EV_z,ones(N_d,1));
        entireRHS=ReturnMatrix_z+beta*entireEV_z*ones(1,N_a,1);
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,z_c)=Vtemp;
        Policy(:,z_c)=maxindex;
        
    end
    
    PolicyIndexesPath(:,:,T-ttr)=Policy;
    Vnext=V;
end

% Free up space on GPU by deleting things no longer needed
clear ReturnMatrix_z entireRHS entireEV_z EV_z Vtemp maxindex V Vnext


%Now we have the full PolicyIndexesPath, we go forward in time from 1
%to T using the policies to generate the AggVarsPath. First though we
%put in it's initial and final values.
AggVarsPath=zeros(T,length(FnsToEvaluate),'gpuArray');
% AggVarsPath(T,:)=SSvalues_AggVars_final;
%Call AgentDist the current periods distn and AgentDistnext
%the next periods distn which we must calculate
AgentDist=AgentDist_initial;
%Now we have the full PolicyIndexesPath, we go forward in time from 1
%to T using the policies to update the agents distribution generating a
%new price path
for tt=1:T%-1
    %Get the current optimal policy
    Policy=PolicyIndexesPath(:,:,tt);
    
    optaprime=shiftdim(ceil(Policy/N_d),-1); % This shipting of dimensions is probably not necessary
    optaprime=reshape(optaprime,[1,N_a*N_z]);
    
    Ptemp=zeros(N_a,N_a*N_z,'gpuArray');
    Ptemp(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=1;
    Ptran=(kron(pi_z',ones(N_a,N_a,'gpuArray'))).*(kron(ones(N_z,1,'gpuArray'),Ptemp));
    AgentDistnext=Ptran*AgentDist;
    
    PolicyTemp=zeros(2,N_a,N_z,'gpuArray'); %NOTE: this is not actually in Kron form
    PolicyTemp(1,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
    PolicyTemp(2,:,:)=shiftdim(ceil(Policy/N_d),-1);
    
    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
    end
    if use_tminus1price==1
        for pp=1:length(tminus1priceNames)
            if tt>1
                Parameters.([tminus1priceNames{pp},'_tminus1'])=Parameters.(tminus1priceNames{pp});
            else
                Parameters.([tminus1priceNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1priceNames{pp});
            end
        end
    end
    if use_tplus1price==1
        for pp=1:length(tplus1priceNames)
            kk=tplus1pricePathkk(pp);
            Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePath(tt+1,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk)); % Make is so that the time t+1 variables can be used
        end
    end
    if use_tminus1AggVars==1
        for pp=1:length(use_tminus1AggVars)
            if tt>1
                % The AggVars have not yet been updated, so they still contain previous period values
                Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=Parameters.(tminus1AggVarsNames{pp});
            else
                Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1AggVarsNames{pp});
            end
        end
    end
    
    PolicyTemp=UnKronPolicyIndexes_Case1(PolicyTemp, n_d, n_a, n_z,unkronoptions);
    AggVars=EvalFnOnAgentDist_AggVars_Case1(AgentDist, PolicyTemp, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, 2,simoptions);

    AggVarsPath(tt,:)=AggVars;
    
    AgentDist=AgentDistnext;
end

% Change the output into a structure
AggVarsPath2=AggVarsPath;
clear AggVarsPath
AggVarsPath=struct();
for ff=1:length(simoptions.AggVarNames)
    AggVarsPath.(simoptions.AggVarNames{ff}).Mean=AggVarsPath2(:,ff);
end

end