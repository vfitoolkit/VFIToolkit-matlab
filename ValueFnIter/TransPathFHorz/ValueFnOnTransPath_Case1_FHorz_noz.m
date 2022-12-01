function [VPath,PolicyPath]=ValueFnOnTransPath_Case1_FHorz_noz(PricePath, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, Policy_final, Parameters, n_d, n_a, N_j, d_grid, a_grid, DiscountFactorParamNames, ReturnFn, transpathoptions, vfoptions)

%% Create ReturnFnParamNames
if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);

temp=getAnonymousFnInputNames(ReturnFn);
if length(temp)>(l_d+l_a+l_a)
    ReturnFnParamNames={temp{l_d+l_a+l_a+1:end}}; % the first inputs will always be (d,aprime,a,z)
else
    ReturnFnParamNames={};
end

%%
if ~strcmp(transpathoptions.exoticpreferences,'None')
    error('Only transpathoptions.exoticpreferences==0 is supported by TransitionPath_Case1')
end

if transpathoptions.parallel~=2
    error('Only transpathoptions.parallel==2 is supported by TransitionPath_Case1')
else
    d_grid=gpuArray(d_grid); a_grid=gpuArray(a_grid);
%     PricePath=gpuArray(PricePath);
end
unkronoptions.parallel=2;

N_d=prod(n_d);
N_a=prod(n_a);
l_p=size(PricePath,2);

if transpathoptions.parallel==2
    % Make sure things are on gpu where appropriate.
    if N_d>0
        d_grid=gather(d_grid);
    end
    a_grid=gather(a_grid);
end

if transpathoptions.verbose==1
    transpathoptions
end

PricePathDist=Inf;
pathcounter=0;

V_final=reshape(V_final,[N_a,N_j]);
if N_d>0
    Policy=zeros(2,N_a,N_j,'gpuArray');
else
    Policy=zeros(N_a,N_j,'gpuArray');
end
if transpathoptions.verbose==1
    DiscountFactorParamNames
    ReturnFnParamNames
    ParamPathNames
    PricePathNames
end

%%
% I DONT THINK THAT _tminus1 and/or _tplus1 variables ARE USED WITH Value fn. 
% AT LEAST NOT IN ANY EXAMPLES I HAVE COME ACROSS. AS SUCH THEY ARE NOT IMPLEMENTED HERE.

%%
VKronPath=zeros(N_a,N_j,T);
VKronPath(:,:,T)=V_final;

if N_d>0
    PolicyIndexesPath=zeros(2,N_a,N_j,T,'gpuArray'); %Periods 1 to T-1
    PolicyIndexesPath(:,:,:,T)=KronPolicyIndexes_FHorz_Case1_noz(Policy_final, n_d, n_a,N_j);
else
    PolicyIndexesPath=zeros(N_a,N_j,T,'gpuArray'); %Periods 1 to T-1
    PolicyIndexesPath(:,:,T)=KronPolicyIndexes_FHorz_Case1_noz(Policy_final, n_d, n_a,N_j);
end

%First, go from T-1 to 1 calculating the Value function and Optimal
%policy function at each step. Since we won't need to keep the value
%functions for anything later we just store the next period one in
%Vnext, and the current period one to be calculated in V
Vnext=V_final;
for ttr=1:T-1 %so t=T-i

    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePath(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
    end
    
    [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep_noz(Vnext,n_d,n_a,N_j,d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    % The VKron input is next period value fn, the VKron output is this period.
    % Policy is kept in the form where it is just a single-value in (d,a')
    
    if N_d>0
        PolicyIndexesPath(:,:,:,T-ttr)=Policy;
    else
        PolicyIndexesPath(:,:,T-ttr)=Policy;
    end
    VKronPath(:,:,T-ttr)=V;
    Vnext=V;
end

%% Unkron to get into the shape for output
VPath=reshape(VKronPath,[n_a,N_j,T]);
PolicyPath=UnKronPolicyIndexes_Case1_TransPathFHorz_noz(PolicyIndexesPath, n_d, n_a, N_j,T,vfoptions);



end