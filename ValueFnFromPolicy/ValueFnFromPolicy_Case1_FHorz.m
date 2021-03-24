function V=ValueFnFromPolicy_Case1_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% vfoptions is an optional input (not required)

if exist('vfoptions','var')==1
    if isfield(vfoptions,'polindorval')==1
        if vfoptions.polindorval==2
            fprintf('ERROR: ValueFnFromPolicy_Case1_FHorz() does not work with policy fn in form of policy values \n');
        end
    end
    if isfield(vfoptions,'exoticpreferences')==1
        if vfoptions.exoticpreferences>0
            fprintf('ERROR: ValueFnFromPolicy_Case1_FHorz() does not yet work with exotic preferences. Please email robertdkirkby@gmail.com if you want/need this feature and I will implement it. \n');
        end
    end
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

%% Evaluate Return Fn (this part is essentially copy-paste from the 'EvaluateFnOnAgentDist' commands)
PolicyValues=PolicyInd2Val_FHorz_Case1(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid, Parallel);
permuteindexes=[1+(1:1:(l_a+l_z)),1,1+l_a+l_z+1];
PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_z,l_d+l_a,N_j]

PolicyValuesPermuteVec=reshape(PolicyValuesPermute,[N_a*N_z*(l_d+l_a),N_j]);

FofPolicy=nan(N_a*N_z,N_j,'gpuArray');
for jj=1:N_j
    
    if fieldexists_ExogShockFn==1
        if fieldexists_ExogShockFnParamNames==1
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,N_j);
            [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsVec);
            z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
        else
            [z_grid,pi_z]=vfoptions.ExogShockFn(N_j);
            z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
        end
    end
    
    FnToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,ReturnFnParamNames,jj));
    FofPolicy(:,jj)=reshape(EvalFnOnAgentDist_Grid_Case1(ReturnFn, FnToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,n_z,l_d+l_a]),n_d,n_a,n_z,a_grid,z_grid,Parallel),[N_a*N_z,1]);
end

% The following will also be needed to calculate the expectation of next
% period value fn, evaluated based on the policy.
PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z,N_j);

%% Calculate the Value Fn by backward iteration
V=zeros(N_a,N_z,N_j,'gpuArray');

V(:,:,N_j)=reshape(FofPolicy(:,N_j),[N_a,N_z]);
for reverse_j=1:N_j-1
    jj=N_j-reverse_j; % current period, counts backwards from J-1
        
    if fieldexists_ExogShockFn==1
        if fieldexists_ExogShockFnParamNames==1
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,N_j);
            [~,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsVec);
%             z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
        else
            [~,pi_z]=vfoptions.ExogShockFn(N_j);
%             z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
        end
    end
    
    beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
    EVnext=sum(V(:,:,jj+1).*pi_z,2); % size N_z-by-1
    
    if N_d==0 %length(n_d)==1 && n_d(1)==0
        optaprime=PolicyIndexesKron(:,:,jj);
    else
        optaprime=shiftdim(PolicyIndexesKron(2,:,:,jj),1);
    end
    EVnextOfPolicy=EVnext(optaprime);
    
    V(:,:,jj)=reshape(FofPolicy(:,N_j),[N_a,N_z])+beta*EVnextOfPolicy;

end

%Transforming Value Fn out of Kronecker Form
V=reshape(V,[n_a,n_z,N_j]);

end