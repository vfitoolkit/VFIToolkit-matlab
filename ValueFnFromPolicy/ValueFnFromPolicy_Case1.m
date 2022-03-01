function V=ValueFnFromPolicy_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParams, vfoptions)

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
    if ~isfield(vfoptions,'parallel')
        vfoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    end
    if ~isfield(vfoptions,'tolerance')
        vfoptions.tolerance=10^(-9);
    end
else
    vfoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    vfoptions.tolerance=10^(-9);
end

%% Evaluate Return Fn (this part is essentially copy-paste from the 'EvaluateFnOnAgentDist' commands)
if vfoptions.parallel==2
    Parallel=2;
    PolicyIndexes=gpuArray(PolicyIndexes);
    n_d=gpuArray(n_d);
    n_a=gpuArray(n_a);
    n_z=gpuArray(n_z);
    d_grid=gpuArray(d_grid);
    a_grid=gpuArray(a_grid);
    z_grid=gpuArray(z_grid);
    
    % l_d not needed with Parallel=2 implementation
    l_a=length(n_a);
    l_z=length(n_z);
    
    N_a=prod(n_a);
    N_z=prod(n_z);
    
    %% Calculate FofPolicy (the return fn evaluated at the Policy)
    
    PolicyValues=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid, Parallel);
    permuteindexes=[1+(1:1:(l_a+l_z)),1];    
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParams);

    FofPolicy=EvalFnOnAgentDist_Grid_Case1(ReturnFn, ReturnFnParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel);
    
    %% Now that we have FofPolicy, calculate V.
    bbb=reshape(shiftdim(pi_z,-1),[1,N_z*N_z]);
    ccc=kron(ones(N_a,1,'gpuArray'),bbb);
    aaa=reshape(ccc,[N_a*N_z,N_z]);

    beta=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames));
    
    currdist=Inf;
    VKron=FofPolicy;
    
    if length(n_d)>1 || length(n_a)>1
        PolicyIndexes=KronPolicyIndexes_Case1(PolicyIndexes, n_d, n_a, n_z);
    end
    
    if n_d(1)>0
%         N_d=prod(n_d);
%         aprimeindexes=ceil(PolicyIndexes/N_d);
        aprimeindexes=shiftdim(ceil(PolicyIndexes(2,:,:)),1);
        while currdist>vfoptions.tolerance
            VKronold=VKron;
            
            EVKrontemp=VKron(aprimeindexes,:);
                        
            EVKrontemp=EVKrontemp.*aaa;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=FofPolicy+beta*EVKrontemp;
            
            currdist=max(max(abs(VKron-VKronold)));
        end
    else
        % No d variable
         while currdist>vfoptions.tolerance
            VKronold=VKron;
            
            EVKrontemp=VKron(PolicyIndexes,:);
            
            EVKrontemp=EVKrontemp.*aaa;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=FofPolicy+beta*EVKrontemp;
            
            currdist=max(max(abs(VKron-VKronold)));
        end
    end

    
else
    if n_d(1)==0
        l_d=0;
    else
        l_d=length(n_d);
    end
    
    N_a=prod(n_a);
    N_z=prod(n_z);
    
    %% Calculate FofPolicy (the return fn evaluated at the Policy)
    [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes,n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
    a_gridvals=CreateGridvals(n_a,a_grid,2);
    z_gridvals=CreateGridvals(n_z,z_grid,2);

    if l_d>0
        ReturnFnParamsCell=num2cell(CreateVectorFromParams(Parameters,ReturnFnParams));
        FofPolicy=zeros(N_a*N_z,1);
        for ii=1:N_a*N_z
            %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
            j1=rem(ii-1,N_a)+1;
            j2=ceil(ii/N_a);
            FofPolicy(ii)=ReturnFn(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},ReturnFnParamsCell{:});
        end
    else %l_d=0
        ReturnFnParamsCell=num2cell(CreateVectorFromParams(Parameters,ReturnFnParams));
        FofPolicy=zeros(N_a*N_z,1);
        for ii=1:N_a*N_z
            %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
            j1=rem(ii-1,N_a)+1;
            j2=ceil(ii/N_a);
            FofPolicy(ii)=ReturnFn(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},ReturnFnParamsCell{:});
        end
    end
    
    %% Now that we have FofPolicy, calculate V.
    bbb=reshape(shiftdim(pi_z,-1),[1,N_z*N_z]);
    ccc=kron(ones(N_a,1),bbb);
    aaa=reshape(ccc,[N_a*N_z,N_z]);
    
    beta=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames));

    if length(n_d)>1 || length(n_a)>1
        PolicyIndexes=KronPolicyIndexes_Case1(PolicyIndexes, n_d, n_a, n_z);
    end
    
    currdist=Inf;
    VKron=FofPolicy;
    if n_d(1)>0
%         N_d=prod(n_d);
%         aprimeindexes=ceil(PolicyIndexes/N_d);
        aprimeindexes=shiftdim(ceil(PolicyIndexes(2,:,:)),1);
        while currdist>vfoptions.tolerance
            VKronold=VKron;
            
            EVKrontemp=VKron(aprimeindexes,:);
            
            EVKrontemp=EVKrontemp.*aaa;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=FofPolicy+beta*EVKrontemp;
            
            currdist=max(max(abs(VKron-VKronold)));
        end
    else
        % No d variable
         while currdist>vfoptions.tolerance
            VKronold=VKron;
            
            EVKrontemp=VKron(PolicyIndexes,:);
            
            EVKrontemp=EVKrontemp.*aaa;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=FofPolicy+beta*EVKrontemp;
            
            currdist=max(max(abs(VKron-VKronold)));
        end
    end
    
end

V=reshape(VKron,[n_a,n_z]);

end