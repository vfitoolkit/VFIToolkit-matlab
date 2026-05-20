function V=ValueFnFromPolicy_FHorz_GI(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% gridinterplayer==1 variant of ValueFnFromPolicy_FHorz.
% Uses the L2 fine-grid index in Policy to build PolicyProbs over the two
% adjacent aprime grid points and weights the continuation value accordingly.

%% Exogenous shock grids
[z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3);

if prod(vfoptions.n_semiz)>0
    error('cannot yet handle semiz, ask on forum if you need/want')
end

%%
N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);
if N_d==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);

if l_a~=1
    error('ValueFnFromPolicy_FHorz_GI currently only handles length(n_a)==1; ask on forum if you need more')
end

%% Implement new way of handling ReturnFn inputs
ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);

%%
a_gridvals=CreateGridvals(n_a,a_grid,1);

% Slot index for the aprime lower index in the gridinterplayer==1 Kron'd Policy
aslot=1+(N_d>0); % 1 if no d, 2 if d

%%
if N_z==0 && N_e==0

    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,0,N_j,d_grid,a_grid,vfoptions,1);
    l_daprime=size(PolicyValues,1);
    PolicyValuesPermute=permute(PolicyValues,[2,1,3]); %[N_a,l_d+l_a,N_j]
    PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1_noz(Policy, n_d, n_a, N_j, vfoptions); % shape (aslot+1, N_a, N_j): aslot row is alower, end row is L2

    alower=reshape(PolicyIndexesKron(aslot,:,:),[N_a,N_j]);
    L2=reshape(PolicyIndexesKron(end,:,:),[N_a,N_j]);
    PolicyProbs=zeros(N_a,N_j,2,'gpuArray');
    PolicyProbs(:,:,2)=(L2-1)/(vfoptions.ngridinterp+1); % prob of upper grid point
    PolicyProbs(:,:,1)=1-PolicyProbs(:,:,2);

    %% Calculate the Value Fn by backward iteration
    V=zeros(N_a,N_j,'gpuArray');

    for reverse_j=0:N_j-1
        jj=N_j-reverse_j; % current period, counts backwards from J-1

        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,jj),l_daprime,n_a,0,a_gridvals,[]);

        if jj==N_j
            V(:,jj)=FofPolicy_jj;
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            EVnext=V(:,jj+1); % (N_a,1)
            EVnextAtPolicy=PolicyProbs(:,jj,1).*EVnext(alower(:,jj))+PolicyProbs(:,jj,2).*EVnext(alower(:,jj)+1);
            V(:,jj)=FofPolicy_jj+beta*EVnextAtPolicy;
        end
    end

    V=reshape(V,[n_a,N_j]);

elseif N_z==0 && N_e>0

    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1); % PolicyInd2Val auto-adds vfoptions.n_e
    l_daprime=size(PolicyValues,1);
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); %[N_a,N_e,l_d+l_a,N_j]
    PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, vfoptions.n_e, N_j, vfoptions); % shape (aslot+1, N_a, N_e, N_j)

    alower=reshape(PolicyIndexesKron(aslot,:,:,:),[N_a,N_e,N_j]);
    L2=reshape(PolicyIndexesKron(end,:,:,:),[N_a,N_e,N_j]);
    PolicyProbs=zeros(N_a,N_e,N_j,2,'gpuArray');
    PolicyProbs(:,:,:,2)=(L2-1)/(vfoptions.ngridinterp+1);
    PolicyProbs(:,:,:,1)=1-PolicyProbs(:,:,:,2);

    %% Calculate the Value Fn by backward iteration
    V=zeros(N_a,N_e,N_j,'gpuArray');

    for reverse_j=0:N_j-1
        jj=N_j-reverse_j;

        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,vfoptions.n_e,a_gridvals,vfoptions.e_gridvals_J(:,:,jj));

        if jj==N_j
            V(:,:,jj)=FofPolicy_jj;
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            EVnext=sum(V(:,:,jj+1).*shiftdim(vfoptions.pi_e_J(:,jj),-1),2); % (N_a,1) integrate over iid e
            % Look up at lower & upper aprime: result shape (N_a, N_e)
            EVlower=reshape(EVnext(alower(:,:,jj)),[N_a,N_e]);
            EVupper=reshape(EVnext(alower(:,:,jj)+1),[N_a,N_e]);
            EVnextAtPolicy=PolicyProbs(:,:,jj,1).*EVlower+PolicyProbs(:,:,jj,2).*EVupper;
            V(:,:,jj)=FofPolicy_jj+beta*EVnextAtPolicy;
        end
    end

    V=reshape(V,[n_a,vfoptions.n_e,N_j]);

elseif N_z>0 && N_e==0

    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
    l_daprime=size(PolicyValues,1);
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); %[N_a,N_z,l_d+l_a,N_j]
    PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z, N_j, vfoptions); % shape (aslot+1, N_a, N_z, N_j)

    alower=reshape(PolicyIndexesKron(aslot,:,:,:),[N_a,N_z,N_j]);
    L2=reshape(PolicyIndexesKron(end,:,:,:),[N_a,N_z,N_j]);
    PolicyProbs=zeros(N_a,N_z,N_j,2,'gpuArray');
    PolicyProbs(:,:,:,2)=(L2-1)/(vfoptions.ngridinterp+1);
    PolicyProbs(:,:,:,1)=1-PolicyProbs(:,:,:,2);

    %% Calculate the Value Fn by backward iteration
    V=zeros(N_a,N_z,N_j,'gpuArray');

    for reverse_j=0:N_j-1
        jj=N_j-reverse_j;

        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj));

        if jj==N_j
            V(:,:,jj)=FofPolicy_jj;
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            % EVnext(aprime, z) = sum_{zprime} pi_z(z,zprime) * V(aprime, zprime, jj+1)
            EVnext=V(:,:,jj+1)*pi_z_J(:,:,jj)'; % (N_a, N_z)
            EVnext(isnan(EVnext))=0;
            % Linear index into (N_a, N_z) at (alower(a,z), z) and (alower+1, z)
            zidxoffset=N_a*gpuArray(0:N_z-1); % (1, N_z)
            lower_lin=alower(:,:,jj)+zidxoffset;
            EVnextAtPolicy=PolicyProbs(:,:,jj,1).*EVnext(lower_lin)+PolicyProbs(:,:,jj,2).*EVnext(lower_lin+1);
            V(:,:,jj)=FofPolicy_jj+beta*EVnextAtPolicy;
        end
    end

    V=reshape(V,[n_a,n_z,N_j]);

elseif N_z>0 && N_e>0

    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1); % PolicyInd2Val auto-adds vfoptions.n_e
    l_daprime=size(PolicyValues,1);
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a,N_z*N_e,l_daprime,N_j] — keep shock dim combined for EvalFnOnAgentDist_Grid
    PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1_e(Policy, n_d, n_a, n_z, vfoptions.n_e, N_j, vfoptions); % shape (aslot+1, N_a, N_z, N_e, N_j)

    alower=reshape(PolicyIndexesKron(aslot,:,:,:,:),[N_a,N_z,N_e,N_j]);
    L2=reshape(PolicyIndexesKron(end,:,:,:,:),[N_a,N_z,N_e,N_j]);
    PolicyProbs=zeros(N_a,N_z,N_e,N_j,2,'gpuArray');
    PolicyProbs(:,:,:,:,2)=(L2-1)/(vfoptions.ngridinterp+1);
    PolicyProbs(:,:,:,:,1)=1-PolicyProbs(:,:,:,:,2);

    %% Calculate the Value Fn by backward iteration
    V=zeros(N_a,N_z,N_e,N_j,'gpuArray');

    for reverse_j=0:N_j-1
        jj=N_j-reverse_j;

        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        FofPolicy_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,[n_z,vfoptions.n_e],a_gridvals,[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)]),[N_a,N_z,N_e]);

        if jj==N_j
            V(:,:,:,jj)=FofPolicy_jj;
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            % Integrate over iid e then over zprime|z
            EVnext=sum(V(:,:,:,jj+1).*shiftdim(vfoptions.pi_e_J(:,jj),-2),3); % (N_a, N_z)
            EVnext=EVnext*pi_z_J(:,:,jj)'; % (N_a, N_z)
            EVnext(isnan(EVnext))=0;
            % For each (a, z, e), look up EVnext at (alower(a,z,e), z) and (alower+1, z)
            zidxoffset=N_a*gpuArray(0:N_z-1); % (1, N_z)
            lower_lin=alower(:,:,:,jj)+zidxoffset; % (N_a, N_z, N_e) — broadcasting
            EVnextAtPolicy=PolicyProbs(:,:,:,jj,1).*EVnext(lower_lin)+PolicyProbs(:,:,:,jj,2).*EVnext(lower_lin+1);
            V(:,:,:,jj)=FofPolicy_jj+beta*EVnextAtPolicy;
        end
    end

    V=reshape(V,[n_a,n_z,vfoptions.n_e,N_j]);
end


end
