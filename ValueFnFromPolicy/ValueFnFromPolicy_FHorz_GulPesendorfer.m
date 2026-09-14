function V=ValueFnFromPolicy_FHorz_GulPesendorfer(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_gridvals_J,pi_z_J,ReturnFn,Parameters,DiscountFactorParamNames,vfoptions)
% Gul-Pesendorfer variant of ValueFnFromPolicy_FHorz: values the given Policy under
%   V_j = u(policy_j) + v(policy_j) - MostTempting_j + beta*E[V_{j+1} at policy_j]
% (no continuation term at j=N_j), where u is the ReturnFn, v is the temptation fn
% (vfoptions.temptationFn, same input signature convention as the ReturnFn, own parameters),
% and MostTempting_j(a,z) is the max of v over the FULL joint (d,aprime) choice set.
%
% Under vfoptions.gridinterplayer==1 the choice set is the FINE aprime grid, so MostTempting
% is the fine-grid max of v, found by the same two-stage scheme as in the GP GI solver raws:
% around v's OWN coarse argmax (otherwise the chosen fine point could be more tempting than
% the coarse max of v, making the self-control cost negative).
%
% This is dispatched from ValueFnFromPolicy_FHorz AFTER ExogShockSetup_FHorz has run, so
% z_gridvals_J/pi_z_J and vfoptions.e_gridvals_J/pi_e_J arrive pre-processed (vfoptions.n_e
% exists, 0-equivalent when there are no e variables). Handles gridinterplayer itself.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

if ~isfield(vfoptions,'temptationFn')
    error('When using Gul-Pesendorfer preferences you must declare vfoptions.temptationFn (the temptation function)')
end
if prod(vfoptions.n_semiz)>0
    % Dispatch to the SemiExo subfn (handles gridinterplayer itself). z/e arrive pre-processed
    % (the parent ValueFnFromPolicy_FHorz ran ExogShockSetup_FHorz); the subfn runs only the
    % semiz setup (SemiExogShockSetup_FHorz) internally.
    V=ValueFnFromPolicy_FHorz_GulPesendorfer_SemiExo(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_gridvals_J,pi_z_J,ReturnFn,Parameters,DiscountFactorParamNames,vfoptions);
    return
end
if vfoptions.experienceasset>=1
    % Dispatch to the ExpAsset subfn (handles gridinterplayer itself): the continuation is at
    % a2prime=aprimeFn(d2,a2), so the policy-lookup machinery differs from the standard case.
    V=ValueFnFromPolicy_FHorz_GulPesendorfer_ExpAsset(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_gridvals_J,pi_z_J,ReturnFn,Parameters,DiscountFactorParamNames,vfoptions);
    return
end
if vfoptions.experienceassetu>=1 || vfoptions.experienceassetz>=1 || vfoptions.experienceassete>=1 || vfoptions.experienceassetze>=1 || vfoptions.experienceassetsemiz>=1
    error('ValueFnFromPolicy: GulPesendorfer is not implemented for the u/z/e/ze/semiz experience-asset variants')
end
if vfoptions.gridinterplayer==1 && length(n_a)>2
    error('GulPesendorfer with gridinterplayer is not implemented for more than two standard endogenous states')
end

TemptationFn=vfoptions.temptationFn;

%% Implement new way of handling ReturnFn inputs
ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);
TemptationFnParamNames=ReturnFnParamNamesFn(TemptationFn,n_d,n_a,n_z,N_j,vfoptions,Parameters); % the temptation fn has the same leading model args as the return fn, so the same convention applies

%%
a_gridvals=CreateGridvals(n_a,a_grid,1);
if N_d>0
    d_gridvals=CreateGridvals(n_d,d_grid,1); % the temptation matrix creators need gridvals
end

%%
if vfoptions.gridinterplayer==1
    % Slot index for the aprime lower index in the gridinterplayer==1 Kron'd Policy
    index_a1=1+(N_d>0); % 1 if no d, 2 if d
    l_a=length(n_a);
    n_a1=n_a(1);
    % GI2A (l_a>=2): interpolation is applied to the first endogenous state only, so
    % PolicyIndexesKron carries a separate a2prime row (index_a1+1). We fold the a2prime offset
    % straight into alower, turning it into a linear index into (N_a1*N_a2); alower+1 then still
    % steps a1 by one, so every lookup below is identical to the l_a==1 case.

    % Grid interpolation
    n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
    n2long=2*n2short+3; % total number of aprime points we end up looking at in second layer
    if l_a==1
        aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
    else % GI2A: the interpolation layer applies to a1prime only (the MostTempting two-stage uses the fine a1prime grid jointly with a2prime, as in the GP GI2A solver raws)
        a1_grid=a_grid(1:n_a1);
        a2_grid=a_grid(n_a1+1:end);
        a1prime_grid=interp1(1:1:n_a1,a1_grid,linspace(1,n_a1,n_a1+(n_a1-1)*n2short))';
    end

    if N_z==0 && N_e==0

        PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,0,N_j,d_grid,a_grid,vfoptions,1);
        l_daprime=size(PolicyValues,1);
        PolicyValuesPermute=permute(PolicyValues,[2,1,3]); %[N_a,l_d+l_a,N_j]
        PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, 1, N_j, vfoptions); % rows: alower (=index_a1), L2 (=end). L2flag dropped by Kron.

        alower=reshape(PolicyIndexesKron(index_a1,:,:),[N_a,N_j]);
        L2=reshape(PolicyIndexesKron(end,:,:),[N_a,N_j]);
        if l_a>=2 % GI2A: fold a2prime into the linear index
            a2prime=reshape(PolicyIndexesKron(index_a1+1,:,:),[N_a,N_j]);
            alower=alower+n_a1*(a2prime-1);
        end
        PolicyProbs=zeros(N_a,N_j,2,'gpuArray');
        PolicyProbs(:,:,2)=(L2-1)/(vfoptions.ngridinterp+1); % prob of upper grid point
        PolicyProbs(:,:,1)=1-PolicyProbs(:,:,2);

        %% Calculate the Value Fn by backward iteration
        V=zeros(N_a,N_j,'gpuArray');

        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;

            FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
            FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,jj),l_daprime,n_a,0,a_gridvals,[]);
            TemptationFnParamsCell=CreateCellFromParams(Parameters,TemptationFnParamNames,jj);
            TofPolicy_jj=EvalFnOnAgentDist_Grid(TemptationFn, TemptationFnParamsCell,PolicyValuesPermute(:,:,jj),l_daprime,n_a,0,a_gridvals,[]);

            TemptationFnParamsVec=CreateVectorFromParams(Parameters,TemptationFnParamNames,jj);
            if N_d==0 && l_a==1
                TemptationMatrix=CreateReturnFnMatrix_Disc_noz(TemptationFn, 0, n_a, 0, a_grid, TemptationFnParamsVec,0);
                % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
                [~,maxindexT]=max(TemptationMatrix,[],1);
                midpointT=max(min(maxindexT,n_a-1),2);
                aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short)';
                TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1_nod_noz(TemptationFn,aprime_grid(aprimeindexesT),a_grid,TemptationFnParamsVec);
                MostTempting=max(TemptationMatrix_Tii,[],1);
            elseif N_d>0 && l_a==1
                TemptationMatrix=CreateReturnFnMatrix_Disc_noz(TemptationFn, n_d, n_a, d_gridvals, a_grid, TemptationFnParamsVec,1);
                % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
                [~,maxindexT]=max(TemptationMatrix,[],2);
                midpointT=max(min(maxindexT,n_a-1),2);
                aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
                TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1_noz(TemptationFn,n_d,d_gridvals,aprime_grid(aprimeindexesT),a_grid,TemptationFnParamsVec,2);
                MostTempting=max(TemptationMatrix_Tii,[],1);
            elseif N_d==0 % l_a==2: GI2A, fine a1prime jointly with a2prime (as in the GP GI2A solver raws)
                TemptationMatrix=CreateReturnFnMatrix_Disc_DC2A_nod_noz(TemptationFn, a1_grid, a2_grid, a1_grid, a2_grid, TemptationFnParamsVec,1);
                % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
                [~,maxindexT]=max(TemptationMatrix,[],1);
                midpointT=max(min(maxindexT,n_a1-1),2);
                a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short)';
                TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC2A_nod_noz(TemptationFn,a1prime_grid(a1primeindexesT),a2_grid,a1_grid,a2_grid,TemptationFnParamsVec,2);
                MostTempting=max(TemptationMatrix_Tii,[],1); % max over the joint (fine a1prime window, a2prime) first dim
            else % N_d>0, l_a==2: GI2A, fine a1prime jointly with d and a2prime (as in the GP GI2A solver raws)
                TemptationMatrix=CreateReturnFnMatrix_Disc_DC2A_noz(TemptationFn,n_d,d_gridvals,a1_grid, a2_grid, a1_grid, a2_grid, TemptationFnParamsVec,1,0);
                % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
                [~,maxindexT]=max(TemptationMatrix,[],2);
                midpointT=max(min(maxindexT,n_a1-1),2);
                a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
                TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC2A_noz(TemptationFn,n_d,d_gridvals,a1prime_grid(a1primeindexesT),a2_grid,a1_grid,a2_grid,TemptationFnParamsVec,2,0);
                MostTempting=max(TemptationMatrix_Tii,[],1); % max over the joint (d, fine a1prime window, a2prime) first dim
            end
            MostTempting=reshape(MostTempting,[N_a,1]);

            if jj==N_j
                V(:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting;
            else
                beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
                EVnext=V(:,jj+1); % (N_a,1)
                EVnextAtPolicy=PolicyProbs(:,jj,1).*EVnext(alower(:,jj))+PolicyProbs(:,jj,2).*EVnext(alower(:,jj)+1);
                EVnextAtPolicy(isnan(EVnextAtPolicy))=0; % zero corner weights times -Inf next-states give NaN
                V(:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting+beta*EVnextAtPolicy;
            end
        end

        V=reshape(V,[n_a,N_j]);

    elseif N_z==0 && N_e>0

        PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1); % PolicyInd2Val auto-adds vfoptions.n_e
        l_daprime=size(PolicyValues,1);
        PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); %[N_a,N_e,l_d+l_a,N_j]
        PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, vfoptions.n_e, N_j, vfoptions); % rows: alower (=index_a1), L2 (=end). L2flag dropped by Kron.

        alower=reshape(PolicyIndexesKron(index_a1,:,:,:),[N_a,N_e,N_j]);
        L2=reshape(PolicyIndexesKron(end,:,:,:),[N_a,N_e,N_j]);
        if l_a>=2 % GI2A: fold a2prime into the linear index
            a2prime=reshape(PolicyIndexesKron(index_a1+1,:,:,:),[N_a,N_e,N_j]);
            alower=alower+n_a1*(a2prime-1);
        end
        PolicyProbs=zeros(N_a,N_e,N_j,2,'gpuArray');
        PolicyProbs(:,:,:,2)=(L2-1)/(vfoptions.ngridinterp+1);
        PolicyProbs(:,:,:,1)=1-PolicyProbs(:,:,:,2);

        %% Calculate the Value Fn by backward iteration
        V=zeros(N_a,N_e,N_j,'gpuArray');

        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;

            FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
            FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,vfoptions.n_e,a_gridvals,vfoptions.e_gridvals_J(:,:,jj));
            TemptationFnParamsCell=CreateCellFromParams(Parameters,TemptationFnParamNames,jj);
            TofPolicy_jj=EvalFnOnAgentDist_Grid(TemptationFn, TemptationFnParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,vfoptions.n_e,a_gridvals,vfoptions.e_gridvals_J(:,:,jj));

            TemptationFnParamsVec=CreateVectorFromParams(Parameters,TemptationFnParamNames,jj);
            if N_d==0 && l_a==1
                TemptationMatrix=CreateReturnFnMatrix_Disc(TemptationFn, 0, n_a, vfoptions.n_e, 0, a_grid, vfoptions.e_gridvals_J(:,:,jj), TemptationFnParamsVec,0); % Because no z, can treat e like z and call Par2 rather than Par2e
                % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
                [~,maxindexT]=max(TemptationMatrix,[],1);
                midpointT=max(min(maxindexT,n_a-1),2);
                aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short)';
                TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1_nod(TemptationFn,vfoptions.n_e,aprime_grid(aprimeindexesT),a_grid,vfoptions.e_gridvals_J(:,:,jj),TemptationFnParamsVec,2);
                MostTempting=max(TemptationMatrix_Tii,[],1);
            elseif N_d>0 && l_a==1
                TemptationMatrix=CreateReturnFnMatrix_Disc(TemptationFn, n_d, n_a, vfoptions.n_e, d_gridvals, a_grid, vfoptions.e_gridvals_J(:,:,jj), TemptationFnParamsVec,1); % Because no z, can treat e like z and call Par2 rather than Par2e
                % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
                [~,maxindexT]=max(TemptationMatrix,[],2);
                midpointT=max(min(maxindexT,n_a-1),2);
                aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
                TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1(TemptationFn,n_d,vfoptions.n_e,d_gridvals,aprime_grid(aprimeindexesT),a_grid,vfoptions.e_gridvals_J(:,:,jj),TemptationFnParamsVec,2);
                MostTempting=max(TemptationMatrix_Tii,[],1);
            elseif N_d==0 % l_a==2: GI2A, fine a1prime jointly with a2prime (as in the GP GI2A solver raws)
                TemptationMatrix=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn,vfoptions.n_e, a1_grid, a2_grid, a1_grid, a2_grid, vfoptions.e_gridvals_J(:,:,jj), TemptationFnParamsVec,1); % Because no z, can treat e like z
                % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
                [~,maxindexT]=max(TemptationMatrix,[],1);
                midpointT=max(min(maxindexT,n_a1-1),2);
                a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short)';
                TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn,vfoptions.n_e,a1prime_grid(a1primeindexesT),a2_grid,a1_grid,a2_grid,vfoptions.e_gridvals_J(:,:,jj),TemptationFnParamsVec,2);
                MostTempting=max(TemptationMatrix_Tii,[],1); % max over the joint (fine a1prime window, a2prime) first dim
            else % N_d>0, l_a==2: GI2A, fine a1prime jointly with d and a2prime (as in the GP GI2A solver raws)
                TemptationMatrix=CreateReturnFnMatrix_Disc_DC2A(TemptationFn,n_d,vfoptions.n_e,d_gridvals,a1_grid, a2_grid, a1_grid, a2_grid, vfoptions.e_gridvals_J(:,:,jj), TemptationFnParamsVec,1,0); % Because no z, can treat e like z
                % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
                [~,maxindexT]=max(TemptationMatrix,[],2);
                midpointT=max(min(maxindexT,n_a1-1),2);
                a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
                TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC2A(TemptationFn,n_d,vfoptions.n_e,d_gridvals,a1prime_grid(a1primeindexesT),a2_grid,a1_grid,a2_grid,vfoptions.e_gridvals_J(:,:,jj),TemptationFnParamsVec,2,0);
                MostTempting=max(TemptationMatrix_Tii,[],1); % max over the joint (d, fine a1prime window, a2prime) first dim
            end
            MostTempting=reshape(MostTempting,[N_a,N_e]);

            if jj==N_j
                V(:,:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting;
            else
                beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
                EVw=V(:,:,jj+1).*shiftdim(vfoptions.pi_e_J(:,jj+1),-1); EVw(isnan(EVw))=0; % a zero weight against an infinite node gives 0*(-Inf)=NaN, so zero the terms BEFORE summing
                EVnext=sum(EVw,2); % (N_a,1) integrate over iid e
                % Look up at lower & upper aprime: result shape (N_a, N_e)
                EVlower=reshape(EVnext(alower(:,:,jj)),[N_a,N_e]);
                EVupper=reshape(EVnext(alower(:,:,jj)+1),[N_a,N_e]);
                EVnextAtPolicy=PolicyProbs(:,:,jj,1).*EVlower+PolicyProbs(:,:,jj,2).*EVupper;
                EVnextAtPolicy(isnan(EVnextAtPolicy))=0; % zero corner weights times -Inf next-states give NaN
                V(:,:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting+beta*EVnextAtPolicy;
            end
        end

        V=reshape(V,[n_a,vfoptions.n_e,N_j]);

    elseif N_z>0 && N_e==0

        PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
        l_daprime=size(PolicyValues,1);
        PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); %[N_a,N_z,l_d+l_a,N_j]
        PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, n_z, N_j, vfoptions); % rows: alower (=index_a1), L2 (=end). L2flag dropped by Kron.

        alower=reshape(PolicyIndexesKron(index_a1,:,:,:),[N_a,N_z,N_j]);
        L2=reshape(PolicyIndexesKron(end,:,:,:),[N_a,N_z,N_j]);
        if l_a>=2 % GI2A: fold a2prime into the linear index
            a2prime=reshape(PolicyIndexesKron(index_a1+1,:,:,:),[N_a,N_z,N_j]);
            alower=alower+n_a1*(a2prime-1);
        end
        PolicyProbs=zeros(N_a,N_z,N_j,2,'gpuArray');
        PolicyProbs(:,:,:,2)=(L2-1)/(vfoptions.ngridinterp+1);
        PolicyProbs(:,:,:,1)=1-PolicyProbs(:,:,:,2);

        %% Calculate the Value Fn by backward iteration
        V=zeros(N_a,N_z,N_j,'gpuArray');

        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;

            FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
            FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj));
            TemptationFnParamsCell=CreateCellFromParams(Parameters,TemptationFnParamNames,jj);
            TofPolicy_jj=EvalFnOnAgentDist_Grid(TemptationFn, TemptationFnParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj));

            TemptationFnParamsVec=CreateVectorFromParams(Parameters,TemptationFnParamNames,jj);
            if N_d==0 && l_a==1
                TemptationMatrix=CreateReturnFnMatrix_Disc(TemptationFn, 0, n_a, n_z, 0, a_grid, z_gridvals_J(:,:,jj), TemptationFnParamsVec,0);
                % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
                [~,maxindexT]=max(TemptationMatrix,[],1);
                midpointT=max(min(maxindexT,n_a-1),2);
                aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short)';
                TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1_nod(TemptationFn,n_z,aprime_grid(aprimeindexesT),a_grid,z_gridvals_J(:,:,jj),TemptationFnParamsVec,2);
                MostTempting=max(TemptationMatrix_Tii,[],1);
            elseif N_d>0 && l_a==1
                TemptationMatrix=CreateReturnFnMatrix_Disc(TemptationFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals_J(:,:,jj), TemptationFnParamsVec,1);
                % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
                [~,maxindexT]=max(TemptationMatrix,[],2);
                midpointT=max(min(maxindexT,n_a-1),2);
                aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
                TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1(TemptationFn,n_d,n_z,d_gridvals,aprime_grid(aprimeindexesT),a_grid,z_gridvals_J(:,:,jj),TemptationFnParamsVec,2);
                MostTempting=max(TemptationMatrix_Tii,[],1);
            elseif N_d==0 % l_a==2: GI2A, fine a1prime jointly with a2prime (as in the GP GI2A solver raws)
                TemptationMatrix=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn,n_z, a1_grid, a2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,jj), TemptationFnParamsVec,1);
                % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
                [~,maxindexT]=max(TemptationMatrix,[],1);
                midpointT=max(min(maxindexT,n_a1-1),2);
                a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short)';
                TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn,n_z,a1prime_grid(a1primeindexesT),a2_grid,a1_grid,a2_grid,z_gridvals_J(:,:,jj),TemptationFnParamsVec,2);
                MostTempting=max(TemptationMatrix_Tii,[],1); % max over the joint (fine a1prime window, a2prime) first dim
            else % N_d>0, l_a==2: GI2A, fine a1prime jointly with d and a2prime (as in the GP GI2A solver raws)
                TemptationMatrix=CreateReturnFnMatrix_Disc_DC2A(TemptationFn,n_d,n_z,d_gridvals,a1_grid, a2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,jj), TemptationFnParamsVec,1,0);
                % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
                [~,maxindexT]=max(TemptationMatrix,[],2);
                midpointT=max(min(maxindexT,n_a1-1),2);
                a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
                TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC2A(TemptationFn,n_d,n_z,d_gridvals,a1prime_grid(a1primeindexesT),a2_grid,a1_grid,a2_grid,z_gridvals_J(:,:,jj),TemptationFnParamsVec,2,0);
                MostTempting=max(TemptationMatrix_Tii,[],1); % max over the joint (d, fine a1prime window, a2prime) first dim
            end
            MostTempting=reshape(MostTempting,[N_a,N_z]);

            if jj==N_j
                V(:,:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting;
            else
                beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
                % EVnext(aprime, z) = sum_{zprime} pi_z(z,zprime) * V(aprime, zprime, jj+1)
                EVnext=V(:,:,jj+1)*pi_z_J(:,:,jj)'; % (N_a, N_z)
                EVnext(isnan(EVnext))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                % Linear index into (N_a, N_z) at (alower(a,z), z) and (alower+1, z)
                zidxoffset=N_a*gpuArray(0:N_z-1); % (1, N_z)
                lower_lin=alower(:,:,jj)+zidxoffset;
                EVnextAtPolicy=PolicyProbs(:,:,jj,1).*EVnext(lower_lin)+PolicyProbs(:,:,jj,2).*EVnext(lower_lin+1);
                EVnextAtPolicy(isnan(EVnextAtPolicy))=0; % zero corner weights times -Inf next-states give NaN
                V(:,:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting+beta*EVnextAtPolicy;
            end
        end

        V=reshape(V,[n_a,n_z,N_j]);

    else % N_z>0 && N_e>0

        PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1); % PolicyInd2Val auto-adds vfoptions.n_e
        l_daprime=size(PolicyValues,1);
        PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a,N_z*N_e,l_daprime,N_j] — keep shock dim combined for EvalFnOnAgentDist_Grid
        PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, [n_z,vfoptions.n_e], N_j, vfoptions); % rows: alower (=index_a1), L2 (=end). L2flag dropped by Kron.

        alower=reshape(PolicyIndexesKron(index_a1,:,:,:),[N_a,N_z,N_e,N_j]);
        L2=reshape(PolicyIndexesKron(end,:,:,:),[N_a,N_z,N_e,N_j]);
        if l_a>=2 % GI2A: fold a2prime into the linear index
            a2prime=reshape(PolicyIndexesKron(index_a1+1,:,:,:),[N_a,N_z,N_e,N_j]);
            alower=alower+n_a1*(a2prime-1);
        end
        PolicyProbs=zeros(N_a,N_z,N_e,N_j,2,'gpuArray');
        PolicyProbs(:,:,:,:,2)=(L2-1)/(vfoptions.ngridinterp+1);
        PolicyProbs(:,:,:,:,1)=1-PolicyProbs(:,:,:,:,2);

        %% Calculate the Value Fn by backward iteration
        V=zeros(N_a,N_z,N_e,N_j,'gpuArray');

        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;

            FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
            FofPolicy_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,[n_z,vfoptions.n_e],a_gridvals,[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)]),[N_a,N_z,N_e]);
            TemptationFnParamsCell=CreateCellFromParams(Parameters,TemptationFnParamNames,jj);
            TofPolicy_jj=reshape(EvalFnOnAgentDist_Grid(TemptationFn, TemptationFnParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,[n_z,vfoptions.n_e],a_gridvals,[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)]),[N_a,N_z,N_e]);

            TemptationFnParamsVec=CreateVectorFromParams(Parameters,TemptationFnParamNames,jj);
            if N_d==0 && l_a==1
                TemptationMatrix=CreateReturnFnMatrix_Disc_e(TemptationFn, 0, n_a, n_z, vfoptions.n_e, 0, a_grid, z_gridvals_J(:,:,jj), vfoptions.e_gridvals_J(:,:,jj), TemptationFnParamsVec,0);
                % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
                [~,maxindexT]=max(TemptationMatrix,[],1);
                midpointT=max(min(maxindexT,n_a-1),2);
                aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short)';
                TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1_nod_e(TemptationFn,n_z,vfoptions.n_e,aprime_grid(aprimeindexesT),a_grid,z_gridvals_J(:,:,jj),vfoptions.e_gridvals_J(:,:,jj),TemptationFnParamsVec,2);
                MostTempting=max(TemptationMatrix_Tii,[],1);
            elseif N_d>0 && l_a==1
                TemptationMatrix=CreateReturnFnMatrix_Disc_e(TemptationFn, n_d, n_a, n_z, vfoptions.n_e, d_gridvals, a_grid, z_gridvals_J(:,:,jj), vfoptions.e_gridvals_J(:,:,jj), TemptationFnParamsVec,1);
                % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
                [~,maxindexT]=max(TemptationMatrix,[],2);
                midpointT=max(min(maxindexT,n_a-1),2);
                aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
                TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1_e(TemptationFn,n_d,n_z,vfoptions.n_e,d_gridvals,aprime_grid(aprimeindexesT),a_grid,z_gridvals_J(:,:,jj),vfoptions.e_gridvals_J(:,:,jj),TemptationFnParamsVec,2);
                MostTempting=max(TemptationMatrix_Tii,[],1);
            elseif N_d==0 % l_a==2: GI2A, fine a1prime jointly with a2prime (as in the GP GI2A solver raws)
                TemptationMatrix=CreateReturnFnMatrix_Disc_DC2A_nod_e(TemptationFn,n_z,vfoptions.n_e, a1_grid, a2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,jj), vfoptions.e_gridvals_J(:,:,jj), TemptationFnParamsVec,1);
                % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
                [~,maxindexT]=max(TemptationMatrix,[],1);
                midpointT=max(min(maxindexT,n_a1-1),2);
                a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short)';
                TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC2A_nod_e(TemptationFn,n_z,vfoptions.n_e,a1prime_grid(a1primeindexesT),a2_grid,a1_grid,a2_grid,z_gridvals_J(:,:,jj),vfoptions.e_gridvals_J(:,:,jj),TemptationFnParamsVec,2);
                MostTempting=max(TemptationMatrix_Tii,[],1); % max over the joint (fine a1prime window, a2prime) first dim
            else % N_d>0, l_a==2: GI2A, fine a1prime jointly with d and a2prime (as in the GP GI2A solver raws)
                TemptationMatrix=CreateReturnFnMatrix_Disc_DC2A_e(TemptationFn,n_d,n_z,vfoptions.n_e,d_gridvals,a1_grid, a2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,jj), vfoptions.e_gridvals_J(:,:,jj), TemptationFnParamsVec,1,0);
                % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
                [~,maxindexT]=max(TemptationMatrix,[],2);
                midpointT=max(min(maxindexT,n_a1-1),2);
                a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
                TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC2A_e(TemptationFn,n_d,n_z,vfoptions.n_e,d_gridvals,a1prime_grid(a1primeindexesT),a2_grid,a1_grid,a2_grid,z_gridvals_J(:,:,jj),vfoptions.e_gridvals_J(:,:,jj),TemptationFnParamsVec,2,0);
                MostTempting=max(TemptationMatrix_Tii,[],1); % max over the joint (d, fine a1prime window, a2prime) first dim
            end
            MostTempting=reshape(MostTempting,[N_a,N_z,N_e]);

            if jj==N_j
                V(:,:,:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting;
            else
                beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
                % Integrate over iid e, then over zprime|z
                EVw=V(:,:,:,jj+1).*shiftdim(vfoptions.pi_e_J(:,jj+1),-2); EVw(isnan(EVw))=0; % a zero weight against an infinite node gives 0*(-Inf)=NaN, so zero the terms BEFORE summing
                EVnext=sum(EVw,3); % (N_a, N_z)
                EVnext=EVnext*pi_z_J(:,:,jj)'; % (N_a, N_z)
                EVnext(isnan(EVnext))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                % For each (a, z, e), look up the EV at (alower(a,z,e), z) and (alower+1, z)
                zidxoffset=N_a*gpuArray(0:N_z-1); % (1, N_z)
                lower_lin=alower(:,:,:,jj)+zidxoffset; % (N_a, N_z, N_e) — broadcasting
                EVnextAtPolicy=PolicyProbs(:,:,:,jj,1).*EVnext(lower_lin)+PolicyProbs(:,:,:,jj,2).*EVnext(lower_lin+1);
                EVnextAtPolicy(isnan(EVnextAtPolicy))=0; % zero corner weights times -Inf next-states give NaN
                V(:,:,:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting+beta*EVnextAtPolicy;
            end
        end

        V=reshape(V,[n_a,n_z,vfoptions.n_e,N_j]);
    end

else % no grid interpolation layer

    if N_z==0 && N_e==0

        PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,0,N_j,d_grid,a_grid,vfoptions,1);
        l_daprime=size(PolicyValues,1);
        PolicyValuesPermute=permute(PolicyValues,[2,1,3]); %[N_a,l_d+l_a,N_j]
        % The following will also be needed to calculate the expectation of next period value fn, evaluated based on the policy.
        PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, 1, N_j, vfoptions);

        %% Calculate the Value Fn by backward iteration
        V=zeros(N_a,N_j,'gpuArray');

        for reverse_j=0:N_j-1
            jj=N_j-reverse_j; % current period, counts backwards from J-1

            % Evaluate Return Fn and Temptation Fn
            FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
            FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,jj),l_daprime,n_a,0,a_gridvals,[]);
            TemptationFnParamsCell=CreateCellFromParams(Parameters,TemptationFnParamNames,jj);
            TofPolicy_jj=EvalFnOnAgentDist_Grid(TemptationFn, TemptationFnParamsCell,PolicyValuesPermute(:,:,jj),l_daprime,n_a,0,a_gridvals,[]);

            % Most-tempting term: max of v over the full joint (d,aprime) choice set
            TemptationFnParamsVec=CreateVectorFromParams(Parameters,TemptationFnParamNames,jj);
            if N_d==0
                TemptationMatrix=CreateReturnFnMatrix_Disc_noz(TemptationFn, 0, n_a, 0, a_grid, TemptationFnParamsVec,0);
            else
                TemptationMatrix=CreateReturnFnMatrix_Disc_noz(TemptationFn, n_d, n_a, d_gridvals, a_grid, TemptationFnParamsVec,0);
            end
            MostTempting=max(TemptationMatrix,[],1);
            MostTempting=reshape(MostTempting,[N_a,1]);

            if jj==N_j
                V(:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting;
            else
                beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
                EVnext=V(:,jj+1);

                if N_d==0
                    optaprime=PolicyIndexesKron(1,:,:,jj);
                else
                    optaprime=shiftdim(PolicyIndexesKron(2,:,:,jj),1);
                end

                aprimez_index=reshape(optaprime,[N_a,1]);

                EVnextOfPolicy=EVnext(aprimez_index);

                V(:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting+beta*reshape(EVnextOfPolicy,[N_a,1]);
            end
        end

        %Transforming Value Fn out of Kronecker Form
        V=reshape(V,[n_a,N_j]);

    elseif N_z==0 && N_e>0

        PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1); % PolicyInd2Val auto-adds vfoptions.n_e
        l_daprime=size(PolicyValues,1);
        PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); %[N_a,N_e,l_d+l_a,N_j]
        % The following will also be needed to calculate the expectation of next period value fn, evaluated based on the policy.
        PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, vfoptions.n_e, N_j, vfoptions);

        %% Calculate the Value Fn by backward iteration
        V=zeros(N_a,N_e,N_j,'gpuArray');

        for reverse_j=0:N_j-1
            jj=N_j-reverse_j; % current period, counts backwards from J-1

            % Evaluate Return Fn and Temptation Fn
            FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
            FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,vfoptions.n_e,a_gridvals,vfoptions.e_gridvals_J(:,:,jj));
            TemptationFnParamsCell=CreateCellFromParams(Parameters,TemptationFnParamNames,jj);
            TofPolicy_jj=EvalFnOnAgentDist_Grid(TemptationFn, TemptationFnParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,vfoptions.n_e,a_gridvals,vfoptions.e_gridvals_J(:,:,jj));

            % Most-tempting term: max of v over the full joint (d,aprime) choice set
            TemptationFnParamsVec=CreateVectorFromParams(Parameters,TemptationFnParamNames,jj);
            if N_d==0
                TemptationMatrix=CreateReturnFnMatrix_Disc(TemptationFn, 0, n_a, vfoptions.n_e, 0, a_grid, vfoptions.e_gridvals_J(:,:,jj), TemptationFnParamsVec,0); % Because no z, can treat e like z and call Par2 rather than Par2e
            else
                TemptationMatrix=CreateReturnFnMatrix_Disc(TemptationFn, n_d, n_a, vfoptions.n_e, d_gridvals, a_grid, vfoptions.e_gridvals_J(:,:,jj), TemptationFnParamsVec,0); % Because no z, can treat e like z and call Par2 rather than Par2e
            end
            MostTempting=max(TemptationMatrix,[],1);
            MostTempting=reshape(MostTempting,[N_a,N_e]);

            if jj==N_j
                V(:,:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting;
            else
                beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
                EVw=V(:,:,jj+1).*shiftdim(vfoptions.pi_e_J(:,jj+1),-1); EVw(isnan(EVw))=0; % a zero weight against an infinite node gives 0*(-Inf)=NaN, so zero the terms BEFORE summing
                EVnext=sum(EVw,2); % expectation over iid

                if N_d==0
                    optaprime=PolicyIndexesKron(1,:,:,jj);
                else
                    optaprime=shiftdim(PolicyIndexesKron(2,:,:,jj),1);
                end

                % e is iid -> EVnext shape [N_a,1] depends only on aprime
                EVnextOfPolicy=EVnext(reshape(optaprime,[N_a*N_e,1]));

                V(:,:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting+beta*reshape(EVnextOfPolicy,[N_a,N_e]);
            end
        end

        %Transforming Value Fn out of Kronecker Form
        V=reshape(V,[n_a,vfoptions.n_e,N_j]);

    elseif N_z>0 && N_e==0

        PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
        l_daprime=size(PolicyValues,1);
        PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); %[N_a,N_z,l_d+l_a,N_j]
        % The following will also be needed to calculate the expectation of next period value fn, evaluated based on the policy.
        PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, n_z, N_j, vfoptions);

        %% Calculate the Value Fn by backward iteration
        V=zeros(N_a,N_z,N_j,'gpuArray');

        for reverse_j=0:N_j-1
            jj=N_j-reverse_j; % current period, counts backwards from J-1

            % Evaluate Return Fn and Temptation Fn
            FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
            FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj));
            TemptationFnParamsCell=CreateCellFromParams(Parameters,TemptationFnParamNames,jj);
            TofPolicy_jj=EvalFnOnAgentDist_Grid(TemptationFn, TemptationFnParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj));

            % Most-tempting term: max of v over the full joint (d,aprime) choice set
            TemptationFnParamsVec=CreateVectorFromParams(Parameters,TemptationFnParamNames,jj);
            if N_d==0
                TemptationMatrix=CreateReturnFnMatrix_Disc(TemptationFn, 0, n_a, n_z, 0, a_grid, z_gridvals_J(:,:,jj), TemptationFnParamsVec,0);
            else
                TemptationMatrix=CreateReturnFnMatrix_Disc(TemptationFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals_J(:,:,jj), TemptationFnParamsVec,0);
            end
            MostTempting=max(TemptationMatrix,[],1);
            MostTempting=reshape(MostTempting,[N_a,N_z]);

            if jj==N_j
                V(:,:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting;
            else
                beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
                % EVnext(a, z_from) = sum_{z_to} pi(z_from, z_to) * V(a, z_to, jj+1)
                EVnext=V(:,:,jj+1)*pi_z_J(:,:,jj)'; % [N_a, N_z_from]
                EVnext(isnan(EVnext))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)

                if N_d==0
                    optaprime=PolicyIndexesKron(1,:,:,jj);
                else
                    optaprime=shiftdim(PolicyIndexesKron(2,:,:,jj),1);
                end

                aprimez_index=reshape(optaprime,[N_a*N_z,1])+N_a*(kron((1:1:N_z)',ones(N_a,1,'gpuArray'))-1); % N_a*(z_index-1), but just with lots of kron

                EVnextOfPolicy=EVnext(aprimez_index);

                V(:,:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting+beta*reshape(EVnextOfPolicy,[N_a,N_z]);
            end
        end

        %Transforming Value Fn out of Kronecker Form
        V=reshape(V,[n_a,n_z,N_j]);

    else % N_z>0 && N_e>0

        PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1); % PolicyInd2Val auto-adds vfoptions.n_e
        l_daprime=size(PolicyValues,1);
        PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a,N_z*N_e,l_d+l_a,N_j] — keep shock dim combined for EvalFnOnAgentDist_Grid
        % The following will also be needed to calculate the expectation of next period value fn, evaluated based on the policy.
        PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, [n_z,vfoptions.n_e], N_j, vfoptions);

        %% Calculate the Value Fn by backward iteration
        V=zeros(N_a,N_z,N_e,N_j,'gpuArray');

        for reverse_j=0:N_j-1
            jj=N_j-reverse_j; % current period, counts backwards from J-1

            % Evaluate Return Fn and Temptation Fn
            FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
            FofPolicy_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,[n_z,vfoptions.n_e],a_gridvals,[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)]),[N_a,N_z,N_e]);
            TemptationFnParamsCell=CreateCellFromParams(Parameters,TemptationFnParamNames,jj);
            TofPolicy_jj=reshape(EvalFnOnAgentDist_Grid(TemptationFn, TemptationFnParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,[n_z,vfoptions.n_e],a_gridvals,[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)]),[N_a,N_z,N_e]);

            % Most-tempting term: max of v over the full joint (d,aprime) choice set
            TemptationFnParamsVec=CreateVectorFromParams(Parameters,TemptationFnParamNames,jj);
            if N_d==0
                TemptationMatrix=CreateReturnFnMatrix_Disc_e(TemptationFn, 0, n_a, n_z, vfoptions.n_e, 0, a_grid, z_gridvals_J(:,:,jj), vfoptions.e_gridvals_J(:,:,jj), TemptationFnParamsVec,0);
            else
                TemptationMatrix=CreateReturnFnMatrix_Disc_e(TemptationFn, n_d, n_a, n_z, vfoptions.n_e, d_gridvals, a_grid, z_gridvals_J(:,:,jj), vfoptions.e_gridvals_J(:,:,jj), TemptationFnParamsVec,0);
            end
            MostTempting=max(TemptationMatrix,[],1);
            MostTempting=reshape(MostTempting,[N_a,N_z,N_e]);

            if jj==N_j
                V(:,:,:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting;
            else
                beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
                EVnext=sum(V(:,:,:,jj+1).*shiftdim(vfoptions.pi_e_J(:,jj+1),-2),3); % expectation over iid
                EVnext=EVnext.*shiftdim(pi_z_J(:,:,jj)',-1); % size N_z-by-1
                EVnext(isnan(EVnext))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EVnext=sum(EVnext,2); % sum over z', leaving a singular second dimension
                % EVnext=reshape(EVnext,[N_a,N_z]); % Not necessary as just index into it

                if N_d==0
                    optaprime=PolicyIndexesKron(1,:,:,jj);
                else
                    optaprime=shiftdim(PolicyIndexesKron(2,:,:,jj),1);
                end

                aprimez_index=reshape(optaprime,[N_a*N_z*N_e,1])+N_a*(kron(kron(ones(N_e,1,'gpuArray'),(1:1:N_z)'),ones(N_a,1,'gpuArray'))-1); % N_a*(z_index-1), but just with lots of kron

                EVnextOfPolicy=EVnext(aprimez_index);

                V(:,:,:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting+beta*reshape(EVnextOfPolicy,[N_a,N_z,N_e]);
            end

        end

        % Transforming Value Fn out of Kronecker Form
        V=reshape(V,[n_a,n_z,vfoptions.n_e,N_j]);
    end

end

end
