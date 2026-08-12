function V=ValueFnFromPolicy_FHorz_EpsteinZin(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Compute V from a given Policy when the model uses Epstein-Zin preferences.
% Evaluates the EZ recursion along the given policy (all three EZ cases:
% consumption-units EZutils=0, utility-units EZutils=1 with positive or
% negative valued utility fn). Includes survival probabilities and warm-glow
% of bequests. Backward-evaluating at the argmax policy of
% ValueFnIter_FHorz_EpsteinZin reproduces its V exactly.

%% Scope limits
if vfoptions.gridinterplayer==1 && ~isscalar(n_a)
    error('ValueFnFromPolicy_FHorz_EpsteinZin: gridinterplayer only supported for a single endogenous state (scalar n_a) for now')
end
if prod(vfoptions.n_semiz)>0
    error('ValueFnFromPolicy_FHorz_EpsteinZin: not yet implemented for semi-exogenous shocks (Epstein-Zin with semiz is itself not yet implemented)')
end
if vfoptions.riskyasset==1
    error('ValueFnFromPolicy_FHorz_EpsteinZin: not yet implemented for riskyasset')
end
if vfoptions.experienceasset>=1 || vfoptions.experienceassetu>=1 || vfoptions.experienceassetz>=1 || vfoptions.experienceassete>=1 || vfoptions.experienceassetze>=1 || vfoptions.experienceassetsemiz>=1
    error('ValueFnFromPolicy_FHorz_EpsteinZin: not yet implemented for the experienceasset family')
end

%% Some Epstein-Zin specific options need to be set if they are not already declared
% (this preamble mirrors ValueFnIter_FHorz_EpsteinZin)
if ~isfield(vfoptions,'EZriskaversion')
    error('When using Epstein-Zin preferences you must declare vfoptions.EZriskaversion (coefficient controlling risk aversion)')
end
if ~isfield(vfoptions,'EZutils')
    vfoptions.EZutils=1; % Use EZ preferences with general utility function (0 gives traditional EZ with exogenous labor, 2 gives traditional EZ with endogenous labor)
end
if vfoptions.EZutils==1
    if ~isfield(vfoptions,'EZpositiveutility')
        warning('Using Epstein-Zin preferences it is assumed the utility/return function is negative valued, if not you need to set vfoptions.EZpositiveutility=1')
        vfoptions.EZpositiveutility=0;
    end
else
    if ~isfield(vfoptions,'EZeis')
        error('When using Epstein-Zin preferences you must declare vfoptions.EZeis (elasticity of intertemporal substitution)')
    end
end
if ~isfield(vfoptions,'EZoneminusbeta')
    vfoptions.EZoneminusbeta=0; % default essentially does nothing
end
% Set up sj
if isfield(vfoptions,'survivalprobability')
    sj=Parameters.(vfoptions.survivalprobability);
    if length(sj)~=N_j
        error('Survival probabilities must be of the same length as N_j')
    end
elseif isfield(vfoptions,'WarmGlowBequestsFn')
    sj=ones(N_j,1); % conditional survival probabilities
    sj(end)=0;
else
    sj=ones(N_j,1); % conditional survival probabilities
end
% Declare warmglow indicator
if isfield(vfoptions,'WarmGlowBequestsFn')
    warmglow=1;
    temp=getAnonymousFnInputNames(vfoptions.WarmGlowBequestsFn);
    vfoptions.WarmGlowBequestsFnParamsNames={temp{2:end}};
else
    warmglow=0;
end

%% Based on the settings, define a bunch of variables that are used to implement the EZ preferences
crisk=Parameters.(vfoptions.EZriskaversion);
if vfoptions.EZutils==0
    ceis=Parameters.(vfoptions.EZeis);
    % Traditional EZ in consumption units
    ezc1=1; % used if vfoptions.EZoneminusbeta=1
    ezc2=1-1./ceis;
    ezc3=1;
    ezc4=1;
    ezc5=1-crisk;
    ezc6=(1-1./ceis)./(1-crisk);
    ezc7=1./(1-1./ceis);
elseif vfoptions.EZutils==1
    % EZ in utility-units
    ezc1=1; % used if vfoptions.EZoneminusbeta=1
    ezc2=1;
    if vfoptions.EZpositiveutility==1
        ezc3=1;
        ezc4=1;
    elseif vfoptions.EZpositiveutility==0
        ezc3=-1;
        ezc4=-1;
    end
    if vfoptions.EZpositiveutility==1
        ezc5=1-crisk;
        ezc6=1./(1-crisk);
    elseif vfoptions.EZpositiveutility==0
        ezc5=1+crisk;
        ezc6=1./(1+crisk);
    end
    ezc7=1;
end
% Can do a double Epstein-Zin (inner EZ is about risk, outer EZ is about mortality-risk)
if isfield(vfoptions,'EZmortalityriskaversion')
    mrisk=Parameters.(vfoptions.EZmortalityriskaversion);
    if vfoptions.EZutils==0
        ezc6=(1-1./ceis)./(1-mrisk);
        ezc8=(1-mrisk)./(1-crisk);
    elseif vfoptions.EZutils==1
        if vfoptions.EZpositiveutility==1
            ezc6=1./(1-mrisk);
            ezc8=(1-mrisk)./(1-crisk);
        elseif vfoptions.EZpositiveutility==0
            ezc6=1./(1+mrisk);
            ezc8=(1+mrisk)./(1+crisk);
        end
    end
else
    ezc8=1;
end

% setup to permit age-dependence of these (and make them column vectors if they are not already)
if size(ezc2,1)==1
    ezc2=ezc2';
end
ezc2=ezc2.*ones(N_j,1);
if size(ezc5,1)==1
    ezc5=ezc5';
end
ezc5=ezc5.*ones(N_j,1);
if size(ezc6,1)==1
    ezc6=ezc6';
end
ezc6=ezc6.*ones(N_j,1);
if size(ezc7,1)==1
    ezc7=ezc7';
end
ezc7=ezc7.*ones(N_j,1);
if size(ezc8,1)==1
    ezc8=ezc8';
end
ezc8=ezc8.*ones(N_j,1);

if vfoptions.EZutils==0
    if crisk<1
        error('Cannot use EZriskaversion parameter less than one (must be risk averse) with Epstein-Zin preferences')
    end
    if ceis<=0
        error('Cannot use EZeis parameter less than zero with Epstein-Zin preferences')
    end
    if ceis==1
        error('Cannot use EZeis parameter equal to one with Epstein-Zin preferences (look at formula, it would mean having to raise to the power of zero; you can always put 0.99 or 1.01)')
    end
end

%% Setup (mirrors ValueFnFromPolicy_FHorz)
% Caller already moved grids and Policy to GPU and ran ExogShockSetup_FHorz.
% Re-run shock setup here to get z_gridvals_J / pi_z_J locally.
[z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3);

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);

a_gridvals=CreateGridvals(n_a,a_grid,1);

%% GI1 variant (gridinterplayer==1): replace integer-aprime lookup with L2-interpolated lookup
% The continuation is looked up as the LINEAR interpolation of the TRANSFORMED
% EV=E[(ezc4*V')^ezc5] between the two adjacent grid points (the same object the
% EZ GI solver interpolates), with the certainty-equivalent power ^ezc6 applied
% pointwise afterwards. The warm-glow fn is evaluated exactly on the fine grid
% and indexed at the policy's fine point.
if vfoptions.gridinterplayer==1
    n2short=vfoptions.ngridinterp;
    index_a1=1+(N_d>0);
    aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
    n2aprime=length(aprime_grid);

    if N_z==0 && N_e==0

        PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,0,N_j,d_grid,a_grid,vfoptions,1);
        l_daprime=size(PolicyValues,1);
        PolicyValuesPermute=permute(PolicyValues,[2,1,3]); %[N_a,l_d+l_a,N_j]
        PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, 1, N_j, vfoptions); % rows: alower (=index_a1), L2 (=end). L2flag dropped by Kron.

        alower=reshape(PolicyIndexesKron(index_a1,:,:),[N_a,N_j]);
        L2=reshape(PolicyIndexesKron(end,:,:),[N_a,N_j]);
        PolicyProbs=zeros(N_a,N_j,2,'gpuArray');
        PolicyProbs(:,:,2)=(L2-1)/(n2short+1); % prob of upper grid point
        PolicyProbs(:,:,1)=1-PolicyProbs(:,:,2);

        V=zeros(N_a,N_j,'gpuArray');

        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;

            FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
            FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,jj),l_daprime,n_a,0,a_gridvals,[]);
            FofPolicy_jj=reshape(FofPolicy_jj,[N_a,1]);

            DiscountFactorParamsVec=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            if vfoptions.EZoneminusbeta==1
                ezc1=1-DiscountFactorParamsVec;
            elseif vfoptions.EZoneminusbeta==2
                ezc1=1-sj(jj)*DiscountFactorParamsVec;
            end

            % If there is a warm-glow, evaluate the warmglowfn on the fine grid, then at the policy's fine point
            if warmglow==1
                WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
                WGmatrixfineraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n2aprime, aprime_grid, WGParamsVec);
                WGmatrixfine=WGmatrixfineraw;
                WGmatrixfine(isfinite(WGmatrixfineraw))=(ezc4*WGmatrixfineraw(isfinite(WGmatrixfineraw))).^ezc5(jj);
                WGmatrixfine(WGmatrixfineraw==0)=0; % otherwise zero to negative power is set to infinity
                finex=(alower(:,jj)-1)*(n2short+1)+L2(:,jj); % the policy's fine-grid index
                WGofPolicy=reshape(WGmatrixfine(finex),[N_a,1]);
            end

            if jj==N_j
                Vjj=FofPolicy_jj;
                becareful=logical(isfinite(FofPolicy_jj).*(FofPolicy_jj~=0)); % finite but not zero
                Vjj(becareful)=(ezc1*FofPolicy_jj(becareful).^ezc2(N_j)).^ezc7(N_j);
                Vjj(FofPolicy_jj==0)=-Inf;
                if warmglow==1
                    becareful2=(WGofPolicy==0);
                    WGofPolicy(isfinite(WGofPolicy))=ezc3*DiscountFactorParamsVec*(((1-sj(N_j))*WGofPolicy(isfinite(WGofPolicy)).^ezc8(N_j)).^ezc6(N_j));
                    WGofPolicy(becareful2)=0;
                    Vjj=Vjj+WGofPolicy;
                end
                V(:,jj)=Vjj;
            else
                % Part of Epstein-Zin is before taking expectation
                EVnextpre=V(:,jj+1);
                temp=EVnextpre;
                temp(isfinite(EVnextpre))=(ezc4*EVnextpre(isfinite(EVnextpre))).^ezc5(jj);
                temp(EVnextpre==0)=0;

                % No shocks, so no expectation to take (the certainty-equivalent is just the identity)
                EVnext=temp;

                EVnextOfPolicy=PolicyProbs(:,jj,1).*EVnext(alower(:,jj))+PolicyProbs(:,jj,2).*EVnext(alower(:,jj)+1);

                temp4=EVnextOfPolicy;
                if warmglow==1
                    becareful=logical(isfinite(temp4).*isfinite(WGofPolicy)); % both are finite
                    temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGofPolicy(becareful).^ezc8(jj)).^ezc6(jj);
                    temp4((EVnextOfPolicy==0)&(WGofPolicy==0))=0; % Is actually zero
                else % not using warmglow
                    temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                    temp4(EVnextOfPolicy==0)=0;
                end

                becareful=logical(isfinite(FofPolicy_jj).*(FofPolicy_jj~=0)); % finite but not zero
                temp2=FofPolicy_jj;
                temp2(becareful)=FofPolicy_jj(becareful).^ezc2(jj);
                temp2(FofPolicy_jj==0)=-Inf;

                Vjj=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4;

                temp5=logical(isfinite(Vjj).*(Vjj~=0));
                Vjj(temp5)=Vjj(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
                Vjj(Vjj==0)=-Inf;

                V(:,jj)=Vjj;
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
        PolicyProbs=zeros(N_a,N_e,N_j,2,'gpuArray');
        PolicyProbs(:,:,:,2)=(L2-1)/(n2short+1);
        PolicyProbs(:,:,:,1)=1-PolicyProbs(:,:,:,2);

        V=zeros(N_a,N_e,N_j,'gpuArray');

        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;

            FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
            FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,vfoptions.n_e,a_gridvals,vfoptions.e_gridvals_J(:,:,jj));
            FofPolicy_jj=reshape(FofPolicy_jj,[N_a,N_e]);

            DiscountFactorParamsVec=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            if vfoptions.EZoneminusbeta==1
                ezc1=1-DiscountFactorParamsVec;
            elseif vfoptions.EZoneminusbeta==2
                ezc1=1-sj(jj)*DiscountFactorParamsVec;
            end

            % If there is a warm-glow, evaluate the warmglowfn on the fine grid, then at the policy's fine point
            if warmglow==1
                WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
                WGmatrixfineraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n2aprime, aprime_grid, WGParamsVec);
                WGmatrixfine=WGmatrixfineraw;
                WGmatrixfine(isfinite(WGmatrixfineraw))=(ezc4*WGmatrixfineraw(isfinite(WGmatrixfineraw))).^ezc5(jj);
                WGmatrixfine(WGmatrixfineraw==0)=0; % otherwise zero to negative power is set to infinity
                finex=(alower(:,:,jj)-1)*(n2short+1)+L2(:,:,jj); % the policy's fine-grid index
                WGofPolicy=reshape(WGmatrixfine(finex),[N_a,N_e]);
            end

            if jj==N_j
                Vjj=FofPolicy_jj;
                becareful=logical(isfinite(FofPolicy_jj).*(FofPolicy_jj~=0)); % finite but not zero
                Vjj(becareful)=(ezc1*FofPolicy_jj(becareful).^ezc2(N_j)).^ezc7(N_j);
                Vjj(FofPolicy_jj==0)=-Inf;
                if warmglow==1
                    becareful2=(WGofPolicy==0);
                    WGofPolicy(isfinite(WGofPolicy))=ezc3*DiscountFactorParamsVec*(((1-sj(N_j))*WGofPolicy(isfinite(WGofPolicy)).^ezc8(N_j)).^ezc6(N_j));
                    WGofPolicy(becareful2)=0;
                    Vjj=Vjj+WGofPolicy;
                end
                V(:,:,jj)=Vjj;
            else
                % Part of Epstein-Zin is before taking expectation
                EVnextpre=V(:,:,jj+1);
                temp=EVnextpre;
                temp(isfinite(EVnextpre))=(ezc4*EVnextpre(isfinite(EVnextpre))).^ezc5(jj);
                temp(EVnextpre==0)=0;

                % Expectation over the iid e
                EVnext=sum(temp.*shiftdim(vfoptions.pi_e_J(:,jj+1),-1),2); % [N_a,1]
                EVnext(isnan(EVnext))=0;

                EVlower=reshape(EVnext(alower(:,:,jj)),[N_a,N_e]);
                EVupper=reshape(EVnext(alower(:,:,jj)+1),[N_a,N_e]);
                EVnextOfPolicy=PolicyProbs(:,:,jj,1).*EVlower+PolicyProbs(:,:,jj,2).*EVupper;

                temp4=EVnextOfPolicy;
                if warmglow==1
                    becareful=logical(isfinite(temp4).*isfinite(WGofPolicy)); % both are finite
                    temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGofPolicy(becareful).^ezc8(jj)).^ezc6(jj);
                    temp4((EVnextOfPolicy==0)&(WGofPolicy==0))=0; % Is actually zero
                else % not using warmglow
                    temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                    temp4(EVnextOfPolicy==0)=0;
                end

                becareful=logical(isfinite(FofPolicy_jj).*(FofPolicy_jj~=0)); % finite but not zero
                temp2=FofPolicy_jj;
                temp2(becareful)=FofPolicy_jj(becareful).^ezc2(jj);
                temp2(FofPolicy_jj==0)=-Inf;

                Vjj=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4;

                temp5=logical(isfinite(Vjj).*(Vjj~=0));
                Vjj(temp5)=Vjj(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
                Vjj(Vjj==0)=-Inf;

                V(:,:,jj)=Vjj;
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
        PolicyProbs=zeros(N_a,N_z,N_j,2,'gpuArray');
        PolicyProbs(:,:,:,2)=(L2-1)/(n2short+1);
        PolicyProbs(:,:,:,1)=1-PolicyProbs(:,:,:,2);

        V=zeros(N_a,N_z,N_j,'gpuArray');

        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;

            FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
            FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj));
            FofPolicy_jj=reshape(FofPolicy_jj,[N_a,N_z]);

            DiscountFactorParamsVec=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            if vfoptions.EZoneminusbeta==1
                ezc1=1-DiscountFactorParamsVec;
            elseif vfoptions.EZoneminusbeta==2
                ezc1=1-sj(jj)*DiscountFactorParamsVec;
            end

            % If there is a warm-glow, evaluate the warmglowfn on the fine grid, then at the policy's fine point
            if warmglow==1
                WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
                WGmatrixfineraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n2aprime, aprime_grid, WGParamsVec);
                WGmatrixfine=WGmatrixfineraw;
                WGmatrixfine(isfinite(WGmatrixfineraw))=(ezc4*WGmatrixfineraw(isfinite(WGmatrixfineraw))).^ezc5(jj);
                WGmatrixfine(WGmatrixfineraw==0)=0; % otherwise zero to negative power is set to infinity
                finex=(alower(:,:,jj)-1)*(n2short+1)+L2(:,:,jj); % the policy's fine-grid index
                WGofPolicy=reshape(WGmatrixfine(finex),[N_a,N_z]);
            end

            if jj==N_j
                Vjj=FofPolicy_jj;
                becareful=logical(isfinite(FofPolicy_jj).*(FofPolicy_jj~=0)); % finite but not zero
                Vjj(becareful)=(ezc1*FofPolicy_jj(becareful).^ezc2(N_j)).^ezc7(N_j);
                Vjj(FofPolicy_jj==0)=-Inf;
                if warmglow==1
                    becareful2=(WGofPolicy==0);
                    WGofPolicy(isfinite(WGofPolicy))=ezc3*DiscountFactorParamsVec*(((1-sj(N_j))*WGofPolicy(isfinite(WGofPolicy)).^ezc8(N_j)).^ezc6(N_j));
                    WGofPolicy(becareful2)=0;
                    Vjj=Vjj+WGofPolicy;
                end
                V(:,:,jj)=Vjj;
            else
                % Part of Epstein-Zin is before taking expectation
                EVnextpre=V(:,:,jj+1);
                temp=EVnextpre;
                temp(isfinite(EVnextpre))=(ezc4*EVnextpre(isfinite(EVnextpre))).^ezc5(jj);
                temp(EVnextpre==0)=0;

                % EVnext(aprime, z_from) = sum_{z_to} pi(z_from, z_to) * temp(aprime, z_to)
                EVnext=temp*pi_z_J(:,:,jj)'; % [N_a, N_z_from]
                EVnext(isnan(EVnext))=0;

                zidxoffset=N_a*gpuArray(0:N_z-1); % (1, N_z)
                lower_lin=alower(:,:,jj)+zidxoffset;
                EVnextOfPolicy=PolicyProbs(:,:,jj,1).*EVnext(lower_lin)+PolicyProbs(:,:,jj,2).*EVnext(lower_lin+1);

                temp4=EVnextOfPolicy;
                if warmglow==1
                    becareful=logical(isfinite(temp4).*isfinite(WGofPolicy)); % both are finite
                    temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGofPolicy(becareful).^ezc8(jj)).^ezc6(jj);
                    temp4((EVnextOfPolicy==0)&(WGofPolicy==0))=0; % Is actually zero
                else % not using warmglow
                    temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                    temp4(EVnextOfPolicy==0)=0;
                end

                becareful=logical(isfinite(FofPolicy_jj).*(FofPolicy_jj~=0)); % finite but not zero
                temp2=FofPolicy_jj;
                temp2(becareful)=FofPolicy_jj(becareful).^ezc2(jj);
                temp2(FofPolicy_jj==0)=-Inf;

                Vjj=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4;

                temp5=logical(isfinite(Vjj).*(Vjj~=0));
                Vjj(temp5)=Vjj(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
                Vjj(Vjj==0)=-Inf;

                V(:,:,jj)=Vjj;
            end
        end

        V=reshape(V,[n_a,n_z,N_j]);

    elseif N_z>0 && N_e>0

        PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1); % PolicyInd2Val auto-adds vfoptions.n_e
        l_daprime=size(PolicyValues,1);
        PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a,N_z*N_e,l_d+l_a,N_j] — keep shock dim combined for EvalFnOnAgentDist_Grid
        PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, [n_z,vfoptions.n_e], N_j, vfoptions); % rows: alower (=index_a1), L2 (=end). L2flag dropped by Kron.

        alower=reshape(PolicyIndexesKron(index_a1,:,:,:),[N_a,N_z,N_e,N_j]);
        L2=reshape(PolicyIndexesKron(end,:,:,:),[N_a,N_z,N_e,N_j]);
        PolicyProbs=zeros(N_a,N_z,N_e,N_j,2,'gpuArray');
        PolicyProbs(:,:,:,:,2)=(L2-1)/(n2short+1);
        PolicyProbs(:,:,:,:,1)=1-PolicyProbs(:,:,:,:,2);

        V=zeros(N_a,N_z,N_e,N_j,'gpuArray');

        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;

            FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
            FofPolicy_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,[n_z,vfoptions.n_e],a_gridvals,[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)]),[N_a,N_z,N_e]);

            DiscountFactorParamsVec=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            if vfoptions.EZoneminusbeta==1
                ezc1=1-DiscountFactorParamsVec;
            elseif vfoptions.EZoneminusbeta==2
                ezc1=1-sj(jj)*DiscountFactorParamsVec;
            end

            % If there is a warm-glow, evaluate the warmglowfn on the fine grid, then at the policy's fine point
            if warmglow==1
                WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
                WGmatrixfineraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n2aprime, aprime_grid, WGParamsVec);
                WGmatrixfine=WGmatrixfineraw;
                WGmatrixfine(isfinite(WGmatrixfineraw))=(ezc4*WGmatrixfineraw(isfinite(WGmatrixfineraw))).^ezc5(jj);
                WGmatrixfine(WGmatrixfineraw==0)=0; % otherwise zero to negative power is set to infinity
                finex=(alower(:,:,:,jj)-1)*(n2short+1)+L2(:,:,:,jj); % the policy's fine-grid index
                WGofPolicy=reshape(WGmatrixfine(finex),[N_a,N_z,N_e]);
            end

            if jj==N_j
                Vjj=FofPolicy_jj;
                becareful=logical(isfinite(FofPolicy_jj).*(FofPolicy_jj~=0)); % finite but not zero
                Vjj(becareful)=(ezc1*FofPolicy_jj(becareful).^ezc2(N_j)).^ezc7(N_j);
                Vjj(FofPolicy_jj==0)=-Inf;
                if warmglow==1
                    becareful2=(WGofPolicy==0);
                    WGofPolicy(isfinite(WGofPolicy))=ezc3*DiscountFactorParamsVec*(((1-sj(N_j))*WGofPolicy(isfinite(WGofPolicy)).^ezc8(N_j)).^ezc6(N_j));
                    WGofPolicy(becareful2)=0;
                    Vjj=Vjj+WGofPolicy;
                end
                V(:,:,:,jj)=Vjj;
            else
                % Part of Epstein-Zin is before taking expectation
                EVnextpre=V(:,:,:,jj+1);
                temp=EVnextpre;
                temp(isfinite(EVnextpre))=(ezc4*EVnextpre(isfinite(EVnextpre))).^ezc5(jj);
                temp(EVnextpre==0)=0;

                % One certainty-equivalent over the joint distribution of (zprime,eprime):
                % expectation over the iid e first, then over zprime (mirrors the EZ VFI raws)
                EVnext=sum(temp.*shiftdim(vfoptions.pi_e_J(:,jj+1),-2),3); % [N_a,N_z_to]
                EVnext=EVnext*pi_z_J(:,:,jj)'; % [N_a,N_z_from]
                EVnext(isnan(EVnext))=0;

                zidxoffset=N_a*gpuArray(0:N_z-1); % (1, N_z)
                lower_lin=alower(:,:,:,jj)+zidxoffset; % (N_a, N_z, N_e) — broadcasting
                EVnextOfPolicy=PolicyProbs(:,:,:,jj,1).*EVnext(lower_lin)+PolicyProbs(:,:,:,jj,2).*EVnext(lower_lin+1);

                temp4=EVnextOfPolicy;
                if warmglow==1
                    becareful=logical(isfinite(temp4).*isfinite(WGofPolicy)); % both are finite
                    temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGofPolicy(becareful).^ezc8(jj)).^ezc6(jj);
                    temp4((EVnextOfPolicy==0)&(WGofPolicy==0))=0; % Is actually zero
                else % not using warmglow
                    temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                    temp4(EVnextOfPolicy==0)=0;
                end

                becareful=logical(isfinite(FofPolicy_jj).*(FofPolicy_jj~=0)); % finite but not zero
                temp2=FofPolicy_jj;
                temp2(becareful)=FofPolicy_jj(becareful).^ezc2(jj);
                temp2(FofPolicy_jj==0)=-Inf;

                Vjj=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4;

                temp5=logical(isfinite(Vjj).*(Vjj~=0));
                Vjj(temp5)=Vjj(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
                Vjj(Vjj==0)=-Inf;

                V(:,:,:,jj)=Vjj;
            end
        end

        V=reshape(V,[n_a,n_z,vfoptions.n_e,N_j]);
    end

    return
end

%%
if N_z==0 && N_e==0

    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,0,N_j,d_grid,a_grid,vfoptions,1);
    l_daprime=size(PolicyValues,1);
    PolicyValuesPermute=permute(PolicyValues,[2,1,3]); %[N_a,l_d+l_a,N_j]
    PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, 1, N_j, vfoptions);

    %% Calculate the Value Fn by backward iteration, evaluating the EZ recursion at the policy
    V=zeros(N_a,N_j,'gpuArray');

    for reverse_j=0:N_j-1
        jj=N_j-reverse_j; % current period, counts backwards from J-1

        % Evaluate Return Fn
        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,jj),l_daprime,n_a,0,a_gridvals,[]);
        FofPolicy_jj=reshape(FofPolicy_jj,[N_a,1]);

        DiscountFactorParamsVec=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
        if vfoptions.EZoneminusbeta==1
            ezc1=1-DiscountFactorParamsVec;
        elseif vfoptions.EZoneminusbeta==2
            ezc1=1-sj(jj)*DiscountFactorParamsVec;
        end

        % Policy's aprime index
        if N_d==0
            optaprime=reshape(PolicyIndexesKron(1,:,:,jj),[N_a,1]);
        else
            optaprime=reshape(PolicyIndexesKron(2,:,:,jj),[N_a,1]);
        end

        % If there is a warm-glow, evaluate the warmglowfn (over aprime), then at the policy
        if warmglow==1
            WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
            WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a, a_grid, WGParamsVec);
            WGmatrix=WGmatrixraw;
            WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5(jj);
            WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
            WGofPolicy=reshape(WGmatrix(optaprime),[N_a,1]);
        end

        if jj==N_j
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            Vjj=FofPolicy_jj;
            becareful=logical(isfinite(FofPolicy_jj).*(FofPolicy_jj~=0)); % finite but not zero
            Vjj(becareful)=(ezc1*FofPolicy_jj(becareful).^ezc2(N_j)).^ezc7(N_j);
            Vjj(FofPolicy_jj==0)=-Inf;
            if warmglow==1
                becareful2=(WGofPolicy==0);
                WGofPolicy(isfinite(WGofPolicy))=ezc3*DiscountFactorParamsVec*(((1-sj(N_j))*WGofPolicy(isfinite(WGofPolicy)).^ezc8(N_j)).^ezc6(N_j));
                WGofPolicy(becareful2)=0;
                Vjj=Vjj+WGofPolicy;
            end
            V(:,jj)=Vjj;
        else
            % Part of Epstein-Zin is before taking expectation
            EVnextpre=V(:,jj+1);
            temp=EVnextpre;
            temp(isfinite(EVnextpre))=(ezc4*EVnextpre(isfinite(EVnextpre))).^ezc5(jj);
            temp(EVnextpre==0)=0;

            % No shocks, so no expectation to take (the certainty-equivalent is just the identity)
            EVnext=temp;

            EVnextOfPolicy=reshape(EVnext(optaprime),[N_a,1]);

            temp4=EVnextOfPolicy;
            if warmglow==1
                becareful=logical(isfinite(temp4).*isfinite(WGofPolicy)); % both are finite
                temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGofPolicy(becareful).^ezc8(jj)).^ezc6(jj);
                temp4((EVnextOfPolicy==0)&(WGofPolicy==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                temp4(EVnextOfPolicy==0)=0;
            end

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(FofPolicy_jj).*(FofPolicy_jj~=0)); % finite but not zero
            temp2=FofPolicy_jj;
            temp2(becareful)=FofPolicy_jj(becareful).^ezc2(jj);
            temp2(FofPolicy_jj==0)=-Inf;

            Vjj=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4;

            temp5=logical(isfinite(Vjj).*(Vjj~=0));
            Vjj(temp5)=Vjj(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
            Vjj(Vjj==0)=-Inf;

            V(:,jj)=Vjj;
        end
    end

    %Transforming Value Fn out of Kronecker Form
    V=reshape(V,[n_a,N_j]);

elseif N_z==0 && N_e>0

    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1); % PolicyInd2Val auto-adds vfoptions.n_e
    l_daprime=size(PolicyValues,1);
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); %[N_a,N_e,l_d+l_a,N_j]
    PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, vfoptions.n_e, N_j, vfoptions);

    %% Calculate the Value Fn by backward iteration, evaluating the EZ recursion at the policy
    V=zeros(N_a,N_e,N_j,'gpuArray');

    for reverse_j=0:N_j-1
        jj=N_j-reverse_j; % current period, counts backwards from J-1

        % Evaluate Return Fn
        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,vfoptions.n_e,a_gridvals,vfoptions.e_gridvals_J(:,:,jj));
        FofPolicy_jj=reshape(FofPolicy_jj,[N_a,N_e]);

        DiscountFactorParamsVec=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
        if vfoptions.EZoneminusbeta==1
            ezc1=1-DiscountFactorParamsVec;
        elseif vfoptions.EZoneminusbeta==2
            ezc1=1-sj(jj)*DiscountFactorParamsVec;
        end

        % Policy's aprime index
        if N_d==0
            optaprime=reshape(PolicyIndexesKron(1,:,:,jj),[N_a*N_e,1]);
        else
            optaprime=reshape(PolicyIndexesKron(2,:,:,jj),[N_a*N_e,1]);
        end

        % If there is a warm-glow, evaluate the warmglowfn (over aprime), then at the policy
        if warmglow==1
            WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
            WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a, a_grid, WGParamsVec);
            WGmatrix=WGmatrixraw;
            WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5(jj);
            WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
            WGofPolicy=reshape(WGmatrix(optaprime),[N_a,N_e]);
        end

        if jj==N_j
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            Vjj=FofPolicy_jj;
            becareful=logical(isfinite(FofPolicy_jj).*(FofPolicy_jj~=0)); % finite but not zero
            Vjj(becareful)=(ezc1*FofPolicy_jj(becareful).^ezc2(N_j)).^ezc7(N_j);
            Vjj(FofPolicy_jj==0)=-Inf;
            if warmglow==1
                becareful2=(WGofPolicy==0);
                WGofPolicy(isfinite(WGofPolicy))=ezc3*DiscountFactorParamsVec*(((1-sj(N_j))*WGofPolicy(isfinite(WGofPolicy)).^ezc8(N_j)).^ezc6(N_j));
                WGofPolicy(becareful2)=0;
                Vjj=Vjj+WGofPolicy;
            end
            V(:,:,jj)=Vjj;
        else
            % Part of Epstein-Zin is before taking expectation
            EVnextpre=V(:,:,jj+1);
            temp=EVnextpre;
            temp(isfinite(EVnextpre))=(ezc4*EVnextpre(isfinite(EVnextpre))).^ezc5(jj);
            temp(EVnextpre==0)=0;

            % Expectation over the iid e
            EVnext=sum(temp.*shiftdim(vfoptions.pi_e_J(:,jj+1),-1),2); % [N_a,1]
            EVnext(isnan(EVnext))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)

            % e is iid -> EVnext depends only on aprime
            EVnextOfPolicy=reshape(EVnext(optaprime),[N_a,N_e]);

            temp4=EVnextOfPolicy;
            if warmglow==1
                becareful=logical(isfinite(temp4).*isfinite(WGofPolicy)); % both are finite
                temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGofPolicy(becareful).^ezc8(jj)).^ezc6(jj);
                temp4((EVnextOfPolicy==0)&(WGofPolicy==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                temp4(EVnextOfPolicy==0)=0;
            end

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(FofPolicy_jj).*(FofPolicy_jj~=0)); % finite but not zero
            temp2=FofPolicy_jj;
            temp2(becareful)=FofPolicy_jj(becareful).^ezc2(jj);
            temp2(FofPolicy_jj==0)=-Inf;

            Vjj=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4;

            temp5=logical(isfinite(Vjj).*(Vjj~=0));
            Vjj(temp5)=Vjj(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
            Vjj(Vjj==0)=-Inf;

            V(:,:,jj)=Vjj;
        end
    end

    %Transforming Value Fn out of Kronecker Form
    V=reshape(V,[n_a,vfoptions.n_e,N_j]);

elseif N_z>0 && N_e==0

    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
    l_daprime=size(PolicyValues,1);
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); %[N_a,N_z,l_d+l_a,N_j]
    PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, n_z, N_j, vfoptions);

    %% Calculate the Value Fn by backward iteration, evaluating the EZ recursion at the policy
    V=zeros(N_a,N_z,N_j,'gpuArray');

    for reverse_j=0:N_j-1
        jj=N_j-reverse_j; % current period, counts backwards from J-1

        % Evaluate Return Fn
        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj));
        FofPolicy_jj=reshape(FofPolicy_jj,[N_a,N_z]);

        DiscountFactorParamsVec=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
        if vfoptions.EZoneminusbeta==1
            ezc1=1-DiscountFactorParamsVec;
        elseif vfoptions.EZoneminusbeta==2
            ezc1=1-sj(jj)*DiscountFactorParamsVec;
        end

        % Policy's aprime index
        if N_d==0
            optaprime=reshape(PolicyIndexesKron(1,:,:,jj),[N_a*N_z,1]);
        else
            optaprime=reshape(PolicyIndexesKron(2,:,:,jj),[N_a*N_z,1]);
        end

        % If there is a warm-glow, evaluate the warmglowfn (over aprime), then at the policy
        if warmglow==1
            WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
            WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a, a_grid, WGParamsVec);
            WGmatrix=WGmatrixraw;
            WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5(jj);
            WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
            WGofPolicy=reshape(WGmatrix(optaprime),[N_a,N_z]);
        end

        if jj==N_j
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            Vjj=FofPolicy_jj;
            becareful=logical(isfinite(FofPolicy_jj).*(FofPolicy_jj~=0)); % finite but not zero
            Vjj(becareful)=(ezc1*FofPolicy_jj(becareful).^ezc2(N_j)).^ezc7(N_j);
            Vjj(FofPolicy_jj==0)=-Inf;
            if warmglow==1
                becareful2=(WGofPolicy==0);
                WGofPolicy(isfinite(WGofPolicy))=ezc3*DiscountFactorParamsVec*(((1-sj(N_j))*WGofPolicy(isfinite(WGofPolicy)).^ezc8(N_j)).^ezc6(N_j));
                WGofPolicy(becareful2)=0;
                Vjj=Vjj+WGofPolicy;
            end
            V(:,:,jj)=Vjj;
        else
            % Part of Epstein-Zin is before taking expectation
            EVnextpre=V(:,:,jj+1);
            temp=EVnextpre;
            temp(isfinite(EVnextpre))=(ezc4*EVnextpre(isfinite(EVnextpre))).^ezc5(jj);
            temp(EVnextpre==0)=0;

            % EVnext(aprime, z_from) = sum_{z_to} pi(z_from, z_to) * temp(aprime, z_to)
            EVnext=temp*pi_z_J(:,:,jj)'; % [N_a, N_z_from]
            EVnext(isnan(EVnext))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)

            aprimez_index=optaprime+N_a*(kron((1:1:N_z)',ones(N_a,1,'gpuArray'))-1); % N_a*(z_index-1), but just with lots of kron

            EVnextOfPolicy=reshape(EVnext(aprimez_index),[N_a,N_z]);

            temp4=EVnextOfPolicy;
            if warmglow==1
                becareful=logical(isfinite(temp4).*isfinite(WGofPolicy)); % both are finite
                temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGofPolicy(becareful).^ezc8(jj)).^ezc6(jj);
                temp4((EVnextOfPolicy==0)&(WGofPolicy==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                temp4(EVnextOfPolicy==0)=0;
            end

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(FofPolicy_jj).*(FofPolicy_jj~=0)); % finite but not zero
            temp2=FofPolicy_jj;
            temp2(becareful)=FofPolicy_jj(becareful).^ezc2(jj);
            temp2(FofPolicy_jj==0)=-Inf;

            Vjj=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4;

            temp5=logical(isfinite(Vjj).*(Vjj~=0));
            Vjj(temp5)=Vjj(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
            Vjj(Vjj==0)=-Inf;

            V(:,:,jj)=Vjj;
        end
    end

    %Transforming Value Fn out of Kronecker Form
    V=reshape(V,[n_a,n_z,N_j]);

elseif N_z>0 && N_e>0

    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1); % PolicyInd2Val auto-adds vfoptions.n_e
    l_daprime=size(PolicyValues,1);
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a,N_z*N_e,l_d+l_a,N_j] — keep shock dim combined for EvalFnOnAgentDist_Grid
    PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, [n_z,vfoptions.n_e], N_j, vfoptions);

    %% Calculate the Value Fn by backward iteration, evaluating the EZ recursion at the policy
    V=zeros(N_a,N_z,N_e,N_j,'gpuArray');

    for reverse_j=0:N_j-1
        jj=N_j-reverse_j; % current period, counts backwards from J-1

        % Evaluate Return Fn
        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        FofPolicy_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,[n_z,vfoptions.n_e],a_gridvals,[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)]),[N_a,N_z,N_e]);

        DiscountFactorParamsVec=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
        if vfoptions.EZoneminusbeta==1
            ezc1=1-DiscountFactorParamsVec;
        elseif vfoptions.EZoneminusbeta==2
            ezc1=1-sj(jj)*DiscountFactorParamsVec;
        end

        % Policy's aprime index
        if N_d==0
            optaprime=reshape(PolicyIndexesKron(1,:,:,jj),[N_a*N_z*N_e,1]);
        else
            optaprime=reshape(PolicyIndexesKron(2,:,:,jj),[N_a*N_z*N_e,1]);
        end

        % If there is a warm-glow, evaluate the warmglowfn (over aprime), then at the policy
        if warmglow==1
            WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
            WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a, a_grid, WGParamsVec);
            WGmatrix=WGmatrixraw;
            WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5(jj);
            WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
            WGofPolicy=reshape(WGmatrix(optaprime),[N_a,N_z,N_e]);
        end

        if jj==N_j
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            Vjj=FofPolicy_jj;
            becareful=logical(isfinite(FofPolicy_jj).*(FofPolicy_jj~=0)); % finite but not zero
            Vjj(becareful)=(ezc1*FofPolicy_jj(becareful).^ezc2(N_j)).^ezc7(N_j);
            Vjj(FofPolicy_jj==0)=-Inf;
            if warmglow==1
                becareful2=(WGofPolicy==0);
                WGofPolicy(isfinite(WGofPolicy))=ezc3*DiscountFactorParamsVec*(((1-sj(N_j))*WGofPolicy(isfinite(WGofPolicy)).^ezc8(N_j)).^ezc6(N_j));
                WGofPolicy(becareful2)=0;
                Vjj=Vjj+WGofPolicy;
            end
            V(:,:,:,jj)=Vjj;
        else
            % Part of Epstein-Zin is before taking expectation
            EVnextpre=V(:,:,:,jj+1);
            temp=EVnextpre;
            temp(isfinite(EVnextpre))=(ezc4*EVnextpre(isfinite(EVnextpre))).^ezc5(jj);
            temp(EVnextpre==0)=0;

            % One certainty-equivalent over the joint distribution of (zprime,eprime):
            % expectation over the iid e first, then over zprime (mirrors the EZ VFI raws)
            EVnext=sum(temp.*shiftdim(vfoptions.pi_e_J(:,jj+1),-2),3); % [N_a,N_z_to]
            EVnext=EVnext.*shiftdim(pi_z_J(:,:,jj)',-1);
            EVnext(isnan(EVnext))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EVnext=sum(EVnext,2); % sum over z', leaving a singular second dimension
            % EVnext is [N_a,1,N_z_from]; just index into it

            aprimez_index=optaprime+N_a*(kron(kron(ones(N_e,1,'gpuArray'),(1:1:N_z)'),ones(N_a,1,'gpuArray'))-1); % N_a*(z_index-1), but just with lots of kron

            EVnextOfPolicy=reshape(EVnext(aprimez_index),[N_a,N_z,N_e]);

            temp4=EVnextOfPolicy;
            if warmglow==1
                becareful=logical(isfinite(temp4).*isfinite(WGofPolicy)); % both are finite
                temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGofPolicy(becareful).^ezc8(jj)).^ezc6(jj);
                temp4((EVnextOfPolicy==0)&(WGofPolicy==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                temp4(EVnextOfPolicy==0)=0;
            end

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(FofPolicy_jj).*(FofPolicy_jj~=0)); % finite but not zero
            temp2=FofPolicy_jj;
            temp2(becareful)=FofPolicy_jj(becareful).^ezc2(jj);
            temp2(FofPolicy_jj==0)=-Inf;

            Vjj=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4;

            temp5=logical(isfinite(Vjj).*(Vjj~=0));
            Vjj(temp5)=Vjj(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
            Vjj(Vjj==0)=-Inf;

            V(:,:,:,jj)=Vjj;
        end
    end

    % Transforming Value Fn out of Kronecker Form
    V=reshape(V,[n_a,n_z,vfoptions.n_e,N_j]);
end

end
