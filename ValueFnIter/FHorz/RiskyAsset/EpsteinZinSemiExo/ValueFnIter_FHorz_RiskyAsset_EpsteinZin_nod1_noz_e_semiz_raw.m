function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_nod1_noz_e_semiz_raw(n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_e,n_u,N_j, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, e_gridvals_J, u_grid, pi_semiz_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9)
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% d4: ReturnFn but not aprimeFn, and determines semiz transitions
% no z; e iid

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_d4=prod(n_d4);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_semiz=prod(n_semiz);
N_e=prod(n_e);
N_u=prod(n_u);

special_n_d4=ones(1,length(n_d4));
d4_gridvals=CreateGridvals(n_d4,d4_grid,1);

N_a=N_a1*N_a2;

n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Policy=zeros(4,N_a,N_semiz,N_e,N_j,'gpuArray'); % (d2,d3,d4,a1prime)

%%
d23_grid=gpuArray(d23_grid);
a2_grid=gpuArray(a2_grid);
a1_grid=gpuArray(a1_grid);
u_grid=gpuArray(u_grid);

d3d4a1_gridvals=gpuArray(CreateGridvals([n_d3,n_d4,n_a1],[d3_grid;d4_grid;a1_grid],1));
a1a2_gridvals=gpuArray(CreateGridvals([n_a1,n_a2],[a1_grid;a2_grid],1));

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_semiz=ones(1,length(n_semiz));
end

semizind=shiftdim(0:1:N_semiz-1,-1);

V_ford4_jj=zeros(N_a,N_semiz,N_e,N_d4,'gpuArray');
Policy_ford4_jj=zeros(N_a,N_semiz,N_e,N_d4,'gpuArray');
d2index_ford4_jj=zeros(N_d3*N_a1,N_semiz,N_d4,'gpuArray');


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
if vfoptions.EZoneminusbeta==1
    ezc1=1-DiscountFactorParamsVec; % Just in case it depends on age
elseif vfoptions.EZoneminusbeta==2
    ezc1=1-sj(N_j)*DiscountFactorParamsVec;
end

% If there is a warm-glow at end of the final period, evaluate the warmglowfn
if warmglow==1
    WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,N_j);
    WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a2, a2_grid, WGParamsVec); % This depends on aprime
    WGmatrix=WGmatrixraw;
    WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5(N_j);
    WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity

    %  Switch WGmatrix from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: a2primeIndex is [N_d23,N_u], whereas a2primeProbs is [N_d23,N_u]

    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1); % [N_d*N_a1,N_u]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d*N_a1,N_u]
    aprimeProbs=repmat(a2primeProbs,N_a1,1);  % [N_d*N_a1,N_u]
    % Note: aprimeIndex corresponds to value of (a1, a2), but has dimension (d,a1)

    % Seems like interpolation has trouble due to numerical precision rounding errors when the two points being interpolated are equal
    % So I will add a check for when this happens, and then overwrite those (by setting aprimeProbs to zero)
    % WG depends only on a2prime (which depends on (d,u), not on a1prime), so use the a2-only indices
    skipinterp=logical(WGmatrix(a2primeIndex)==WGmatrix(a2primeIndex+1)); % Note, probably just do this off of a2prime values
    a2primeProbsWG=a2primeProbs;
    a2primeProbsWG(skipinterp)=0;

    WG1=WGmatrix(a2primeIndex); % (d,u), the lower a2prime
    WG2=WGmatrix(a2primeIndex+1); % (d,u), the upper a2prime
    % Apply the a2primeProbs
    WG1=reshape(WG1,[N_d23,N_u]).*a2primeProbsWG; % probability of lower grid point
    WG2=reshape(WG2,[N_d23,N_u]).*(1-a2primeProbsWG); % probability of upper grid point
    % If WG1 or WG2 is infinite, and probability is zero, we will get a nan, so get rid of these
    WG1(isnan(WG1))=0;
    WG2(isnan(WG2))=0;
    % Expectation over u (using pi_u), and then add the lower and upper
    WGmatrix=sum((WG1.*pi_u'),2)+sum((WG2.*pi_u'),2); % (d,1), sum over u
    WGmatrix=repmat(WGmatrix,N_a1,1); % (d-a1prime,1): WG does not depend on a1prime, expand to the (d,a1prime) rows

    % WGmatrix is over (d-a1prime,1)
    if ~isfield(vfoptions,'V_Jplus1')
        becareful=(WGmatrix==0);
        WGmatrix(isfinite(WGmatrix))=ezc3*DiscountFactorParamsVec*(((1-sj(N_j))*WGmatrix(isfinite(WGmatrix)).^ezc8(N_j)).^ezc6(N_j));
        WGmatrix(becareful)=0;
    end
    % Now just make it the right shape (temp4 spans semiz in all lowmemory tiers of the e raws)
    if isfield(vfoptions,'V_Jplus1')
        WGmatrix=WGmatrix.*ones(1,1,N_semiz);
    end
else
    WGmatrix=0;
end

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d3,n_d4,n_a1], [n_a1,n_a2], n_semiz, n_e, d3d4a1_gridvals, a1a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0)); % finite and not zero
        ReturnMatrix(becareful)=(ezc1*ReturnMatrix(becareful).^ezc2(N_j)).^ezc7(N_j); % Otherwise can get things like 0 to negative power equals infinity
        ReturnMatrix(ReturnMatrix==0)=-Inf;

        if warmglow==1
            % Time to refine
            % First: ReturnMatrix, we can refine out d1
            % no d1 here
            % Second: warm-glow, we can refine out d2
            [WGmatrix_onlyd3,d2index]=max(ezc9*reshape((~isinf(WGmatrix)).*WGmatrix,[N_d2,N_d3*N_a1]),[],1);
            % WGmatrix_onlyd3 is over (d3,a1prime); rows of ReturnMatrix are (d3,d4,a1prime), so spread it over d4
            WGmatrix_onlyd3=reshape(repmat(reshape(WGmatrix_onlyd3,[N_d3,1,N_a1]),1,N_d4,1),[N_d3*N_d4*N_a1,1]);
            % Now put together entireRHS, which just depends on (d3,d4,a1prime)
            entireRHS=ReturnMatrix+ezc9*WGmatrix_onlyd3;

            [Vtemp,maxindex]=max(entireRHS,[],1);
            V(:,:,:,N_j)=Vtemp;
            dindex=rem(maxindex-1,N_d3*N_d4)+1;
            d3index=rem(dindex-1,N_d3)+1;
            a1primeindex=ceil(maxindex/(N_d3*N_d4));
            Policy(1,:,:,:,N_j)=d2index(d3index+N_d3*(a1primeindex-1)); % d2 (note: no a nor semiz in WGmatrix)
            Policy(2,:,:,:,N_j)=d3index; % d3
            Policy(3,:,:,:,N_j)=shiftdim(ceil(dindex/N_d3),-1); % d4
            Policy(4,:,:,:,N_j)=shiftdim(a1primeindex,-1); % a1prime
        elseif warmglow==0
            [Vtemp,maxindex]=max(ReturnMatrix,[],1);
            V(:,:,:,N_j)=Vtemp;
            dindex=rem(maxindex-1,N_d3*N_d4)+1;
            Policy(1,:,:,:,N_j)=1;
            Policy(2,:,:,:,N_j)=rem(dindex-1,N_d3)+1;
            Policy(3,:,:,:,N_j)=shiftdim(ceil(dindex/N_d3),-1);
            Policy(4,:,:,:,N_j)=shiftdim(ceil(maxindex/(N_d3*N_d4)),-1);
        end

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d3,n_d4,n_a1], [n_a1,n_a2], n_semiz, special_n_e, d3d4a1_gridvals, a1a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_e).*(ReturnMatrix_e~=0)); % finite and not zero
            ReturnMatrix_e(becareful)=(ezc1*ReturnMatrix_e(becareful).^ezc2(N_j)).^ezc7(N_j); % Otherwise can get things like 0 to negative power equals infinity
            ReturnMatrix_e(ReturnMatrix_e==0)=-Inf;

            if warmglow==1
                % Time to refine
                % First: ReturnMatrix, we can refine out d1
                % no d1 here
                % Second: warm-glow, we can refine out d2
                [WGmatrix_onlyd3,d2index]=max(ezc9*reshape((~isinf(WGmatrix)).*WGmatrix,[N_d2,N_d3*N_a1]),[],1);
                % WGmatrix_onlyd3 is over (d3,a1prime); rows of ReturnMatrix are (d3,d4,a1prime), so spread it over d4
                WGmatrix_onlyd3=reshape(repmat(reshape(WGmatrix_onlyd3,[N_d3,1,N_a1]),1,N_d4,1),[N_d3*N_d4*N_a1,1]);
                % Now put together entireRHS, which just depends on (d3,d4,a1prime)
                entireRHS=ReturnMatrix_e+ezc9*WGmatrix_onlyd3;

                [Vtemp,maxindex]=max(entireRHS,[],1);
                V(:,:,e_c,N_j)=Vtemp;
                dindex=rem(maxindex-1,N_d3*N_d4)+1;
                d3index=rem(dindex-1,N_d3)+1;
                a1primeindex=ceil(maxindex/(N_d3*N_d4));
                Policy(1,:,:,e_c,N_j)=d2index(d3index+N_d3*(a1primeindex-1)); % d2 (note: no a nor semiz in WGmatrix)
                Policy(2,:,:,e_c,N_j)=d3index; % d3
                Policy(3,:,:,e_c,N_j)=shiftdim(ceil(dindex/N_d3),-1); % d4
                Policy(4,:,:,e_c,N_j)=shiftdim(a1primeindex,-1); % a1prime
            elseif warmglow==0
                [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
                V(:,:,e_c,N_j)=Vtemp;
                dindex=rem(maxindex-1,N_d3*N_d4)+1;
                Policy(1,:,:,e_c,N_j)=1;
                Policy(2,:,:,e_c,N_j)=rem(dindex-1,N_d3)+1;
                Policy(3,:,:,e_c,N_j)=shiftdim(ceil(dindex/N_d3),-1);
                Policy(4,:,:,e_c,N_j)=shiftdim(ceil(maxindex/(N_d3*N_d4)),-1);
            end
        end

    elseif vfoptions.lowmemory>=2 % lm2 already does the most-looped variant, so it also serves the higher lowmemory values
        for z_c=1:N_semiz % outer loop over semiz
            z_val=semiz_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e % inner loop over e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d3,n_d4,n_a1], [n_a1,n_a2], special_n_semiz, special_n_e, d3d4a1_gridvals, a1a2_gridvals, z_val, e_val, ReturnFnParamsVec);

                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(ReturnMatrix_ze).*(ReturnMatrix_ze~=0)); % finite and not zero
                ReturnMatrix_ze(becareful)=(ezc1*ReturnMatrix_ze(becareful).^ezc2(N_j)).^ezc7(N_j); % Otherwise can get things like 0 to negative power equals infinity
                ReturnMatrix_ze(ReturnMatrix_ze==0)=-Inf;

                if warmglow==1
                    % Time to refine
                    % First: ReturnMatrix, we can refine out d1
                    % no d1 here
                    % Second: warm-glow, we can refine out d2
                    [WGmatrix_onlyd3,d2index]=max(ezc9*reshape((~isinf(WGmatrix)).*WGmatrix,[N_d2,N_d3*N_a1]),[],1);
                    % WGmatrix_onlyd3 is over (d3,a1prime); rows of ReturnMatrix are (d3,d4,a1prime), so spread it over d4
                    WGmatrix_onlyd3=reshape(repmat(reshape(WGmatrix_onlyd3,[N_d3,1,N_a1]),1,N_d4,1),[N_d3*N_d4*N_a1,1]);
                    % Now put together entireRHS, which just depends on (d3,d4,a1prime)
                    entireRHS=ReturnMatrix_ze+ezc9*WGmatrix_onlyd3;

                    [Vtemp,maxindex]=max(entireRHS,[],1);
                    V(:,z_c,e_c,N_j)=Vtemp;
                    dindex=rem(maxindex-1,N_d3*N_d4)+1;
                    d3index=rem(dindex-1,N_d3)+1;
                    a1primeindex=ceil(maxindex/(N_d3*N_d4));
                    Policy(1,:,z_c,e_c,N_j)=d2index(d3index+N_d3*(a1primeindex-1)); % d2 (note: no a nor semiz in WGmatrix)
                    Policy(2,:,z_c,e_c,N_j)=d3index; % d3
                    Policy(3,:,z_c,e_c,N_j)=shiftdim(ceil(dindex/N_d3),-1); % d4
                    Policy(4,:,z_c,e_c,N_j)=shiftdim(a1primeindex,-1); % a1prime
                elseif warmglow==0
                    [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                    V(:,z_c,e_c,N_j)=Vtemp;
                    dindex=rem(maxindex-1,N_d3*N_d4)+1;
                    Policy(1,:,z_c,e_c,N_j)=1;
                    Policy(2,:,z_c,e_c,N_j)=rem(dindex-1,N_d3)+1;
                    Policy(3,:,z_c,e_c,N_j)=shiftdim(ceil(dindex/N_d3),-1);
                    Policy(4,:,z_c,e_c,N_j)=shiftdim(ceil(maxindex/(N_d3*N_d4)),-1);
                end
            end
        end
    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]);

    if warmglow==0 % if warmglow==1 these were already created above
        aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
        [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
        % Note: a2primeIndex is [N_d23,N_u], whereas a2primeProbs is [N_d23,N_u]

        aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1); % [N_d*N_a1,N_u]
        aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d*N_a1,N_u]
        % aprimeProbs=repmat(a2primeProbs,N_a1,1);  % [N_d*N_a1,N_u]
        % Note: aprimeIndex corresponds to value of (a1, a2), but has dimension (d,a1)
    end

    % Part of Epstein-Zin is before taking expectation
    temp=V_Jplus1;
    temp(isfinite(V_Jplus1))=(ezc4*V_Jplus1(isfinite(V_Jplus1))).^ezc5(N_j);
    temp(V_Jplus1==0)=0; % otherwise zero to negative power is set to infinity

    % Integrate over e' first (e is iid); part of the same joint certainty-equivalent as (u,semiz')
    temp=sum(temp.*shiftdim(pi_e_J(:,N_j+1),-2),3);

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(N_j)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,N_j);
    end

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d3_special_d4_a1_gridvals=gpuArray(CreateGridvals([n_d3,special_n_d4,n_a1], [d3_grid; d4_gridvals(d4_c,:)'; a1_grid], 1));
            ReturnMatrix_d4=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d3,special_n_d4,n_a1], [n_a1,n_a2], n_semiz, n_e, d3_special_d4_a1_gridvals, a1a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_d4).*(ReturnMatrix_d4~=0)); % finite and not zero
            temp2=ReturnMatrix_d4;
            temp2(becareful)=ReturnMatrix_d4(becareful).^ezc2(N_j);
            temp2(ReturnMatrix_d4==0)=-Inf;

            EV=temp.*shiftdim(pi_semizd4',-1);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Seems like interpolation has trouble due to numerical precision rounding errors when the two points being interpolated are equal
            % So I will add a check for when this happens, and then overwrite those (by setting aprimeProbs to zero)
            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1))); % Note, probably just do this off of a2prime values
            aprimeProbs=repmat(a2primeProbs,N_a1,N_semiz);  % [N_d*N_a1,N_u]
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_semiz]);

            % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
            EV1=EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1)); % (d,u,semiz), the lower aprime
            EV2=EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)); % (d,u,semiz), the upper aprime

            % Apply the aprimeProbs
            EV1=reshape(EV1,[N_d23*N_a1,N_u,N_semiz]).*aprimeProbs; % probability of lower grid point
            EV2=reshape(EV2,[N_d23*N_a1,N_u,N_semiz]).*(1-aprimeProbs); % probability of upper grid point

            % Expectation over u (using pi_u), and then add the lower and upper
            EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2); % (d,1,semiz), sum over u
            % EV is over (d,1,semiz)

            % Part of Epstein-Zin is after taking expectation
            temp4=EV;
            if warmglow==1
                becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
                temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrix(becareful).^ezc8(N_j)).^ezc6(N_j);
                temp4((EV==0)&(WGmatrix==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4)).^ezc8(N_j)).^ezc6(N_j);
                temp4(EV==0)=0;
            end

            % Time to refine
            % First: ReturnMatrix, we can refine out d1
            % no d1 here
            % Second: EV, we can refine out d2
            [temp4_onlyd3,d2index]=max(ezc9*ezc3*reshape((~isinf(temp4)).*temp4,[N_d2,N_d3*N_a1,1,N_semiz]),[],1);
            % Now put together entireRHS, which just depends on d3
            entireRHS=ezc1*temp2+DiscountFactorParamsVec*ezc9*shiftdim(temp4_onlyd3,1);
            % entireRHS=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4;

            temp5=logical(isfinite(entireRHS).*(entireRHS~=0));
            entireRHS(temp5)=entireRHS(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
            entireRHS(entireRHS==0)=-Inf;

            [Vtemp,maxindex]=max(entireRHS,[],1);

            V_ford4_jj(:,:,:,d4_c)=shiftdim(Vtemp,1);
            Policy_ford4_jj(:,:,:,d4_c)=shiftdim(maxindex,1);
            d2index_ford4_jj(:,:,d4_c)=squeeze(d2index);
        end

        [V_jj,maxindex]=max(V_ford4_jj,[],4);
        V(:,:,:,N_j)=V_jj;
        Policy(3,:,:,:,N_j)=maxindex;
        maxindex_d4=reshape(maxindex,[N_a*N_semiz*N_e,1]);
        d3a1prime_ind=reshape(Policy_ford4_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex_d4-1)),[1,N_a,N_semiz,N_e]);
        Policy(1,:,:,:,N_j)=shiftdim(d2index_ford4_jj(d3a1prime_ind+N_d3*N_a1*semizind+N_d3*N_a1*N_semiz*shiftdim(maxindex-1,-1)),-1);
        Policy(2,:,:,:,N_j)=shiftdim(rem(d3a1prime_ind-1,N_d3)+1,-1);
        Policy(4,:,:,:,N_j)=shiftdim(ceil(d3a1prime_ind/N_d3),-1);

    elseif vfoptions.lowmemory>=1 % terminal lm1 already does the most-looped variant, so it also serves the higher lowmemory values
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d3_special_d4_a1_gridvals=gpuArray(CreateGridvals([n_d3,special_n_d4,n_a1], [d3_grid; d4_gridvals(d4_c,:)'; a1_grid], 1));

            EV=temp.*shiftdim(pi_semizd4',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)));
            aprimeProbs=repmat(a2primeProbs,N_a1,N_semiz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_semiz]);

            EV1=EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1));
            EV2=EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1));

            EV1=reshape(EV1,[N_d23*N_a1,N_u,N_semiz]).*aprimeProbs;
            EV2=reshape(EV2,[N_d23*N_a1,N_u,N_semiz]).*(1-aprimeProbs);

            EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2);

            % Part of Epstein-Zin is after taking expectation
            temp4=EV;
            if warmglow==1
                becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
                temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrix(becareful).^ezc8(N_j)).^ezc6(N_j);
                temp4((EV==0)&(WGmatrix==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4)).^ezc8(N_j)).^ezc6(N_j);
                temp4(EV==0)=0;
            end

            % Time to refine
            % Second (out of order): EV, we can refine out d2
            [temp4_onlyd3,d2index]=max(ezc9*ezc3*reshape((~isinf(temp4)).*temp4,[N_d2,N_d3*N_a1,1,N_semiz]),[],1);
            DiscountedEV_onlyd3=DiscountFactorParamsVec*ezc9*shiftdim(temp4_onlyd3,1);
            d2index_ford4_jj(:,:,d4_c)=squeeze(d2index);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d4e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d3,special_n_d4,n_a1], [n_a1,n_a2], n_semiz, special_n_e, d3_special_d4_a1_gridvals, a1a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);

                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(ReturnMatrix_d4e).*(ReturnMatrix_d4e~=0)); % finite and not zero
                temp2=ReturnMatrix_d4e;
                temp2(becareful)=ReturnMatrix_d4e(becareful).^ezc2(N_j);
                temp2(ReturnMatrix_d4e==0)=-Inf;

                entireRHS_e=ezc1*temp2+DiscountedEV_onlyd3;

                temp5=logical(isfinite(entireRHS_e).*(entireRHS_e~=0));
                entireRHS_e(temp5)=entireRHS_e(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
                entireRHS_e(entireRHS_e==0)=-Inf;

                [Vtemp,maxindex]=max(entireRHS_e,[],1);
                V_ford4_jj(:,:,e_c,d4_c)=shiftdim(Vtemp,1);
                Policy_ford4_jj(:,:,e_c,d4_c)=shiftdim(maxindex,1);
            end
        end

        [V_jj,maxindex]=max(V_ford4_jj,[],4);
        V(:,:,:,N_j)=V_jj;
        Policy(3,:,:,:,N_j)=maxindex;
        maxindex_d4=reshape(maxindex,[N_a*N_semiz*N_e,1]);
        d3a1prime_ind=reshape(Policy_ford4_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex_d4-1)),[1,N_a,N_semiz,N_e]);
        Policy(1,:,:,:,N_j)=shiftdim(d2index_ford4_jj(d3a1prime_ind+N_d3*N_a1*semizind+N_d3*N_a1*N_semiz*shiftdim(maxindex-1,-1)),-1);
        Policy(2,:,:,:,N_j)=shiftdim(rem(d3a1prime_ind-1,N_d3)+1,-1);
        Policy(4,:,:,:,N_j)=shiftdim(ceil(d3a1prime_ind/N_d3),-1);
    end
end



%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    if vfoptions.EZoneminusbeta==1
        ezc1=1-DiscountFactorParamsVec; % Just in case it depends on age
    elseif vfoptions.EZoneminusbeta==2
        ezc1=1-sj(jj)*DiscountFactorParamsVec;
    end

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: a2primeIndex is [N_d23,N_u], whereas a2primeProbs is [N_d23,N_u]
    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1); % [N_d*N_a1,N_u]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d*N_a1,N_u]
    % aprimeProbs=repmat(a2primeProbs,N_a1,1);  % [N_d*N_a1,N_u]
    % Note: aprimeIndex corresponds to value of (a1, a2), but has dimension (d,a1)

    % If there is a warm-glow, evaluate the warmglowfn
    if warmglow==1
        WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
        WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a2, a2_grid, WGParamsVec);
        WGmatrix=WGmatrixraw;
        WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5(jj);
        WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
        %  Switch WGmatrix from being in terms of aprime to being in terms of d (in expectation because of the u shocks)

        % Seems like interpolation has trouble due to numerical precision rounding errors when the two points being interpolated are equal
        % So I will add a check for when this happens, and then overwrite those (by setting aprimeProbs to zero)
        % WG depends only on a2prime (which depends on (d,u), not on a1prime), so use the a2-only indices
        skipinterp=logical(WGmatrix(a2primeIndex)==WGmatrix(a2primeIndex+1)); % Note, probably just do this off of a2prime values
        a2primeProbsWG=a2primeProbs;
        a2primeProbsWG(skipinterp)=0;

        WG1=WGmatrix(a2primeIndex); % (d,u), the lower a2prime
        WG2=WGmatrix(a2primeIndex+1); % (d,u), the upper a2prime
        % Apply the a2primeProbs
        WG1=reshape(WG1,[N_d23,N_u]).*a2primeProbsWG; % probability of lower grid point
        WG2=reshape(WG2,[N_d23,N_u]).*(1-a2primeProbsWG); % probability of upper grid point
        % If WG1 or WG2 is infinite, and probability is zero, we will get a nan, so get rid of these
        WG1(isnan(WG1))=0;
        WG2(isnan(WG2))=0;
        % Expectation over u (using pi_u), and then add the lower and upper
        WGmatrix=sum((WG1.*pi_u'),2)+sum((WG2.*pi_u'),2); % (d,1), sum over u
        WGmatrix=repmat(WGmatrix,N_a1,1); % (d-a1prime,1): WG does not depend on a1prime, expand to the (d,a1prime) rows
        % WGmatrix is over (d-a1prime,1)
        % Now just make it the right shape (temp4 spans semiz in all lowmemory tiers of the e raws)
        WGmatrix=WGmatrix.*ones(1,1,N_semiz);
    end

    EVpre=V(:,:,:,jj+1);

    % Part of Epstein-Zin is before taking expectation
    temp=EVpre;
    temp(isfinite(EVpre))=(ezc4*EVpre(isfinite(EVpre))).^ezc5(jj);
    temp(EVpre==0)=0;

    % Integrate over e' first (e is iid); part of the same joint certainty-equivalent as (u,semiz')
    temp=sum(temp.*shiftdim(pi_e_J(:,jj+1),-2),3);

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(jj)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,jj);
    end

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d3_special_d4_a1_gridvals=gpuArray(CreateGridvals([n_d3,special_n_d4,n_a1], [d3_grid; d4_gridvals(d4_c,:)'; a1_grid], 1));
            ReturnMatrix_d4=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d3,special_n_d4,n_a1], [n_a1,n_a2], n_semiz, n_e, d3_special_d4_a1_gridvals, a1a2_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec);

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_d4).*(ReturnMatrix_d4~=0)); % finite and not zero
            temp2=ReturnMatrix_d4;
            temp2(becareful)=ReturnMatrix_d4(becareful).^ezc2(jj);
            temp2(ReturnMatrix_d4==0)=-Inf;

            EV=temp.*shiftdim(pi_semizd4',-1);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Seems like interpolation has trouble due to numerical precision rounding errors when the two points being interpolated are equal
            % So I will add a check for when this happens, and then overwrite those (by setting aprimeProbs to zero)
            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1))); % Note, probably just do this off of a2prime values
            aprimeProbs=repmat(a2primeProbs,N_a1,N_semiz);  % [N_d*N_a1,N_u]
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_semiz]);

            % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
            EV1=EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1)); % (d,u,semiz), the lower aprime
            EV2=EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)); % (d,u,semiz), the upper aprime

            % Apply the aprimeProbs
            EV1=reshape(EV1,[N_d23*N_a1,N_u,N_semiz]).*aprimeProbs; % probability of lower grid point
            EV2=reshape(EV2,[N_d23*N_a1,N_u,N_semiz]).*(1-aprimeProbs); % probability of upper grid point

            % Expectation over u (using pi_u), and then add the lower and upper
            EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2); % (d,1,semiz), sum over u
            % EV is over (d,1,semiz)

            % Part of Epstein-Zin is after taking expectation
            temp4=EV;
            if warmglow==1
                becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
                temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrix(becareful).^ezc8(jj)).^ezc6(jj);
                temp4((EV==0)&(WGmatrix==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                temp4(EV==0)=0;
            end

            % Time to refine
            % First: ReturnMatrix, we can refine out d1
            % no d1 here
            % Second: EV, we can refine out d2
            [temp4_onlyd3,d2index]=max(ezc9*ezc3*reshape((~isinf(temp4)).*temp4,[N_d2,N_d3*N_a1,1,N_semiz]),[],1);
            % Now put together entireRHS, which just depends on d3
            entireRHS=ezc1*temp2+DiscountFactorParamsVec*ezc9*shiftdim(temp4_onlyd3,1);
            % entireRHS=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4;

            temp5=logical(isfinite(entireRHS).*(entireRHS~=0));
            entireRHS(temp5)=entireRHS(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
            entireRHS(entireRHS==0)=-Inf;

            [Vtemp,maxindex]=max(entireRHS,[],1);

            V_ford4_jj(:,:,:,d4_c)=shiftdim(Vtemp,1);
            Policy_ford4_jj(:,:,:,d4_c)=shiftdim(maxindex,1);
            d2index_ford4_jj(:,:,d4_c)=squeeze(d2index);
        end

        [V_jj,maxindex]=max(V_ford4_jj,[],4);
        V(:,:,:,jj)=V_jj;
        Policy(3,:,:,:,jj)=maxindex;
        maxindex_d4=reshape(maxindex,[N_a*N_semiz*N_e,1]);
        d3a1prime_ind=reshape(Policy_ford4_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex_d4-1)),[1,N_a,N_semiz,N_e]);
        Policy(1,:,:,:,jj)=shiftdim(d2index_ford4_jj(d3a1prime_ind+N_d3*N_a1*semizind+N_d3*N_a1*N_semiz*shiftdim(maxindex-1,-1)),-1);
        Policy(2,:,:,:,jj)=shiftdim(rem(d3a1prime_ind-1,N_d3)+1,-1);
        Policy(4,:,:,:,jj)=shiftdim(ceil(d3a1prime_ind/N_d3),-1);

    elseif vfoptions.lowmemory==1
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d3_special_d4_a1_gridvals=gpuArray(CreateGridvals([n_d3,special_n_d4,n_a1], [d3_grid; d4_gridvals(d4_c,:)'; a1_grid], 1));

            EV=temp.*shiftdim(pi_semizd4',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)));
            aprimeProbs=repmat(a2primeProbs,N_a1,N_semiz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_semiz]);

            EV1=EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1));
            EV2=EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1));

            EV1=reshape(EV1,[N_d23*N_a1,N_u,N_semiz]).*aprimeProbs;
            EV2=reshape(EV2,[N_d23*N_a1,N_u,N_semiz]).*(1-aprimeProbs);

            EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2);

            % Part of Epstein-Zin is after taking expectation
            temp4=EV;
            if warmglow==1
                becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
                temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrix(becareful).^ezc8(jj)).^ezc6(jj);
                temp4((EV==0)&(WGmatrix==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                temp4(EV==0)=0;
            end

            % Time to refine
            % Second (out of order): EV, we can refine out d2
            [temp4_onlyd3,d2index]=max(ezc9*ezc3*reshape((~isinf(temp4)).*temp4,[N_d2,N_d3*N_a1,1,N_semiz]),[],1);
            DiscountedEV_onlyd3=DiscountFactorParamsVec*ezc9*shiftdim(temp4_onlyd3,1);
            d2index_ford4_jj(:,:,d4_c)=squeeze(d2index);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_d4e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d3,special_n_d4,n_a1], [n_a1,n_a2], n_semiz, special_n_e, d3_special_d4_a1_gridvals, a1a2_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec);

                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(ReturnMatrix_d4e).*(ReturnMatrix_d4e~=0)); % finite and not zero
                temp2=ReturnMatrix_d4e;
                temp2(becareful)=ReturnMatrix_d4e(becareful).^ezc2(jj);
                temp2(ReturnMatrix_d4e==0)=-Inf;

                entireRHS_e=ezc1*temp2+DiscountedEV_onlyd3;

                temp5=logical(isfinite(entireRHS_e).*(entireRHS_e~=0));
                entireRHS_e(temp5)=entireRHS_e(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
                entireRHS_e(entireRHS_e==0)=-Inf;

                [Vtemp,maxindex]=max(entireRHS_e,[],1);
                V_ford4_jj(:,:,e_c,d4_c)=shiftdim(Vtemp,1);
                Policy_ford4_jj(:,:,e_c,d4_c)=shiftdim(maxindex,1);
            end
        end

        [V_jj,maxindex]=max(V_ford4_jj,[],4);
        V(:,:,:,jj)=V_jj;
        Policy(3,:,:,:,jj)=maxindex;
        maxindex_d4=reshape(maxindex,[N_a*N_semiz*N_e,1]);
        d3a1prime_ind=reshape(Policy_ford4_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex_d4-1)),[1,N_a,N_semiz,N_e]);
        Policy(1,:,:,:,jj)=shiftdim(d2index_ford4_jj(d3a1prime_ind+N_d3*N_a1*semizind+N_d3*N_a1*N_semiz*shiftdim(maxindex-1,-1)),-1);
        Policy(2,:,:,:,jj)=shiftdim(rem(d3a1prime_ind-1,N_d3)+1,-1);
        Policy(4,:,:,:,jj)=shiftdim(ceil(d3a1prime_ind/N_d3),-1);

    elseif vfoptions.lowmemory>=2 % lm2 already does the most-looped variant, so it also serves the higher lowmemory values
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d3_special_d4_a1_gridvals=gpuArray(CreateGridvals([n_d3,special_n_d4,n_a1], [d3_grid; d4_gridvals(d4_c,:)'; a1_grid], 1));

            EV=temp.*shiftdim(pi_semizd4',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)));
            aprimeProbs=repmat(a2primeProbs,N_a1,N_semiz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_semiz]);

            EV1=EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1));
            EV2=EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1));

            EV1=reshape(EV1,[N_d23*N_a1,N_u,N_semiz]).*aprimeProbs;
            EV2=reshape(EV2,[N_d23*N_a1,N_u,N_semiz]).*(1-aprimeProbs);

            EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2);

            % Part of Epstein-Zin is after taking expectation
            temp4=EV;
            if warmglow==1
                becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
                temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrix(becareful).^ezc8(jj)).^ezc6(jj);
                temp4((EV==0)&(WGmatrix==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                temp4(EV==0)=0;
            end

            % Time to refine
            % Second (out of order): EV, we can refine out d2
            [temp4_onlyd3,d2index]=max(ezc9*ezc3*reshape((~isinf(temp4)).*temp4,[N_d2,N_d3*N_a1,1,N_semiz]),[],1);
            d2index_ford4_jj(:,:,d4_c)=squeeze(d2index);

            for z_c=1:N_semiz % outer loop over semiz
                DiscountedEV_onlyd3=DiscountFactorParamsVec*ezc9*shiftdim(temp4_onlyd3(:,:,:,z_c),1);
                z_val=semiz_gridvals_J(z_c,:,jj);
                for e_c=1:N_e % inner loop over e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_d4ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d3,special_n_d4,n_a1], [n_a1,n_a2], special_n_semiz, special_n_e, d3_special_d4_a1_gridvals, a1a2_gridvals, z_val, e_val, ReturnFnParamsVec);

                    % Modify the Return Function appropriately for Epstein-Zin Preferences
                    becareful=logical(isfinite(ReturnMatrix_d4ze).*(ReturnMatrix_d4ze~=0)); % finite and not zero
                    temp2=ReturnMatrix_d4ze;
                    temp2(becareful)=ReturnMatrix_d4ze(becareful).^ezc2(jj);
                    temp2(ReturnMatrix_d4ze==0)=-Inf;

                    entireRHS_ze=ezc1*temp2+DiscountedEV_onlyd3;

                    temp5=logical(isfinite(entireRHS_ze).*(entireRHS_ze~=0));
                    entireRHS_ze(temp5)=entireRHS_ze(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
                    entireRHS_ze(entireRHS_ze==0)=-Inf;

                    [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                    V_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(Vtemp,1);
                    Policy_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(maxindex,1);
                end
            end
        end

        [V_jj,maxindex]=max(V_ford4_jj,[],4);
        V(:,:,:,jj)=V_jj;
        Policy(3,:,:,:,jj)=maxindex;
        maxindex_d4=reshape(maxindex,[N_a*N_semiz*N_e,1]);
        d3a1prime_ind=reshape(Policy_ford4_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex_d4-1)),[1,N_a,N_semiz,N_e]);
        Policy(1,:,:,:,jj)=shiftdim(d2index_ford4_jj(d3a1prime_ind+N_d3*N_a1*semizind+N_d3*N_a1*N_semiz*shiftdim(maxindex-1,-1)),-1);
        Policy(2,:,:,:,jj)=shiftdim(rem(d3a1prime_ind-1,N_d3)+1,-1);
        Policy(4,:,:,:,jj)=shiftdim(ceil(d3a1prime_ind/N_d3),-1);
    end
end


end
