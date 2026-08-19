function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_nod1_noz_e_raw(n_d2,n_d3,n_a1,n_a2,n_e,n_u, N_j, d2_grid, d3_grid, a1_grid, a2_grid, e_gridvals_J, u_grid, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9)
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_e=prod(n_e);
N_u=prod(n_u);

N_a=N_a1*N_a2;

% For ReturnFn
% n_d3
% N_d3
% d3_grid
% For aprimeFn
n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(3,N_a,N_e,N_j,'gpuArray'); % d2, d3, a1prime

%%
pi_u=shiftdim(pi_u,-1); % 2nd dimension

d3a1_gridvals=CreateGridvals([n_d3,n_a1],[d3_grid;a1_grid],1);
a1a2_gridvals=CreateGridvals([n_a1,n_a2],[a1_grid;a2_grid],1);

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
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
    % Note: a2primeIndex is [N_d,N_u], whereas a2primeProbs is [N_d,N_u]

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
    WGmatrix=sum((WG1.*pi_u),2)+sum((WG2.*pi_u),2); % (d,1), sum over u
    WGmatrix=repmat(WGmatrix,N_a1,1); % (d-a1prime,1): WG does not depend on a1prime, expand to the (d,a1prime) rows
    % WGmatrix is over (d-a1prime,1)
    if ~isfield(vfoptions,'V_Jplus1')
        becareful=(WGmatrix==0);
        WGmatrix(isfinite(WGmatrix))=ezc3*DiscountFactorParamsVec*(((1-sj(N_j))*WGmatrix(isfinite(WGmatrix)).^ezc8(N_j)).^ezc6(N_j));
        WGmatrix(becareful)=0;
    end
else
    WGmatrix=0;
end


if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d3,n_a1], [n_a1,n_a2], n_e, d3a1_gridvals, a1a2_gridvals,e_gridvals_J(:,:,N_j), ReturnFnParamsVec);

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0)); % finite and not zero
        ReturnMatrix(becareful)=(ezc1*ReturnMatrix(becareful).^ezc2(N_j)).^ezc7(N_j); % Otherwise can get things like 0 to negative power equals infinity
        ReturnMatrix(ReturnMatrix==0)=-Inf;

        %Calc the max and it's index
        if warmglow==1
            % Time to refine
            % First: ReturnMatrix, we can refine out d1
            % no d1 here
            % Second: EV, we can refine out d2
            [WGmatrix_onlyd3,d2index]=max(ezc9*reshape((~isinf(WGmatrix)).*WGmatrix,[N_d2,N_d3*N_a1]),[],1);
            % Now put together entireRHS, which just depends on d3
            entireRHS=ReturnMatrix+ezc9*shiftdim(WGmatrix_onlyd3,1);

            [Vtemp,maxindex]=max(entireRHS,[],1);

            V(:,:,N_j)=shiftdim(Vtemp,1);
            Policy(2,:,:,N_j)=shiftdim(rem(maxindex-1,N_d3)+1,1); % d3
            Policy(3,:,:,N_j)=shiftdim(ceil(maxindex/N_d3),1); % a1prime
            Policy(1,:,:,N_j)=shiftdim(d2index(maxindex),1); % d2, note: no a nor e in WGmatrix
        elseif warmglow==0
            [Vtemp,maxindex]=max(ReturnMatrix,[],1);

            V(:,:,N_j)=shiftdim(Vtemp,1);
            Policy(1,:,:,N_j)=1; % d2, is meaningless anyway
            Policy(2,:,:,N_j)=shiftdim(rem(maxindex-1,N_d3)+1,1); % d3
            Policy(3,:,:,N_j)=shiftdim(ceil(maxindex/N_d3),-1); % a1prime
        end

    elseif vfoptions.lowmemory>=1 % lm1 already does the most-looped variant, so it also serves the higher lowmemory values
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d3,n_a1], [n_a1,n_a2], special_n_e, d3a1_gridvals, a1a2_gridvals, e_val, ReturnFnParamsVec);

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_e).*(ReturnMatrix_e~=0)); % finite and not zero
            ReturnMatrix_e(becareful)=(ezc1*ReturnMatrix_e(becareful).^ezc2(N_j)).^ezc7(N_j); % Otherwise can get things like 0 to negative power equals infinity
            ReturnMatrix_e(ReturnMatrix_e==0)=-Inf;

            %Calc the max and it's index
            if warmglow==1
                % Time to refine
                % First: ReturnMatrix, we can refine out d1
                % no d1 here
                % Second: EV, we can refine out d2
                [WGmatrix_onlyd3,d2index]=max(ezc9*reshape((~isinf(WGmatrix)).*WGmatrix,[N_d2,N_d3*N_a1]),[],1);
                % Now put together entireRHS, which just depends on d3
                entireRHS_e=ReturnMatrix_e+ezc9*shiftdim(WGmatrix_onlyd3,1);

                [Vtemp,maxindex]=max(entireRHS_e,[],1);

                V(:,e_c,N_j)=Vtemp;
                Policy(2,:,e_c,N_j)=shiftdim(rem(maxindex-1,N_d3)+1,1); % d3
                Policy(3,:,e_c,N_j)=shiftdim(ceil(maxindex/N_d3),1); % a1prime
                Policy(1,:,e_c,N_j)=shiftdim(d2index(maxindex),1); % d2
            elseif warmglow==0
                [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);

                V(:,e_c,N_j)=Vtemp;
                Policy(1,:,e_c,N_j)=1; % d2, is meaningless anyway
                Policy(2,:,e_c,N_j)=shiftdim(rem(maxindex-1,N_d3)+1,1); % d3
                Policy(3,:,e_c,N_j)=shiftdim(ceil(maxindex/N_d3),-1); % a1prime
            end
        end
    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_e]);    % First, switch V_Jplus1 into Kron form

    if warmglow==0 % if warmglow==1 these were already created above
        aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
        [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
        % Note: a2primeIndex is [N_d,N_u], whereas a2primeProbs is [N_d,N_u]

        aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1); % [N_d*N_a1,N_u]
        aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d*N_a1,N_u]
        % Note: aprimeIndex corresponds to value of (a1, a2), but has dimension (d,a1)
    end

    % Part of Epstein-Zin is before taking expectation
    temp=V_Jplus1;
    temp(isfinite(V_Jplus1))=(ezc4*V_Jplus1(isfinite(V_Jplus1))).^ezc5(N_j);
    temp(V_Jplus1==0)=0; % otherwise zero to negative power is set to infinity

    % Take expectation over e
    temp=sum(temp.*pi_e_J(:,N_j+1)',2);

    % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
    EVlower=reshape(temp(aprimeIndex),[N_d23*N_a1,N_u]); % the lower aprime
    EVupper=reshape(temp(aprimeplus1Index),[N_d23*N_a1,N_u]); % the upper aprime
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    aprimeProbs=repmat(a2primeProbs,N_a1,1);  % [N_d*N_a1,N_u]
    skipinterp=(EVlower==EVupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*EVlower+(1-aprimeProbs).*EVupper; % (d23 & a1prime,u)
    % Already applied the probabilities from interpolating onto grid
    EV=squeeze(sum((EV.*pi_u),2)); % (d23 & a1prime,1)

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
    [temp4_onlyd3,d2index]=max(ezc9*ezc3*reshape((~isinf(temp4)).*temp4,[N_d2,N_d3*N_a1]),[],1);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d3,n_a1], [n_a1,n_a2],n_e, d3a1_gridvals, a1a2_gridvals,e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        % (d,a,e)

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0)); % finite and not zero
        temp2=ReturnMatrix;
        temp2(becareful)=ReturnMatrix(becareful).^ezc2(N_j);
        temp2(ReturnMatrix==0)=-Inf;

        % Time to refine
        % First: ReturnMatrix, we can refine out d1
        % no d1 here
        % Now put together entireRHS, which just depends on d3
        entireRHS=ezc1*temp2+DiscountFactorParamsVec*ezc9*shiftdim(temp4_onlyd3,1);
        % entireRHS=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4;

        temp5=logical(isfinite(entireRHS).*(entireRHS~=0));
        entireRHS(temp5)=entireRHS(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
        entireRHS(entireRHS==0)=-Inf;

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,N_j)=shiftdim(Vtemp,1);
        Policy(2,:,:,N_j)=shiftdim(rem(maxindex-1,N_d3)+1,1);
        Policy(3,:,:,N_j)=shiftdim(ceil(maxindex/N_d3),-1);
        Policy(1,:,:,N_j)=shiftdim(d2index(maxindex),1);

    elseif vfoptions.lowmemory>=1 % lm1 already does the most-looped variant, so it also serves the higher lowmemory values

       for e_c=1:N_e
           e_val=e_gridvals_J(e_c,:,N_j);
           ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d3,n_a1], [n_a1,n_a2], special_n_e, d3a1_gridvals, a1a2_gridvals, e_val, ReturnFnParamsVec);

           % Modify the Return Function appropriately for Epstein-Zin Preferences
           becareful=logical(isfinite(ReturnMatrix_e).*(ReturnMatrix_e~=0)); % finite and not zero
           temp2=ReturnMatrix_e;
           temp2(becareful)=ReturnMatrix_e(becareful).^ezc2(N_j);
           temp2(ReturnMatrix_e==0)=-Inf;

           % Time to refine
           % First: ReturnMatrix, we can refine out d1
           % no d1 here
           % Now put together entireRHS, which just depends on d3
           entireRHS_e=ezc1*temp2+DiscountFactorParamsVec*ezc9*shiftdim(temp4_onlyd3,1);

           temp5=logical(isfinite(entireRHS_e).*(entireRHS_e~=0));
           entireRHS_e(temp5)=entireRHS_e(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
           entireRHS_e(entireRHS_e==0)=-Inf;

           %Calc the max and it's index
           [Vtemp,maxindex]=max(entireRHS_e,[],1);
           V(:,e_c,N_j)=Vtemp;
           Policy(2,:,e_c,N_j)=shiftdim(rem(maxindex-1,N_d3)+1,1);
           Policy(3,:,e_c,N_j)=shiftdim(ceil(maxindex/N_d3),-1);
           Policy(1,:,e_c,N_j)=shiftdim(d2index(maxindex),1);
        end

    end
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    % Create a vector containing all the return function parameters (in order)
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
    % Note: a2primeIndex is [N_d,N_u], whereas a2primeProbs is [N_d,N_u]

    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1); % [N_d*N_a1,N_u]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d*N_a1,N_u]
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
        WGmatrix=sum((WG1.*pi_u),2)+sum((WG2.*pi_u),2); % (d,1), sum over u
        WGmatrix=repmat(WGmatrix,N_a1,1); % (d-a1prime,1): WG does not depend on a1prime, expand to the (d,a1prime) rows
        % WGmatrix is over (d-a1prime,1)
    end

    EVpre=V(:,:,jj+1);

    % Part of Epstein-Zin is before taking expectation
    temp=EVpre;
    temp(isfinite(EVpre))=(ezc4*EVpre(isfinite(EVpre))).^ezc5(jj);
    temp(EVpre==0)=0;

    % Take expectation over e
    temp=sum(temp.*pi_e_J(:,jj+1)',2);

    % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
    EVlower=reshape(temp(aprimeIndex),[N_d23*N_a1,N_u]); % the lower aprime
    EVupper=reshape(temp(aprimeplus1Index),[N_d23*N_a1,N_u]); % the upper aprime
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    aprimeProbs=repmat(a2primeProbs,N_a1,1);  % [N_d*N_a1,N_u]
    skipinterp=(EVlower==EVupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*EVlower+(1-aprimeProbs).*EVupper; % (d23 & a1prime,u)
    % Already applied the probabilities from interpolating onto grid
    EV=squeeze(sum((EV.*pi_u),2)); % (d23 & a1prime,1)

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
    [temp4_onlyd3,d2index]=max(ezc9*ezc3*reshape((~isinf(temp4)).*temp4,[N_d2,N_d3*N_a1]),[],1);

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d3,n_a1], [n_a1,n_a2], n_e, d3a1_gridvals, a1a2_gridvals, e_gridvals_J(:,:,jj), ReturnFnParamsVec);
        % (d,a,e)

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0)); % finite and not zero
        temp2=ReturnMatrix;
        temp2(becareful)=ReturnMatrix(becareful).^ezc2(jj);
        temp2(ReturnMatrix==0)=-Inf;

        % Time to refine
        % First: ReturnMatrix, we can refine out d1
        % no d1 here
        % Now put together entireRHS, which just depends on d3
        entireRHS=ezc1*temp2+DiscountFactorParamsVec*ezc9*shiftdim(temp4_onlyd3,1);
        % entireRHS=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4;

        temp5=logical(isfinite(entireRHS).*(entireRHS~=0));
        entireRHS(temp5)=entireRHS(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
        entireRHS(entireRHS==0)=-Inf;

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,jj)=shiftdim(Vtemp,1);
        Policy(2,:,:,jj)=shiftdim(rem(maxindex-1,N_d3)+1,1);
        Policy(3,:,:,jj)=shiftdim(ceil(maxindex/N_d3),-1);
        Policy(1,:,:,jj)=shiftdim(d2index(maxindex),1);

    elseif vfoptions.lowmemory>=1 % lm1 already does the most-looped variant, so it also serves the higher lowmemory values

       for e_c=1:N_e
           e_val=e_gridvals_J(e_c,:,jj);
           ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d3,n_a1], [n_a1,n_a2], special_n_e, d3a1_gridvals, a1a2_gridvals, e_val, ReturnFnParamsVec);

           % Modify the Return Function appropriately for Epstein-Zin Preferences
           becareful=logical(isfinite(ReturnMatrix_e).*(ReturnMatrix_e~=0)); % finite and not zero
           temp2=ReturnMatrix_e;
           temp2(becareful)=ReturnMatrix_e(becareful).^ezc2(jj);
           temp2(ReturnMatrix_e==0)=-Inf;

           % Time to refine
           % First: ReturnMatrix, we can refine out d1
           % no d1 here
           % Now put together entireRHS, which just depends on d3
           entireRHS_e=ezc1*temp2+DiscountFactorParamsVec*ezc9*shiftdim(temp4_onlyd3,1);

           temp5=logical(isfinite(entireRHS_e).*(entireRHS_e~=0));
           entireRHS_e(temp5)=entireRHS_e(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
           entireRHS_e(entireRHS_e==0)=-Inf;

           %Calc the max and it's index
           [Vtemp,maxindex]=max(entireRHS_e,[],1);

           V(:,e_c,jj)=Vtemp;
           Policy(2,:,e_c,jj)=shiftdim(rem(maxindex-1,N_d3)+1,1);
           Policy(3,:,e_c,jj)=shiftdim(ceil(maxindex/N_d3),-1);
           Policy(1,:,e_c,jj)=shiftdim(d2index(maxindex),1);
        end
    end
end




end
