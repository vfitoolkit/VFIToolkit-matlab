function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_noz_e_raw(n_d,n_a1,n_a2,n_e,n_u, N_j, d_grid, a1_grid,a2_grid, e_gridvals_J, u_grid, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)

N_d=prod(n_d);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_e=prod(n_e);
N_u=prod(n_u);

N_a=N_a1*N_a2;

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
d_grid=gpuArray(d_grid);
a1_grid=gpuArray(a1_grid);
a2_grid=gpuArray(a2_grid);
u_grid=gpuArray(u_grid);

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
    WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a2, a2_grid, WGParamsVec);
    WGmatrix=WGmatrixraw;
    WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5;
    WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
    
    % Switch WGmatrix from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateaprimeFnMatrix_RiskyAsset(aprimeFn, n_d, n_a2, n_u, d_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d*N_u,1], whereas aprimeProbs is [N_d,N_u]
    aprimeIndex=kron((1:1:N_a1)',ones(N_d,N_u))+N_a1*kron(ones(N_a1,1),(a2primeIndex-1)); % [N_d*N_a1,N_u]
    aprimeplus1Index=kron((1:1:N_a1)',ones(N_d,N_u))+N_a1*kron(ones(N_a1,1),a2primeIndex); % [N_d*N_a1,N_u]
    aprimeProbs=kron(ones(N_a1,1),a2primeProbs);  % [N_d*N_a1,N_u]

    WG1=WGmatrix(aprimeIndex); % (d,u), the lower aprime
    WG2=WGmatrix(aprimeplus1Index); % (d,u), the upper aprime
    % Apply the aprimeProbs
    WG1=reshape(WG1,[N_d,N_u]).*aprimeProbs; % probability of lower grid point
    WG2=reshape(WG2,[N_d,N_u]).*(1-aprimeProbs); % probability of upper grid point
    % Expectation over u (using pi_u), and then add the lower and upper
    WGmatrix=sum((WG1.*pi_u'),2)+sum((WG2.*pi_u'),2); % (d,1), sum over u
    % WGmatrix is over (d,1)

    if ~isfield(vfoptions,'V_Jplus1')
        becareful=(WGmatrix==0);
        WGmatrix(isfinite(WGmatrix))=ezc3*DiscountFactorParamsVec*(((1-sj(N_j))*WGmatrix(isfinite(WGmatrix)).^ezc8).^ezc6);
        WGmatrix(becareful)=0;
    end
    % Now just make it the right shape (currently has aprime, needs the d,a,z dimensions)
    if ~isfield(vfoptions,'V_Jplus1')
        if vfoptions.lowmemory==0
            WGmatrix=WGmatrix.*ones(1,N_a2,N_e);
        elseif vfoptions.lowmemory==1
            WGmatrix=WGmatrix.*ones(1,N_a2);
        end
    else
        if vfoptions.lowmemory==0 
            WGmatrix=WGmatrix.*ones(1,N_a2,N_e);
        elseif vfoptions.lowmemory==1
            WGmatrix=WGmatrix.*ones(1,N_a2);
        end
    end
else
    WGmatrix=0;
end


if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, [n_d,n_a1], [n_a1,n_a2], n_e,[d_grid; a1_grid], [a1_grid; a2_grid],e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0)); % finite but not zero
        ReturnMatrix(becareful)=(ezc1*ReturnMatrix(becareful).^ezc2).^ezc7; % Otherwise can get things like 0 to negative power equals infinity
        ReturnMatrix(ReturnMatrix==0)=-Inf;

        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix+WGmatrix,[],1);
        V(:,N_j)=Vtemp;
        Policy(:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(a_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, [n_d,n_a1], [n_a1,n_a2], special_n_e, [d_grid; a1_grid], [a1_grid; a2_grid], e_val, ReturnFnParamsVec,0);

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_e).*(ReturnMatrix_e~=0)); % finite and not zero
            ReturnMatrix_e(becareful)=(ezc1*ReturnMatrix_e(becareful).^ezc2).^ezc7; % Otherwise can get things like 0 to negative power equals infinity
            ReturnMatrix_e(ReturnMatrix_e==0)=-Inf;

            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e+WGmatrix,[],1);
            V(:,e_c,N_j)=Vtemp;
            Policy(:,e_c,N_j)=maxindex;
        end
    end
else
    % Using V_Jplus1
    V_Jplus1=vfoptions.V_Jplus1;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateaprimeFnMatrix_RiskyAsset(aprimeFn, n_d, n_a2, n_u, d_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d*N_u,1], whereas aprimeProbs is [N_d,N_u]
    aprimeIndex=kron((1:1:N_a1)',ones(N_d,N_u))+N_a1*kron(ones(N_a1,1),(a2primeIndex-1)); % [N_d*N_a1,N_u]
    aprimeplus1Index=kron((1:1:N_a1)',ones(N_d,N_u))+N_a1*kron(ones(N_a1,1),a2primeIndex); % [N_d*N_a1,N_u]
    aprimeProbs=kron(ones(N_a1,1),a2primeProbs);  % [N_d*N_a1,N_u]

    % Part of Epstein-Zin is before taking expectation
    temp=V_Jplus1;
    temp(isfinite(V_Jplus1))=(ezc4*V_Jplus1(isfinite(V_Jplus1))).^ezc5;
    temp(V_Jplus1==0)=0; % otherwise zero to negative power is set to infinity

    % Take expectation over e
    temp=sum(reshape(temp,[N_a2,N_e]).*pi_e_J(:,:,N_j)',2);    % First, switch V_Jplus1 into Kron form ,take expecation over e
        
    % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
    EV1=aprimeProbs.*reshape(temp(aprimeIndex),[N_d*N_a1,N_u]); % (d&a1prime,u), the lower aprime
    EV2=(1-aprimeProbs).*reshape(temp(aprimeplus1Index),[N_d*N_a1,N_u]); % (d&a1prime,u), the upper aprime
    % Already applied the probabilities from interpolating onto grid
    
    % Expectation over u (using pi_u), and then add the lower and upper
    EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2); % (d&a1prime,u), sum over u
    % EV is over (d&a1prime,1)
    
    % Part of Epstein-Zin is after taking expectation
    temp4=EV;
    if warmglow==1
        becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
        temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8+(1-sj(N_j))*WGmatrix(becareful).^ezc8).^ezc6;
        temp4((EV==0)&(WGmatrix==0))=0; % Is actually zero
    else % not using warmglow
        temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4)).^ezc8).^ezc6;
        temp4(EV==0)=0;
    end
    
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, [n_d,n_a1], [n_a1,n_a2],n_e, [d_grid; a1_grid], [a1_grid; a2_grid],e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        % (d,a,e)
        
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0)); % finite and not zero
        temp2=ReturnMatrix;
        temp2(becareful)=ReturnMatrix(becareful).^ezc2;
        temp2(ReturnMatrix==0)=-Inf;

        entireRHS=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4.*ones(1,N_a,N_e); % d-by-a-by-e

        temp5=logical(isfinite(entireRHS).*(entireRHS~=0));
        entireRHS(temp5)=ezc1*entireRHS(temp5).^ezc7;  % matlab otherwise puts 0 to negative power to infinity
        entireRHS(entireRHS==0)=-Inf;
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,N_j)=shiftdim(maxindex,1);
        
    elseif vfoptions.lowmemory==1

        betaEV=DiscountFactorParamsVec*temp4.*ones(1,N_a);

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, [n_d,n_a1], [n_a1,n_a2], n_e, special_n_e, [d_grid; a1_grid], [a1_grid; a2_grid], e_val, ReturnFnParamsVec);

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_e).*(ReturnMatrix_e~=0)); % finite and not zero
            temp2=ReturnMatrix_e;
            temp2(becareful)=ReturnMatrix_e(becareful).^ezc2;
            temp2(ReturnMatrix_e==0)=-Inf;

            entireRHS_e=ezc1*temp2+ezc3*betaEV; % d-by-a

            temp5=logical(isfinite(entireRHS_e).*(entireRHS_e~=0));
            entireRHS_e(temp5)=ezc1*entireRHS_e(temp5).^ezc7;  % matlab otherwise puts 0 to negative power to infinity
            entireRHS_e(entireRHS_e==0)=-Inf;

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            V(:,e_c,N_j)=Vtemp;
            Policy(:,e_c,N_j)=maxindex;
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
    [a2primeIndex,a2primeProbs]=CreateaprimeFnMatrix_RiskyAsset(aprimeFn, n_d, n_a2, n_u, d_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d*N_u,1], whereas aprimeProbs is [N_d,N_u]
    aprimeIndex=kron((1:1:N_a1)',ones(N_d,N_u))+N_a1*kron(ones(N_a1,1),(a2primeIndex-1)); % [N_d*N_a1,N_u]
    aprimeplus1Index=kron((1:1:N_a1)',ones(N_d,N_u))+N_a1*kron(ones(N_a1,1),a2primeIndex); % [N_d*N_a1,N_u]
    aprimeProbs=kron(ones(N_a1,1),a2primeProbs);  % [N_d*N_a1,N_u]
    
    % If there is a warm-glow, evaluate the warmglowfn
    if warmglow==1
        WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
        WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a2, a2_grid, WGParamsVec);
        WGmatrix=WGmatrixraw;
        WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5;
        WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
        %  Switch WGmatrix from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
        % Note: aprimeIndex is [N_d*N_u,1], whereas aprimeProbs is [N_d,N_u]
        WG1=WGmatrix(aprimeIndex); % (d,u), the lower aprime
        WG2=WGmatrix(aprimeplus1Index); % (d,u), the upper aprime
        % Apply the aprimeProbs
        WG1=reshape(WG1,[N_d,N_u]).*aprimeProbs; % probability of lower grid point
        WG2=reshape(WG2,[N_d,N_u]).*(1-aprimeProbs); % probability of upper grid point
        % Expectation over u (using pi_u), and then add the lower and upper
        WGmatrix=sum((WG1.*pi_u'),2)+sum((WG2.*pi_u'),2); % (d,1), sum over u
        % Now just make it the right shape (currently has aprime, needs the d,a,z dimensions)
        % if vfoptions.lowmemory==0 || vfoptions.lowmemory==1
        % WGmatrix=WGmatrix; % note, actually the same as no z
    end
    
    VKronNext_j=V(:,:,jj+1);

    % Part of Epstein-Zin is before taking expectation
    temp=VKronNext_j;
    temp(isfinite(VKronNext_j))=(ezc4*VKronNext_j(isfinite(VKronNext_j))).^ezc5;
    temp(VKronNext_j==0)=0; % otherwise zero to negative power is set to infinity 

    temp=sum(temp.*pi_e_J(:,:,jj)',2); % Expectation over e
    
    % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
    EV1=aprimeProbs.*reshape(temp(aprimeIndex),[N_d*N_a1,N_u]); % (d&a1prime,u), the lower aprime
    EV2=(1-aprimeProbs).*reshape(temp(aprimeplus1Index),[N_d*N_a1,N_u]); % (d&a1prime,u), the upper aprime
    % Already applied the probabilities from interpolating onto grid
    
    % Expectation over u (using pi_u), and then add the lower and upper
    EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2); % (d&a1prime,u), sum over u
    % EV is over (d&a1prime,1)
    
    % Part of Epstein-Zin is after taking expectation
    temp4=EV;
    if warmglow==1
        becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
        temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8+(1-sj(jj))*WGmatrix(becareful).^ezc8).^ezc6;
        temp4((EV==0)&(WGmatrix==0))=0; % Is actually zero
    else % not using warmglow
        temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8).^ezc6;
        temp4(EV==0)=0;
    end

    if vfoptions.lowmemory==0
        
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, [n_d,n_a1], [n_a1,n_a2], n_e, [d_grid; a1_grid], [a1_grid; a2_grid], e_gridvals_J(:,:,jj), ReturnFnParamsVec);
        % (d,a,e)
        
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0)); % finite and not zero
        temp2=ReturnMatrix;
        temp2(becareful)=ReturnMatrix(becareful).^ezc2;
        temp2(ReturnMatrix==0)=-Inf;

        entireRHS=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4.*ones(1,N_a,N_e); % d-by-a-by-e

        temp5=logical(isfinite(entireRHS).*(entireRHS~=0));
        entireRHS(temp5)=ezc1*entireRHS(temp5).^ezc7;  % matlab otherwise puts 0 to negative power to infinity
        entireRHS(entireRHS==0)=-Inf;
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,jj)=shiftdim(maxindex,1);
        
    elseif vfoptions.lowmemory==1
       
       betaEV=DiscountFactorParamsVec*temp4.*ones(1,N_a);
        
       for e_c=1:N_e
           e_val=e_gridvals_J(e_c,:,jj);
           ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, [n_d,n_a1], [n_a1,n_a2], special_n_e, [d_grid; a1_grid], [a1_grid; a2_grid], e_val, ReturnFnParamsVec);
           
           % Modify the Return Function appropriately for Epstein-Zin Preferences
           becareful=logical(isfinite(ReturnMatrix_e).*(ReturnMatrix_e~=0)); % finite and not zero
           temp2=ReturnMatrix_e;
           temp2(becareful)=ReturnMatrix_e(becareful).^ezc2;
           temp2(ReturnMatrix_e==0)=-Inf;

           entireRHS_e=ezc1*temp2+ezc3*betaEV; % d-by-a

           temp5=logical(isfinite(entireRHS_e).*(entireRHS_e~=0));
           entireRHS_e(temp5)=ezc1*entireRHS_e(temp5).^ezc7;  % matlab otherwise puts 0 to negative power to infinity
           entireRHS_e(entireRHS_e==0)=-Inf;
           
           %Calc the max and it's index
           [Vtemp,maxindex]=max(entireRHS_e,[],1);
           V(:,e_c,jj)=Vtemp;
           Policy(:,e_c,jj)=maxindex;
        end
        
    end
end


end