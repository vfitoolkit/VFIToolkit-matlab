function [V, Policy]=ValueFnIter_Case1_FHorz_EpsteinZin_nod_noz_e_raw(n_a,n_e,N_j, a_grid,e_grid, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7)

N_a=prod(n_a);
N_e=prod(n_e);

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%%
eval('fieldexists_EiidShockFn=1;vfoptions.EiidShockFn;','fieldexists_EiidShockFn=0;')
eval('fieldexists_EiidShockFnParamNames=1;vfoptions.EiidShockFnParamNames;','fieldexists_EiidShockFnParamNames=0;')
eval('fieldexists_pi_e_J=1;vfoptions.pi_e_J;','fieldexists_pi_e_J=0;')

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
    % e_gridvals is created below
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
if vfoptions.EZoneminusbeta==1
    ezc1=1-DiscountFactorParamsVec; % Just in case it depends on age
end

if fieldexists_pi_e_J==1
    e_grid=vfoptions.e_grid_J(:,N_j);
    pi_e=vfoptions.pi_e_J(:,N_j);
elseif fieldexists_EiidShockFn==1
    if fieldexists_EiidShockFnParamNames==1
        EiidShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.EiidShockFnParamNames,N_j);
        EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
        for ii=1:length(EiidShockFnParamsVec)
            EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
        end
        [e_grid,pi_e]=vfoptions.EiidShockFn(EiidShockFnParamsCell{:});
        e_grid=gpuArray(e_grid); pi_e=gpuArray(pi_e);
    else
        [e_grid,pi_e]=vfoptions.ExogShockFn(N_j);
        e_grid=gpuArray(e_grid); pi_e=gpuArray(pi_e);
    end
end

pi_e=shiftdim(pi_e,-2); % Move to third dimension

if vfoptions.lowmemory>0
    if all(size(e_grid)==[sum(n_e),1]) % kronecker (cross-product) grid
        e_gridvals=CreateGridvals(n_e,e_grid,1); % The 1 at end indicates want output in form of matrix.
    elseif all(size(e_grid)==[prod(n_e),length(n_e)]) % joint-grid
        e_gridvals=e_grid;
    end
end


% If there is a warm-glow at end of the final period, evaluate the warmglowfn
if warmglow==1
    WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,N_j);
    WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a, a_grid, WGParamsVec);
    WGmatrix=WGmatrixraw;
    WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5;
    WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
    if ~isfield(vfoptions,'V_Jplus1')
        WGmatrix(isfinite(WGmatrixraw))=ezc3*DiscountFactorParamsVec*(((1-sj(N_j))*WGmatrix(isfinite(WGmatrixraw))).^ezc6);
        WGmatrix(WGmatrixraw==0)=0;
    end
    % Now just make it the right shape (currently has aprime, needs the d,a,z dimensions)
    if ~isfield(vfoptions,'V_Jplus1')
        if vfoptions.lowmemory==0
            WGmatrix=WGmatrix.*ones(1,N_a,N_e);
        elseif vfoptions.lowmemory==1
            WGmatrix=WGmatrix.*ones(1,N_a);
        end
    else
        if vfoptions.lowmemory==0 || vfoptions.lowmemory==1
            WGmatrix=WGmatrix.*ones(1,N_a);
        end
    end
else
    WGmatrix=0;
end


if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a,n_e, 0, a_grid, e_grid, ReturnFnParamsVec);

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0)); % finite but not zero
        ReturnMatrix(becareful)=(ezc1*ReturnMatrix(becareful).^ezc2).^ezc7; % Otherwise can get things like 0 to negative power equals infinity
        ReturnMatrix(ReturnMatrix==0)=-Inf;

        % Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix+WGmatrix,[],1);
        V(:,:,N_j)=Vtemp;
        Policy(:,:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1

        %if vfoptions.returnmatrix==2 % GPU
        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_e, 0, a_grid, e_val, ReturnFnParamsVec);

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_e).*(ReturnMatrix_e~=0)); % finite and not zero
            ReturnMatrix_e(becareful)=(ezc1*ReturnMatrix_e(becareful).^ezc2).^ezc7; % Otherwise can get things like 0 to negative power equals infinity
            ReturnMatrix_e(ReturnMatrix_e==0)=-Inf;

            % Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e+WGmatrix,[],1);
            V(:,e_c,N_j)=Vtemp;
            Policy(:,e_c,N_j)=maxindex;
        end

    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_e]);    % First, switch V_Jplus1 into Kron form

    % Part of Epstein-Zin is before taking expectation
    temp=V_Jplus1;
    temp(isfinite(V_Jplus1))=(ezc4*V_Jplus1(isfinite(V_Jplus1))).^ezc5;
    temp(V_Jplus1==0)=0;

    % Take expectation over e
    temp=sum(temp.*pi_e,3);
    
    if vfoptions.lowmemory==0
        
        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_e, 0, a_grid, e_grid, ReturnFnParamsVec);
        
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0)); % finite but not zero
        temp2=ReturnMatrix;
        temp2(becareful)=ReturnMatrix(becareful).^ezc2;
        temp2(ReturnMatrix==0)=-Inf;

        %Calc the expectation term (except beta)
        temp4=temp;
        if warmglow==1
            becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
            temp4(becareful)=(sj(N_j)*temp4(becareful)+(1-sj(N_j))*WGmatrix(becareful)).^ezc6;
            temp4((temp==0)&(WGmatrix==0))=0; % Is actually zero
        else % not using warmglow
            temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4))).^ezc6;
            temp4(temp==0)=0;
        end

        entireRHS=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4*ones(1,N_a,N_e);

        temp5=logical(isfinite(entireRHS).*(entireRHS~=0));
        entireRHS(temp5)=ezc1*entireRHS(temp5).^ezc7;  % matlab otherwise puts 0 to negative power to infinity
        entireRHS(entireRHS==0)=-Inf;

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,:,N_j)=Vtemp;
        Policy(:,:,N_j)=maxindex;
        
    elseif vfoptions.lowmemory==1
        
        %Calc the expectation term (except beta)
        temp4=temp;
        if warmglow==1
            becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
            temp4(becareful)=(sj(N_j)*temp4(becareful)+(1-sj(N_j))*WGmatrix(becareful)).^ezc6;
            temp4((temp==0)&(WGmatrix==0))=0; % Is actually zero
        else % not using warmglow
            temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4))).^ezc6;
            temp4(temp==0)=0;
        end
        
        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_e, 0, a_grid, e_val, ReturnFnParamsVec);
            
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_e).*(ReturnMatrix_e~=0)); % finite but not zero
            temp2=ReturnMatrix_e;
            temp2(becareful)=ReturnMatrix_e(becareful).^ezc2;
            temp2(ReturnMatrix_e==0)=-Inf;

            entireRHS_e=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4*ones(1,N_a,1);

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
        fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
    end
    
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    if vfoptions.EZoneminusbeta==1
        ezc1=1-DiscountFactorParamsVec; % Just in case it depends on age
    end
    
    if fieldexists_pi_e_J==1
        e_grid=vfoptions.e_grid_J(:,jj);
        pi_e=vfoptions.pi_e_J(:,jj);
        pi_e=shiftdim(pi_e,-2); % Move to thrid dimension
    elseif fieldexists_EiidShockFn==1
        if fieldexists_EiidShockFnParamNames==1
            EiidShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.EiidShockFnParamNames,jj);
            EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
            for ii=1:length(EiidShockFnParamsVec)
                EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
            end
            [e_grid,pi_e]=vfoptions.EiidShockFn(EiidShockFnParamsCell{:});
            e_grid=gpuArray(e_grid); pi_e=gpuArray(pi_e);
        else
            [e_grid,pi_e]=vfoptions.EiidShockFn(jj);
            e_grid=gpuArray(e_grid); pi_e=gpuArray(pi_e);
        end
        pi_e=shiftdim(pi_e,-2); % Move to third dimension
    end
    
    if vfoptions.lowmemory>0
        if (fieldexists_pi_e_J==1 || fieldexists_EiidShockFn==1)
            if all(size(e_grid)==[sum(n_e),1]) % kronecker (cross-product) grid
                e_gridvals=CreateGridvals(n_e,e_grid,1); % The 1 at end indicates want output in form of matrix.
            elseif all(size(e_grid)==[prod(n_e),length(n_e)]) % joint-grid
                e_gridvals=e_grid;
            end
        end
    end
    

    % If there is a warm-glow, evaluate the warmglowfn
    if warmglow==1
        WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
        WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a, a_grid, WGParamsVec);
        WGmatrix=WGmatrixraw;
        WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5;
        WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
        % Now just make it the right shape (currently has aprime, needs the d,a,z dimensions)
        % if vfoptions.lowmemory==0 || vfoptions.lowmemory==1
        % WGmatrix=WGmatrix; % note, actually the same as no z
    end

    VKronNext_j=V(:,:,jj+1);
    % Part of Epstein-Zin is before taking expectation
    temp=VKronNext_j;
    temp(isfinite(VKronNext_j))=(ezc4*VKronNext_j(isfinite(VKronNext_j))).^ezc5;
    temp(VKronNext_j==0)=0;
    
    % Take expectation over e
    temp=sum(temp.*pi_e,3);
    
    if vfoptions.lowmemory==0
        
        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_e, 0, a_grid, e_grid, ReturnFnParamsVec);
        
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0)); % finite but not zero
        temp2=ReturnMatrix;
        temp2(becareful)=ReturnMatrix(becareful).^ezc2;
        temp2(ReturnMatrix==0)=-Inf;

        %Calc the expectation term (except beta)
        temp4=temp;
        if warmglow==1
            becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
            temp4(becareful)=(sj(jj)*temp4(becareful)+(1-sj(jj))*WGmatrix(becareful)).^ezc6;
            temp4((temp==0)&(WGmatrix==0))=0; % Is actually zero
        else % not using warmglow
            temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4))).^ezc6;
            temp4(temp==0)=0;
        end

        entireRHS=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4*ones(1,N_a,N_e);

        temp5=logical(isfinite(entireRHS).*(entireRHS~=0));
        entireRHS(temp5)=ezc1*entireRHS(temp5).^ezc7;  % matlab otherwise puts 0 to negative power to infinity
        entireRHS(entireRHS==0)=-Inf;
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,:,jj)=Vtemp;
        Policy(:,:,jj)=maxindex;
        
    elseif vfoptions.lowmemory==1
        
        %Calc the expectation term (except beta)
        temp4=temp;
        if warmglow==1
            becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
            temp4(becareful)=(sj(jj)*temp4(becareful)+(1-sj(jj))*WGmatrix(becareful)).^ezc6;
            temp4((temp==0)&(WGmatrix==0))=0; % Is actually zero
        else % not using warmglow
            temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4))).^ezc6;
            temp4(temp==0)=0;
        end
        
        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_e, 0, a_grid, e_val, ReturnFnParamsVec);
            
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_e).*(ReturnMatrix_e~=0)); % finite but not zero
            temp2=ReturnMatrix_e;
            temp2(becareful)=ReturnMatrix_e(becareful).^ezc2;
            temp2(ReturnMatrix_e==0)=-Inf;

            entireRHS_e=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4*ones(1,N_a,1);

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