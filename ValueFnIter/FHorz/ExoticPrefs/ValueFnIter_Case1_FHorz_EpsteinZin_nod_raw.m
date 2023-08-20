function [V, Policy]=ValueFnIter_Case1_FHorz_EpsteinZin_nod_raw(n_a,n_z,N_j, a_grid, z_grid,pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7)

N_a=prod(n_a);
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%%
eval('fieldexists_pi_z_J=1;vfoptions.pi_z_J;','fieldexists_pi_z_J=0;')
eval('fieldexists_ExogShockFn=1;vfoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;vfoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')

if vfoptions.lowmemory>1
    special_n_a=ones(1,length(n_a));
    a_gridvals=CreateGridvals(n_a,a_grid,1);
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
if vfoptions.EZoneminusbeta==1
    ezc1=1-DiscountFactorParamsVec; % Just in case it depends on age
elseif vfoptions.EZoneminusbeta==2
    ezc1=1-sj(N_j)*DiscountFactorParamsVec;
end

if fieldexists_pi_z_J==1
    z_grid=vfoptions.z_grid_J(:,N_j);
    pi_z=vfoptions.pi_z_J(:,:,N_j);
elseif fieldexists_ExogShockFn==1
    if fieldexists_ExogShockFnParamNames==1
        ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,N_j);
        ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
        for ii=1:length(ExogShockFnParamsVec)
            ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
        end
        [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
        z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
    else
        [z_grid,pi_z]=vfoptions.ExogShockFn(N_j);
        z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
    end
end
if vfoptions.lowmemory>0
    l_z=length(n_z);
    special_n_z=ones(1,l_z);
    if all(size(z_grid)==[sum(n_z),1])
        z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
    elseif all(size(z_grid)==[prod(n_z),l_z])
        z_gridvals=z_grid;
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
            WGmatrix=WGmatrix.*ones(1,N_a,N_z);
        elseif vfoptions.lowmemory==1
            WGmatrix=WGmatrix.*ones(1,N_a);
        elseif vfoptions.lowmemory==2
            % WGmatrix=WGmatrix;
        end
    else
        if vfoptions.lowmemory==0 || vfoptions.lowmemory==1
            WGmatrix=WGmatrix.*ones(1,N_a);
        elseif vfoptions.lowmemory==2
            % WGmatrix=WGmatrix;
        end
    end
else
    WGmatrix=0;
end

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_grid, ReturnFnParamsVec);

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0)); % finite but not zero
        ReturnMatrix(becareful)=(ezc1*ReturnMatrix(becareful).^ezc2).^ezc7; % Otherwise can get things like 0 to negative power equals infinity
        ReturnMatrix(ReturnMatrix==0)=-Inf;

        % Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix+WGmatrix,[],1);
        V(:,:,N_j)=Vtemp;
        Policy(:,:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1

        for z_c=1:N_z
            z_val=z_gridvals(z_c,:);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec);

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_z).*(ReturnMatrix_z~=0)); % finite and not zero
            ReturnMatrix_z(becareful)=(ezc1*ReturnMatrix_z(becareful).^ezc2).^ezc7; % Otherwise can get things like 0 to negative power equals infinity
            ReturnMatrix_z(ReturnMatrix_z==0)=-Inf;

            % Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_z+WGmatrix,[],1);
            V(:,z_c,N_j)=Vtemp;
            Policy(:,z_c,N_j)=maxindex;
        end

    elseif vfoptions.lowmemory==2

        for z_c=1:N_z
            z_val=z_gridvals(z_c,:);
            for a_c=1:N_a
                a_val=a_gridvals(a_c,:);
                ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, special_n_a, special_n_z, 0, a_val, z_val, ReturnFnParamsVec);
                
                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(ReturnMatrix_az).*(ReturnMatrix_az~=0)); % finite and not zero
                ReturnMatrix_az(becareful)=(ezc1*ReturnMatrix_az(becareful).^ezc2).^ezc7; % Otherwise can get things like 0 to negative power equals infinity
                ReturnMatrix_az(ReturnMatrix_az==0)=-Inf;

                % Calc the max and it's index
                [Vtemp,maxindex]=max(ReturnMatrix_az+WGmatrix);
                V(a_c,z_c,N_j)=Vtemp;
                Policy(a_c,z_c,N_j)=maxindex;

            end
        end

    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_z]);    % First, switch V_Jplus1 into Kron form

    % Part of Epstein-Zin is before taking expectation
    temp=V_Jplus1;
    temp(isfinite(V_Jplus1))=(ezc4*V_Jplus1(isfinite(V_Jplus1))).^ezc5;
    temp(V_Jplus1==0)=0;
    
    if vfoptions.lowmemory==0
        
        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_grid, ReturnFnParamsVec);
        
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0)); % finite but not zero
        temp2=ReturnMatrix;
        temp2(becareful)=ReturnMatrix(becareful).^ezc2;
        temp2(ReturnMatrix==0)=-Inf;

        %Calc the expectation term (except beta)
        EV=temp.*shiftdim(pi_z',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        temp4=EV;
        if warmglow==1
            becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
            temp4(becareful)=(sj(N_j)*temp4(becareful)+(1-sj(N_j))*WGmatrix(becareful)).^ezc6;
            temp4((EV==0)&(WGmatrix==0))=0; % Is actually zero
        else % not using warmglow
            temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4))).^ezc6;
            temp4(EV==0)=0;
        end
            
        entireRHS=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4.*ones(1,N_a,1);

        temp5=logical(isfinite(entireRHS).*(entireRHS~=0));
        entireRHS(temp5)=ezc1*entireRHS(temp5).^ezc7;  % matlab otherwise puts 0 to negative power to infinity
        entireRHS(entireRHS==0)=-Inf;

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,:,N_j)=Vtemp;
        Policy(:,:,N_j)=maxindex;
        
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals(z_c,:);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec);

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_z).*(ReturnMatrix_z~=0)); % finite but not zero
            temp2=ReturnMatrix_z;
            temp2(becareful)=ReturnMatrix_z(becareful).^ezc2;
            temp2(ReturnMatrix_z==0)=-Inf;

            %Calc the expectation term (except beta)
            EV_z=temp.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            temp4=EV_z;
             if warmglow==1
                becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
                temp4(becareful)=(sj(N_j)*temp4(becareful)+(1-sj(N_j))*WGmatrix(becareful)).^ezc6;
                temp4((EV_z==0)&(WGmatrix==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4))).^ezc6;
                temp4(EV_z==0)=0;
            end
            
            entireRHS_z=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4.*ones(1,N_a,1);

            temp5=logical(isfinite(entireRHS_z).*(entireRHS_z~=0));
            entireRHS_z(temp5)=ezc1*entireRHS_z(temp5).^ezc7;  % matlab otherwise puts 0 to negative power to infinity
            entireRHS_z(entireRHS_z==0)=-Inf;

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            Policy(:,z_c,N_j)=maxindex;
        end
        
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            %Calc the expectation term (except beta)
            EV_z=temp.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            temp4=EV_z;
            if warmglow==1
                becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
                temp4(becareful)=(sj(N_j)*temp4(becareful)+(1-sj(N_j))*WGmatrix(becareful)).^ezc6;
                temp4((EV_z==0)&(WGmatrix==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4))).^ezc6;
                temp4(EV_z==0)=0;
            end
              
            z_val=z_gridvals(z_c,:);
            for a_c=1:N_a
                a_val=a_gridvals(a_c,:);
                ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, special_n_a, special_n_z, 0, a_val, z_val, ReturnFnParamsVec);

                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(ReturnMatrix_az).*(ReturnMatrix_az~=0)); % finite but not zero
                temp2=ReturnMatrix_az;
                temp2(becareful)=ReturnMatrix_az(becareful).^ezc2;
                temp2(ReturnMatrix_az==0)=-Inf; % matlab otherwise puts 0 to negative power to infinity

                entireRHS_az=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4;

                temp5=logical(isfinite(entireRHS_az).*(entireRHS_az~=0));
                entireRHS_az(temp5)=ezc1*entireRHS_az(temp5).^ezc7;  % matlab otherwise puts 0 to negative power to infinity
                entireRHS_az(entireRHS_az==0)=-Inf;

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_az);
                V(a_c,z_c,N_j)=Vtemp;
                Policy(a_c,z_c,N_j)=maxindex;
            end
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
    elseif vfoptions.EZoneminusbeta==2
        ezc1=1-sj(jj)*DiscountFactorParamsVec;
    end

    if fieldexists_pi_z_J==1
        z_grid=vfoptions.z_grid_J(:,jj);
        pi_z=vfoptions.pi_z_J(:,:,jj);
    elseif fieldexists_ExogShockFn==1
        if fieldexists_ExogShockFnParamNames==1
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
            ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
            for ii=1:length(ExogShockFnParamsVec)
                ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
            end
            [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
            z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
        else
            [z_grid,pi_z]=vfoptions.ExogShockFn(jj);
            z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
        end
    end
    if vfoptions.lowmemory>0 && (fieldexists_pi_z_J==1 || fieldexists_ExogShockFn==1)
        if all(size(z_grid)==[sum(n_z),1])
            z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
        elseif all(size(z_grid)==[prod(n_z),l_z])
            z_gridvals=z_grid;
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
        if vfoptions.lowmemory==0
            WGmatrix=WGmatrix.*ones(1,1,N_z);
        elseif vfoptions.lowmemory==1 || vfoptions.lowmemory==2
            % WGmatrix=WGmatrix;
        end
    end
    
    VKronNext_j=V(:,:,jj+1);
    % Part of Epstein-Zin is before taking expectation
    temp=VKronNext_j;
    temp(isfinite(VKronNext_j))=(ezc4*VKronNext_j(isfinite(VKronNext_j))).^ezc5;
    temp(VKronNext_j==0)=0;

    if vfoptions.lowmemory==0
        
        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_grid, ReturnFnParamsVec);
        
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0)); % finite but not zero
        temp2=ReturnMatrix;
        temp2(becareful)=ReturnMatrix(becareful).^ezc2;
        temp2(ReturnMatrix==0)=-Inf; % matlab otherwise puts 0 to negative power to infinity

        %Calc the expectation term (except beta)
        EV=temp.*shiftdim(pi_z',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        temp4=EV;
        if warmglow==1
            becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
            temp4(becareful)=(sj(jj)*temp4(becareful)+(1-sj(jj))*WGmatrix(becareful)).^ezc6;
            temp4((EV==0)&(WGmatrix==0))=0; % Is actually zero
        else % not using warmglow
            temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4))).^ezc6;
            temp4(EV==0)=0;
        end
        
        entireRHS=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4.*ones(1,N_a,1);

        temp5=logical(isfinite(entireRHS).*(entireRHS~=0));
        entireRHS(temp5)=ezc1*entireRHS(temp5).^ezc7;  % matlab otherwise puts 0 to negative power to infinity
        entireRHS(entireRHS==0)=-Inf;

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,:,jj)=Vtemp;
        Policy(:,:,jj)=maxindex;
        
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals(z_c,:);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec);

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_z).*(ReturnMatrix_z~=0)); % finite but not zero
            temp2=ReturnMatrix_z;
            temp2(becareful)=ReturnMatrix_z(becareful).^ezc2;
            temp2(ReturnMatrix_z==0)=-Inf;

            %Calc the expectation term (except beta)
            EV_z=temp.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            temp4=EV_z;
            if warmglow==1
                becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
                temp4(becareful)=(sj(jj)*temp4(becareful)+(1-sj(jj))*WGmatrix(becareful)).^ezc6;
                temp4((EV_z==0)&(WGmatrix==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4))).^ezc6;
                temp4(EV_z==0)=0;
            end
            
            entireRHS_z=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4.*ones(1,N_a,1);
            
            temp5=logical(isfinite(entireRHS_z).*(entireRHS_z~=0));
            entireRHS_z(temp5)=ezc1*entireRHS_z(temp5).^ezc7;  % matlab otherwise puts 0 to negative power to infinity
            entireRHS_z(entireRHS_z==0)=-Inf;

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,z_c,jj)=Vtemp;
            Policy(:,z_c,jj)=maxindex;
        end
        
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            %Calc the expectation term (except beta)
            EV_z=temp.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            temp4=EV_z;
            if warmglow==1
                becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
                temp4(becareful)=(sj(jj)*temp4(becareful)+(1-sj(jj))*WGmatrix(becareful)).^ezc6;
                temp4((EV_z==0)&(WGmatrix==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4))).^ezc6;
                temp4(EV_z==0)=0;
            end
                   
            z_val=z_gridvals(z_c,:);
            for a_c=1:N_a
                a_val=a_gridvals(a_c,:);
                ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, special_n_a, special_n_z, 0, a_val, z_val, ReturnFnParamsVec);
                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(ReturnMatrix_z).*(ReturnMatrix_z~=0)); % finite but not zero
                temp2=ReturnMatrix_az;
                temp2(becareful)=ReturnMatrix_az(becareful).^ezc2;
                temp2(ReturnMatrix_az==0)=-Inf; % matlab otherwise puts 0 to negative power to infinity

                entireRHS_az=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4;

                temp5=logical(isfinite(entireRHS_az).*(entireRHS_az~=0));
                entireRHS_az(temp5)=ezc1*entireRHS_az(temp5).^ezc7;  % matlab otherwise puts 0 to negative power to infinity
                entireRHS_az(entireRHS_az==0)=-Inf;

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_az);
                V(a_c,z_c,jj)=Vtemp;
                Policy(a_c,z_c,jj)=maxindex;
            end
        end
        
    end
end


end