function [V,Policy]=ValueFnIter_Case2_3_FHorz_EpsteinZin_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions, sj, warmglowweight, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7)

if warmglow==1
    error('Have not yet implemented warm-glow of bequests for Epstein-Zin when using Case2=3, email me if you need/want this')
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); %indexes the optimal choice for d given rest of dimensions a,z

%%
eval('fieldexists_pi_z_J=1;vfoptions.pi_z_J;','fieldexists_pi_z_J=0;')
eval('fieldexists_ExogShockFn=1;vfoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;vfoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')

if vfoptions.lowmemory>0
    special_n_z=ones(1,length(n_z));
    z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
end
if vfoptions.lowmemory>1
    special_n_a=ones(1,length(n_a));
    a_gridvals=CreateGridvals(n_a,a_grid,1); % The 1 at end indicates want output in form of matrix.
end

%%

%% j=N_j

if vfoptions.verbose==1
    sprintf('Age j is currently %i \n',N_j)
end

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
if vfoptions.EZoneminusbeta==1
    ezc1=1-DiscountFactorParamsVec; % Just in case it depends on age
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


if vfoptions.lowmemory==0
    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);

    % Modify the Return Function appropriately for Epstein-Zin Preferences
    becareful=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0)); % finite and not zero
    ReturnMatrix(becareful)=(ezc1*ReturnMatrix(becareful).^ezc2).^ezc7; % Otherwise can get things like 0 to negative power equals infinity
    ReturnMatrix(ReturnMatrix==0)=-Inf;
    
    % Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,:,N_j)=Vtemp;
    Policy(:,:,N_j)=maxindex;
    
elseif vfoptions.lowmemory==1
    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_grid, a_grid, z_val, ReturnFnParamsVec);

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix_z).*(ReturnMatrix_z~=0)); % finite and not zero
        ReturnMatrix_z(becareful)=(ezc1*ReturnMatrix_z(becareful).^ezc2).^ezc7; % Otherwise can get things like 0 to negative power equals infinity
        ReturnMatrix_z(ReturnMatrix_z==0)=-Inf;

        % Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
        V(:,z_c,N_j)=Vtemp;
        Policy(:,z_c,N_j)=maxindex;
    end
    
elseif vfoptions.lowmemory==2
    for a_c=1:N_a
        a_val=a_gridvals(a_c,:);
        ReturnMatrix_a=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, special_n_a, n_z, d_grid, a_val, z_grid, ReturnFnParamsVec);

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix_a).*(ReturnMatrix_a~=0)); % finite and not zero
        ReturnMatrix_a(becareful)=(ezc1*ReturnMatrix_a(becareful).^ezc2).^ezc7; % Otherwise can get things like 0 to negative power equals infinity
        ReturnMatrix_a(ReturnMatrix_a==0)=-Inf;
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix_a,[],1);
        V(a_c,:,N_j)=Vtemp;
        Policy(a_c,:,N_j)=maxindex;
    end
    
end

%%
% Case2_Type==3  % phi_a'(d,z')
if vfoptions.phiaprimedependsonage==0
    PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
    Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
end

for reverse_j=1:N_j-1
    jj=N_j-reverse_j;
    
    if vfoptions.verbose==1
        sprintf('Age j is currently %i \n',jj)
    end
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    if vfoptions.EZoneminusbeta==1
        ezc1=1-DiscountFactorParamsVec; % Just in case it depends on age
    end

    if vfoptions.phiaprimedependsonage==1
        PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
        Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
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
    
    VKronNext_j=V(:,:,jj+1);
    % Part of Epstein-Zin is before taking expectation
    temp=VKronNext_j;
    temp(isfinite(VKronNext_j))=(ezc4*VKronNext_j(isfinite(VKronNext_j))).^ezc5;
    temp(VKronNext_j==0)=0;
        
    if vfoptions.lowmemory==0
        Phi_aprimeMatrix=reshape(Phi_aprimeMatrix,[N_d*N_z,1]);
        zprime_ToMatchPhi=kron((1:1:N_z)',ones(N_d,1));
        EV=temp(Phi_aprimeMatrix+N_a*(zprime_ToMatchPhi-1));
        EV(isfinite(EV))=EV(isfinite(EV)).^ezc5;
        EV=reshape(EV,[N_d,N_z]);
        
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
        
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0)); % finite but not zero
        temp2=ReturnMatrix;
        temp2(becareful)=ReturnMatrix(becareful).^ezc2; % Otherwise can get things like 0 to negative power equals infinity
        temp2(ReturnMatrix==0)=-Inf; % Otherwise these ReturnMatrix=zero points get a finite amount added to them (from expectations) and were mishandled later

        EV=EV.*shiftdim(pi_z',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        temp4=EV;
        temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4))).^ezc6;
        temp4(EV==0)=0;

        entireRHS=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4.*ones(1,N_a,1);

        temp5=logical(isfinite(entireRHS).*(entireRHS~=0));
        entireRHS(temp5)=ezc1*entireRHS(temp5).^ezc7;  % matlab otherwise puts 0 to negative power to infinity
        entireRHS(entireRHS==0)=-Inf; % Dont want to consider these
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,:,jj)=ezc1*shiftdim(Vtemp,1);
        Policy(:,:,jj)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1
        Phi_aprimeMatrix=reshape(Phi_aprimeMatrix,[N_d*N_z,1]);
        zprime_ToMatchPhi=kron((1:1:N_z)',ones(N_d,1));
        EV=temp(Phi_aprimeMatrix+N_a*(zprime_ToMatchPhi-1));
        EV(isfinite(EV))=EV(isfinite(EV)).^ezc5;
        EV=reshape(EV,[N_d,N_z]);
        
        for z_c=1:N_z
            z_val=z_gridvals(z_c,:); % Value of z (not of z')

            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_grid, a_grid, z_val, ReturnFnParamsVec);
            
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_z).*(ReturnMatrix_z~=0)); % finite but not zero
            temp2=ReturnMatrix_z;
            temp2(becareful)=ReturnMatrix_z(becareful).^ezc2; % Otherwise can get things like 0 to negative power equals infinity
            temp2(ReturnMatrix_z==0)=-Inf; % Otherwise these ReturnMatrix=zero points get a finite amount added to them (from expectations) and were mishandled later

            EV_z=EV.*kron(pi_z(z_c,:),ones(N_d,1,'gpuArray'));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=reshape(sum(EV_z,2),[N_d,1]);

            temp4=EV_z;
            temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4))).^ezc6;
            temp4(EV_z==0)=0;

            entireRHS_z=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4*ones(1,N_a,1);

            temp5=logical(isfinite(entireRHS_z).*(entireRHS_z~=0));
            entireRHS_z(temp5)=ezc1*entireRHS_z(temp5).^ezc7;  % matlab otherwise puts 0 to negative power to infinity
            entireRHS_z(entireRHS_z==0)=-Inf;
            
            %calculate in order, the maximizing aprime indexes
            [Vtemp,Policy(:,z_c,jj)]=max(entireRHS_z,[],1);
            V(:,z_c,jj)=Vtemp;
        end
    elseif vfoptions.lowmemory==2
        Phi_aprimeMatrix=reshape(Phi_aprimeMatrix,[N_d*N_z,1]);
        zprime_ToMatchPhi=kron((1:1:N_z)',ones(N_d,1));
        EV=temp(Phi_aprimeMatrix+N_a*(zprime_ToMatchPhi-1));
        EV(isfinite(EV))=EV(isfinite(EV)).^ezc5;
        EV=reshape(EV,[N_d,N_z]);
        
        for z_c=1:N_Z
            EV_z=EV.*kron(pi_z(z_c,:),ones(N_d,1,'gpuArray'));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=reshape(sum(EV_z,2),[N_d,1]);

            temp4=EV_z;
            temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4))).^ezc6;
            temp4(EV_z==0)=0;
            
            for a_c=1:N_a
                a_val=a_gridvals(z_c,:);
                ReturnMatrix_az=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, special_n_a, special_n_z, d_grid, a_val, z_val, ReturnFnParamsVec);

                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(ReturnMatrix_az).*(ReturnMatrix_az~=0)); % finite but not zero
                temp2=ReturnMatrix_az;
                temp2(becareful)=ReturnMatrix_az(becareful).^ezc2; % Otherwise can get things like 0 to negative power equals infinity
                temp2(ReturnMatrix_az==0)=-Inf; % Otherwise these ReturnMatrix=zero points get a finite amount added to them (from expectations) and were mishandled later

                entireRHS_az=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4;

                temp5=logical(isfinite(entireRHS_az).*(entireRHS_az~=0));
                entireRHS_az(temp5)=entireRHS_az(temp5);  % matlab otherwise puts 0 to negative power to infinity
                entireRHS_az(entireRHS_az==0)=-Inf;
                
                %calculate in order, the maximizing aprime indexes
                [Vtemp,Policy(a_c,z_c,jj)]=max(entireRHS_az,[],1);
                V(a_c,z_c,jj)=Vtemp;
            end
        end
    end
end


end