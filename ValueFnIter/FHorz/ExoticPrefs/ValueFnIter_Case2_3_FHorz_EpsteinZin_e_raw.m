function [V,Policy]=ValueFnIter_Case2_3_FHorz_EpsteinZin_e_raw(n_d,n_a,n_z,n_e,N_j, d_grid, a_grid, z_grid,e_grid, pi_z,pi_e, Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7)

if warmglow==1
    error('Have not yet implemented warm-glow of bequests for Epstein-Zin when using Case2=3, email me if you need/want this')
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%

eval('fieldexists_pi_z_J=1;vfoptions.pi_z_J;','fieldexists_pi_z_J=0;')
eval('fieldexists_ExogShockFn=1;vfoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;vfoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')

eval('fieldexists_EiidShockFn=1;vfoptions.EiidShockFn;','fieldexists_EiidShockFn=0;')
eval('fieldexists_EiidShockFnParamNames=1;vfoptions.EiidShockFnParamNames;','fieldexists_EiidShockFnParamNames=0;')
eval('fieldexists_pi_e_J=1;vfoptions.pi_e_J;','fieldexists_pi_e_J=0;')

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
    e_gridvals=CreateGridvals(n_e,e_grid,1); % The 1 at end indicates want output in form of matrix.
end
if vfoptions.lowmemory>1
    special_n_z=ones(1,length(n_z));
    z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
end


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
    if all(size(z_grid)==[sum(n_z),1])
        z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
    elseif all(size(z_grid)==[prod(n_z),l_z])
        z_gridvals=z_grid;
    end
    if all(size(e_grid)==[sum(n_e),1]) % kronecker (cross-product) grid
        e_gridvals=CreateGridvals(n_e,e_grid,1); % The 1 at end indicates want output in form of matrix.
    elseif all(size(e_grid)==[prod(n_e),length(n_e)]) % joint-grid
        e_gridvals=e_grid;
    end
end

if vfoptions.lowmemory==0
    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2e(ReturnFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_grid, e_grid, ReturnFnParamsVec);

    % Modify the Return Function appropriately for Epstein-Zin Preferences
    becareful=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0)); % finite but not zero
    ReturnMatrix(becareful)=(ezc1*ReturnMatrix(becareful).^ezc2).^ezc7; % Otherwise can get things like 0 to negative power equals infinity
    ReturnMatrix(ReturnMatrix==0)=-Inf;

    % Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,:,:,N_j)=Vtemp;
    Policy(:,:,:,N_j)=maxindex;

elseif vfoptions.lowmemory==1
    for e_c=1:N_e
        e_val=e_gridvals(e_c,:);
        ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_Par2e(ReturnFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_grid, e_val, ReturnFnParamsVec);

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix_e).*(ReturnMatrix_e~=0)); % finite and not zero
        ReturnMatrix_e(becareful)=(ezc1*ReturnMatrix_e(becareful).^ezc2).^ezc7; % Otherwise can get things like 0 to negative power equals infinity
        ReturnMatrix_e(ReturnMatrix_e==0)=-Inf;

        % Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
        V(:,:,e_c,N_j)=Vtemp;
        Policy(:,:,e_c,N_j)=maxindex;
    end
elseif vfoptions.lowmemory==2
    for e_c=1:N_e
        e_val=e_gridvals(e_c,:);
        for z_c=1:N_z
            z_val=z_gridvals(z_c,:);
            ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_Par2e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_grid, a_grid, z_val, e_val, ReturnFnParamsVec);

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_ze).*(ReturnMatrix_ze~=0)); % finite and not zero
            ReturnMatrix_ze(becareful)=(ezc1*ReturnMatrix_ze(becareful).^ezc2).^ezc7; % Otherwise can get things like 0 to negative power equals infinity
            ReturnMatrix_ze(ReturnMatrix_ze==0)=-Inf;

            % Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
            V(:,z_c,e_c,N_j)=Vtemp;
            Policy(:,z_c,e_c,N_j)=maxindex;
        end
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
        if vfoptions.lowmemory==2 && (fieldexists_pi_z_J==1 || fieldexists_ExogShockFn==1)
            if all(size(z_grid)==[sum(n_z),1])
                z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
            elseif all(size(z_grid)==[prod(n_z),l_z])
                z_gridvals=z_grid;
            end
        end
        if (fieldexists_pi_e_J==1 || fieldexists_EiidShockFn==1)
            if all(size(e_grid)==[sum(n_e),1]) % kronecker (cross-product) grid
                e_gridvals=CreateGridvals(n_e,e_grid,1); % The 1 at end indicates want output in form of matrix.
            elseif all(size(e_grid)==[prod(n_e),length(n_e)]) % joint-grid
                e_gridvals=e_grid;
            end
        end
    end

    VKronNext_j=V(:,:,:,jj+1);

    % Part of Epstein-Zin is before taking expectation
    temp=VKronNext_j;
    temp(isfinite(VKronNext_j))=(ezc4*VKronNext_j(isfinite(VKronNext_j))).^ezc5;
    temp(VKronNext_j==0)=0; % otherwise zero to negative power is set to infinity

    % Expectation over e
    temp=sum(temp.*pi_e,3);

    if vfoptions.lowmemory==0
        Phi_aprimeMatrix=reshape(Phi_aprimeMatrix,[N_d*N_z,1]);
        zprime_ToMatchPhi=kron((1:1:N_z)',ones(N_d,1));
        EV=temp(Phi_aprimeMatrix+N_a*(zprime_ToMatchPhi-1));
        EV(isfinite(EV))=EV(isfinite(EV)).^ezc5;
        EV=reshape(EV,[N_d,N_z]);

        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2e(ReturnFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_grid, e_grid, ReturnFnParamsVec);

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0)); % finite and not zero
        temp2=ReturnMatrix;
        temp2(becareful)=ReturnMatrix(becareful).^ezc2;
        temp2(ReturnMatrix==0)=-Inf;

        % Calc the expectation term (except beta)
        EV=EV.*shiftdim(pi_z',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension

        temp4=EV;
        temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4))).^ezc6;
        temp4(EV==0)=0;

        entireRHS=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4*ones(1,N_a,1,N_e);

        temp5=logical(isfinite(entireRHS).*(entireRHS~=0));
        entireRHS(temp5)=ezc1*entireRHS(temp5).^ezc7;  % matlab otherwise puts 0 to negative power to infinity
        entireRHS(entireRHS==0)=-Inf;

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,:,:,jj)=ezc1*shiftdim(Vtemp,1);
        Policy(:,:,:,jj)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1
        Phi_aprimeMatrix=reshape(Phi_aprimeMatrix,[N_d*N_z,1]);
        zprime_ToMatchPhi=kron((1:1:N_z)',ones(N_d,1));
        EV=temp(Phi_aprimeMatrix+N_a*(zprime_ToMatchPhi-1));
        EV(isfinite(EV))=EV(isfinite(EV)).^ezc5;
        EV=reshape(EV,[N_d,N_z]);

        % Calc the expectation term (except beta)
        EV=EV.*shiftdim(pi_z',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension

        temp4=EV;
        temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4))).^ezc6;
        temp4(EV==0)=0;

        for e_c=1:N_e
            e_val=e_gridvals(e_c,:); % Value of e (not of e')

            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_Par2e(ReturnFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_grid, e_val, ReturnFnParamsVec);

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_e).*(ReturnMatrix_e~=0)); % finite and not zero
            temp2=ReturnMatrix_e;
            temp2(becareful)=ReturnMatrix_e(becareful).^ezc2;
            temp2(ReturnMatrix_e==0)=-Inf;

            entireRHS_e=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4*ones(1,N_a,1);

            temp5=logical(isfinite(entireRHS_e).*(entireRHS_e~=0));
            entireRHS_e(temp5)=ezc1*entireRHS_e(temp5).^ezc7;  % matlab otherwise puts 0 to negative power to infinity
            entireRHS_e(entireRHS_e==0)=-Inf;

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            V(:,:,e_c,jj)=ezc1*shiftdim(Vtemp,1);
            Policy(:,:,e_c,jj)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==2
        Phi_aprimeMatrix=reshape(Phi_aprimeMatrix,[N_d*N_z,1]);
        zprime_ToMatchPhi=kron((1:1:N_z)',ones(N_d,1));
        EV=temp(Phi_aprimeMatrix+N_a*(zprime_ToMatchPhi-1));
        EV(isfinite(EV))=EV(isfinite(EV)).^ezc5;
        EV=reshape(EV,[N_d,N_z]);

        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            for z_c=1:N_z
                z_val=z_gridvals(z_c,:); % Value of z (not of z')

                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_Par2e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_grid, a_grid, z_val, e_val, ReturnFnParamsVec);

                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(ReturnMatrix_ze).*(ReturnMatrix_ze~=0)); % finite and not zero
                temp2=ReturnMatrix_ze;
                temp2(becareful)=ReturnMatrix_ze(becareful).^ezc2;
                temp2(ReturnMatrix_ze==0)=-Inf;

                EV_z=EV.*kron(pi_z(z_c,:),ones(N_d,1,'gpuArray'));
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=reshape(sum(EV_z,2),[N_d,1]);

                temp4=EV_z;
                temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4))).^ezc6;
                temp4(EV_z==0)=0;

                entireRHS_ez=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4*ones(1,N_a);

                temp5=logical(isfinite(entireRHS_ez).*(entireRHS_ez~=0));
                entireRHS_ez(temp5)=ezc1*entireRHS_ez(temp5).^ezc7;  % matlab otherwise puts 0 to negative power to infinity
                entireRHS_ez(entireRHS_ez==0)=-Inf;

                %calculate in order, the maximizing aprime indexes
                [Vtemp,Policy(:,z_c,e_c,jj)]=max(entireRHS_ez,[],1);
                V(:,z_c,e_c,jj)=Vtemp;
            end
        end
    end
end


end