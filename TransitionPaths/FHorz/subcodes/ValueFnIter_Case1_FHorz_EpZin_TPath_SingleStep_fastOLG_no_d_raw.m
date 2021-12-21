function [V, Policy]=ValueFnIter_Case1_FHorz_EpZin_TPath_SingleStep_fastOLG_no_d_raw(V,n_a,n_z,N_j, a_grid, z_grid,pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% DiscountFactorParamNames contains the names for the three parameters relating to
% Epstein-Zin preferences. Calling them beta, gamma, and psi,
% respectively the Epstein-Zin preferences are given by
% U_t= [ (1-beta)*u_t^(1-1/psi) + beta (E[(U_{t+1}^(1-gamma)])^((1-1/psi)/(1-gamma))]^(1/(1-1/psi))
% where
%  u_t is per-period utility function. c_t if just consuption, or ((c_t)^v(1-l_t)^(1-v)) if consumption and leisure (1-l_t)


N_a=prod(n_a);
N_z=prod(n_z);

Policy=zeros(N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%%
eval('fieldexists_pi_z_J=1;vfoptions.pi_z_J;','fieldexists_pi_z_J=0;')
eval('fieldexists_ExogShockFn=1;vfoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;vfoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')

if length(DiscountFactorParamNames)<3
    disp('ERROR: There should be at least three variables in DiscountFactorParamNames when using Epstein-Zin Preferences')
    dbstack
end

%% First, create the big 'next period (of transition path) expected value fn.

% VfastOLG will be N_d*N_aprime by N_a*N_z*N_j (note: N_aprime is just equal to N_a)

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

if fieldexists_pi_z_J==0 && fieldexists_ExogShockFn==0
    z_grid_J=z_grid.*ones(1,N_j);
    pi_z_J=pi_z.*ones(1,1,N_j);
elseif fieldexists_pi_z_J==1
    z_grid_J=vfoptions.z_grid_J;
    pi_z_J=vfoptions.pi_z_J;
elseif fieldexists_ExogShockFn==1
    z_grid_J=zeros(sum(n_z),N_j,'gpuArray');
    pi_z_J=zeros(N_z,N_z,N_j,'gpuArray');
    for jj=1:N_j
        if fieldexists_ExogShockFnParamNames==1
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
            ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
            for ii=1:length(ExogShockFnParamsVec)
                ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
            end
            [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
            z_grid_J(:,jj)=gpuArray(z_grid); pi_z_J(:,jj)=gpuArray(pi_z);
        else
            [z_grid,pi_z]=vfoptions.ExogShockFn(jj);
            z_grid_J(:,jj)=gpuArray(z_grid); pi_z_J(:,:,jj)=gpuArray(pi_z);
        end
    end
end
z_grid_J=z_grid_J'; % Give it the size required for CreateReturnFnMatrix_Case1_Disc_Par2_fastOLG(): N_j-by-sum(n_z)
pi_z_J=permute(pi_z_J,[3,2,1]); % Give it the size best for the loop below, namely (j,z',z)

ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2_fastOLG(ReturnFn, 0, n_a, n_z, N_j, [], a_grid, z_grid_J, ReturnFnParamsAgeMatrix);

% ReturnMatrix=permute(ReturnMatrix,[1 2 4 3]); % Swap j and z (so that z
% is last) % Modified CreateReturnFnMatrix_Case1_Disc_Par2_fastOLG to
% eliminate this step.
ReturnMatrix=reshape(ReturnMatrix,[N_a,N_a*N_j,N_z]);

DiscountFactorParamsVec=CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=[prod(DiscountFactorParamsVec(:,1:end-2),2),DiscountFactorParamsVec(:,end-1),DiscountFactorParamsVec(:,end)];
DiscountFactorParamsVec=DiscountFactorParamsVec';
DiscountFactorParamsVec=kron(ones(N_a,1),DiscountFactorParamsVec);

VKronNext=zeros(N_a,N_j,N_z,'gpuArray');
VKronNext(:,1:N_j-1,:)=permute(V(:,:,2:end),[1 3 2]); % Swap j and z
VKronNext=reshape(VKronNext,[N_a*N_j,N_z]);

% Modify the Return Function appropriately for Epstein-Zin Preferences
temp2=ReturnMatrix;
temp2(isfinite(ReturnMatrix))=ReturnMatrix(isfinite(ReturnMatrix)).^(1-1/DiscountFactorParamsVec(:,3));
temp2=(1-DiscountFactorParamsVec(:,1))*temp2;

for z_c=1:N_z
    ReturnMatrix_z=temp2(:,:,z_c); % ReturnMatrix(:,:,z_c);

    temp=VKronNext;
    temp(isfinite(VKronNext))=VKronNext(isfinite(VKronNext)).^(1-DiscountFactorParamsVec(:,2));
    temp(VKronNext==0)=0;
    
    %Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV_z=temp.*(ones(N_a*N_j,1,'gpuArray')*pi_z(z_c,:));
    EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV_z=sum(EV_z,2);
    
    % Could prob do all this without creating temp3, just operate
    % direct on EV_z, as last third line is just
    % EV_z(isnan(EV_z))=0; to deal with NaN resulting from 0 to
    % negative power (as earlier, replace them with zeros (as the
    % zeros come from the transition probabilities)
    temp3=EV_z;
    temp3(isfinite(temp3))=temp3(isfinite(temp3)).^((1-1/DiscountFactorParamsVec(:,3))/(1-DiscountFactorParamsVec(:,2)));
    temp3(EV_z==0)=0;

    discountedEV_z=DiscountFactorParamsVec(:,1).*reshape(temp3,[N_a,N_j]); % (aprime)-by-(j) % Note: jprime and j are essentially the same thing.

    entirediscountedEV_z=kron(discountedEV_z,ones(1,N_a));
    
    entirediscountedEV_z=ReturnMatrix_z+entirediscountedEV_z; %(aprime)-by-(a,j)
    
    %Calc the max and it's index
    [Vtemp,maxindex]=max(entirediscountedEV_z,[],1);
    V(:,z_c,:)=reshape(Vtemp.^(1/(1-1/DiscountFactorParamsVec(:,3))),[N_a,1,N_j]); % V is over (a,z,j)
    Policy(:,z_c,:)=reshape(maxindex,[N_a,1,N_j]); % Policy is over (a,z,j)

end

end