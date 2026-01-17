function [VKron,Policy]=ValueFnIter_Refine_postGI_raw(VKron,n_d,n_a,n_z,d_gridvals,a_grid,z_gridvals,pi_z,ReturnFn,DiscountFactorParamsVec,ReturnFnParams,vfoptions)
% Optimized version: Replaces interp1 with pre-computed interpolation matrix
% OPTIMIZATION: Line 177 bottleneck fixed with matrix multiplication approach

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_aNz = N_a*N_z;

n_da=[n_d,n_a];
da_gridvals=[repmat(d_gridvals,N_a,1),repelem(a_grid,N_d,1)];

%% CreateReturnFnMatrix_Case1_Disc creates a matrix of dimension (d and aprime)-by-a-by-z.
if vfoptions.lowmemory==0
    ReturnMatrixraw=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,n_da, n_a, n_z, da_gridvals, a_grid, z_gridvals, ReturnFnParams);
    ReturnMatrixraw=reshape(ReturnMatrixraw,[N_d,N_a,N_a,N_z]);
    
    [ReturnMatrix,~]=max(ReturnMatrixraw,[],1);
    ReturnMatrix=shiftdim(ReturnMatrix,1);

elseif vfoptions.lowmemory==1
    ReturnMatrix=zeros(N_a,N_a,N_z,'gpuArray');
    dstar=zeros(N_a,N_a,N_z,'gpuArray');
    l_z=length(n_z);
    special_n_z=ones(1,l_z);
    for z_c=1:N_z
        zvals=z_gridvals(z_c,:);
        ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,n_da, n_a, special_n_z, da_gridvals, a_grid, zvals, ReturnFnParams);
        ReturnMatrix_z=reshape(ReturnMatrix_z,[N_d,N_a,N_a]);
        [ReturnMatrix_z,dstar_z]=max(ReturnMatrix_z,[],1);
        ReturnMatrix(:,:,z_c)=shiftdim(ReturnMatrix_z,1);
        dstar(:,:,z_c)=shiftdim(dstar_z,1);
    end
end

pi_z_alt=shiftdim(pi_z',-1);
pi_z_howards=repelem(pi_z,N_a,1);

addindexforaz=gpuArray(N_a*(0:1:N_a-1)'+N_a*N_a*(0:1:N_z-1));

%% First, just consider a_grid for next period
tempcounter=1;
currdist=1;
while currdist>(vfoptions.multigridswitch*vfoptions.tolerance) && tempcounter<=vfoptions.maxiter
    VKronold=VKron;
    
    EV=VKronold.*pi_z_alt;
    EV(isnan(EV))=0;
    EV=sum(EV,2);

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV;

    [VKron,Policy_a]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1);

    VKrondist=VKron(:)-VKronold(:); 
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards 
        tempmaxindex=shiftdim(Policy_a,1)+addindexforaz;
        Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a,N_z]);
        Policy_a=Policy_a(:);
        for Howards_counter=1:vfoptions.howards
            EVKrontemp=VKron(Policy_a,:);
            EVKrontemp=EVKrontemp.*pi_z_howards;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=Ftemp+DiscountFactorParamsVec*EVKrontemp;
        end
    end

    tempcounter=tempcounter+1;
end
Policy_a=reshape(Policy_a,[1,N_a,N_z]);

%% Now that we have solved on the rough grid, we resolve on the fine grid
n_aprimediff=1+2*vfoptions.maxaprimediff;
N_aprimediff=prod(n_aprimediff);
aprimeshifter=min(max(Policy_a,1+vfoptions.maxaprimediff),N_a-vfoptions.maxaprimediff);
aprimeindex=(-vfoptions.maxaprimediff:1:vfoptions.maxaprimediff)' +aprimeshifter;
aprime_grid=a_grid(aprimeindex);

n2short=vfoptions.ngridinterp;
n_aprime=n_aprimediff+(n_aprimediff-1)*vfoptions.ngridinterp;
N_aprime=prod(n_aprime);
aprime_grid=interp1((1:1:N_aprimediff)',aprime_grid,linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp)');

if vfoptions.lowmemory==0
    ReturnMatrixfine=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_z, d_gridvals, aprime_grid, a_grid, z_gridvals, ReturnFnParams,1);
    
    [ReturnMatrixfine,dstar]=max(ReturnMatrixfine,[],1);
    ReturnMatrixfine=shiftdim(ReturnMatrixfine,1);

elseif vfoptions.lowmemory==1
    ReturnMatrixfine=zeros(N_aprime,N_a,N_z,'gpuArray');
    dstar=zeros(N_aprime,N_a,N_z,'gpuArray');
    l_z=length(n_z);
    special_n_z=ones(1,l_z);
    for z_c=1:N_z
        zvals=z_gridvals(z_c,:);
        ReturnMatrixfine_z=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_z, d_gridvals, aprime_grid(:,:,z_c), a_grid, zvals, ReturnFnParams,1);
        ReturnMatrixfine_z=reshape(ReturnMatrixfine_z,[N_d,N_aprime,N_a]);
        [ReturnMatrixfine_z,dstar_z]=max(ReturnMatrixfine_z,[],1);
        ReturnMatrixfine(:,:,z_c)=shiftdim(ReturnMatrixfine_z,1);
        dstar(:,:,z_c)=shiftdim(dstar_z,1);
    end
end

EVinterpindex2=gpuArray.linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp)';

% OPTIMIZATION: Pre-compute interpolation matrix (all on GPU)
idx_low = floor(EVinterpindex2);
idx_low = max(1, min(idx_low, N_aprimediff-1));
idx_high = idx_low + 1;
weight_high = EVinterpindex2 - idx_low;
weight_low = 1 - weight_high;

i_indices = [gpuArray.colon(1,N_aprime)'; gpuArray.colon(1,N_aprime)'];
j_indices = [idx_low; idx_high];
weights = [weight_low; weight_high];
interpMatrix_sparse = sparse(i_indices, j_indices, weights, N_aprime, N_aprimediff);
%interpMatrix = full(interpMatrix_sparse);  % Keep on GPU, convert to dense for fast multiplication
interpMatrix = interpMatrix_sparse;

addindexforazfine=gpuArray(N_aprime*(0:1:N_a-1)'+N_aprime*N_a*(0:1:N_z-1));

pi_z_alt2=shiftdim(pi_z,-2);

%% Now switch to considering the fine/interpolated aprime_grid
currdist=1;
while currdist>vfoptions.tolerance && tempcounter<=vfoptions.maxiter
    VKronold=VKron;

    EVpre=reshape(VKron(aprimeindex,:),[N_aprimediff,N_a,N_z,N_z]);
    EV=EVpre.*pi_z_alt2;
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,4));
    
    % OPTIMIZED: Matrix multiplication instead of interp1
    EVinterp = reshape(interpMatrix * reshape(EV, N_aprimediff, N_aNz), N_aprime, N_a, N_z);

    entireRHS=ReturnMatrixfine+DiscountFactorParamsVec*EVinterp;

    [VKron,Policy_a]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1);

    VKrondist=VKron(:)-VKronold(:); 
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards 
        tempmaxindex=shiftdim(Policy_a,1)+addindexforazfine;
        Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a,N_z]);
        tempmaxindex2=Policy_a(:)+N_aprime*(0:1:N_a*N_z-1)';
        
        N_aNz_Nz = N_aNz*N_z;
        
        for Howards_counter=1:vfoptions.howards
            EVpre=reshape(VKron(aprimeindex,:),[N_aprimediff,N_aNz,N_z]);
            
            % OPTIMIZED: Replace interp1 with matrix multiplication (LINE 177 BOTTLENECK)
            EVpre2D = reshape(EVpre, N_aprimediff, N_aNz_Nz);
            EVKrontemp = reshape(interpMatrix * EVpre2D, N_aprime*N_aNz, N_z);
            
            EVKrontemp=EVKrontemp(tempmaxindex2,:);
            EVKrontemp=EVKrontemp.*pi_z_howards;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=Ftemp+DiscountFactorParamsVec*EVKrontemp;
        end
    end

    tempcounter=tempcounter+1;
end

%% Do another post-GI layer
while vfoptions.postGIrepeat>0
    vfoptions.postGIrepeat=vfoptions.postGIrepeat-1;

    Policy_a=reshape(Policy_a,[1,N_a,N_z]);
    Policy_a=ceil((Policy_a-1)/(n2short+1))-vfoptions.maxaprimediff+aprimeshifter;

    n_aprimediff=1+2*vfoptions.maxaprimediff;
    N_aprimediff=prod(n_aprimediff);
    aprimeshifter=min(max(Policy_a,1+vfoptions.maxaprimediff),N_a-vfoptions.maxaprimediff);
    aprimeindex=(-vfoptions.maxaprimediff:1:vfoptions.maxaprimediff)' +aprimeshifter;
    aprime_grid=a_grid(aprimeindex);
    
    n2short=vfoptions.ngridinterp;
    n_aprime=n_aprimediff+(n_aprimediff-1)*vfoptions.ngridinterp;
    N_aprime=prod(n_aprime);
    aprime_grid=interp1((1:1:N_aprimediff)',aprime_grid,linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp)');

    if vfoptions.lowmemory==0
        ReturnMatrixfine=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_z, d_gridvals, aprime_grid, a_grid, z_gridvals, ReturnFnParams,1);

        [ReturnMatrixfine,dstar]=max(ReturnMatrixfine,[],1);
        ReturnMatrixfine=shiftdim(ReturnMatrixfine,1);

    elseif vfoptions.lowmemory==1
        ReturnMatrixfine=zeros(N_aprime,N_a,N_z,'gpuArray');
        dstar=zeros(N_aprime,N_a,N_z,'gpuArray');
        l_z=length(n_z);
        special_n_z=ones(1,l_z);
        for z_c=1:N_z
            zvals=z_gridvals(z_c,:);
            ReturnMatrixfine_z=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_z, d_gridvals, aprime_grid(:,:,z_c), a_grid, zvals, ReturnFnParams,1);
            ReturnMatrixfine_z=reshape(ReturnMatrixfine_z,[N_d,N_aprime,N_a]);
            [ReturnMatrixfine_z,dstar_z]=max(ReturnMatrixfine_z,[],1);
            ReturnMatrixfine(:,:,z_c)=shiftdim(ReturnMatrixfine_z,1);
            dstar(:,:,z_c)=shiftdim(dstar_z,1);
        end
    end

    EVinterpindex2=gpuArray.linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp)';

    % OPTIMIZATION: Recompute interpolation matrix (all on GPU)
    idx_low = floor(EVinterpindex2);
    idx_low = max(1, min(idx_low, N_aprimediff-1));
    idx_high = idx_low + 1;
    weight_high = EVinterpindex2 - idx_low;
    weight_low = 1 - weight_high;
    
    i_indices = [gpuArray.colon(1,N_aprime)'; gpuArray.colon(1,N_aprime)'];
    j_indices = [idx_low; idx_high];
    weights = [weight_low; weight_high];
    interpMatrix_sparse = sparse(i_indices, j_indices, weights, N_aprime, N_aprimediff);
    interpMatrix = full(interpMatrix_sparse);  % Keep on GPU

    addindexforazfine=gpuArray(N_aprime*(0:1:N_a-1)'+N_aprime*N_a*(0:1:N_z-1));

    pi_z_alt2=shiftdim(pi_z,-2);

    currdist=1;
    while currdist>vfoptions.tolerance && tempcounter<=vfoptions.maxiter
        VKronold=VKron;

        EVpre=reshape(VKron(aprimeindex,:),[N_aprimediff,N_a,N_z,N_z]);
        EV=EVpre.*pi_z_alt2;
        EV(isnan(EV))=0;
        EV=squeeze(sum(EV,4));

        % OPTIMIZED: Matrix multiplication
        EVinterp = reshape(interpMatrix * reshape(EV, N_aprimediff, N_aNz), N_aprime, N_a, N_z);

        entireRHS=ReturnMatrixfine+DiscountFactorParamsVec*EVinterp;

        [VKron,Policy_a]=max(entireRHS,[],1);
        VKron=shiftdim(VKron,1);

        VKrondist=VKron(:)-VKronold(:);
        VKrondist(isnan(VKrondist))=0;
        currdist=max(abs(VKrondist));

        if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
            tempmaxindex=shiftdim(Policy_a,1)+addindexforazfine;
            Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a,N_z]);
            tempmaxindex2=Policy_a(:)+N_aprime*(0:1:N_a*N_z-1)';
            
            N_aNz_Nz = N_aNz*N_z;
            
            for Howards_counter=1:vfoptions.howards
                EVpre=reshape(VKron(aprimeindex,:),[N_aprimediff,N_aNz,N_z]);
                
                % OPTIMIZED: Matrix multiplication (LINE 286 BOTTLENECK)
                EVpre2D = reshape(EVpre, N_aprimediff, N_aNz_Nz);
                EVKrontemp = reshape(interpMatrix * EVpre2D, N_aprime*N_aNz, N_z);
                
                EVKrontemp=EVKrontemp(tempmaxindex2,:);
                EVKrontemp=EVKrontemp.*pi_z_howards;
                EVKrontemp(isnan(EVKrontemp))=0;
                EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
                VKron=Ftemp+DiscountFactorParamsVec*EVKrontemp;
            end
        end

        tempcounter=tempcounter+1;
    end

end

%% For refinement, add d back into Policy
Policy=zeros(3,N_a,N_z,'gpuArray');
temppolicyindex=reshape(Policy_a,[1,N_a*N_z])+N_aprime*(0:1:N_a*N_z-1);
Policy(1,:,:)=reshape(dstar(temppolicyindex),[N_a,N_z]);

%% Switch policy to lower grid index and L2 index
fineindex=reshape(Policy_a,[N_a*N_z,1]);
L1a=ceil((fineindex-1)/(n2short+1))-1;
L1=max(L1a-vfoptions.maxaprimediff+1+aprimeshifter(:)-1,1);
L1intermediate=max(L1a,0)+1;
L2=fineindex-(L1intermediate-1)*(n2short+1);

Policy(2,:,:)=reshape(L1,[1,N_a,N_z]);
Policy(3,:,:)=reshape(L2,[1,N_a,N_z]);

end