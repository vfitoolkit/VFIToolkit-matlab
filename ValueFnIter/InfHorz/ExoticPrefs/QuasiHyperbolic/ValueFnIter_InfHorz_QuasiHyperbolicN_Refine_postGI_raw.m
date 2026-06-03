function [VKron, Policy, ValtKron, Policyalt]=ValueFnIter_InfHorz_QuasiHyperbolicN_Refine_postGI_raw(VKron, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, beta0, ReturnFn, ReturnFnParams, vfoptions)
% Naive quasi-hyperbolic on InfHorz with refinement + grid-interpolation-layer (postGI default).
% d-refinement is preference-independent (d only enters the return function, never the
% continuation), so the rough and fine return matrices are collapsed with the same
% max-over-d as in ValueFnIter_InfHorz_Refine_postGI_raw, regardless of QH discounts.
% After std V_std converges, take one extra fine-grid argmax using beta0*beta to get
% the QH-naive policy. Both Policy and Policyalt recover d via dstar lookup and pass
% through the same L1/L2/L2flag extraction.
%
% Outputs:
%   VKron     = Vtilde (N_a-by-N_z)
%   Policy    = QH policy as (4,N_a,N_z): rows (d, L1, L2, L2flag)
%   ValtKron  = V_std (N_a-by-N_z)
%   Policyalt = std policy as (4,N_a,N_z): rows (d, L1, L2, L2flag)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

beta=prod(DiscountFactorParamsVec);
beta0beta=beta0*beta;

n_da=[n_d,n_a];
da_gridvals=[repmat(d_gridvals,N_a,1),repelem(a_grid,N_d,1)];

%% Refinement: build rough ReturnMatrix and collapse d (rough dstar is not needed downstream)
if vfoptions.lowmemory==0
    ReturnMatrixraw=CreateReturnFnMatrix_Case2_Disc(ReturnFn,n_da, n_a, n_z, da_gridvals, a_grid, z_gridvals, ReturnFnParams);
    ReturnMatrixraw=reshape(ReturnMatrixraw,[N_d,N_a,N_a,N_z]);
    [ReturnMatrix,~]=max(ReturnMatrixraw,[],1);
    ReturnMatrix=shiftdim(ReturnMatrix,1);
elseif vfoptions.lowmemory==1
    ReturnMatrix=zeros(N_a,N_a,N_z,'gpuArray');
    l_z=length(n_z);
    special_n_z=ones(1,l_z);
    for z_c=1:N_z
        zvals=z_gridvals(z_c,:);
        ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn,n_da, n_a, special_n_z, da_gridvals, a_grid, zvals, ReturnFnParams);
        ReturnMatrix_z=reshape(ReturnMatrix_z,[N_d,N_a,N_a]);
        [ReturnMatrix_z,~]=max(ReturnMatrix_z,[],1);
        ReturnMatrix(:,:,z_c)=shiftdim(ReturnMatrix_z,1);
    end
end

pi_z_alt=shiftdim(pi_z',-1);
pi_z_howards=repelem(pi_z,N_a,1);

addindexforaz=gpuArray(N_a*(0:1:N_a-1)'+N_a*N_a*(0:1:N_z-1));

%% Rough-grid VFI for V_std
tempcounter=1;
currdist=Inf;
while currdist>(vfoptions.multigridswitch*vfoptions.tolerance) && tempcounter<=vfoptions.maxiter
    VKronold=VKron;

    EV=VKronold.*pi_z_alt;
    EV(isnan(EV))=0;
    EV=sum(EV,2);

    entireRHS=ReturnMatrix+beta*EV;

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
            VKron=Ftemp+beta*EVKrontemp;
        end
    end

    tempcounter=tempcounter+1;
end
Policy_a=reshape(Policy_a,[1,N_a,N_z]);

%% Fine-grid setup with d-refinement
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
    ReturnMatrixfine=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, aprime_grid, a_grid, z_gridvals, ReturnFnParams,1);
    [ReturnMatrixfine,dstar]=max(ReturnMatrixfine,[],1);
    ReturnMatrixfine=shiftdim(ReturnMatrixfine,1);
elseif vfoptions.lowmemory==1
    ReturnMatrixfine=zeros(N_aprime,N_a,N_z,'gpuArray');
    dstar=zeros(N_aprime,N_a,N_z,'gpuArray');
    l_z=length(n_z);
    special_n_z=ones(1,l_z);
    for z_c=1:N_z
        zvals=z_gridvals(z_c,:);
        ReturnMatrixfine_z=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_z, d_gridvals, aprime_grid(:,:,z_c), a_grid, zvals, ReturnFnParams,1);
        ReturnMatrixfine_z=reshape(ReturnMatrixfine_z,[N_d,N_aprime,N_a]);
        [ReturnMatrixfine_z,dstar_z]=max(ReturnMatrixfine_z,[],1);
        ReturnMatrixfine(:,:,z_c)=shiftdim(ReturnMatrixfine_z,1);
        dstar(:,:,z_c)=shiftdim(dstar_z,1);
    end
end

EVinterpindex1=gpuArray(1:1:N_aprimediff)';
EVinterpindex2=gpuArray(linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp))';

addindexforazfine=gpuArray(N_aprime*(0:1:N_a-1)'+N_aprime*N_a*(0:1:N_z-1));

pi_z_alt2=shiftdim(pi_z,-2);

%% Fine-grid VFI for V_std
tempcounter=1;
currdist=1;
while currdist>vfoptions.tolerance && tempcounter<=vfoptions.maxiter
    VKronold=VKron;

    EVpre=reshape(VKron(aprimeindex,:),[N_aprimediff,N_a,N_z,N_z]);
    EV=EVpre.*pi_z_alt2;
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,4));

    EVinterp=interp1(EVinterpindex1,EV,EVinterpindex2);

    entireRHS=ReturnMatrixfine+beta*EVinterp;

    [VKron,Policy_a]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1);

    VKrondist=VKron(:)-VKronold(:);
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
        tempmaxindex=shiftdim(Policy_a,1)+addindexforazfine;
        Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a,N_z]);
        tempmaxindex2=Policy_a(:)+N_aprime*gpuArray(0:1:N_a*N_z-1)';
        for Howards_counter=1:vfoptions.howards
            EVpre=reshape(VKron(aprimeindex,:),[N_aprimediff,N_a*N_z,N_z]);
            EVKrontemp=interp1(EVinterpindex1,EVpre,EVinterpindex2);
            EVKrontemp=reshape(EVKrontemp,[N_aprime*N_a*N_z,N_z]);
            EVKrontemp=EVKrontemp(tempmaxindex2,:);
            EVKrontemp=EVKrontemp.*pi_z_howards;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=Ftemp+beta*EVKrontemp;
        end
    end

    tempcounter=tempcounter+1;
end

%% postGIrepeat layer(s)
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
        ReturnMatrixfine=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, aprime_grid, a_grid, z_gridvals, ReturnFnParams,1);
        [ReturnMatrixfine,dstar]=max(ReturnMatrixfine,[],1);
        ReturnMatrixfine=shiftdim(ReturnMatrixfine,1);
    elseif vfoptions.lowmemory==1
        ReturnMatrixfine=zeros(N_aprime,N_a,N_z,'gpuArray');
        dstar=zeros(N_aprime,N_a,N_z,'gpuArray');
        l_z=length(n_z);
        special_n_z=ones(1,l_z);
        for z_c=1:N_z
            zvals=z_gridvals(z_c,:);
            ReturnMatrixfine_z=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_z, d_gridvals, aprime_grid(:,:,z_c), a_grid, zvals, ReturnFnParams,1);
            ReturnMatrixfine_z=reshape(ReturnMatrixfine_z,[N_d,N_aprime,N_a]);
            [ReturnMatrixfine_z,dstar_z]=max(ReturnMatrixfine_z,[],1);
            ReturnMatrixfine(:,:,z_c)=shiftdim(ReturnMatrixfine_z,1);
            dstar(:,:,z_c)=shiftdim(dstar_z,1);
        end
    end

    EVinterpindex1=gpuArray(1:1:N_aprimediff)';
    EVinterpindex2=gpuArray(linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp))';

    addindexforazfine=gpuArray(N_aprime*(0:1:N_a-1)'+N_aprime*N_a*(0:1:N_z-1));

    pi_z_alt2=shiftdim(pi_z,-2);

    tempcounter=1;
    currdist=1;
    while currdist>vfoptions.tolerance && tempcounter<=vfoptions.maxiter
        VKronold=VKron;

        EVpre=reshape(VKron(aprimeindex,:),[N_aprimediff,N_a,N_z,N_z]);
        EV=EVpre.*pi_z_alt2;
        EV(isnan(EV))=0;
        EV=squeeze(sum(EV,4));

        EVinterp=interp1(EVinterpindex1,EV,EVinterpindex2);

        entireRHS=ReturnMatrixfine+beta*EVinterp;

        [VKron,Policy_a]=max(entireRHS,[],1);
        VKron=shiftdim(VKron,1);

        VKrondist=VKron(:)-VKronold(:);
        VKrondist(isnan(VKrondist))=0;
        currdist=max(abs(VKrondist));

        if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
            tempmaxindex=shiftdim(Policy_a,1)+addindexforazfine;
            Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a,N_z]);
            tempmaxindex2=Policy_a(:)+N_aprime*(0:1:N_a*N_z-1)';
            for Howards_counter=1:vfoptions.howards
                EVpre=reshape(VKron(aprimeindex,:),[N_aprimediff,N_a*N_z,N_z]);
                EVKrontemp=interp1(EVinterpindex1,EVpre,EVinterpindex2);
                EVKrontemp=reshape(EVKrontemp,[N_aprime*N_a*N_z,N_z]);
                EVKrontemp=EVKrontemp(tempmaxindex2,:);
                EVKrontemp=EVKrontemp.*pi_z_howards;
                EVKrontemp(isnan(EVKrontemp))=0;
                EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
                VKron=Ftemp+beta*EVKrontemp;
            end
        end

        tempcounter=tempcounter+1;
    end
end

%% V_std converged; preserve as Valt and take one QH-naive fine-grid step
ValtKron=VKron;
Policy_std_a=reshape(Policy_a,[1,N_a,N_z]);

EVpre=reshape(ValtKron(aprimeindex,:),[N_aprimediff,N_a,N_z,N_z]);
EV=EVpre.*pi_z_alt2;
EV(isnan(EV))=0;
EV=squeeze(sum(EV,4));
EVinterp=interp1(EVinterpindex1,EV,EVinterpindex2);

entireRHS=ReturnMatrixfine+beta0beta*EVinterp;
[VKron,Policy_QH_a]=max(entireRHS,[],1);
VKron=shiftdim(VKron,1);
Policy_QH_a=reshape(Policy_QH_a,[1,N_a,N_z]);

%% Recover d via dstar and extract L1/L2/L2flag for both policies
Policy=local_extract_postGI_d(Policy_QH_a, dstar, aprimeshifter, n2short, vfoptions, N_a, N_z, N_aprime, ReturnMatrixfine, addindexforazfine);
Policyalt=local_extract_postGI_d(Policy_std_a, dstar, aprimeshifter, n2short, vfoptions, N_a, N_z, N_aprime, ReturnMatrixfine, addindexforazfine);

if tempcounter>=vfoptions.maxiter
    warning('Value fn iteration has stopped due to reaching the maximum number of iterations (not due to convergence); can be set by vfoptions.maxiter.')
end

end


function P=local_extract_postGI_d(Policy_a, dstar, aprimeshifter, n2short, vfoptions, N_a, N_z, N_aprime, ReturnMatrixfine, addindexforazfine)
% Convert fine-grid Policy_a into (d, L1, L2, L2flag) quadruple in shape (4,N_a,N_z).
P=zeros(4,N_a,N_z,'gpuArray');
temppolicyindex=reshape(Policy_a,[1,N_a*N_z])+N_aprime*(0:1:N_a*N_z-1);
P(1,:,:)=reshape(dstar(temppolicyindex),[N_a,N_z]);

fineindex=reshape(Policy_a,[N_a*N_z,1]);
L1a=ceil((fineindex-1)/(n2short+1))-1;
L1=max(L1a-vfoptions.maxaprimediff+1+aprimeshifter(:)-1,1);
L1intermediate=max(L1a,0)+1;
L2=fineindex-(L1intermediate-1)*(n2short+1);

P(2,:,:)=reshape(L1,[1,N_a,N_z]);
P(3,:,:)=reshape(L2,[1,N_a,N_z]);

fineindex_lower = (L1intermediate-1)*(n2short+1) + 1;
fineindex_upper = L1intermediate*(n2short+1) + 1;
linidx_lower = reshape(fineindex_lower,[N_a,N_z]) + addindexforazfine;
linidx_upper = reshape(fineindex_upper,[N_a,N_z]) + addindexforazfine;
isInfLower = (ReturnMatrixfine(linidx_lower(:)) == -Inf);
isInfUpper = (ReturnMatrixfine(linidx_upper(:)) == -Inf);
inInterior = (L2 >= 2) & (L2 <= n2short+1);
P(4,:,:) = reshape(2 + (inInterior & isInfLower) - (inInterior & isInfUpper), [1,N_a,N_z]);

end
