function [VKron, Policy, ValtKron, Policyalt]=ValueFnIter_InfHorz_QuasiHyperbolicN_postGI_nod_raw(VKron, n_a, n_z, a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, beta0, ReturnFn, ReturnFnParams, vfoptions)
% Naive quasi-hyperbolic on InfHorz with grid-interpolation-layer (postGI default).
% Strategy: run the standard postGI VFI to convergence on V_std, then take one extra
% fine-grid argmax using beta0*beta and V_std as continuation to get the QH-naive
% policy and value Vtilde. L1/L2/L2flag are extracted for both Policy (QH) and
% Policyalt (std). Since d-refinement is preference-independent, the V_std fine grid
% reuses the same ReturnMatrixfine for the QH step.
%
% Caveat: the fine grid (+-vfoptions.maxaprimediff around Policyalt) may not contain
% the QH-optimal aprime if the QH policy drifts far from the std policy. Widen
% vfoptions.maxaprimediff if convergence diagnostics suggest the QH policy is
% pinned to the fine-grid boundary.
%
% Outputs:
%   VKron     = Vtilde (N_a-by-N_z)
%   Policy    = QH policy as (3,N_a,N_z): rows (L1, L2, L2flag)
%   ValtKron  = V_std (N_a-by-N_z)
%   Policyalt = std policy as (3,N_a,N_z): rows (L1, L2, L2flag)

N_a=prod(n_a);
N_z=prod(n_z);

beta=prod(DiscountFactorParamsVec);
beta0beta=beta0*beta;

ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_a, n_a, n_z, a_grid, a_grid, z_gridvals, ReturnFnParams);

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

    [VKron,Policy]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1);

    VKrondist=VKron(:)-VKronold(:);
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
        tempmaxindex=shiftdim(Policy,1)+addindexforaz;
        Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a,N_z]);
        Policy=Policy(:);
        for Howards_counter=1:vfoptions.howards
            EVKrontemp=VKron(Policy,:);
            EVKrontemp=EVKrontemp.*pi_z_howards;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=Ftemp+beta*EVKrontemp;
        end
    end

    tempcounter=tempcounter+1;
end
Policy=reshape(Policy,[1,N_a,N_z]);

%% Fine-grid setup
n_aprimediff=1+2*vfoptions.maxaprimediff;
N_aprimediff=prod(n_aprimediff);
aprimeshifter=min(max(Policy,1+vfoptions.maxaprimediff),N_a-vfoptions.maxaprimediff);
aprimeindex=(-vfoptions.maxaprimediff:1:vfoptions.maxaprimediff)' +aprimeshifter;
aprime_grid=a_grid(aprimeindex);

n2short=vfoptions.ngridinterp;
n_aprime=n_aprimediff+(n_aprimediff-1)*vfoptions.ngridinterp;
N_aprime=prod(n_aprime);
aprime_grid=interp1((1:1:N_aprimediff)',aprime_grid,linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp)');

ReturnMatrixfine=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_z, aprime_grid, a_grid, z_gridvals, ReturnFnParams,1);

EVinterpindex1=(1:1:N_aprimediff)';
EVinterpindex2=linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp)';

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

    [VKron,Policy]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1);

    VKrondist=VKron(:)-VKronold(:);
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
        tempmaxindex=shiftdim(Policy,1)+addindexforazfine;
        Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a,N_z]);
        tempmaxindex2=Policy(:)+N_aprime*(0:1:N_a*N_z-1)';
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

    Policy=reshape(Policy,[1,N_a,N_z]);
    Policy=ceil((Policy-1)/(n2short+1))-vfoptions.maxaprimediff+aprimeshifter;

    n_aprimediff=1+2*vfoptions.maxaprimediff;
    N_aprimediff=prod(n_aprimediff);
    aprimeshifter=min(max(Policy,1+vfoptions.maxaprimediff),N_a-vfoptions.maxaprimediff);
    aprimeindex=(-vfoptions.maxaprimediff:1:vfoptions.maxaprimediff)' +aprimeshifter;
    aprime_grid=a_grid(aprimeindex);

    n2short=vfoptions.ngridinterp;
    n_aprime=n_aprimediff+(n_aprimediff-1)*vfoptions.ngridinterp;
    N_aprime=prod(n_aprime);
    aprime_grid=interp1((1:1:N_aprimediff)',aprime_grid,linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp)');

    ReturnMatrixfine=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_z, aprime_grid, a_grid, z_gridvals, ReturnFnParams,1);

    EVinterpindex1=(1:1:N_aprimediff)';
    EVinterpindex2=linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp)';

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

        [VKron,Policy]=max(entireRHS,[],1);
        VKron=shiftdim(VKron,1);

        VKrondist=VKron(:)-VKronold(:);
        VKrondist(isnan(VKrondist))=0;
        currdist=max(abs(VKrondist));

        if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
            tempmaxindex=shiftdim(Policy,1)+addindexforazfine;
            Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a,N_z]);
            tempmaxindex2=Policy(:)+N_aprime*(0:1:N_a*N_z-1)';
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
Policy_std_a=reshape(Policy,[1,N_a,N_z]);

EVpre=reshape(ValtKron(aprimeindex,:),[N_aprimediff,N_a,N_z,N_z]);
EV=EVpre.*pi_z_alt2;
EV(isnan(EV))=0;
EV=squeeze(sum(EV,4));
EVinterp=interp1(EVinterpindex1,EV,EVinterpindex2);

entireRHS=ReturnMatrixfine+beta0beta*EVinterp;
[VKron,Policy_QH_a]=max(entireRHS,[],1);
VKron=shiftdim(VKron,1);
Policy_QH_a=reshape(Policy_QH_a,[1,N_a,N_z]);

%% L1/L2/L2flag extraction for both QH (Policy) and std (Policyalt)
Policy=local_extract_postGI(Policy_QH_a, aprimeshifter, n2short, vfoptions, N_a, N_z, N_aprime, ReturnMatrixfine, addindexforazfine);
Policyalt=local_extract_postGI(Policy_std_a, aprimeshifter, n2short, vfoptions, N_a, N_z, N_aprime, ReturnMatrixfine, addindexforazfine);

if currdist > vfoptions.tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, vfoptions.tolerance)
end

end


function P=local_extract_postGI(Policy_a, aprimeshifter, n2short, vfoptions, N_a, N_z, N_aprime, ReturnMatrixfine, addindexforazfine)
% Convert fine-grid Policy_a into (L1, L2, L2flag) triple in shape (3,N_a,N_z).
fineindex=reshape(Policy_a,[N_a*N_z,1]);
L1a=ceil((fineindex-1)/(n2short+1))-1;
L1=max(L1a-vfoptions.maxaprimediff+1+aprimeshifter(:)-1,1);
L1intermediate=max(L1a,0)+1;
L2=fineindex-(L1intermediate-1)*(n2short+1);

P=zeros(3,N_a,N_z,'gpuArray');
P(1,:,:)=reshape(L1,[1,N_a,N_z]);
P(2,:,:)=reshape(L2,[1,N_a,N_z]);

fineindex_lower = (L1intermediate-1)*(n2short+1) + 1;
fineindex_upper = L1intermediate*(n2short+1) + 1;
linidx_lower = reshape(fineindex_lower,[N_a,N_z]) + addindexforazfine;
linidx_upper = reshape(fineindex_upper,[N_a,N_z]) + addindexforazfine;
isInfLower = (ReturnMatrixfine(linidx_lower(:)) == -Inf);
isInfUpper = (ReturnMatrixfine(linidx_upper(:)) == -Inf);
inInterior = (L2 >= 2) & (L2 <= n2short+1);
P(3,:,:) = reshape(2 + (inInterior & isInfLower) - (inInterior & isInfUpper), [1,N_a,N_z]);

end
