function [VKron, Policy, ValtKron, Policyalt]=ValueFnIter_InfHorz_QuasiHyperbolicS_postGI_nod_raw(Vunderbar, n_a, n_z, a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, beta0, ReturnFn, ReturnFnParams, vfoptions)
% Sophisticated quasi-hyperbolic on InfHorz with grid-interpolation-layer (postGI default).
% Joint Vhat/Vunderbar iteration on rough and fine grids:
%   Vhat       = u + beta0*beta * E[Vunderbar_next]   (the value the agent uses to choose)
%   Vunderbar  = u + beta      * E[Vunderbar_next]   (realised continuation under future QH choices)
% Convergence is tracked on Vunderbar. Howards is skipped (existing InfHorz Sophisticated
% raws also skip it for nod).
%
% Outputs:
%   VKron     = Vhat (N_a-by-N_z)
%   Policy    = QH policy as (3,N_a,N_z): rows (L1, L2, L2flag)
%   ValtKron  = Vunderbar (N_a-by-N_z)
%   Policyalt = [] (Sophisticated only has one policy)
% NOTE (2026-09-01): this raw runs plain value function iteration -- no Howards.
% Adding it is step 5 of InfHorz_QHSophisticated_Howards_proposal.md, and is DEFERRED on purpose
% until Howards-without-GI (S_raw and S_nod_raw, done 2026-09-01) has been validated on GPU.
% Two reasons to wait: the postGI Howards operator is a two-point interpolation with weights
% rather than a single lookup, so it should be built on a known-good simpler case; and while these
% two raws still run plain VFI they are the control that separates "Howards helped" from "the
% finer grid helped" when comparing currdist against the non-GI paths.

N_a=prod(n_a);
N_z=prod(n_z);

beta=prod(DiscountFactorParamsVec);
beta0beta=beta0*beta;

ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_a, n_a, n_z, a_grid, a_grid, z_gridvals, ReturnFnParams);

pi_z_alt=shiftdim(pi_z',-1);

addindexforaz=gpuArray(N_a*(0:1:N_a-1)'+N_a*N_a*(0:1:N_z-1));

Vhat=zeros(N_a,N_z,'gpuArray');

%% Rough-grid joint iteration
tempcounter=1;
currdist=Inf;
while currdist>(vfoptions.multigridswitch*vfoptions.tolerance) && tempcounter<=vfoptions.maxiter
    Vunderbarold=Vunderbar;

    EV=Vunderbarold.*pi_z_alt;
    EV(isnan(EV))=0;
    EV=sum(EV,2);

    entireRHS_QH=ReturnMatrix+beta0beta*EV;
    [Vhat,Policy]=max(entireRHS_QH,[],1);
    Vhat=shiftdim(Vhat,1);

    entireRHS_beta=ReturnMatrix+beta*EV;
    tempmaxindex=shiftdim(Policy,1)+addindexforaz;
    Vunderbar=reshape(entireRHS_beta(tempmaxindex),[N_a,N_z]);

    Vudist=Vunderbar(:)-Vunderbarold(:);
    Vudist(isnan(Vudist))=0;
    currdist=max(abs(Vudist));

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
pi_z_howards=repelem(pi_z,N_a,1); % row a+N_a*(z-1) is pi_z(z,:), matching how Vunderbar(:) linearises

%% Fine-grid joint iteration
tempcounter=1;
currdist=1;
while currdist>vfoptions.tolerance && tempcounter<=vfoptions.maxiter
    Vunderbarold=Vunderbar;

    EVpre=reshape(Vunderbarold(aprimeindex,:),[N_aprimediff,N_a,N_z,N_z]);
    EV=EVpre.*pi_z_alt2;
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,4));

    EVinterp=interp1(EVinterpindex1,EV,EVinterpindex2);

    entireRHS_QH=ReturnMatrixfine+beta0beta*EVinterp;
    [Vhat,Policy]=max(entireRHS_QH,[],1);
    Vhat=shiftdim(Vhat,1);

    entireRHS_beta=ReturnMatrixfine+beta*EVinterp;
    tempmaxindex=shiftdim(Policy,1)+addindexforazfine;
    Vunderbar=reshape(entireRHS_beta(tempmaxindex),[N_a,N_z]);

    Vudist=Vunderbar(:)-Vunderbarold(:);
    Vudist(isnan(Vudist))=0;
    currdist=max(abs(Vudist));

    % Howards: iterate Vunderbar ONLY, at the fine policy just chosen, sourced from Vunderbar.
    % Vhat is NOT iterated; it is one step off Vunderbar and the main loop recomputes it next pass.
    % See InfHorz_QHSophisticated_Howards_proposal.md sections 3.1 and 3.4.
    % The TWO-POINT form is the whole point. Recomputing EVinterp inside the inner loop would cost
    % one interp1 per inner step, which is what a whole outer iteration costs, so it would buy
    % nothing at all. At a FIXED fine index the interpolated continuation is just a two-point
    % weighted combination of coarse EV values, so the window position is resolved once here and
    % each inner step is two indexed lookups. The index arithmetic is this file's own post-process
    % block (below) applied in-loop, and the weighted-lookup pattern is the one in
    % ValueFnFromPolicy_InfHorz_GI.
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
        fineindexH=reshape(shiftdim(Policy,1),[N_a*N_z,1]);
        L1aH=ceil((fineindexH-1)/(n2short+1))-1;
        L1intermediateH=max(L1aH,0)+1;
        lowerH=reshape(max(L1aH,0)-vfoptions.maxaprimediff+aprimeshifter(:),[N_a,N_z]);
        upperH=min(lowerH+1,N_a);
        wupperH=reshape((fineindexH-(L1intermediateH-1)*(n2short+1)-1)/(n2short+1),[N_a,N_z]);
        FtempH=reshape(ReturnMatrixfine(tempmaxindex),[N_a,N_z]);
        for Howards_counter=1:vfoptions.howards
            EVlowerH=Vunderbar(lowerH,:).*pi_z_howards;
            EVlowerH(isnan(EVlowerH))=0;
            EVlowerH=reshape(sum(EVlowerH,2),[N_a,N_z]);
            EVupperH=Vunderbar(upperH,:).*pi_z_howards;
            EVupperH(isnan(EVupperH))=0;
            EVupperH=reshape(sum(EVupperH,2),[N_a,N_z]);
            Vunderbar=FtempH+beta*((1-wupperH).*EVlowerH+wupperH.*EVupperH);
        end
    end

    tempcounter=tempcounter+1;
end

%% postGIrepeat layer(s)
% NOTE: Howards is deliberately NOT added to the postGIrepeat loop below. vfoptions.postGIrepeat
% defaults to 0 and the quasi-hyperbolic dispatcher sets it to 0, so that loop does not run in any
% default configuration and would be shipping a second untested copy of the block above. Add it
% there only once the block above has been validated on a GPU run.
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
        Vunderbarold=Vunderbar;

        EVpre=reshape(Vunderbarold(aprimeindex,:),[N_aprimediff,N_a,N_z,N_z]);
        EV=EVpre.*pi_z_alt2;
        EV(isnan(EV))=0;
        EV=squeeze(sum(EV,4));

        EVinterp=interp1(EVinterpindex1,EV,EVinterpindex2);

        entireRHS_QH=ReturnMatrixfine+beta0beta*EVinterp;
        [Vhat,Policy]=max(entireRHS_QH,[],1);
        Vhat=shiftdim(Vhat,1);

        entireRHS_beta=ReturnMatrixfine+beta*EVinterp;
        tempmaxindex=shiftdim(Policy,1)+addindexforazfine;
        Vunderbar=reshape(entireRHS_beta(tempmaxindex),[N_a,N_z]);

        Vudist=Vunderbar(:)-Vunderbarold(:);
        Vudist(isnan(Vudist))=0;
        currdist=max(abs(Vudist));

        tempcounter=tempcounter+1;
    end
end

%% Outputs: Vhat -> VKron, Vunderbar -> ValtKron, L1/L2/L2flag for Policy
VKron=Vhat;
ValtKron=Vunderbar;
Policy_a=reshape(Policy,[1,N_a,N_z]);

% Separate fine-grid Policy_a into L1 and L2
fineindex=reshape(Policy_a,[N_a*N_z,1]);
L1a=ceil((fineindex-1)/(n2short+1))-1;
% the max(L1a,0) matters only at the window's bottom node, where L1a is -1; L2 below is built off the same thing, so L1 has to be too
L1=max(L1a,0)-vfoptions.maxaprimediff+aprimeshifter(:);
L1intermediate=max(L1a,0)+1;
L2=fineindex-(L1intermediate-1)*(n2short+1);

Policy=zeros(3,N_a,N_z,'gpuArray'); % +1 channel for PolicyL2flag
Policy(1,:,:)=reshape(L1,[1,N_a,N_z]);
Policy(2,:,:)=reshape(L2,[1,N_a,N_z]);

% L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
fineindex_lower = (L1intermediate-1)*(n2short+1) + 1;
fineindex_upper = L1intermediate*(n2short+1) + 1;
linidx_lower = reshape(fineindex_lower,[N_a,N_z]) + addindexforazfine;
linidx_upper = reshape(fineindex_upper,[N_a,N_z]) + addindexforazfine;
isInfLower = (ReturnMatrixfine(linidx_lower(:)) == -Inf);
isInfUpper = (ReturnMatrixfine(linidx_upper(:)) == -Inf);
inInterior = (L2 >= 2) & (L2 <= n2short+1);
Policy(3,:,:) = reshape(2 + (inInterior & isInfLower) - (inInterior & isInfUpper), [1,N_a,N_z]);
Policyalt=[];

if currdist > vfoptions.tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, vfoptions.tolerance)
end

end
