function [VKron, Policy]=ValueFnIter_InfHorz_SemiExo_postGI_sparse_nod1_noz_e_raw(VKron, n_d2, n_a, n_semiz, n_e, d2_gridvals, a_grid, semiz_gridvals, e_gridvals, pi_semiz, pi_e, DiscountFactorParamsVec, ReturnMatrix, ReturnFn, ReturnFnParamsVec, vfoptions)
% Sparse-matrix (iterated) Howards: Howards is done with a sparse matrix from each state to the (aprime,semizprime) points
% of the expectation over eprime, with the probabilities pi_semiz(semiz,semizprime,d2) at the optimal d2 (and on the fine
% stage the two aprime grid points with the interpolation weights), rather than by indexing.
% postGI (grid interpolation layer), with semi-exogenous state semiz and iid e, no z. The only decision variable is d2,
% which determines the transition probabilities of semiz: pi_semiz(semiz,semizprime,d2).
% VKron is a-by-semiz-by-e, ReturnMatrix is the coarse (d2,aprime)-by-a-by-semiz-by-e [semiz varies first, then e]
% pi_e is a column vector of probabilities
% Policy is returned as [d2; aprime lower grid index; L2 index; L2flag]
%
% First solve on the coarse grid (as ValueFnIter_InfHorz_SemiExo_nod1_noz_e_raw) until near convergence. Then, around the
% coarse optimal aprime of each state, consider +-vfoptions.maxaprimediff coarse points with vfoptions.ngridinterp
% points between each, and solve choosing (d2,aprime) jointly on these fine points: d2 cannot be refined out, as it
% determines EV through pi_semiz, so EV(aprime,semiz,d2) is interpolated for every d2. EV does not depend on e (it
% is iid), so it is the same for every e, but the window (around the coarse optimum) differs by e.

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_e=prod(n_e);
N_bothz=N_semiz; % (no z)
n_bothz=n_semiz;

ReturnMatrix=reshape(ReturnMatrix,[N_d2,N_a,N_a,N_semiz,N_e]);

pi_e_alt=shiftdim(pi_e,-2); % 1-by-1-by-eprime
pi_semiz_alt=permute(pi_semiz,[4,2,1,5,3]); % 1-by-semizprime-by-semiz-by-1-by-d2

% For Howards we need
addindexforstate=gpuArray(N_d2*N_a*(0:1:N_a*N_semiz*N_e-1)'); % index for (a,semiz,e), for indexing ReturnMatrix at (d2,aprime)
semizind=gpuArray(repmat(repelem((1:1:N_semiz)',N_a,1),N_e,1)); % semiz of each state (a,semiz,e)
EVcolindex=N_a*gpuArray(0:1:N_semiz-1); % [1,N_semiz], index for semizprime in EVe (which is aprime-by-semizprime)
Pindex=semizind+N_semiz*gpuArray(0:1:N_semiz-1); % [N_a*N_semiz*N_e,N_semiz], index for (semiz,semizprime) in pi_semiz
stateind=gpuArray((1:1:N_a*N_bothz*N_e)'); % rows of the Howards matrix

%% First, just consider a_grid for next period
tempcounter=1;
currdist=Inf;
while currdist>(vfoptions.multigridswitch*vfoptions.tolerance) && tempcounter<=vfoptions.maxiter
    VKronold=VKron;

    % Calc the condl expectation term (except beta), which depends on semiz and d2
    % First the expectation over eprime, which is iid
    EVe=reshape(VKronold,[N_a,N_semiz,N_e]).*pi_e_alt;
    EVe(isnan(EVe))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
    EVe=sum(EVe,3); % sum over eprime, aprime-by-semizprime
    % Then the expectation over semizprime, conditional on semiz and d2
    EV=EVe.*pi_semiz_alt;
    EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=sum(EV,2); % sum over semizprime, aprime-by-1-by-semiz-by-1-by-d2
    EV=permute(EV,[5,1,2,3,4]); % d2-by-aprime-by-1-by-semiz

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; % d2 by aprime by a by semiz by e (EV autoexpands over a and e)

    % Calc the max and it's index
    [VKron,Policy]=max(reshape(entireRHS,[N_d2*N_a,N_a*N_bothz*N_e]),[],1);
    VKron=reshape(VKron,[N_a,N_bothz,N_e]);

    VKrondist=VKron(:)-VKronold(:);
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
        Policy=Policy(:); % (d2,aprime) index for each state (a,semiz,e)
        Ftemp=ReturnMatrix(Policy+addindexforstate); % keep return function of optimal policy for using in Howards
        Policy_d2=rem(Policy-1,N_d2)+1;
        Policy_aprime=ceil(Policy/N_d2);
        EVindex=Policy_aprime+EVcolindex; % [N_a*N_semiz*N_e,N_semiz], index for (aprime,semizprime) in EVe
        Pdz=pi_semiz(Pindex+N_semiz*N_semiz*(Policy_d2-1)); % [N_a*N_semiz*N_e,N_semiz], pi_semiz(semiz,semizprime,d2) at the optimal d2

        T=sparse(repmat(stateind,1,N_semiz),EVindex,Pdz,N_a*N_bothz*N_e,N_a*N_bothz); % sparse() drops the zero probabilities
        for Howards_counter=1:vfoptions.howards
            EVe=reshape(VKron,[N_a,N_semiz,N_e]).*pi_e_alt;
            EVe(isnan(EVe))=0;
            EVcol=reshape(sum(EVe,3),[N_a*N_semiz,1]); % aprime-by-semizprime (as a column)
            % GPU sparse*vector gives NaN when the vector holds -Inf, even where the row has no entry for that column (CPU gives the finite
            % product), so the -Inf entries are zeroed for the product and put back after: a row that puts positive weight on a -Inf entry is
            % -Inf itself. (CoreInfHorzVFIAlgoTests doPart(11); InfHorzGI_InfNaN_proposal.md, class B)
            EVcolisdead=(EVcol==-Inf);
            EVcol(EVcolisdead)=0;
            VKron=reshape(Ftemp+DiscountFactorParamsVec*(T*EVcol),size(VKron));
            VKron(T*double(EVcolisdead)>0)=-Inf;
        end
    end

    tempcounter=tempcounter+1;

end
Policy_aprime=reshape(ceil(Policy/N_d2),[1,N_a,N_bothz,N_e]); % coarse optimal aprime (at the optimal d2)

%% Now that we have solved on the rough grid, we resolve on the fine grid
% Consider +-vfoptions.maxaprimediff coarse points around the coarse optimal aprime, and interpolate between them
n_aprimediff=1+2*vfoptions.maxaprimediff;
N_aprimediff=prod(n_aprimediff);
aprimeshifter=min(max(Policy_aprime,1+vfoptions.maxaprimediff),N_a-vfoptions.maxaprimediff);
aprimeindex=(-vfoptions.maxaprimediff:1:vfoptions.maxaprimediff)' +aprimeshifter; % size n_aprimediff-by-n_a-by-n_bothz-by-n_e
aprime_grid=a_grid(aprimeindex);
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n_aprime=n_aprimediff+(n_aprimediff-1)*vfoptions.ngridinterp;
N_aprime=prod(n_aprime);
aprime_grid=interp1((1:1:N_aprimediff)',aprime_grid,linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp)');
% Note: aprime_grid is N_aprime-by-N_a-by-N_semiz-by-N_e

ReturnMatrixfine=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d2, n_bothz, n_e, d2_gridvals, aprime_grid, a_grid, semiz_gridvals, e_gridvals, ReturnFnParamsVec,1); % d2-by-aprime(fine)-by-a-by-semiz-by-e

EVinterpindex1=(1:1:N_aprimediff)';
EVinterpindex2=linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp)';

% Index for EV(aprime,(semiz,z),d2) at the window points of each state (a,semiz,e), for every d2
aprimeindexbothzd2=aprimeindex+N_a*shiftdim(gpuArray(0:1:N_bothz-1),-1)+N_a*N_bothz*shiftdim(gpuArray(0:1:N_d2-1),-3); % [N_aprimediff,N_a,N_bothz,N_e,N_d2]

% For Howards we need
addindexforstatefine=gpuArray(N_d2*N_aprime*(0:1:N_a*N_bothz*N_e-1)'); % index for (a,semiz,e), for indexing ReturnMatrixfine at (d2,aprime)

%% Now switch to considering the fine/interpolated aprime_grid
tempcounter=1; % reset the counter
currdist=1; % force going into the next while loop at least one iteration
while currdist>vfoptions.tolerance && tempcounter<=vfoptions.maxiter
    VKronold=VKron;

    % Calc the condl expectation term (except beta), which depends on semiz and d2
    EVe=reshape(VKronold,[N_a,N_semiz,N_e]).*pi_e_alt;
    EVe(isnan(EVe))=0;
    EVe=sum(EVe,3); % sum over eprime, aprime-by-semizprime
    EV=EVe.*pi_semiz_alt;
    EV(isnan(EV))=0;
    EV=reshape(sum(EV,2),[N_a,N_semiz,N_d2]); % sum over semizprime, aprime-by-semiz-by-d2
    EV=EV(aprimeindexbothzd2); % [N_aprimediff,N_a,N_bothz,N_e,N_d2]
    % Interpolate EV over aprime_grid
    EVinterp=interp1(EVinterpindex1,EV,EVinterpindex2); % [N_aprime,N_a,N_bothz,N_e,N_d2]
    EVinterp=permute(EVinterp,[5,1,2,3,4]); % d2-by-aprime(fine)-by-a-by-semiz-by-e

    entireRHS=ReturnMatrixfine+DiscountFactorParamsVec*EVinterp; % d2 by aprime(fine) by a by (semiz,z) by e

    % Calc the max and it's index
    [VKron,Policy]=max(reshape(entireRHS,[N_d2*N_aprime,N_a*N_bothz*N_e]),[],1);
    VKron=reshape(VKron,[N_a,N_bothz,N_e]);

    VKrondist=VKron(:)-VKronold(:);
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
        Policy=Policy(:); % (d2,aprime(fine)) index for each state (a,semiz,e)
        Ftemp=ReturnMatrixfine(Policy+addindexforstatefine); % keep return function of optimal policy for using in Howards
        Policy_d2=rem(Policy-1,N_d2)+1;
        Policy_fine=ceil(Policy/N_d2);
        % Resolve the interpolation to a (lower grid index, weight) pair once, here
        Policy_L1a=ceil((Policy_fine-1)/(n2short+1))-1;
        Policy_lowerind=max(Policy_L1a,0)-vfoptions.maxaprimediff+aprimeshifter(:); % index on the full a_grid
        Policy_lowerprob=1- ((Policy_fine-max(Policy_L1a,0)*(n2short+1))-1)/(n2short+1);
        EVindex=Policy_lowerind+EVcolindex; % [N_a*N_semiz*N_e,N_semiz], index for (aprime lower,semizprime) in EVe
        Pdz=pi_semiz(Pindex+N_semiz*N_semiz*(Policy_d2-1)); % [N_a*N_semiz*N_e,N_semiz], pi_semiz(semiz,semizprime,d2) at the optimal d2

        T=sparse(repmat(stateind,1,2*N_semiz),[EVindex,EVindex+1],[Policy_lowerprob.*Pdz,(1-Policy_lowerprob).*Pdz],N_a*N_bothz*N_e,N_a*N_bothz); % two aprime points; sparse() drops the zero probabilities
        for Howards_counter=1:vfoptions.howards
            EVe=reshape(VKron,[N_a,N_semiz,N_e]).*pi_e_alt;
            EVe(isnan(EVe))=0;
            EVcol=reshape(sum(EVe,3),[N_a*N_semiz,1]); % aprime-by-semizprime (as a column)
            % GPU sparse*vector gives NaN when the vector holds -Inf, even where the row has no entry for that column (CPU gives the finite
            % product), so the -Inf entries are zeroed for the product and put back after: a row that puts positive weight on a -Inf entry is
            % -Inf itself. (CoreInfHorzVFIAlgoTests doPart(11); InfHorzGI_InfNaN_proposal.md, class B)
            EVcolisdead=(EVcol==-Inf);
            EVcol(EVcolisdead)=0;
            VKron=reshape(Ftemp+DiscountFactorParamsVec*(T*EVcol),size(VKron));
            VKron(T*double(EVcolisdead)>0)=-Inf;
        end
    end

    tempcounter=tempcounter+1;
end

%% Switch policy to d2, lower grid index and L2 index (aprime is currently index on fine grid)
Policy=Policy(:);
Policy_d2=rem(Policy-1,N_d2)+1;
fineindex=ceil(Policy/N_d2);
L1a=ceil((fineindex-1)/(n2short+1))-1;  % this ranges -1:0:2*vfoptions.maxaprimediff-1
L1=max(L1a,0)-vfoptions.maxaprimediff+aprimeshifter(:); % lower grid point index on the full grid
L1intermediate=max(L1a,0)+1; % lower grid point index (on the small grid, in form so we can get L2)
L2=fineindex-(L1intermediate-1)*(n2short+1); % L2 index

Policy=zeros(4,N_a,N_bothz,N_e,'gpuArray'); % [d2; aprime lower; L2; L2flag]
Policy(1,:,:,:)=reshape(Policy_d2,[1,N_a,N_bothz,N_e]);
Policy(2,:,:,:)=reshape(L1,[1,N_a,N_bothz,N_e]);
Policy(3,:,:,:)=reshape(L2,[1,N_a,N_bothz,N_e]);

% L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper), at the optimal d2
fineindex_lower=(L1intermediate-1)*(n2short+1)+1; % fine pos of lower coarse for chosen L1 segment
fineindex_upper=L1intermediate*(n2short+1)+1; % fine pos of upper coarse
isInfLower=(ReturnMatrixfine(Policy_d2+N_d2*(fineindex_lower-1)+addindexforstatefine)==-Inf);
isInfUpper=(ReturnMatrixfine(Policy_d2+N_d2*(fineindex_upper-1)+addindexforstatefine)==-Inf);
inInterior=(L2>=2) & (L2<=n2short+1);
Policy(4,:,:,:)=reshape(2+(inInterior & isInfLower)-(inInterior & isInfUpper),[1,N_a,N_bothz,N_e]);

if currdist > vfoptions.tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, vfoptions.tolerance)
end

end
