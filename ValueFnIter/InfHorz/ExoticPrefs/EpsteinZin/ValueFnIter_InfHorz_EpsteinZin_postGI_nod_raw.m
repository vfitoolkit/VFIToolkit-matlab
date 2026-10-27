function [VKron, Policy]=ValueFnIter_InfHorz_EpsteinZin_postGI_nod_raw(VKron, n_a, n_z, a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParams, vfoptions, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7)
% Epstein-Zin preferences, infinite horizon, grid interpolation layer (postGI), no decision variable.
% The structure is ValueFnIter_InfHorz_postGI_nod_raw (coarse-grid solve, then a fine/interpolated
% aprime grid within +-vfoptions.maxaprimediff of the coarse argmax, dense two-point Howards), with
% the Epstein-Zin recursion of ValueFnIter_InfHorz_EpsteinZin_nod_raw at every step:
%   V = [ ezc1*F^ezc2 + ezc3*beta*( E[(ezc4*V')^ezc5 | z] )^ezc6 ]^ezc7
% What is interpolated over aprime is the TRANSFORMED expectation EV=E[(ezc4*V')^ezc5], before the
% certainty-equivalent power ^ezc6 (as ValueFnIter_FHorz_EpsteinZin_GI1_nod_raw does): EV is linear
% in the transformed V', so a two-point weighted combination of coarse nodes IS its interpolation,
% and the Howards step and the main loop evaluate the same object.
%
% Conventions carried from the plain raws (2026-10-27): a -Inf (no feasible choice) is kept out of
% every power and, with ezc4=-1, becomes +Inf inside the transform so that ezc3*beta*(.) returns it
% to -Inf; a zero return is a value; an Inf-Inf is infeasible (-Inf); +Inf goes through the final
% power (Inf^ezc7 is the floor 0 when ezc7<0). Carried from the InfHorz grid-interpolation raws
% (InfHorzGI_InfNaN_proposal.md): in the two-point Howards step each weighted term is zeroed BEFORE
% the sum where its weight is exactly zero, because 0*(-Inf) is NaN and the isnan line that follows
% is meant for pi_z==0 only.
%
% Not implemented here, and refused by the dispatcher: preGI, postGIrepeat>0, howardsgreedy>0.
% vfoptions.howardssparse is not read: the Howards step is the dense two-point form (the sparse
% form is a matrix product with V, and the Epstein-Zin step is not linear in V).

N_a=prod(n_a);
N_z=prod(n_z);

ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_a, n_a, n_z, a_grid, a_grid, z_gridvals, ReturnFnParams);
% Modify the Return Function appropriately for Epstein-Zin Preferences (aprime-by-a-by-z)
becareful=isfinite(ReturnMatrix); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
temp2=ReturnMatrix;
temp2(becareful)=ReturnMatrix(becareful).^ezc2;

pi_z_alt=shiftdim(pi_z',-1);
pi_z_howards=repelem(pi_z,N_a,1);
addindexforaz=gpuArray(N_a*(0:1:N_a-1)'+N_a*N_a*(0:1:N_z-1));

DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

% Progress print for vfoptions.verbose (the fine-grid loop only; the coarse loop is a pre-solve)
distvstolstr=['ValueFnIter (Epstein-Zin, postGI): after %i iterations the dist is %4.',num2str(-round(log10(vfoptions.tolerance))),'f \n'];

%%
tempcounter=1;
currdist=Inf;
%% First, just consider a_grid for next period
while currdist>(vfoptions.multigridswitch*vfoptions.tolerance) && tempcounter<=vfoptions.maxiter
    VKronold=VKron;

    % Part of Epstein-Zin is before taking expectation
    temp=VKronold;
    temp(isfinite(VKronold))=(ezc4*VKronold(isfinite(VKronold))).^ezc5;
    temp(VKronold==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf

    % Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=temp.*pi_z_alt;
    EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension: aprime-by-1-by-z

    temp4=EV;
    temp4(~(temp4==-Inf))=temp4(~(temp4==-Inf)).^ezc6;
    temp4(EV==0)=0;

    entireRHS=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4; % aprime by a by z
    entireRHS(isnan(entireRHS))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
    temp5=~(entireRHS==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
    entireRHS(temp5)=entireRHS(temp5).^ezc7;

    % Calc the max and it's index
    [VKron,Policy]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1); % a by z

    VKrondist=VKron(:)-VKronold(:);
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
        tempmaxindex=shiftdim(Policy,1)+addindexforaz; % aprime index, add the index for a and z
        Ftemp=reshape(temp2(tempmaxindex),[N_a,N_z]); % the EZ-transformed return of the optimal policy, for Howards
        Policy=Policy(:); % a by z (this shape is just convenient for Howards)
        for Howards_counter=1:vfoptions.howards
            temp=VKron;
            temp(isfinite(VKron))=(ezc4*VKron(isfinite(VKron))).^ezc5;
            temp(VKron==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf
            EVKrontemp=temp(Policy,:);
            EVKrontemp=EVKrontemp.*pi_z_howards;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            temp4=EVKrontemp;
            temp4(~(temp4==-Inf))=temp4(~(temp4==-Inf)).^ezc6;
            temp4(EVKrontemp==0)=0;
            VKron=ezc1*Ftemp+ezc3*DiscountFactorParamsVec*temp4;
            VKron(isnan(VKron))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
            temp5=~(VKron==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
            VKron(temp5)=VKron(temp5).^ezc7;
        end
    end

    tempcounter=tempcounter+1;
end
Policy=reshape(Policy,[1,N_a,N_z]); % Howards can mess with the size

%% Now that we have solved on the rough grid, we resolve on the fine grid
% Only consider +-vfoptions.maxaprimediff coarse points around the coarse argmax, with all the
% interpolation points between them (see ValueFnIter_InfHorz_postGI_nod_raw for the reasoning).
n_aprimediff=1+2*vfoptions.maxaprimediff;
N_aprimediff=prod(n_aprimediff);
aprimeshifter=min(max(Policy,1+vfoptions.maxaprimediff),N_a-vfoptions.maxaprimediff);
aprimeindex=(-vfoptions.maxaprimediff:1:vfoptions.maxaprimediff)' +aprimeshifter; % size n_aprime-by-n_a
aprime_grid=a_grid(aprimeindex);
% Second, interpolate this
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n_aprime=n_aprimediff+(n_aprimediff-1)*vfoptions.ngridinterp;
N_aprime=prod(n_aprime);
aprime_grid=interp1((1:1:N_aprimediff)',aprime_grid,linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp)');
% Note: aprime_grid is N_aprime-by-N_a-by-N_z
ReturnMatrixfine=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_z, aprime_grid, a_grid, z_gridvals, ReturnFnParams,1);
becareful=isfinite(ReturnMatrixfine); % a zero return is a value, not infeasible
temp2fine=ReturnMatrixfine;
temp2fine(becareful)=ReturnMatrixfine(becareful).^ezc2;

EVinterpindex1=(1:1:N_aprimediff)';
EVinterpindex2=linspace(1,N_aprimediff,N_aprimediff+(N_aprimediff-1)*vfoptions.ngridinterp)';
% For Howards we need
addindexforazfine=gpuArray(N_aprime*(0:1:N_a-1)'+N_aprime*N_a*(0:1:N_z-1));
pi_z_alt2=shiftdim(pi_z,-2);

%% Now switch to considering the fine/interpolated aprime_grid
tempcounter=1; % reset the counter
currdist=1; % force going into the next while loop at least one iteration
while currdist>vfoptions.tolerance && tempcounter<=vfoptions.maxiter
    VKronold=VKron;

    % Part of Epstein-Zin is before taking expectation
    temp=VKronold;
    temp(isfinite(VKronold))=(ezc4*VKronold(isfinite(VKronold))).^ezc5;
    temp(VKronold==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf

    % Switch the transformed V into being over vfoptions.maxaprimediff
    EVpre=reshape(temp(aprimeindex,:),[N_aprimediff,N_a,N_z,N_z]); % last dimension is zprime
    % Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=EVpre.*pi_z_alt2;
    EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=squeeze(sum(EV,4)); % sum over z', leaving a singular second dimension
    % EV is now [N_aprimediff,N_a,N_z]
    % Interpolate the TRANSFORMED expectation over aprime_grid (BEFORE the certainty-equivalent power ^ezc6)
    EVinterp=interp1(EVinterpindex1,EV,EVinterpindex2);
    temp4interp=EVinterp;
    temp4interp(~(temp4interp==-Inf))=temp4interp(~(temp4interp==-Inf)).^ezc6;
    temp4interp(EVinterp==0)=0;

    entireRHS=ezc1*temp2fine+ezc3*DiscountFactorParamsVec*temp4interp; % aprime by a by z
    entireRHS(isnan(entireRHS))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
    temp5=~(entireRHS==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
    entireRHS(temp5)=entireRHS(temp5).^ezc7;

    % Calc the max and it's index
    [VKron,Policy]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1); % a by z

    VKrondist=VKron(:)-VKronold(:);
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    if vfoptions.verbose==1
        if rem(tempcounter,10)==0 % Every 10 iterations
            fprintf(distvstolstr, tempcounter,currdist)
        end
    end

    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards
        tempmaxindex=shiftdim(Policy,1)+addindexforazfine; % aprime index, add the index for a and z, size is [N_a,N_z]
        Ftemp=reshape(temp2fine(tempmaxindex),[N_a,N_z]); % the EZ-transformed return of the optimal policy, for Howards
        % Resolve the interpolation to a (lower grid index, weight) pair once, here, instead of
        % re-interpolating the whole of EV inside every Howards step. At a FIXED fine index the
        % interpolated transformed expectation is a two-point weighted combination of the
        % transformed V' at the two coarse nodes, pi-summed; the certainty-equivalent power is then
        % applied pointwise, exactly as in the main loop.
        Policy_L1a=ceil((Policy(:)-1)/(n2short+1))-1;
        % as above, max(Policy_L1a,0) so the lower index and the weight agree at the window's bottom node
        Policy_lowerind=max(Policy_L1a,0)-vfoptions.maxaprimediff+aprimeshifter(:); % index on the full a_grid
        Policy_lowerprob=1- ((Policy(:)-max(Policy_L1a,0)*(n2short+1))-1)/(n2short+1);
        lowerweightzero=(Policy_lowerprob==0); % the chosen point is a coarse node, coded as all the weight on the upper node
        upperweightzero=(Policy_lowerprob==1); % the chosen point is the window's bottom node
        for Howards_counter=1:vfoptions.howards
            temp=VKron;
            temp(isfinite(VKron))=(ezc4*VKron(isfinite(VKron))).^ezc5;
            temp(VKron==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf
            EVlower=Policy_lowerprob.*temp(Policy_lowerind,:); % [N_a*N_z,N_z], last dimension is zprime
            EVlower(lowerweightzero,:)=0; % zero weight times -Inf gives NaN, a zero weight contributes nothing
            EVupper=(1-Policy_lowerprob).*temp(Policy_lowerind+1,:);
            EVupper(upperweightzero,:)=0;
            EVKrontemp=(EVlower+EVupper).*pi_z_howards;
            EVKrontemp(isnan(EVKrontemp))=0; % now only pi_z==0 against -Inf
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            temp4=EVKrontemp;
            temp4(~(temp4==-Inf))=temp4(~(temp4==-Inf)).^ezc6;
            temp4(EVKrontemp==0)=0;
            VKron=ezc1*Ftemp+ezc3*DiscountFactorParamsVec*temp4;
            VKron(isnan(VKron))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
            temp5=~(VKron==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
            VKron(temp5)=VKron(temp5).^ezc7;
        end
    end

    tempcounter=tempcounter+1;
end

%% Switch policy to lower grid index and L2 index (is currently index on fine grid)
% Separate Policy into L1 and L2
fineindex=reshape(Policy,[N_a*N_z,1]);
L1a=ceil((fineindex-1)/(n2short+1))-1;  % this ranges -1:0:2*vfoptions.maxaprimediff-1
% the max(L1a,0) matters only at the window's bottom node, where L1a is -1; L2 below is built off the same thing, so L1 has to be too
L1=max(L1a,0)-vfoptions.maxaprimediff+aprimeshifter(:); % lower grid point index on the full grid; the window bounds put this in 1:N_a-1, so no clamp is needed
L1intermediate=max(L1a,0)+1; % lower grid point index (on the small grid, in form so we can get L2)
L2=fineindex-(L1intermediate-1)*(n2short+1); % L2 index
Policy=zeros(3,N_a,N_z,'gpuArray'); % +1 channel for PolicyL2flag
Policy(1,:,:)=reshape(L1,[1,N_a,N_z]);
Policy(2,:,:)=reshape(L2,[1,N_a,N_z]);
% L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
% Computed once, post-convergence, using final ReturnMatrixfine and final aprimeshifter
fineindex_lower = (L1intermediate-1)*(n2short+1) + 1;          % fine pos of lower coarse for chosen L1 segment
fineindex_upper = L1intermediate*(n2short+1) + 1;              % fine pos of upper coarse
linidx_lower = reshape(fineindex_lower,[N_a,N_z]) + addindexforazfine;
linidx_upper = reshape(fineindex_upper,[N_a,N_z]) + addindexforazfine;
isInfLower = (ReturnMatrixfine(linidx_lower(:)) == -Inf);
isInfUpper = (ReturnMatrixfine(linidx_upper(:)) == -Inf);
inInterior = (L2 >= 2) & (L2 <= n2short+1);
Policy(3,:,:) = reshape(2 + (inInterior & isInfLower) - (inInterior & isInfUpper), [1,N_a,N_z]);

if currdist > vfoptions.tolerance
    warning(['Value fn iteration (Epstein-Zin, postGI) has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, vfoptions.tolerance)
end

end
