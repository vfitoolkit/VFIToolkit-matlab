function [VKron, Policy]=ValueFnIter_GI_nod_raw(VKron, n_a, n_z, a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn,ReturnFnParamsVec, vfoptions)
%Does pretty much exactly the same as ValueFnIter_Case1, only without any decision variable (n_d=0)

N_a=prod(n_a);
N_z=prod(n_z);

pi_z_alt=shiftdim(pi_z',-1);
pi_z_howards=repelem(pi_z,N_a,1);

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short))';
n2aprime=length(aprime_grid);

n_aprime=n_a+(n_a-1)*vfoptions.ngridinterp;
N_aprime=prod(n_aprime);

addindexforaz=gpuArray(N_a*(0:1:N_a-1)'+N_a*N_a*(0:1:N_z-1));
addindexforazL2=gpuArray(n2long*(0:1:N_a-1)'+n2long*N_a*(0:1:N_z-1));

% Create just the basic ReturnMatrix
ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_nod_Par2(ReturnFn, n_a, n_z, a_grid, z_gridvals, ReturnFnParamsVec);


%%
tempcounter=1;
currdist=Inf;

%% First, just consider a_grid for next period
while currdist>(vfoptions.multigridswitch*vfoptions.tolerance) && tempcounter<=vfoptions.maxiter
    VKronold=VKron;
    
    % Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=VKronold.*pi_z_alt;
    EV(isnan(EV))=0; % multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; %aprime by a by z

    %Calc the max and it's index
    [VKron,Policy]=max(entireRHS,[],1);

    tempmaxindex=shiftdim(Policy,1)+addindexforaz; % aprime index, add the index for a and z

    Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a,N_z]); % keep return function of optimal policy for using in Howards

    VKron=shiftdim(VKron,1); % a by z

    VKrondist=VKron(:)-VKronold(:); 
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    
    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards 
        Policy=Policy(:); % a by z (this shape is just convenient for Howards)
        for Howards_counter=1:vfoptions.howards
            EVKrontemp=VKron(Policy,:);
            EVKrontemp=EVKrontemp.*pi_z_howards;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=Ftemp+DiscountFactorParamsVec*EVKrontemp;
        end
    end

    tempcounter=tempcounter+1;

end


%% Now switch to considering the fine/interpolated aprime_grid
aind=(1:(1+vfoptions.ngridinterp):N_aprime)'; % indexes for a_grid in aprime_grid
currdist=1; % force going into the next while loop at least one iteration
while currdist>vfoptions.tolerance && tempcounter<vfoptions.maxiter

    VKronold=VKron;
    
    % Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=VKronold.*pi_z_alt;
    EV(isnan(EV))=0; % multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    % Interpolate EV over aprime_grid
    EVinterp=interp1(a_grid,EV,aprime_grid);

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; %aprime by a by z

    % Calc the max and it's index
    [~,maxindex]=max(entireRHS,[],1);

    % Turn this into the 'midpoint'
    midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is 1-by-n_a-by-n_z
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
    % aprime possibilities are n2long-by-n_a-by-n_z
    ReturnMatrixL2=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,n_z,aprime_grid(aprimeindexes),a_grid,z_gridvals,ReturnFnParamsVec,2);
    aprimez=aprimeindexes+n2aprime*shiftdim((0:1:N_z-1),-1);
    entireRHS_ii=ReturnMatrixL2+DiscountFactorParamsVec*reshape(EVinterp(aprimez(:)),[n2long,N_a,N_z]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    VKron=shiftdim(Vtempii,1);

    tempmaxindex=shiftdim(maxindexL2,1)+addindexforazL2; % aprime index, add the index for a and z
    Ftemp=reshape(ReturnMatrixL2(tempmaxindex),[N_a,N_z]); % keep return function of optimal policy for using in Howards

    VKrondist=VKron(:)-VKronold(:); 
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    
    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards 
        PolicyHowards=squeeze(midpoint)*(vfoptions.ngridinterp+1)+(squeeze(maxindexL2)-1-vfoptions.ngridinterp);
        PolicyHowards=reshape(PolicyHowards,[N_a*N_z,1]);
        for Howards_counter=1:vfoptions.howards
            EVKrontemp=interp1(aind,VKron,PolicyHowards);
            EVKrontemp=EVKrontemp.*pi_z_howards;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=Ftemp+DiscountFactorParamsVec*EVKrontemp;
        end
    end

    tempcounter=tempcounter+1;

end



%%
% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(maxindexL2<1+n2short+1); % if second layer is choosing below midpoint
midpoint=midpoint-adjust; % lower grid point
maxindexL2=adjust.*maxindexL2+(1-adjust).*(maxindexL2-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=zeros([2,N_a,N_z],'gpuArray');
Policy(1,:,:)=midpoint;
Policy(2,:,:)=maxindexL2;



end
