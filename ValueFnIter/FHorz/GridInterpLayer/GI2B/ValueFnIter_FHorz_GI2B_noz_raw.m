function [V, Policy]=ValueFnIter_FHorz_GI2B_noz_raw(n_d,n_a, N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(4,N_a,N_j,'gpuArray'); % first dim is (d,a1prime midpoint,a2prime,a1prime L2)

%%
n_a1=n_a(1);
n_a2=n_a(2:end);
N_a1=n_a1;
N_a2=n_a2;
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);
% aprime_grid=[a1prime_grid; a2_grid];

% For debugging, uncomment next two lines, with this 'aprime_grid' you
% should get exact same value fn as without interpolation (as it doesn't
% really interpolate, it just repeats points)
% aprime_grid=repelem(a_grid,1+n2short,1);
% aprime_grid=aprime_grid(1:(N_a+(N_a-1)*n2short));

% precompute
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-1); % already includes -1

a12ind=repmat(gpuArray(0:1:N_a1-1),1,N_a2)+N_a1*repelem(gpuArray(0:1:N_a2-1),1,N_a1);


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_DC2B_noz_Par2(ReturnFn,n_d,d_gridvals,a1_grid, a2_grid, a1_grid, a2_grid, ReturnFnParamsVec,1);

    % Calc the max and it's index: a1prime(d,1,a2prime,a1,a2)
    [~,maxindex]=max(ReturnMatrix,[],2);

    % Turn this into the 'midpoint'
    midpoint=max(min(maxindex,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2
    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
    % aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_noz_Par2(ReturnFn,n_d,d_gridvals,a1prime_grid(a1primeindexes),a2_grid,a1_grid,a2_grid,ReturnFnParamsVec,2);
    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
    maxindexL2d=rem(maxindexL2-1,N_d)+1;
    maxindexL2a=ceil(maxindexL2/N_d);
    maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
    maxindexL2a2=ceil(maxindexL2a/n2long);
    V(:,N_j)=shiftdim(Vtempii,1);
    Policy(1,:,N_j)=maxindexL2d; % d
    Policy(2,:,N_j)=midpoint(maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind); % a1prime midpoint
    Policy(3,:,N_j)=maxindexL2a2; % a2prime
    Policy(4,:,N_j)=maxindexL2a1; % a1primeL2ind
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=reshape(vfoptions.V_Jplus1,[N_a,1]);    % First, switch V_Jplus1 into Kron form

    EV=reshape(EV,[N_a1,N_a2]);
    % Interpolate EV over aprime_grid
    EVinterp=interp1(a1_grid,EV,a1prime_grid);

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_DC2B_noz_Par2(ReturnFn,n_d,d_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, ReturnFnParamsVec,1);
    entireRHS=ReturnMatrix+DiscountFactorParamsVec*shiftdim(EV,-1);

    % Calc the max and it's index: a1prime(d,1,a2prime,a1,a2)
    [~,maxindex]=max(entireRHS,[],2);

    % Turn this into the 'midpoint'
    midpoint=max(min(maxindex,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2
    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
    % aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_noz_Par2(ReturnFn,n_d,d_gridvals,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid,ReturnFnParamsVec,2);
    aprime=a1primeindexes+N_a1fine*a2ind;
    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EVinterp(aprime),[N_d*n2long*N_a2,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    maxindexL2d=rem(maxindexL2-1,N_d)+1;
    maxindexL2a=ceil(maxindexL2/N_d);
    maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
    maxindexL2a2=ceil(maxindexL2a/n2long);
    V(:,N_j)=shiftdim(Vtempii,1);
    Policy(1,:,N_j)=maxindexL2d; % d
    Policy(2,:,N_j)=midpoint(maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind); % a1prime midpoint
    Policy(3,:,N_j)=maxindexL2a2; % a2prime
    Policy(4,:,N_j)=maxindexL2a1; % a1primeL2ind
end


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
    end
    
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=V(:,jj+1);
    
    EV=reshape(EV,[N_a1,N_a2]);
    % Interpolate EV over aprime_grid
    EVinterp=interp1(a1_grid,EV,a1prime_grid);

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_DC2B_noz_Par2(ReturnFn,n_d,d_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, ReturnFnParamsVec,1);
    entireRHS=ReturnMatrix+DiscountFactorParamsVec*shiftdim(EV,-1);
    
    % Calc the max and it's index: a1prime(d,1,a2prime,a1,a2)
    [~,maxindex]=max(entireRHS,[],2);

    % Turn this into the 'midpoint'
    midpoint=max(min(maxindex,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2
    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
    % aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_noz_Par2(ReturnFn,n_d,d_gridvals,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid,ReturnFnParamsVec,2);
    aprime=a1primeindexes+N_a1fine*a2ind;
    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EVinterp(aprime),[N_d*n2long*N_a2,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    maxindexL2d=rem(maxindexL2-1,N_d)+1;
    maxindexL2a=ceil(maxindexL2/N_d);
    maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
    maxindexL2a2=ceil(maxindexL2a/n2long);
    V(:,jj)=shiftdim(Vtempii,1);
    Policy(1,:,jj)=maxindexL2d; % d
    Policy(2,:,jj)=midpoint(maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind); % a1prime midpoint
    Policy(3,:,jj)=maxindexL2a2; % a2prime
    Policy(4,:,jj)=maxindexL2a1; % a1primeL2ind
end


%% Currently Policy(2,:) is the midpoint, and Policy(4,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(4,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(4,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:)=Policy(2,:,:)-adjust; % lower grid point
Policy(4,:,:)=adjust.*Policy(4,:,:)+(1-adjust).*(Policy(4,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=Policy(1,:,:)+N_d*(Policy(2,:,:)-1)+N_d*N_a1*(Policy(3,:,:)-1)+N_d*N_a1*N_a2*(Policy(4,:,:)-1);


end