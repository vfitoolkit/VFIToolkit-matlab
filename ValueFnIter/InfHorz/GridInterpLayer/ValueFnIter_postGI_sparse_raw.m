function [VKron, Policy] = ValueFnIter_postGI_sparse_raw(VKronold,n_d,n_a,n_z,d_gridvals,a_grid,z_gridvals,pi_z,ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,vfoptions)
% Alternative version of ValueFnIter_postGI_nod_raw, using sparse matrix for Howard improvement.
% OUTPUTS
%   VKron:  Value function, size: [N_a,N_z]
%   Policy: Policy function, size: [3,N_a,N_z], where
%           Policy(1,:,:) is index of grid point of d_grid
%           Policy(2,:,:) is index of lower grid point of a_grid, which is
%           an integer from 1 to n_a-1
%           Policy(3,:,:) is index of second layer, which is integer from 1
%           to n2+2, where n2 = ngridinterp
% INPUTS
% VKronold:   Initial guess for value function, size: [N_a,N_z]
% n_a,n_z:    Grid dimensions
% a_grid:     Grid for endogenous state variable
% z_gridvals: Grid for exogenous state variables, size: [prod(n_z),length(n_z)]
% pi_z:       Transition matrix for Markov shock, size: [prod(n_z),prod(n_z)]
% DiscountFactorParamsVec
% ReturnFn:   Function handle
% ReturnFnParamsVec: Row vector of ReturnFn parameters
% vfoptions:  Structure with options for VFI.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

%% Create return function matrix on coarse grid
a_gridvals = a_grid; % only one endogenous state, else wouldn't end up here

% ReturnMatrix(d,a',a,z) with a' on coarse grid
ReturnMatrixraw = CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_z, d_gridvals, a_gridvals, a_gridvals, z_gridvals, ReturnFnParamsVec,1);
% Refine away d
ReturnMatrix = max(ReturnMatrixraw,[],1);
ReturnMatrix = reshape(ReturnMatrix,[N_a,N_a,N_z]);

a_ind_howard = repmat((1:1:N_a)',N_z,1); % a varies first, size: [N_a*N_z,1]
z_ind_howard = repelem((1:1:N_z)',N_a,1); % z varies second, size: [N_a*N_z,1]
ind_howard  = a_ind_howard+N_a*(z_ind_howard-1);

pi_z_transpose = pi_z';


%% Create finer grid
n2short = vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long  = vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));

special_n_z=ones(1,length(n_z),'gpuArray'); % as lowmemory=1

aind=gpuArray(0:1:N_a-1);

%%

% preallocate
VKron  = zeros(N_a,N_z,'gpuArray');
Policy = zeros(3,N_a,N_z,'gpuArray'); % first dim indexes the optimal choice for aprime and aprime2 (in GI layer)
Ftemp  = zeros(N_a,N_z,'gpuArray'); % useful for Howard

tempcounter=1;
currdist=Inf;
while currdist>vfoptions.tolerance && tempcounter<=vfoptions.maxiter

    %% Given VKronold, obtain VKron
    
    % Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV = VKronold*pi_z_transpose; % (a',z)

    for z_c=1:N_z
        z_val = z_gridvals(z_c,:);   % scalars z1,z2,etc.
        EV_z  = EV(:,z_c);           % (a',1)
        ReturnMatrix_z=ReturnMatrix(:,:,z_c); % (a',a) a' on coarse grid

        % First layer
        entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*EV_z; %(a',a)

        % No need to compute the maximum value in first layer
        [~,maxindex]=max(entireRHS_z,[],1); % Note: d has been refined away from this

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is 1-by-n_a
        % aprimeindexes has size [n2long,n_a]
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a
        aprime_ii = aprime_grid(aprimeindexes);

        % ReturnMatrix_ii_z is (N_d,n2long,n_a)
        ReturnMatrix_ii_z = CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_z, d_gridvals, aprime_ii, a_gridvals, z_val, ReturnFnParamsVec,1);
        % EV_z_interp is (n2long,n_a)
        EV_z_interp = interp1(a_grid,EV_z,aprime_ii,'linear');
        % entireRHS_ii_z is (N_d,n2long,n_a)
        entireRHS_ii_z = ReturnMatrix_ii_z + DiscountFactorParamsVec*reshape(EV_z_interp,[1,n2long,n_a]); % autofill d into first dimension of EV_z_interp
        
        [Vtempii,maxindex]=max(reshape(entireRHS_ii_z,[N_d*n2long,n_a]),[],1);
        VKron(:,z_c) = Vtempii;
        maxindex_d=rem(maxindex-1,N_d)+1;
        maxindex_aL2=ceil(maxindex/N_d);
        Policy(1,:,z_c) = maxindex_d; % d
        Policy(2,:,z_c) = midpoint; % midpoint
        Policy(3,:,z_c) = maxindex_aL2; % aprimeL2ind from 1 to n2long=3+2*n2short
        ReturnMatrixind=maxindex+N_d*n2long*aind;
        Ftemp(:, z_c)   = ReturnMatrix_ii_z(ReturnMatrixind);
    end

    %---------------------------------------------------------------------%
    if currdist>vfoptions.tolerance*10
        % Howard update
        % Ftemp(a,z) = ReturnMatrix_fine(g(a,z),a,z)
        Ftemp_vec = reshape(Ftemp,[N_a*N_z,1]);

        % Find interp indexes and weights
        adjust      = (Policy(3,:,:)<1+n2short+1); % if second layer is choosing below midpoint
        % (1) lower grid point: from 1 to N_a-1
        aprime_left = Policy(2,:,:)-adjust;
        % (2) Index on fine grid: from 1 (lower grid point) to 1+n2short+1 (upper grid point)
        ind_L2      = adjust.*Policy(3,:,:)+(1-adjust).*(Policy(3,:,:)-n2short-1);
        % (3) Weight on left grid point
        weight_left = 1-(ind_L2-1)/(n2short+1);

        % Left grid point
        aprime_opt_vec = aprime_left(:);
        % Mass on left grid point
        weight_opt_vec = weight_left(:);

        indp = aprime_opt_vec+N_a*(z_ind_howard-1);
        indpp = (aprime_opt_vec+1)+N_a*(z_ind_howard-1);
        Qmat = sparse(ind_howard,indp,weight_opt_vec,N_a*N_z,N_a*N_z)+...
            sparse(ind_howard,indpp,1-weight_opt_vec,N_a*N_z,N_a*N_z);

        for h_c=1:vfoptions.howards
            EV_howard = VKron*pi_z_transpose; % (a',z)
            EV_howard = reshape(EV_howard,[N_a*N_z,1]);
            VKron = Ftemp_vec+DiscountFactorParamsVec*Qmat*EV_howard;
            VKron = reshape(VKron,[N_a,N_z]);
        end
    end
    %---------------------------------------------------------------------%


    %% Compute distance and update
    VKrondist=VKron(:)-VKronold(:);
    currdist=max(abs(VKrondist));

    if vfoptions.verbose==1
        disp(currdist)
    end

    VKronold = VKron;
    tempcounter=tempcounter+1;

end


%% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:)=Policy(2,:,:)-adjust; % lower grid point
Policy(3,:,:)=adjust.*Policy(3,:,:)+(1-adjust).*(Policy(3,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

%Policy=Policy(1,:,:)+N_d*(Policy(2,:,:)-1)+N_d*N_a*(Policy(3,:,:)-1);

if tempcounter>=vfoptions.maxiter
    warning('Value fn iteration has stopped due to reaching the maximum number of iterations (not due to convergence); can be set by vfoptions.maxiter.')
end

end
