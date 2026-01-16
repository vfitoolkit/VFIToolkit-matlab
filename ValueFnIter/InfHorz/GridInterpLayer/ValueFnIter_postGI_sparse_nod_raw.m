function [VKron, Policy] = ValueFnIter_postGI_sparse_nod_raw(VKronold,n_a,n_z,a_grid,z_gridvals,pi_z,DiscountFactorParamsVec,ReturnFn,ReturnFnParamsVec,vfoptions)
% Improved version of ValueFnIter_postGI_nod_raw, using sparse matrix for
% Howard improvement.
% OUTPUTS
%   VKron:  Value function, size: [N_a,N_z]
%   Policy: Policy function, size: [2,N_a,N_z], where
%           Policy(1,:,:) is index of lower grid point of a_grid, which is
%           an integer from 1 to n_a-1
%           Policy(2,:,:) is index of second layer, which is integer from 1
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

N_a=prod(n_a);
N_z=prod(n_z);

ParamCell=cell(length(ReturnFnParamsVec),1);
for ii=1:length(ReturnFnParamsVec)
    if ~isequal(size(ReturnFnParamsVec(ii)), [1,1])
        error('Using GPU for the return fn does not allow for any of ReturnFnParams to be anything but a scalar')
    end
    ParamCell(ii,1)={ReturnFnParamsVec(ii)};
end

%% Create return function matrix on coarse grid
aprime_gridvals = reshape(a_grid,[n_a,1,1]);     % (a',1,1)
a_gridvals      = reshape(a_grid,[1,n_a,1]);     % (1,a,1)
z_gridvals3     = reshape(z_gridvals,[1,1,n_z]); % (1,1,z)

% ReturnMatrix(a',a,z) with a' on coarse grid
ReturnMatrix = arrayfun(ReturnFn, aprime_gridvals, a_gridvals, z_gridvals3, ParamCell{:});

%NA = gpuArray.colon(1,N_a)';
%NAZ = gpuArray.colon(1,N_a*N_z)';

[a_ind,z_ind] = ndgrid((1:N_a)',(1:N_z)');
a_ind = a_ind(:); % a varies first, size: [N_a*N_z,1]
z_ind = z_ind(:); % z varies second, size: [N_a*N_z,1]

pi_z_transpose = pi_z.';

Policy   = zeros(2,N_a,N_z,'gpuArray');

Tolerance = vfoptions.tolerance;
maxiter   = vfoptions.maxiter;
howards   = vfoptions.howards;
verbose   = vfoptions.verbose;
n2short   = vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)


%% Create finer grid
aprime_fine=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));

%% Value function iteration
% ReturnMatrix on coarse grid is precomputed, but ReturnMatrix_fine is
% computed at every iteration in the VFI loop.
tempcounter=1;
currdist=Inf;
while currdist>Tolerance && tempcounter<=maxiter

    % Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV = VKronold*pi_z_transpose; % (a',z)

    VKron = zeros(N_a,N_z,'gpuArray');
    Ftemp = zeros(N_a,N_z,'gpuArray'); % useful for Howard
    for z_c=1:N_z
        EV_z = EV(:,z_c);
        z_vals = z_gridvals(z_c,:);
        % ReturnMatrix(:,:,z_c) is (a',a), EV_z is (a',1)
        % entireRHS is (a',a) due to implicit expansion/broadcasting
        entireRHS=ReturnMatrix(:,:,z_c)+DiscountFactorParamsVec*EV_z; % size: [N_a,N_a]
        [~,max_ind]=max(entireRHS,[],1); % size: [1,N_a]

        % Refinement with a' on finer grid
        % Turn max_ind into the 'midpoint'
        midpoint=max(min(max_ind,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is 1-by-n_a
        % aprimeindexes has size [3+2*n2short,n_a]
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
        aprime_fine_small = aprime_fine(aprimeindexes);
        % ReturnMatrix_fine(a',a) has size: [3+2*n2short,n_a]
        ReturnMatrix_fine = CreateReturnFnMatrix_GI_nod(ReturnFn,aprime_fine_small,a_gridvals,z_vals,ParamCell);
        EV_z_interp       = interp1(a_grid,EV_z,aprime_fine_small,'linear','extrap');
        % entireRHS_fine has size [3+2*n2short,n_a]
        entireRHS_fine    = ReturnMatrix_fine+DiscountFactorParamsVec*EV_z_interp;
        [max_val,maxindexL2]=max(entireRHS_fine,[],1); % (1,N_a)
        VKron(:,z_c)    = max_val;
        % midpoint from 2 to n_a-1
        Policy(1,:,z_c) = midpoint;
        % aprimeL2ind from 1 to 3+2*n2short
        Policy(2,:,z_c) = maxindexL2;
        lin_ind         = sub2ind(size(ReturnMatrix_fine), maxindexL2, 1:N_a);
        Ftemp(:, z_c)   = ReturnMatrix_fine(lin_ind);
    end % end z

    %---------------------------------------------------------------------%
    % Howard update
    % Ftemp(a,z) = ReturnMatrix_fine(g(a,z),a,z)
    Ftemp_vec = reshape(Ftemp,[N_a*N_z,1]);

    % Find interp indexes and weights
    adjust      = (Policy(2,:,:)<1+n2short+1); % if second layer is choosing below midpoint
    % (1) lower grid point: from 1 to N_a-1
    aprime_left = Policy(1,:,:)-adjust;
    % (2) Index on fine grid: from 1 (lower grid point) to 1+n2short+1 (upper grid point)
    ind_L2      = adjust.*Policy(2,:,:)+(1-adjust).*(Policy(2,:,:)-n2short-1);
    % (3) Weight on left grid point
    weight_left = 1-(ind_L2-1)/(n2short+1);

    % Left grid point
    aprime_opt_vec = aprime_left(:);
    % Mass on left grid point
    weight_opt_vec = weight_left(:);

    ind  = a_ind+(z_ind-1)*N_a;
    indp = aprime_opt_vec+(z_ind-1)*N_a;
    indpp = aprime_opt_vec+1+(z_ind-1)*N_a;
    Qmat = sparse(ind,indp,weight_opt_vec,N_a*N_z,N_a*N_z)+...
        sparse(ind,indpp,1-weight_opt_vec,N_a*N_z,N_a*N_z);

    for h_c=1:howards
        EV_howard = VKron*pi_z_transpose; % (a',z)
        EV_howard = reshape(EV_howard,[N_a*N_z,1]);
        VKron = Ftemp_vec+DiscountFactorParamsVec*Qmat*EV_howard;
        VKron = reshape(VKron,[N_a,N_z]);
    end
    %---------------------------------------------------------------------%

    VKrondist=VKron(:)-VKronold(:);
    currdist=max(abs(VKrondist));

    if verbose==1
        disp(currdist)
    end

    VKronold = VKron;
    tempcounter=tempcounter+1;

end %end while

% Policy=reshape(Policy,[N_a,N_z]);
% VKron = reshape(VKron,[N_a,N_z]);

% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(1,:,:)=Policy(1,:,:)-adjust; % lower grid point
Policy(2,:,:)=adjust.*Policy(2,:,:)+(1-adjust).*(Policy(2,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

end %end function

%-------------------------------------------------------------------------%

function Fmatrix = CreateReturnFnMatrix_GI_nod(ReturnFn,aprime_grid,a_grid,z_gridvals,ParamCell)
% Assumption: z_gridvals has size [1,length(n_z)]


%ReturnMatrix_fine = arrayfun(ReturnFn,aprime_grid,a_grid,z_grid,ParamCell{:});

l_z = length(z_gridvals);
if l_z>3
    error('ERROR: not allow for more than 3 of z variable (you have length(n_z)>3)')
end

if l_z==1
    Fmatrix=arrayfun(ReturnFn, aprime_grid, a_grid, z_gridvals(1), ParamCell{:});
elseif l_z==2
    Fmatrix=arrayfun(ReturnFn, aprime_grid, a_grid, z_gridvals(1), z_gridvals(2), ParamCell{:});
elseif l_z==3
    Fmatrix=arrayfun(ReturnFn, aprime_grid, a_grid, z_gridvals(1), z_gridvals(2), z_gridvals(3), ParamCell{:});
end

end % end subfunction