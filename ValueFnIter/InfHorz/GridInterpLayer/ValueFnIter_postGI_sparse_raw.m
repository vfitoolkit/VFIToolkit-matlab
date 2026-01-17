function [VKron, Policy] = ValueFnIter_postGI_sparse_raw(VKronold,n_d,n_a,n_z,d_gridvals,a_grid,z_gridvals,pi_z,ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,vfoptions)

% Improved version of ValueFnIter_postGI_nod_raw, using sparse matrix for
% Howard improvement.
% OUTPUTS
%   VKron:  Value function, size: [N_a,N_z]
%   Policy: Policy function, size: [2,N_a,N_z], where
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
%d_gridvals      = d_gridvals;                 % (d,1,1,1)
aprime_gridvals = reshape(a_grid,[1,n_a,1,1]); % (1,a',1)
a_gridvals      = reshape(a_grid,[1,1,n_a]);     % (1,a,1)

% ReturnMatrix(d,a',a,z) with a' on coarse grid
ReturnMatrixraw = CreateReturnFnMatrix_GI(ReturnFn,d_gridvals,aprime_gridvals,a_gridvals,z_gridvals,ParamCell,n_z);

ReturnMatrix = max(ReturnMatrixraw,[],1);
ReturnMatrix = reshape(ReturnMatrix,[N_a,N_a,N_z]);

%NA = gpuArray.colon(1,N_a)';
%NAZ = gpuArray.colon(1,N_a*N_z)';

[a_ind,z_ind] = ndgrid((1:N_a)',(1:N_z)');
a_ind = a_ind(:); % a varies first, size: [N_a*N_z,1]
z_ind = z_ind(:); % z varies second, size: [N_a*N_z,1]

pi_z_transpose = pi_z.';

Tolerance = vfoptions.tolerance;
maxiter   = vfoptions.maxiter;
howards   = vfoptions.howards;
verbose   = vfoptions.verbose;

%% Create finer grid
n2short = vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long  = vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));


VKron  = zeros(N_a,N_z,'gpuArray');
Policy = zeros(3,N_a,N_z,'gpuArray'); % first dim indexes the optimal choice for aprime and aprime2 (in GI layer)
Ftemp  = zeros(N_a,N_z,'gpuArray'); % useful for Howard

tempcounter=1;
currdist=Inf;
while currdist>Tolerance && tempcounter<=maxiter

    %% Given VKronold, obtain VKron


    % Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV = VKronold*pi_z_transpose; % (a',z)

    for z_c=1:N_z
        z_val = z_gridvals(z_c,:);   % scalars z1,z2,etc.
        EV_z  = EV(:,z_c);           % (a',1)

        ReturnMatrix_z=ReturnMatrix(:,:,z_c); % (a',a) a' on coarse grid

        entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*EV_z; %(a',a)

        % No need to compute the maximum value in first layer
        [~,maxindex]=max(entireRHS_z,[],1);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is 1-by-n_a
        % % aprimeindexes has size [n2long,n_a]
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a
        aprime_ii = aprime_grid(aprimeindexes);
        % ReturnMatrix_ii_z is (N_d,n2long,n_a)
        ReturnMatrix_ii_z = CreateReturnFnMatrix_GI_lowmem(ReturnFn,d_gridvals,aprime_ii,a_gridvals,z_val,ParamCell,n_z);
        % EV_z_interp is (n2long,n_a)
        EV_z_interp = interp1(a_grid,EV_z,aprime_ii,'linear','extrap');
       
        % entireRHS_ii_z is (N_d,n2long,n_a)
        entireRHS_ii_z = ReturnMatrix_ii_z + DiscountFactorParamsVec*reshape(EV_z_interp,[1,n2long,n_a]);
        [Vtempii,maxindexL2]=max(reshape(entireRHS_ii_z,[n_d*n2long,n_a]),[],1);
        VKron(:,z_c) = Vtempii;
        [maxindexL2_d, maxindexL2_a] = ind2sub([n_d,n2long],maxindexL2);
        Policy(1,:,z_c) = maxindexL2_d; % d
        Policy(2,:,z_c) = midpoint; % midpoint
        Policy(3,:,z_c) = maxindexL2_a; % aprimeL2ind from 1 to n2long=3+2*n2short
        lin_ind         = sub2ind(size(ReturnMatrix_ii_z), maxindexL2_d, maxindexL2_a, 1:N_a);
        Ftemp(:, z_c)   = ReturnMatrix_ii_z(lin_ind);
    end % end z

    %---------------------------------------------------------------------%
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


    %% Compute distance and update
    VKrondist=VKron(:)-VKronold(:);
    currdist=max(abs(VKrondist));

    if verbose==1
        disp(currdist)
    end

    VKronold = VKron;
    tempcounter=tempcounter+1;

end %end while

% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:)=Policy(2,:,:)-adjust; % lower grid point
Policy(3,:,:)=adjust.*Policy(3,:,:)+(1-adjust).*(Policy(3,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

%Policy=Policy(1,:,:)+N_d*(Policy(2,:,:)-1)+N_d*N_a*(Policy(3,:,:)-1);



end %end function

%-------------------------------------------------------------------------%

function Fmatrix = CreateReturnFnMatrix_GI(ReturnFn,d_gridvals,aprime_grid,a_grid,z_gridvals,ParamCell,n_z)
% Assumption: z_gridvals has size [1,length(n_z)]


%ReturnMatrix_fine = arrayfun(ReturnFn,aprime_grid,a_grid,z_grid,ParamCell{:});

l_z = length(n_z);
if l_z>3
    error('ERROR: not allow for more than 3 of z variable (you have length(n_z)>3)')
end

if l_z==1
    Fmatrix=arrayfun(ReturnFn, d_gridvals,aprime_grid, a_grid, shiftdim(z_gridvals(:,1),-3), ParamCell{:});
elseif l_z==2
    Fmatrix=arrayfun(ReturnFn, d_gridvals,aprime_grid, a_grid, shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), ParamCell{:});
elseif l_z==3
    Fmatrix=arrayfun(ReturnFn, d_gridvals,aprime_grid, a_grid, shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), ParamCell{:});
end

end % end subfunction

function Fmatrix = CreateReturnFnMatrix_GI_lowmem(ReturnFn,d_gridvals,aprime_grid,a_grid,z_gridvals,ParamCell,n_z)
% Assumption: z_gridvals has size [1,length(n_z)]

l_z = length(n_z);
if l_z>3
    error('ERROR: not allow for more than 3 of z variable (you have length(n_z)>3)')
end

if l_z==1
    Fmatrix=arrayfun(ReturnFn,d_gridvals,shiftdim(aprime_grid,-1),a_grid, z_gridvals(1), ParamCell{:});
elseif l_z==2
    Fmatrix=arrayfun(ReturnFn,d_gridvals,shiftdim(aprime_grid,-1),a_grid, z_gridvals(1), z_gridvals(2), ParamCell{:});
elseif l_z==3
    Fmatrix=arrayfun(ReturnFn,d_gridvals,shiftdim(aprime_grid,-1),a_grid, z_gridvals(1), z_gridvals(2), z_gridvals(3), ParamCell{:});
end

end % end subfunction