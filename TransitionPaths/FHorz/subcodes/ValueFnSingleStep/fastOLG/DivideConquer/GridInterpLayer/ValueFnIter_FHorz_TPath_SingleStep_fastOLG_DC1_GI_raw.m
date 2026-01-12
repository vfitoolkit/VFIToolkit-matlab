function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_GI_raw(V,n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z), rather than standard (a,z,j)
% V is (a,j)-by-z
% pi_z_J is (j,z',z) for fastOLG
% z_gridvals_J is (j,N_z,l_z) for fastOLG

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

% fastOLG, so a-j-z
Policy=zeros(3,N_a,N_j,N_z,'gpuArray'); % first dim indexes the optimal choice for d and aprime

z_gridvals_J=shiftdim(z_gridvals_J,-3); % [1,1,1,N_j,N_z,l_z]

%% Grid interpolation

% Preallocate
if vfoptions.lowmemory==0
    midpoints_jj=zeros(N_d,1,N_a,N_j,N_z,'gpuArray');
elseif vfoptions.lowmemory==1
    midpoints_jj=zeros(N_d,1,N_a,N_j,'gpuArray');
end

% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

% n-Monotonicity
% vfoptions.level1n=5;
level1ii=round(linspace(1,n_a,vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

jind=shiftdim(gpuArray(0:1:N_j-1),-2);
zind=shiftdim(gpuArray(0:1:N_z-1),-3);
aBind=gpuArray(0:1:N_a-1);
jBind=shiftdim(gpuArray(0:1:N_j-1),-1);
zBind=shiftdim(gpuArray(0:1:N_z-1),-2);

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_z (note: N_aprime is just equal to N_a)

DiscountFactorParamsVec=CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec,2);
DiscountFactorParamsVec=shiftdim(DiscountFactorParamsVec,-2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

if vfoptions.EVpre==0
    EVpre=zeros(N_a,1,N_j,N_z);
    EVpre(:,1,1:N_j-1,:)=reshape(V(N_a+1:end,:),[N_a,1,N_j-1,N_z]); % I use zeros in j=N_j so that can just use pi_z_J to create expectations
    EV=EVpre.*shiftdim(pi_z_J,-2);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a
elseif vfoptions.EVpre==1
    % This is used for 'Matched Expecations Path'
    EV=reshape(V,[N_a,1,N_j,N_z]).*shiftdim(pi_z_J,-2); % input V is already of size [N_a,N_j,N_z] and we want to use the whole thing
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a
end

% Interpolate EV over aprime_grid
EVinterp=interp1(a_grid,EV,aprime_grid);

DiscountedEV=DiscountFactorParamsVec.*EV;
DiscountedEV=repelem(shiftdim(DiscountedEV,-1),N_d,1,1,1); % [d,aprime,1,j,z]

DiscountedEVinterp=DiscountFactorParamsVec.*EVinterp;
DiscountedEVinterp=repelem(shiftdim(DiscountedEVinterp,-1),N_d,1,1,1); % [d,aprime,1,j,z]

if vfoptions.lowmemory==0
    
    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_Par2(ReturnFn, n_d, n_z, N_j, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J, ReturnFnParamsAgeMatrix,1);

    entireRHS_ii=ReturnMatrix_ii+DiscountedEV; % (d,aprime,a and j,z), autofills a for expectation term

    % First, we want aprime conditional on (d,1,a,j)
    [~,maxindex1]=max(entireRHS_ii,[],2);

    % Just keep the 'midpoint' vesion of maxindex1 [as GI]
    midpoints_jj(:,1,level1ii,:,:)=maxindex1;

    % Attempt for improved version
    maxgap=max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(:,1,ii,:,:)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1-by-1-by-N_j-by-n_z
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-N_j-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_Par2(ReturnFn, n_d, n_z, N_j, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J, ReturnFnParamsAgeMatrix,3);
            daprimejz=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*N_a*jind+N_d*N_a*N_j*zind;
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV(daprimejz(:)),[N_d,(maxgap(ii)+1),1,N_j,N_z]); % note: 3rd dim autofills to level1iidiff(ii)
            [~,maxindex]=max(entireRHS_ii,[],2);
            midpoints_jj(:,1,curraindex,:,:)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(:,1,ii,:,:);
            midpoints_jj(:,1,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1);
        end
    end

    % Turn this into the 'midpoint'
    midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is n_d-by-1-by-n_a-by-N_j-by-n_z
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
    % aprime possibilities are n_d-by-n2long-by-n_a-by-N_j-by-n_z
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_Par2(ReturnFn,n_d, n_z, N_j,d_gridvals,aprime_grid(aprimeindexes),a_grid, z_gridvals_J, ReturnFnParamsAgeMatrix,2);
    daprimejz=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*n2aprime*jind+N_d*n2aprime*N_j*zind;
    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(daprimejz(:)),[N_d*n2long,N_a,N_j,N_z]);
    [V,maxindexL2]=max(entireRHS_ii,[],1);
    V=reshape(V,[N_a*N_j,N_z]);
    d_ind=rem(maxindexL2-1,N_d)+1;
    allind=d_ind+N_d*aBind+N_d*N_a*jBind+N_d*N_a*N_j*zBind; % midpoint is n_d-by-1-by-n_a-by-N_j-by-n_z
    Policy(1,:,:,:)=d_ind; % d
    Policy(2,:,:,:)=shiftdim(squeeze(midpoints_jj(allind)),-1); % midpoint
    Policy(3,:,:,:)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind

elseif vfoptions.lowmemory==1

    special_n_z=ones(1,length(n_z));
    V=zeros(N_a*N_j,N_z,'gpuArray'); % V is over (a,j,z)

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,1,:,z_c,:); % z_gridvals_J has shape (1,1,1,N_j,N_z,l_z) for fastOLG
        DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);
        DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,z_c);

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_Par2(ReturnFn, n_d, special_n_z, N_j, d_gridvals, a_grid, a_grid(level1ii), z_vals, ReturnFnParamsAgeMatrix,1);

        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z; % (d,aprime,a and j,z), autofills j for expectation term

        % First, we want aprime conditional on (d,1,a,j)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Just keep the 'midpoint' vesion of maxindex1 [as GI]
        midpoints_jj(:,1,level1ii,:)=maxindex1;

        % Attempt for improved version
        maxgap=max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(:,1,ii,:)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-1-by-N_j
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-N_j
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_Par2(ReturnFn, n_d, special_n_z, N_j, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_vals, ReturnFnParamsAgeMatrix,3);
                daprimej=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*N_a*jind;
                entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV_z(daprimej(:)),[N_d,(maxgap(ii)+1),1,N_j]); % note: 3rd dim autofills to level1iidiff(ii)
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoints_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:);
                midpoints_jj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-by-1-by-n_a-by-N_j
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-N_j
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_Par2(ReturnFn,n_d, special_n_z, N_j,d_gridvals,aprime_grid(aprimeindexes),a_grid, z_vals, ReturnFnParamsAgeMatrix,2);
        daprimej=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*n2aprime*jind;
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp_z(daprimej(:)),[N_d*n2long,N_a,N_j]);
        [Vtemp,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,z_c)=reshape(Vtemp,[N_a*N_j,1]);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aBind+N_d*N_a*jBind; % midpoint is n_d-by-1-by-n_a-by-N_j
        Policy(1,:,:,z_c)=d_ind; % d
        Policy(2,:,:,z_c)=shiftdim(squeeze(midpoints_jj(allind)),-1); % midpoint
        Policy(3,:,:,z_c)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
    end
end



%% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:,:)=Policy(2,:,:,:)-adjust; % lower grid point
Policy(3,:,:,:)=adjust.*Policy(3,:,:,:)+(1-adjust).*(Policy(3,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

% Leave the first dimension as is
% Policy=squeeze(Policy(1,:,:,:)+N_d*(Policy(2,:,:,:)-1)+N_d*N_a*(Policy(3,:,:,:)-1));

%% fastOLG with z, so need to output to take certain shapes
% V=reshape(V,[N_a*N_j,N_z]);
% Policy=reshape(Policy,[N_a,N_j,N_z]);
% Note that in fastOLG, we do not separate d from aprime in Policy

end