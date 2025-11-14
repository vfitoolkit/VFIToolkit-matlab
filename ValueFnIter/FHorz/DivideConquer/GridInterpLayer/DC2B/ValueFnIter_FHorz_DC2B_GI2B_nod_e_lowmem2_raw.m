function [V,Policy]=ValueFnIter_FHorz_DC2B_GI2B_nod_e_lowmem2_raw(n_a,n_z,n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% divide-and-conquer in the first endo state

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(3,N_a,N_z,N_e,N_j,'gpuArray'); % first dim is (a1prime midpoint,a2prime,a1prime L2)

%%
special_n_z=ones(1,length(n_z));
special_n_e=ones(1,length(n_e));

n_a1=n_a(1);
n_a2=n_a(2:end);
N_a1=n_a1;
N_a2=n_a2;
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% preallocate
midpoints_jj=zeros(1,N_a2,N_a1,N_a2,'gpuArray');

% n-Monotonicity
% vfoptions.level1n=7;
level1ii=round(linspace(1,N_a1,vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);
% aprime_grid=[a1prime_grid; a2_grid];

pi_e_J=shiftdim(pi_e_J,-2); % Move to thrid dimension

% precompute
a2ind=gpuArray(0:1:N_a2-1); % already includes -1
zind=shiftdim(gpuArray(0:1:N_z-1),-1); % already includes -1
zBind=shiftdim(gpuArray(0:1:N_z-1),-3); % already includes -1

a12ind=repmat(gpuArray(0:1:N_a1-1),1,N_a2)+N_a1*repelem(gpuArray(0:1:N_a2-1),1,N_a1);

%% j=N_j
% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    for z_c=1:N_z
        z_val=z_gridvals_J(z_c,:,N_j);
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2e(ReturnFn, special_n_z, special_n_e, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_val, e_val, ReturnFnParamsVec,1);

            %Calc the max and it's index
            [~,maxindex1]=max(ReturnMatrix_ii,[],1);

            % Just keep the 'midpoint' vesion of maxindex1 [as GI]
            midpoints_jj(1,:,level1ii,:)=maxindex1;

            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(1,:,2:end,:)-maxindex1(1,:,1:end-1,:),[],4),[],2));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,:,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-n_a2-by-1-by-n_a2
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2e(ReturnFn, special_n_z, special_n_e, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, e_val, ReturnFnParamsVec,3);
                    [~,maxindex]=max(ReturnMatrix_ii,[],1);
                    midpoints_jj(1,:,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(1,:,ii,:);
                    midpoints_jj(1,:,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
                end
            end

            % Turn this into the 'midpoint'
            midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is 1-by-n_a2-by-n_a1-by-n_a2
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
            % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2e(ReturnFn,special_n_z,special_n_e,a1prime_grid(a1primeindexes),a2_grid,a1_grid,a2_grid,z_val, e_val, ReturnFnParamsVec,2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);
            V(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(1,:,z_c,e_c,N_j)=midpoints_jj(maxindexL2a2+N_a2*a12ind); % a1prime midpoint
            Policy(2,:,z_c,e_c,N_j)=maxindexL2a2; % a2prime
            Policy(3,:,z_c,e_c,N_j)=maxindexL2a1; % a1primeL2ind
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=sum(reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]).*pi_e_J(1,1,:,N_j),3); % Using V_Jplus1

    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-1);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_a1,N_a2,1,1,N_z]);
    % Interpolate EV over aprime_grid
    DiscountedEVinterp=interp1(a1_grid,DiscountedEV,a1prime_grid);

    for z_c=1:N_z
        z_val=z_gridvals_J(z_c,:,N_j);
        DiscountedEV_z=DiscountedEV(:,:,1,1,z_c);
        DiscountedEVinterp_z=DiscountedEVinterp(:,:,1,1,z_c);
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2e(ReturnFn, special_n_z, special_n_e, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_val, e_val, ReturnFnParamsVec,1);

            entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z; % autofill e

            %Calc the max and it's index
            [~,maxindex1]=max(entireRHS_ii,[],1);

            % Just keep the 'midpoint' vesion of maxindex1 [as GI]
            midpoints_jj(1,:,level1ii,:)=maxindex1;

            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(1,:,2:end,:)-maxindex1(1,:,1:end-1,:),[],4),[],2));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,:,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-n_a2-by-1-by-n_a2
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2e(ReturnFn, special_n_z, special_n_e, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, e_val, ReturnFnParamsVec,3);
                    aprime=aprimeindexes+N_a1*a2ind;
                    entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z(reshape(aprime,[(maxgap(ii)+1),N_a2,1,N_a2])); % autoexpand level1iidiff(ii) in 3rd-dim
                    [~,maxindex]=max(entireRHS_ii,[],1);
                    midpoints_jj(1,:,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(1,:,ii,:);
                    midpoints_jj(1,:,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
                end
            end

            % Turn this into the 'midpoint'
            midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is 1-by-n_a2-by-n_a1-by-n_a2
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
            % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2e(ReturnFn,special_n_z,special_n_e,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid,z_val, e_val, ReturnFnParamsVec,2);
            aprime=a1primeindexes+N_a1fine*a2ind;
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp_z(aprime),[n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);
            V(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(1,:,z_c,e_c,N_j)=midpoints_jj(maxindexL2a2+N_a2*a12ind); % a1prime midpoint
            Policy(2,:,z_c,e_c,N_j)=maxindexL2a2; % a2prime
            Policy(3,:,z_c,e_c,N_j)=maxindexL2a1; % a1primeL2ind
        end
    end
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

    EV=sum(V(:,:,:,jj+1).*pi_e_J(1,1,:,jj),3);

    EV=EV.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_a1,N_a2,1,1,N_z]);
    % Interpolate EV over aprime_grid
    DiscountedEVinterp=interp1(a1_grid,DiscountedEV,a1prime_grid);

    for z_c=1:N_z
        z_val=z_gridvals_J(z_c,:,jj);
        DiscountedEV_z=DiscountedEV(:,:,1,1,z_c);
        DiscountedEVinterp_z=DiscountedEVinterp(:,:,1,1,z_c);

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);

            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2e(ReturnFn, special_n_z, special_n_e, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_val, e_val, ReturnFnParamsVec,1);

            entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z;

            %Calc the max and it's index
            [~,maxindex1]=max(entireRHS_ii,[],1);

            % Just keep the 'midpoint' vesion of maxindex1 [as GI]
            midpoints_jj(1,:,level1ii,:)=maxindex1;

            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(1,:,2:end,:)-maxindex1(1,:,1:end-1,:),[],4),[],2));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,:,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-n_a2-by-1-by-n_a2
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2e(ReturnFn, special_n_z, special_n_e, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, e_val, ReturnFnParamsVec,3);
                    aprime=aprimeindexes+N_a1*a2ind;
                    entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z(reshape(aprime,[(maxgap(ii)+1),N_a2,1,N_a2])); % autoexpand level1iidiff(ii) in 3rd-dim
                    [~,maxindex]=max(entireRHS_ii,[],1);
                    midpoints_jj(1,:,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(1,:,ii,:);
                    midpoints_jj(1,:,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
                end
            end

            % Turn this into the 'midpoint'
            midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is 1-by-n_a2-by-n_a1-by-n_a2
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
            % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2e(ReturnFn,special_n_z,special_n_e,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid,z_val, e_val, ReturnFnParamsVec,2);
            aprime=a1primeindexes+N_a1fine*a2ind;
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp_z(aprime),[n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);
            V(:,z_c,e_c,jj)=shiftdim(Vtempii,1);
            Policy(1,:,z_c,e_c,jj)=midpoints_jj(maxindexL2a2+N_a2*a12ind); % a1prime midpoint
            Policy(2,:,z_c,e_c,jj)=maxindexL2a2; % a2prime
            Policy(3,:,z_c,e_c,jj)=maxindexL2a1; % a1primeL2ind
        end
    end
end



%% Currently Policy(1,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(1,:,:,:,:)=Policy(1,:,:,:,:)-adjust; % lower grid point
Policy(3,:,:,:,:)=adjust.*Policy(3,:,:,:,:)+(1-adjust).*(Policy(3,:,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=Policy(1,:,:,:,:)+N_a1*(Policy(2,:,:,:,:)-1)+N_a1*N_a2*(Policy(3,:,:,:,:)-1);



end