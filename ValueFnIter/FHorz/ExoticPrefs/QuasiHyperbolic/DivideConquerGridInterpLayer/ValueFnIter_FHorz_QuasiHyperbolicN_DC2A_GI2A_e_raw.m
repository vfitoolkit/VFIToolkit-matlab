function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicN_DC2A_GI2A_e_raw(n_d,n_a,n_z,n_e, N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive quasi-hyperbolic discounting variant of ValueFnIter_FHorz_DC2A_GI2A_e_raw.
% divide-and-conquer in the first endo state (a2 enumerated in full), plus grid interpolation layer.
% Has d. Has z (z-expectation via pi_z_J). Has e (iid, e-expectation via pi_e_J). GPU (parallel==2 only).
% lowmemory: =0 vectorize, =1 loop over e, =2 loop over e and z
%
% Naive: Valt_j   = max u + beta*E[V_{j+1}]         (used as EVsource)
%        Vtilde_j = max u + beta_0*beta*E[V_{j+1}]  (agent's choice)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

Valt=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(4,N_a,N_z,N_e,N_j,'gpuArray'); % first dim is (d,a1prime midpoint,a2prime,a1prime L2)
PolicyL2flag=2*ones(1,N_a,N_z,N_e,N_j,'gpuArray'); % L2 flag: 1=all to lower, 2=usual, 3=all to upper
Policyalt=zeros(4,N_a,N_z,N_e,N_j,'gpuArray'); % exponential discounter optimal choice
PolicyL2flagalt=2*ones(1,N_a,N_z,N_e,N_j,'gpuArray');

%%
n_a1=n_a(1);
n_a2=n_a(2:end);
N_a1=n_a1;
N_a2=n_a2;
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% n-Monotonicity
level1ii=round(linspace(1,n_a(1),vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);
% aprime_grid=[a1prime_grid; a2_grid];

pi_e_J=shiftdim(pi_e_J,-2); % Move to third dimension

% precompute
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-1); % already includes -1
a2Bind=shiftdim(gpuArray(0:1:N_a2-1),-1); % already includes -1
a12ind=repmat(gpuArray(0:1:N_a1-1),1,N_a2)+N_a1*repelem(gpuArray(0:1:N_a2-1),1,N_a1);
if vfoptions.lowmemory==0
    midpoints_jj=zeros(N_d,1,N_a2,N_a1,N_a2,N_z,N_e,'gpuArray');
    zind=shiftdim(gpuArray(0:1:N_z-1),-1); % already includes -1
    eind=shiftdim(gpuArray(0:1:N_e-1),-2); % already includes -1
    zBind=shiftdim(gpuArray(0:1:N_z-1),-4); % already includes -1
elseif vfoptions.lowmemory==1
    midpoints_jj=zeros(N_d,1,N_a2,N_a1,N_a2,N_z,'gpuArray');
    zind=shiftdim(gpuArray(0:1:N_z-1),-1); % already includes -1
    zBind=shiftdim(gpuArray(0:1:N_z-1),-4); % already includes -1
    special_n_e=ones(1,length(n_e),'gpuArray');
elseif vfoptions.lowmemory==2
    midpoints_jj=zeros(N_d,1,N_a2,N_a1,N_a2,'gpuArray');
    special_n_z=ones(1,length(n_z),'gpuArray');
    special_n_e=ones(1,length(n_e),'gpuArray');
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

        % First, we want a1prime conditional on (d,1,a2prime,a,z,e)
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);

        % Just keep the 'midpoint' version of maxindex1 [as GI]
        midpoints_jj(:,1,:,level1ii,:,:,:)=maxindex1;

        % Attempt for improved version
        maxgap=squeeze(max(max(max(max(max(maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z-by-n_e
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-n_a2-by-1-by-n_a2-by-n_z-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                [~,maxindex]=max(ReturnMatrix_ii,[],2);
                midpoints_jj(:,1,:,curraindex,:,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                midpoints_jj(:,1,:,curraindex,:,:,:)=repelem(loweredge,1,1,1,length(curraindex),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2-by-n_z-by-n_e
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,n_z,n_e,d_gridvals,a1prime_grid(a1primeindexes),a2_grid,a1_grid,a2_grid, z_gridvals_J(:,:,N_j),e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        maxindexL2d=rem(maxindexL2-1,N_d)+1;
        maxindexL2a=ceil(maxindexL2/N_d);
        maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2a/n2long);
        Valt(:,:,:,N_j)=shiftdim(Vtempii,1);
        Policy(1,:,:,:,N_j)=maxindexL2d; % d
        Policy(2,:,:,:,N_j)=midpoints_jj(maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind+N_d*N_a2*N_a*zind+N_d*N_a2*N_a*N_z*eind); % a1prime midpoint
        Policy(3,:,:,:,N_j)=maxindexL2a2; % a2prime
        Policy(4,:,:,:,N_j)=maxindexL2a1; % a1primeL2ind

        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        linidx_lower = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind + N_d*n2long*N_a2*N_a*N_z*eind;
        linidx_upper = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind + N_d*n2long*N_a2*N_a*N_z*eind;
        isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        PolicyL2flag(1,:,:,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);

            % First, we want a1prime conditional on (d,1,a2prime,a,z)
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoints_jj(:,1,:,level1ii,:,:)=maxindex1;

            % Attempt for improved version
            maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-n_a2-by-1-by-n_a2-by-n_z
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                    [~,maxindex]=max(ReturnMatrix_ii,[],2);
                    midpoints_jj(:,1,:,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    midpoints_jj(:,1,:,curraindex,:,:)=repelem(loweredge,1,1,1,length(curraindex),1);
                end
            end

            % Turn this into the 'midpoint'
            midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2-by-n_z
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,n_z,special_n_e,d_gridvals,a1prime_grid(a1primeindexes),a2_grid,a1_grid,a2_grid, z_gridvals_J(:,:,N_j),e_val, ReturnFnParamsVec,2,0);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            maxindexL2d=rem(maxindexL2-1,N_d)+1;
            maxindexL2a=ceil(maxindexL2/N_d);
            maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2a/n2long);
            Valt(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(1,:,:,e_c,N_j)=maxindexL2d; % d
            Policy(2,:,:,e_c,N_j)=midpoints_jj(maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind+N_d*N_a2*N_a*zind); % a1prime midpoint
            Policy(3,:,:,e_c,N_j)=maxindexL2a2; % a2prime
            Policy(4,:,:,e_c,N_j)=maxindexL2a1; % a1primeL2ind

            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            linidx_lower = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind;
            linidx_upper = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind;
            isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            PolicyL2flag(1,:,:,e_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                % n-Monotonicity
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_val,e_val, ReturnFnParamsVec,1,0);

                % First, we want a1prime conditional on (d,1,a2prime,a)
                [~,maxindex1]=max(ReturnMatrix_ii,[],2);

                % Just keep the 'midpoint' version of maxindex1 [as GI]
                midpoints_jj(:,1,:,level1ii,:)=maxindex1;

                % Attempt for improved version
                maxgap=squeeze(max(max(max(maxindex1(:,1,:,2:end,:)-maxindex1(:,1,:,1:end-1,:),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d-by-maxgap(ii)+1-by-n_a2-by-1-by-n_a2
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val,e_val, ReturnFnParamsVec,3,0);
                        [~,maxindex]=max(ReturnMatrix_ii,[],2);
                        midpoints_jj(:,1,:,curraindex,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:);
                        midpoints_jj(:,1,:,curraindex,:)=repelem(loweredge,1,1,1,length(curraindex),1);
                    end
                end

                % Turn this into the 'midpoint'
                midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2
                a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,special_n_z,special_n_e,d_gridvals,a1prime_grid(a1primeindexes),a2_grid,a1_grid,a2_grid, z_val,e_val, ReturnFnParamsVec,2,0);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                maxindexL2d=rem(maxindexL2-1,N_d)+1;
                maxindexL2a=ceil(maxindexL2/N_d);
                maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
                maxindexL2a2=ceil(maxindexL2a/n2long);
                Valt(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                Policy(1,:,z_c,e_c,N_j)=maxindexL2d; % d
                Policy(2,:,z_c,e_c,N_j)=midpoints_jj(maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind); % a1prime midpoint
                Policy(3,:,z_c,e_c,N_j)=maxindexL2a2; % a2prime
                Policy(4,:,z_c,e_c,N_j)=maxindexL2a1; % a1primeL2ind

                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                linidx_lower = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind;
                linidx_upper = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind;
                isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                PolicyL2flag(1,:,z_c,e_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
            end
        end
    end


    Vtilde=Valt;
    % terminal period: QH and exponential discounter coincide
    Policyalt(:,:,:,:,N_j)=Policy(:,:,:,:,N_j);
    PolicyL2flagalt(1,:,:,:,N_j)=PolicyL2flag(1,:,:,:,N_j);

else

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    Vtilde=zeros(N_a,N_z,N_e,N_j,'gpuArray');

    EV=sum(reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]).*pi_e_J(1,1,:,N_j+1),3); % Using V_Jplus1

    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-1);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    DiscountedEV=beta*reshape(EV,[N_a1,N_a2,1,1,N_z]); % will autoexand d in 1st-dim
    DiscountedEV_tilde=beta0beta*reshape(EV,[N_a1,N_a2,1,1,N_z]); % will autoexand d in 1st-dim
    % Interpolate EV over aprime_grid
    DiscountedEVinterp=interp1(a1_grid,DiscountedEV,a1prime_grid);
    DiscountedEVinterp_tilde=interp1(a1_grid,DiscountedEV_tilde,a1prime_grid);
    DiscountedEV=shiftdim(DiscountedEV,-1); % will autoexand d in 1st-dim
    DiscountedEV_tilde=shiftdim(DiscountedEV_tilde,-1); % will autoexand d in 1st-dim
    DiscountedEVinterp=shiftdim(DiscountedEVinterp,-1); % will autoexand d in 1st-dim
    DiscountedEVinterp_tilde=shiftdim(DiscountedEVinterp_tilde,-1); % will autoexand d in 1st-dim

    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

        %% Valt (beta)
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV; % autofill e

        % First, we want a1prime conditional on (d,1,a2prime,a,z,e)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Just keep the 'midpoint' version of maxindex1 [as GI]
        midpoints_jj(:,1,:,level1ii,:,:,:)=maxindex1;

        % Attempt for improved version
        maxgap_V=squeeze(max(max(max(max(max(maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap_V(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap_V(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap_V(ii) points
                % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z-by-n_e
                a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                % aprime possibilities are n_d-by-maxgap_V(ii)+1-by-n_a2-by-1-by-n_a2-by-n_z-by-n_e
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                aprimez=a1primeindexes+N_a1*a2Bind+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV(reshape(aprimez,[N_d,(maxgap_V(ii)+1),N_a2,1,N_a2,N_z,N_e])); % autoexpand level1iidiff(ii) into 4th-dim
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoints_jj(:,1,:,curraindex,:,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                midpoints_jj(:,1,:,curraindex,:,:,:)=repelem(loweredge,1,1,1,length(curraindex),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2-by-n_z-by-n_e
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z-by-n_e
        ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,n_z,n_e,d_gridvals,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,N_j),e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
        entireRHS_L2=ReturnMatrix_L2+reshape(DiscountedEVinterp(aprime),[N_d*n2long*N_a2,N_a,N_z,N_e]);
        [Vtempii,maxindexL2alt]=max(entireRHS_L2,[],1);
        maxindexL2dalt=rem(maxindexL2alt-1,N_d)+1;
        maxindexL2aalt=ceil(maxindexL2alt/N_d);
        maxindexL2a1alt=rem(maxindexL2aalt-1,n2long)+1;
        maxindexL2a2alt=ceil(maxindexL2aalt/n2long);
        Valt(:,:,:,N_j)=shiftdim(Vtempii,1);
        Policyalt(1,:,:,:,N_j)=maxindexL2dalt; % d
        Policyalt(2,:,:,:,N_j)=midpoints_jj(maxindexL2dalt+N_d*(maxindexL2a2alt-1)+N_d*N_a2*a12ind+N_d*N_a2*N_a*zind+N_d*N_a2*N_a*N_z*eind); % a1prime midpoint
        Policyalt(3,:,:,:,N_j)=maxindexL2a2alt; % a2prime
        Policyalt(4,:,:,:,N_j)=maxindexL2a1alt; % a1primeL2ind

        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        linidx_loweralt = maxindexL2dalt                  + N_d*n2long*(maxindexL2a2alt-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind + N_d*n2long*N_a2*N_a*N_z*eind;
        linidx_upperalt = maxindexL2dalt + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2alt-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind + N_d*n2long*N_a2*N_a*N_z*eind;
        isInfLoweralt = (ReturnMatrix_L2(linidx_loweralt) == -Inf);
        isInfUpperalt = (ReturnMatrix_L2(linidx_upperalt) == -Inf);
        inLowerStrictalt = (maxindexL2a1alt >= 2)         & (maxindexL2a1alt <= n2short+1);
        inUpperStrictalt = (maxindexL2a1alt >= n2short+3) & (maxindexL2a1alt <= n2long-1);
        PolicyL2flagalt(1,:,:,:,N_j) = 2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);
        %% Vtilde (beta0*beta)
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_tilde; % autofill e

        % First, we want a1prime conditional on (d,1,a2prime,a,z,e)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Just keep the 'midpoint' version of maxindex1 [as GI]
        midpoints_jj(:,1,:,level1ii,:,:,:)=maxindex1;

        % Attempt for improved version
        maxgap=squeeze(max(max(max(max(max(maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z-by-n_e
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-n_a2-by-1-by-n_a2-by-n_z-by-n_e
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                aprimez=a1primeindexes+N_a1*a2Bind+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_tilde(reshape(aprimez,[N_d,(maxgap(ii)+1),N_a2,1,N_a2,N_z,N_e])); % autoexpand level1iidiff(ii) into 4th-dim
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoints_jj(:,1,:,curraindex,:,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                midpoints_jj(:,1,:,curraindex,:,:,:)=repelem(loweredge,1,1,1,length(curraindex),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2-by-n_z-by-n_e
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z-by-n_e
        ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,n_z,n_e,d_gridvals,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,N_j),e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
        entireRHS_L2=ReturnMatrix_L2+reshape(DiscountedEVinterp_tilde(aprime),[N_d*n2long*N_a2,N_a,N_z,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
        maxindexL2d=rem(maxindexL2-1,N_d)+1;
        maxindexL2a=ceil(maxindexL2/N_d);
        maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2a/n2long);
        Vtilde(:,:,:,N_j)=shiftdim(Vtempii,1);
        Policy(1,:,:,:,N_j)=maxindexL2d; % d
        Policy(2,:,:,:,N_j)=midpoints_jj(maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind+N_d*N_a2*N_a*zind+N_d*N_a2*N_a*N_z*eind); % a1prime midpoint
        Policy(3,:,:,:,N_j)=maxindexL2a2; % a2prime
        Policy(4,:,:,:,N_j)=maxindexL2a1; % a1primeL2ind

        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        linidx_lower = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind + N_d*n2long*N_a2*N_a*N_z*eind;
        linidx_upper = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind + N_d*n2long*N_a2*N_a*N_z*eind;
        isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        PolicyL2flag(1,:,:,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);

            %% Valt (beta)
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV; % autofill e

            % First, we want a1prime conditional on (d,1,a2prime,a,z)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoints_jj(:,1,:,level1ii,:,:)=maxindex1;

            % Attempt for improved version
            maxgap_V=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap_V(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap_V(ii) points
                    % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
                    a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                    % aprime possibilities are n_d-by-maxgap_V(ii)+1-by-n_a2-by-1-by-n_a2-by-n_z
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                    aprimez=a1primeindexes+N_a1*a2Bind+N_a*zBind;
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV(reshape(aprimez,[N_d,(maxgap_V(ii)+1),N_a2,1,N_a2,N_z])); % autoexpand level1iidiff(ii) into 4th-dim
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoints_jj(:,1,:,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    midpoints_jj(:,1,:,curraindex,:,:)=repelem(loweredge,1,1,1,length(curraindex),1);
                end
            end

            % Turn this into the 'midpoint'
            midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2-by-n_z
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z
            ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,n_z,special_n_e,d_gridvals,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,N_j),e_val, ReturnFnParamsVec,2,0);
            aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
            entireRHS_L2=ReturnMatrix_L2+reshape(DiscountedEVinterp(aprime),[N_d*n2long*N_a2,N_a,N_z]);
            [Vtempii,maxindexL2alt]=max(entireRHS_L2,[],1);
            maxindexL2dalt=rem(maxindexL2alt-1,N_d)+1;
            maxindexL2aalt=ceil(maxindexL2alt/N_d);
            maxindexL2a1alt=rem(maxindexL2aalt-1,n2long)+1;
            maxindexL2a2alt=ceil(maxindexL2aalt/n2long);
            Valt(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            Policyalt(1,:,:,e_c,N_j)=maxindexL2dalt; % d
            Policyalt(2,:,:,e_c,N_j)=midpoints_jj(maxindexL2dalt+N_d*(maxindexL2a2alt-1)+N_d*N_a2*a12ind+N_d*N_a2*N_a*zind); % a1prime midpoint
            Policyalt(3,:,:,e_c,N_j)=maxindexL2a2alt; % a2prime
            Policyalt(4,:,:,e_c,N_j)=maxindexL2a1alt; % a1primeL2ind

            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            linidx_loweralt = maxindexL2dalt                  + N_d*n2long*(maxindexL2a2alt-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind;
            linidx_upperalt = maxindexL2dalt + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2alt-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind;
            isInfLoweralt = (ReturnMatrix_L2(linidx_loweralt) == -Inf);
            isInfUpperalt = (ReturnMatrix_L2(linidx_upperalt) == -Inf);
            inLowerStrictalt = (maxindexL2a1alt >= 2)         & (maxindexL2a1alt <= n2short+1);
            inUpperStrictalt = (maxindexL2a1alt >= n2short+3) & (maxindexL2a1alt <= n2long-1);
            PolicyL2flagalt(1,:,:,e_c,N_j) = 2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);
            %% Vtilde (beta0*beta)
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV_tilde; % autofill e

            % First, we want a1prime conditional on (d,1,a2prime,a,z)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoints_jj(:,1,:,level1ii,:,:)=maxindex1;

            % Attempt for improved version
            maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-n_a2-by-1-by-n_a2-by-n_z
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                    aprimez=a1primeindexes+N_a1*a2Bind+N_a*zBind;
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_tilde(reshape(aprimez,[N_d,(maxgap(ii)+1),N_a2,1,N_a2,N_z])); % autoexpand level1iidiff(ii) into 4th-dim
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoints_jj(:,1,:,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    midpoints_jj(:,1,:,curraindex,:,:)=repelem(loweredge,1,1,1,length(curraindex),1);
                end
            end

            % Turn this into the 'midpoint'
            midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2-by-n_z
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z
            ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,n_z,special_n_e,d_gridvals,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,N_j),e_val, ReturnFnParamsVec,2,0);
            aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
            entireRHS_L2=ReturnMatrix_L2+reshape(DiscountedEVinterp_tilde(aprime),[N_d*n2long*N_a2,N_a,N_z]);
            [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
            maxindexL2d=rem(maxindexL2-1,N_d)+1;
            maxindexL2a=ceil(maxindexL2/N_d);
            maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2a/n2long);
            Vtilde(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(1,:,:,e_c,N_j)=maxindexL2d; % d
            Policy(2,:,:,e_c,N_j)=midpoints_jj(maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind+N_d*N_a2*N_a*zind); % a1prime midpoint
            Policy(3,:,:,e_c,N_j)=maxindexL2a2; % a2prime
            Policy(4,:,:,e_c,N_j)=maxindexL2a1; % a1primeL2ind

            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            linidx_lower = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind;
            linidx_upper = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind;
            isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            PolicyL2flag(1,:,:,e_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            DiscountedEV_z=DiscountedEV(:,:,:,:,:,z_c);
            DiscountedEV_tilde_z=DiscountedEV_tilde(:,:,:,:,:,z_c);
            DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,:,z_c);
            DiscountedEVinterp_tilde_z=DiscountedEVinterp_tilde(:,:,:,:,:,z_c);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                % n-Monotonicity
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_val,e_val, ReturnFnParamsVec,1,0);

                %% Valt (beta)
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z; % autofill e

                % First, we want a1prime conditional on (d,1,a2prime,a)
                [~,maxindex1]=max(entireRHS_ii,[],2);

                % Just keep the 'midpoint' version of maxindex1 [as GI]
                midpoints_jj(:,1,:,level1ii,:)=maxindex1;

                % Attempt for improved version
                maxgap_V=squeeze(max(max(max(maxindex1(:,1,:,2:end,:)-maxindex1(:,1,:,1:end-1,:),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap_V(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:),N_a1-maxgap_V(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap_V(ii) points
                        % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
                        a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                        % aprime possibilities are n_d-by-maxgap_V(ii)+1-by-n_a2-by-1-by-n_a2
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val,e_val, ReturnFnParamsVec,3,0);
                        aprime=a1primeindexes+N_a1*a2Bind;
                        entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_z(reshape(aprime,[N_d,(maxgap_V(ii)+1),N_a2,1,N_a2])); % autoexpand level1iidiff(ii) into 4th-dim
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoints_jj(:,1,:,curraindex,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:);
                        midpoints_jj(:,1,:,curraindex,:)=repelem(loweredge,1,1,1,length(curraindex),1);
                    end
                end

                % Turn this into the 'midpoint'
                midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2
                a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2
                ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,special_n_z,special_n_e,d_gridvals,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid, z_val,e_val, ReturnFnParamsVec,2,0);
                aprime=a1primeindexes+N_a1fine*a2ind;
                entireRHS_L2=ReturnMatrix_L2+reshape(DiscountedEVinterp_z(aprime),[N_d*n2long*N_a2,N_a]);
                [Vtempii,maxindexL2alt]=max(entireRHS_L2,[],1);
                maxindexL2dalt=rem(maxindexL2alt-1,N_d)+1;
                maxindexL2aalt=ceil(maxindexL2alt/N_d);
                maxindexL2a1alt=rem(maxindexL2aalt-1,n2long)+1;
                maxindexL2a2alt=ceil(maxindexL2aalt/n2long);
                Valt(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                Policyalt(1,:,z_c,e_c,N_j)=maxindexL2dalt; % d
                Policyalt(2,:,z_c,e_c,N_j)=midpoints_jj(maxindexL2dalt+N_d*(maxindexL2a2alt-1)+N_d*N_a2*a12ind); % a1prime midpoint
                Policyalt(3,:,z_c,e_c,N_j)=maxindexL2a2alt; % a2prime
                Policyalt(4,:,z_c,e_c,N_j)=maxindexL2a1alt; % a1primeL2ind

                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                linidx_loweralt = maxindexL2dalt                  + N_d*n2long*(maxindexL2a2alt-1) + N_d*n2long*N_a2*a12ind;
                linidx_upperalt = maxindexL2dalt + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2alt-1) + N_d*n2long*N_a2*a12ind;
                isInfLoweralt = (ReturnMatrix_L2(linidx_loweralt) == -Inf);
                isInfUpperalt = (ReturnMatrix_L2(linidx_upperalt) == -Inf);
                inLowerStrictalt = (maxindexL2a1alt >= 2)         & (maxindexL2a1alt <= n2short+1);
                inUpperStrictalt = (maxindexL2a1alt >= n2short+3) & (maxindexL2a1alt <= n2long-1);
                PolicyL2flagalt(1,:,z_c,e_c,N_j) = 2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);
                %% Vtilde (beta0*beta)
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_tilde_z; % autofill e

                % First, we want a1prime conditional on (d,1,a2prime,a)
                [~,maxindex1]=max(entireRHS_ii,[],2);

                % Just keep the 'midpoint' version of maxindex1 [as GI]
                midpoints_jj(:,1,:,level1ii,:)=maxindex1;

                % Attempt for improved version
                maxgap=squeeze(max(max(max(maxindex1(:,1,:,2:end,:)-maxindex1(:,1,:,1:end-1,:),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d-by-maxgap(ii)+1-by-n_a2-by-1-by-n_a2
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val,e_val, ReturnFnParamsVec,3,0);
                        aprime=a1primeindexes+N_a1*a2Bind;
                        entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_tilde_z(reshape(aprime,[N_d,(maxgap(ii)+1),N_a2,1,N_a2])); % autoexpand level1iidiff(ii) into 4th-dim
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoints_jj(:,1,:,curraindex,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:);
                        midpoints_jj(:,1,:,curraindex,:)=repelem(loweredge,1,1,1,length(curraindex),1);
                    end
                end

                % Turn this into the 'midpoint'
                midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2
                a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2
                ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,special_n_z,special_n_e,d_gridvals,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid, z_val,e_val, ReturnFnParamsVec,2,0);
                aprime=a1primeindexes+N_a1fine*a2ind;
                entireRHS_L2=ReturnMatrix_L2+reshape(DiscountedEVinterp_tilde_z(aprime),[N_d*n2long*N_a2,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                maxindexL2d=rem(maxindexL2-1,N_d)+1;
                maxindexL2a=ceil(maxindexL2/N_d);
                maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
                maxindexL2a2=ceil(maxindexL2a/n2long);
                Vtilde(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                Policy(1,:,z_c,e_c,N_j)=maxindexL2d; % d
                Policy(2,:,z_c,e_c,N_j)=midpoints_jj(maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind); % a1prime midpoint
                Policy(3,:,z_c,e_c,N_j)=maxindexL2a2; % a2prime
                Policy(4,:,z_c,e_c,N_j)=maxindexL2a1; % a1primeL2ind

                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                linidx_lower = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind;
                linidx_upper = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind;
                isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
                inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                PolicyL2flag(1,:,z_c,e_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
            end
        end
    end
end


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    EV=sum(Valt(:,:,:,jj+1).*pi_e_J(1,1,:,jj+1),3); % naive: continuation is the exponential value fn

    EV=EV.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    DiscountedEV=beta*reshape(EV,[N_a1,N_a2,1,1,N_z]); % will autoexand d in 1st-dim
    DiscountedEV_tilde=beta0beta*reshape(EV,[N_a1,N_a2,1,1,N_z]); % will autoexand d in 1st-dim
    % Interpolate EV over aprime_grid
    DiscountedEVinterp=interp1(a1_grid,DiscountedEV,a1prime_grid);
    DiscountedEVinterp_tilde=interp1(a1_grid,DiscountedEV_tilde,a1prime_grid);
    DiscountedEV=shiftdim(DiscountedEV,-1); % will autoexand d in 1st-dim
    DiscountedEV_tilde=shiftdim(DiscountedEV_tilde,-1); % will autoexand d in 1st-dim
    DiscountedEVinterp=shiftdim(DiscountedEVinterp,-1); % will autoexand d in 1st-dim
    DiscountedEVinterp_tilde=shiftdim(DiscountedEVinterp_tilde,-1); % will autoexand d in 1st-dim

    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);

        %% Valt (beta)
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV; % autofill e

        % First, we want a1prime conditional on (d,1,a2prime,a,z,e)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Just keep the 'midpoint' version of maxindex1 [as GI]
        midpoints_jj(:,1,:,level1ii,:,:,:)=maxindex1;

        % Attempt for improved version
        maxgap_V=squeeze(max(max(max(max(max(maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap_V(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap_V(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap_V(ii) points
                % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z-by-n_e
                a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                % aprime possibilities are n_d-by-maxgap_V(ii)+1-by-n_a2-by-1-by-n_a2-by-n_z-by-n_e
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                aprimez=a1primeindexes+N_a1*a2Bind+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV(reshape(aprimez,[N_d,(maxgap_V(ii)+1),N_a2,1,N_a2,N_z,N_e])); % autoexpand level1iidiff(ii) into 4th-dim
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoints_jj(:,1,:,curraindex,:,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                midpoints_jj(:,1,:,curraindex,:,:,:)=repelem(loweredge,1,1,1,length(curraindex),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2-by-n_z-by-n_e
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z-by-n_e
        ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,n_z,n_e,d_gridvals,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,jj),e_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0);
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
        entireRHS_L2=ReturnMatrix_L2+reshape(DiscountedEVinterp(aprime),[N_d*n2long*N_a2,N_a,N_z,N_e]);
        [Vtempii,maxindexL2alt]=max(entireRHS_L2,[],1);
        maxindexL2dalt=rem(maxindexL2alt-1,N_d)+1;
        maxindexL2aalt=ceil(maxindexL2alt/N_d);
        maxindexL2a1alt=rem(maxindexL2aalt-1,n2long)+1;
        maxindexL2a2alt=ceil(maxindexL2aalt/n2long);
        Valt(:,:,:,jj)=shiftdim(Vtempii,1);
        Policyalt(1,:,:,:,jj)=maxindexL2dalt; % d
        Policyalt(2,:,:,:,jj)=midpoints_jj(maxindexL2dalt+N_d*(maxindexL2a2alt-1)+N_d*N_a2*a12ind+N_d*N_a2*N_a*zind+N_d*N_a2*N_a*N_z*eind); % a1prime midpoint
        Policyalt(3,:,:,:,jj)=maxindexL2a2alt; % a2prime
        Policyalt(4,:,:,:,jj)=maxindexL2a1alt; % a1primeL2ind

        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        linidx_loweralt = maxindexL2dalt                  + N_d*n2long*(maxindexL2a2alt-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind + N_d*n2long*N_a2*N_a*N_z*eind;
        linidx_upperalt = maxindexL2dalt + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2alt-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind + N_d*n2long*N_a2*N_a*N_z*eind;
        isInfLoweralt = (ReturnMatrix_L2(linidx_loweralt) == -Inf);
        isInfUpperalt = (ReturnMatrix_L2(linidx_upperalt) == -Inf);
        inLowerStrictalt = (maxindexL2a1alt >= 2)         & (maxindexL2a1alt <= n2short+1);
        inUpperStrictalt = (maxindexL2a1alt >= n2short+3) & (maxindexL2a1alt <= n2long-1);
        PolicyL2flagalt(1,:,:,:,jj) = 2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);
        %% Vtilde (beta0*beta)
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_tilde; % autofill e

        % First, we want a1prime conditional on (d,1,a2prime,a,z,e)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Just keep the 'midpoint' version of maxindex1 [as GI]
        midpoints_jj(:,1,:,level1ii,:,:,:)=maxindex1;

        % Attempt for improved version
        maxgap=squeeze(max(max(max(max(max(maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z-by-n_e
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-n_a2-by-1-by-n_a2-by-n_z-by-n_e
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                aprimez=a1primeindexes+N_a1*a2Bind+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_tilde(reshape(aprimez,[N_d,(maxgap(ii)+1),N_a2,1,N_a2,N_z,N_e])); % autoexpand level1iidiff(ii) into 4th-dim
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoints_jj(:,1,:,curraindex,:,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                midpoints_jj(:,1,:,curraindex,:,:,:)=repelem(loweredge,1,1,1,length(curraindex),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2-by-n_z-by-n_e
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z-by-n_e
        ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,n_z,n_e,d_gridvals,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,jj),e_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0);
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
        entireRHS_L2=ReturnMatrix_L2+reshape(DiscountedEVinterp_tilde(aprime),[N_d*n2long*N_a2,N_a,N_z,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
        maxindexL2d=rem(maxindexL2-1,N_d)+1;
        maxindexL2a=ceil(maxindexL2/N_d);
        maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2a/n2long);
        Vtilde(:,:,:,jj)=shiftdim(Vtempii,1);
        Policy(1,:,:,:,jj)=maxindexL2d; % d
        Policy(2,:,:,:,jj)=midpoints_jj(maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind+N_d*N_a2*N_a*zind+N_d*N_a2*N_a*N_z*eind); % a1prime midpoint
        Policy(3,:,:,:,jj)=maxindexL2a2; % a2prime
        Policy(4,:,:,:,jj)=maxindexL2a1; % a1primeL2ind

        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        linidx_lower = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind + N_d*n2long*N_a2*N_a*N_z*eind;
        linidx_upper = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind + N_d*n2long*N_a2*N_a*N_z*eind;
        isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        PolicyL2flag(1,:,:,:,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1,0);

            %% Valt (beta)
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV; % autofill e

            % First, we want a1prime conditional on (d,1,a2prime,a,z)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoints_jj(:,1,:,level1ii,:,:)=maxindex1;

            % Attempt for improved version
            maxgap_V=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap_V(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap_V(ii) points
                    % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
                    a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                    % aprime possibilities are n_d-by-maxgap_V(ii)+1-by-n_a2-by-1-by-n_a2-by-n_z
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0);
                    aprimez=a1primeindexes+N_a1*a2Bind+N_a*zBind;
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV(reshape(aprimez,[N_d,(maxgap_V(ii)+1),N_a2,1,N_a2,N_z])); % autoexpand level1iidiff(ii) into 4th-dim
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoints_jj(:,1,:,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    midpoints_jj(:,1,:,curraindex,:,:)=repelem(loweredge,1,1,1,length(curraindex),1);
                end
            end

            % Turn this into the 'midpoint'
            midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2-by-n_z
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z
            ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,n_z,special_n_e,d_gridvals,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,jj),e_val, ReturnFnParamsVec,2,0);
            aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
            entireRHS_L2=ReturnMatrix_L2+reshape(DiscountedEVinterp(aprime),[N_d*n2long*N_a2,N_a,N_z]);
            [Vtempii,maxindexL2alt]=max(entireRHS_L2,[],1);
            maxindexL2dalt=rem(maxindexL2alt-1,N_d)+1;
            maxindexL2aalt=ceil(maxindexL2alt/N_d);
            maxindexL2a1alt=rem(maxindexL2aalt-1,n2long)+1;
            maxindexL2a2alt=ceil(maxindexL2aalt/n2long);
            Valt(:,:,e_c,jj)=shiftdim(Vtempii,1);
            Policyalt(1,:,:,e_c,jj)=maxindexL2dalt; % d
            Policyalt(2,:,:,e_c,jj)=midpoints_jj(maxindexL2dalt+N_d*(maxindexL2a2alt-1)+N_d*N_a2*a12ind+N_d*N_a2*N_a*zind); % a1prime midpoint
            Policyalt(3,:,:,e_c,jj)=maxindexL2a2alt; % a2prime
            Policyalt(4,:,:,e_c,jj)=maxindexL2a1alt; % a1primeL2ind

            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            linidx_loweralt = maxindexL2dalt                  + N_d*n2long*(maxindexL2a2alt-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind;
            linidx_upperalt = maxindexL2dalt + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2alt-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind;
            isInfLoweralt = (ReturnMatrix_L2(linidx_loweralt) == -Inf);
            isInfUpperalt = (ReturnMatrix_L2(linidx_upperalt) == -Inf);
            inLowerStrictalt = (maxindexL2a1alt >= 2)         & (maxindexL2a1alt <= n2short+1);
            inUpperStrictalt = (maxindexL2a1alt >= n2short+3) & (maxindexL2a1alt <= n2long-1);
            PolicyL2flagalt(1,:,:,e_c,jj) = 2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);
            %% Vtilde (beta0*beta)
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV_tilde; % autofill e

            % First, we want a1prime conditional on (d,1,a2prime,a,z)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoints_jj(:,1,:,level1ii,:,:)=maxindex1;

            % Attempt for improved version
            maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-n_a2-by-1-by-n_a2-by-n_z
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0);
                    aprimez=a1primeindexes+N_a1*a2Bind+N_a*zBind;
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_tilde(reshape(aprimez,[N_d,(maxgap(ii)+1),N_a2,1,N_a2,N_z])); % autoexpand level1iidiff(ii) into 4th-dim
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoints_jj(:,1,:,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    midpoints_jj(:,1,:,curraindex,:,:)=repelem(loweredge,1,1,1,length(curraindex),1);
                end
            end

            % Turn this into the 'midpoint'
            midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2-by-n_z
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z
            ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,n_z,special_n_e,d_gridvals,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,jj),e_val, ReturnFnParamsVec,2,0);
            aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
            entireRHS_L2=ReturnMatrix_L2+reshape(DiscountedEVinterp_tilde(aprime),[N_d*n2long*N_a2,N_a,N_z]);
            [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
            maxindexL2d=rem(maxindexL2-1,N_d)+1;
            maxindexL2a=ceil(maxindexL2/N_d);
            maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2a/n2long);
            Vtilde(:,:,e_c,jj)=shiftdim(Vtempii,1);
            Policy(1,:,:,e_c,jj)=maxindexL2d; % d
            Policy(2,:,:,e_c,jj)=midpoints_jj(maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind+N_d*N_a2*N_a*zind); % a1prime midpoint
            Policy(3,:,:,e_c,jj)=maxindexL2a2; % a2prime
            Policy(4,:,:,e_c,jj)=maxindexL2a1; % a1primeL2ind

            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            linidx_lower = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind;
            linidx_upper = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind;
            isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            PolicyL2flag(1,:,:,e_c,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_z=DiscountedEV(:,:,:,:,:,z_c);
            DiscountedEV_tilde_z=DiscountedEV_tilde(:,:,:,:,:,z_c);
            DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,:,z_c);
            DiscountedEVinterp_tilde_z=DiscountedEVinterp_tilde(:,:,:,:,:,z_c);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                % n-Monotonicity
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_val,e_val, ReturnFnParamsVec,1,0);

                %% Valt (beta)
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z; % autofill e

                % First, we want a1prime conditional on (d,1,a2prime,a)
                [~,maxindex1]=max(entireRHS_ii,[],2);

                % Just keep the 'midpoint' version of maxindex1 [as GI]
                midpoints_jj(:,1,:,level1ii,:)=maxindex1;

                % Attempt for improved version
                maxgap_V=squeeze(max(max(max(maxindex1(:,1,:,2:end,:)-maxindex1(:,1,:,1:end-1,:),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap_V(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:),N_a1-maxgap_V(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap_V(ii) points
                        % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
                        a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                        % aprime possibilities are n_d-by-maxgap_V(ii)+1-by-n_a2-by-1-by-n_a2
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val,e_val, ReturnFnParamsVec,3,0);
                        aprime=a1primeindexes+N_a1*a2Bind;
                        entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_z(reshape(aprime,[N_d,(maxgap_V(ii)+1),N_a2,1,N_a2])); % autoexpand level1iidiff(ii) into 4th-dim
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoints_jj(:,1,:,curraindex,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:);
                        midpoints_jj(:,1,:,curraindex,:)=repelem(loweredge,1,1,1,length(curraindex),1);
                    end
                end

                % Turn this into the 'midpoint'
                midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2
                a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2
                ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,special_n_z,special_n_e,d_gridvals,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid, z_val,e_val, ReturnFnParamsVec,2,0);
                aprime=a1primeindexes+N_a1fine*a2ind;
                entireRHS_L2=ReturnMatrix_L2+reshape(DiscountedEVinterp_z(aprime),[N_d*n2long*N_a2,N_a]);
                [Vtempii,maxindexL2alt]=max(entireRHS_L2,[],1);
                maxindexL2dalt=rem(maxindexL2alt-1,N_d)+1;
                maxindexL2aalt=ceil(maxindexL2alt/N_d);
                maxindexL2a1alt=rem(maxindexL2aalt-1,n2long)+1;
                maxindexL2a2alt=ceil(maxindexL2aalt/n2long);
                Valt(:,z_c,e_c,jj)=shiftdim(Vtempii,1);
                Policyalt(1,:,z_c,e_c,jj)=maxindexL2dalt; % d
                Policyalt(2,:,z_c,e_c,jj)=midpoints_jj(maxindexL2dalt+N_d*(maxindexL2a2alt-1)+N_d*N_a2*a12ind); % a1prime midpoint
                Policyalt(3,:,z_c,e_c,jj)=maxindexL2a2alt; % a2prime
                Policyalt(4,:,z_c,e_c,jj)=maxindexL2a1alt; % a1primeL2ind

                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                linidx_loweralt = maxindexL2dalt                  + N_d*n2long*(maxindexL2a2alt-1) + N_d*n2long*N_a2*a12ind;
                linidx_upperalt = maxindexL2dalt + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2alt-1) + N_d*n2long*N_a2*a12ind;
                isInfLoweralt = (ReturnMatrix_L2(linidx_loweralt) == -Inf);
                isInfUpperalt = (ReturnMatrix_L2(linidx_upperalt) == -Inf);
                inLowerStrictalt = (maxindexL2a1alt >= 2)         & (maxindexL2a1alt <= n2short+1);
                inUpperStrictalt = (maxindexL2a1alt >= n2short+3) & (maxindexL2a1alt <= n2long-1);
                PolicyL2flagalt(1,:,z_c,e_c,jj) = 2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);
                %% Vtilde (beta0*beta)
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_tilde_z; % autofill e

                % First, we want a1prime conditional on (d,1,a2prime,a)
                [~,maxindex1]=max(entireRHS_ii,[],2);

                % Just keep the 'midpoint' version of maxindex1 [as GI]
                midpoints_jj(:,1,:,level1ii,:)=maxindex1;

                % Attempt for improved version
                maxgap=squeeze(max(max(max(maxindex1(:,1,:,2:end,:)-maxindex1(:,1,:,1:end-1,:),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d-by-maxgap(ii)+1-by-n_a2-by-1-by-n_a2
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val,e_val, ReturnFnParamsVec,3,0);
                        aprime=a1primeindexes+N_a1*a2Bind;
                        entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_tilde_z(reshape(aprime,[N_d,(maxgap(ii)+1),N_a2,1,N_a2])); % autoexpand level1iidiff(ii) into 4th-dim
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoints_jj(:,1,:,curraindex,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:);
                        midpoints_jj(:,1,:,curraindex,:)=repelem(loweredge,1,1,1,length(curraindex),1);
                    end
                end

                % Turn this into the 'midpoint'
                midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2
                a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2
                ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn,n_d,special_n_z,special_n_e,d_gridvals,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid, z_val,e_val, ReturnFnParamsVec,2,0);
                aprime=a1primeindexes+N_a1fine*a2ind;
                entireRHS_L2=ReturnMatrix_L2+reshape(DiscountedEVinterp_tilde_z(aprime),[N_d*n2long*N_a2,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                maxindexL2d=rem(maxindexL2-1,N_d)+1;
                maxindexL2a=ceil(maxindexL2/N_d);
                maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
                maxindexL2a2=ceil(maxindexL2a/n2long);
                Vtilde(:,z_c,e_c,jj)=shiftdim(Vtempii,1);
                Policy(1,:,z_c,e_c,jj)=maxindexL2d; % d
                Policy(2,:,z_c,e_c,jj)=midpoints_jj(maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind); % a1prime midpoint
                Policy(3,:,z_c,e_c,jj)=maxindexL2a2; % a2prime
                Policy(4,:,z_c,e_c,jj)=maxindexL2a1; % a1primeL2ind

                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                linidx_lower = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind;
                linidx_upper = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind;
                isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
                inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                PolicyL2flag(1,:,z_c,e_c,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
            end
        end
    end
end



%% Currently Policy(2,:) is the midpoint, and Policy(4,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(4,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(4,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:,:,:)=Policy(2,:,:,:,:)-adjust; % lower grid point
Policy(4,:,:,:,:)=adjust.*Policy(4,:,:,:,:)+(1-adjust).*(Policy(4,:,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=[Policy; PolicyL2flag];

adjustalt=(Policyalt(4,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policyalt(2,:,:,:,:)=Policyalt(2,:,:,:,:)-adjustalt; % lower grid point
Policyalt(4,:,:,:,:)=adjustalt.*Policyalt(4,:,:,:,:)+(1-adjustalt).*(Policyalt(4,:,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policyalt=[Policyalt; PolicyL2flagalt];

% Policy=Policy(1,:,:,:,:)+N_d*(Policy(2,:,:,:,:)-1)+N_d*N_a1*(Policy(3,:,:,:,:)-1)+N_d*N_a1*N_a2*(Policy(4,:,:,:,:)-1)+N_d*N_a1*N_a2*(n2short+2)*(PolicyL2flag-1);



end
