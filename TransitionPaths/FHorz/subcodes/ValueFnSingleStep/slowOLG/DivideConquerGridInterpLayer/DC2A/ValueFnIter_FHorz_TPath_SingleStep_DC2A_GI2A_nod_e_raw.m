function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_DC2A_GI2A_nod_e_raw(V,n_a,n_z,n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The V input is next period value fn (across all ages), the V output is this period.
% divide-and-conquer in the first endo state
% lowmemory: =0 vectorize, =1 loop over e, =2 loop over e and z

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

Policy=zeros(3,N_a,N_z,N_e,N_j,'gpuArray'); % first dim is (a1prime midpoint,a2prime,a1prime L2)
PolicyL2flag=2*ones(1,N_a,N_z,N_e,N_j,'gpuArray'); % L2 flag: 1=all to lower, 2=usual, 3=all to upper

% e is start-of-period: precompute the expectation of V over e for use as continuation
Vnext=sum(V.*shiftdim(pi_e_J,-2),3); % Take expectations over e

%%
n_a1=n_a(1);
n_a2=n_a(2:end);
N_a1=n_a1;
N_a2=n_a2;
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% n-Monotonicity
level1ii=round(linspace(1,N_a1,vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);
% aprime_grid=[a1prime_grid; a2_grid];

% precompute
a2ind=gpuArray(0:1:N_a2-1); % already includes -1
a12ind=repmat(gpuArray(0:1:N_a1-1),1,N_a2)+N_a1*repelem(gpuArray(0:1:N_a2-1),1,N_a1);
if vfoptions.lowmemory==0
    midpoints_jj=zeros(1,N_a2,N_a1,N_a2,N_z,N_e,'gpuArray');
    zind=shiftdim(gpuArray(0:1:N_z-1),-1); % already includes -1
    eind=shiftdim(gpuArray(0:1:N_e-1),-2); % already includes -1
    zBind=shiftdim(gpuArray(0:1:N_z-1),-3); % already includes -1
elseif vfoptions.lowmemory==1
    midpoints_jj=zeros(1,N_a2,N_a1,N_a2,N_z,'gpuArray');
    zind=shiftdim(gpuArray(0:1:N_z-1),-1); % already includes -1
    zBind=shiftdim(gpuArray(0:1:N_z-1),-3); % already includes -1
    special_n_e=ones(1,length(n_e),'gpuArray');
elseif vfoptions.lowmemory==2
    midpoints_jj=zeros(1,N_a2,N_a1,N_a2,'gpuArray');
    special_n_z=ones(1,length(n_z),'gpuArray');
    special_n_e=ones(1,length(n_e),'gpuArray');
end

%% j=N_j: terminal age has no continuation in TPath
% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if vfoptions.lowmemory==0
    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod_e(ReturnFn, n_z, n_e, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

    %Calc the max and it's index
    [~,maxindex1]=max(ReturnMatrix_ii,[],1);

    % Just keep the 'midpoint' version of maxindex1 [as GI]
    midpoints_jj(1,:,level1ii,:,:,:)=maxindex1;

    % Attempt for improved version
    maxgap=squeeze(max(max(max(max(maxindex1(1,:,2:end,:,:,:)-maxindex1(1,:,1:end-1,:,:,:),[],6),[],5),[],4),[],2));
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(1,:,ii,:,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_z-by-n_e
            aprimeindexes=loweredge+(0:1:maxgap(ii))';
            % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_z-by-n_e
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod_e(ReturnFn, n_z, n_e, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3);
            [~,maxindex]=max(ReturnMatrix_ii,[],1);
            midpoints_jj(1,:,curraindex,:,:,:)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(1,:,ii,:,:,:);
            midpoints_jj(1,:,curraindex,:,:,:)=repelem(loweredge,1,1,length(curraindex),1);
        end
    end

    % Turn this into the 'midpoint'
    midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is 1-by-n_a2-by-n_a1-by-n_a2-by-n_z-by-n_e
    a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
    % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z-by-n_e
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod_e(ReturnFn,n_z,n_e,a1prime_grid(a1primeindexes),a2_grid,a1_grid,a2_grid,z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
    maxindexL2a1=rem(maxindexL2-1,n2long)+1;
    maxindexL2a2=ceil(maxindexL2/n2long);
    V(:,:,:,N_j)=shiftdim(Vtempii,1);
    Policy(1,:,:,:,N_j)=midpoints_jj(maxindexL2a2+N_a2*a12ind+N_a2*N_a*zind+N_a2*N_a*N_z*eind); % a1prime midpoint
    Policy(2,:,:,:,N_j)=maxindexL2a2; % a2prime
    Policy(3,:,:,:,N_j)=maxindexL2a1; % a1primeL2ind

    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
    linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind + n2long*N_a2*N_a*N_z*eind;
    linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind + n2long*N_a2*N_a*N_z*eind;
    isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
    isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
    inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
    inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
    PolicyL2flag(1,:,:,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

elseif vfoptions.lowmemory==1
    for e_c=1:N_e
        e_val=e_gridvals_J(e_c,:,N_j);
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod_e(ReturnFn, n_z, special_n_e, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);

        %Calc the max and it's index
        [~,maxindex1]=max(ReturnMatrix_ii,[],1);

        % Just keep the 'midpoint' version of maxindex1 [as GI]
        midpoints_jj(1,:,level1ii,:,:)=maxindex1;

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_z
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_z
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod_e(ReturnFn, n_z, special_n_e, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3);
                [~,maxindex]=max(ReturnMatrix_ii,[],1);
                midpoints_jj(1,:,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(1,:,ii,:,:);
                midpoints_jj(1,:,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is 1-by-n_a2-by-n_a1-by-n_a2-by-n_z
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
        % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod_e(ReturnFn,n_z,special_n_e,a1prime_grid(a1primeindexes),a2_grid,a1_grid,a2_grid,z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);
        V(:,:,e_c,N_j)=shiftdim(Vtempii,1);
        Policy(1,:,:,e_c,N_j)=midpoints_jj(maxindexL2a2+N_a2*a12ind+N_a2*N_a*zind); % a1prime midpoint
        Policy(2,:,:,e_c,N_j)=maxindexL2a2; % a2prime
        Policy(3,:,:,e_c,N_j)=maxindexL2a1; % a1primeL2ind

        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
        linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
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
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod_e(ReturnFn, special_n_z, special_n_e, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_val, e_val, ReturnFnParamsVec,1);

            %Calc the max and it's index
            [~,maxindex1]=max(ReturnMatrix_ii,[],1);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
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
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod_e(ReturnFn, special_n_z, special_n_e, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, e_val, ReturnFnParamsVec,3);
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
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod_e(ReturnFn,special_n_z,special_n_e,a1prime_grid(a1primeindexes),a2_grid,a1_grid,a2_grid,z_val, e_val, ReturnFnParamsVec,2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);
            V(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(1,:,z_c,e_c,N_j)=midpoints_jj(maxindexL2a2+N_a2*a12ind); % a1prime midpoint
            Policy(2,:,z_c,e_c,N_j)=maxindexL2a2; % a2prime
            Policy(3,:,z_c,e_c,N_j)=maxindexL2a1; % a1primeL2ind

            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            PolicyL2flag(1,:,z_c,e_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end
    end
end


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    VKronNext_j=Vnext(:,:,1,jj+1);

    EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_a1,N_a2,1,1,N_z]);
    % Interpolate EV over aprime_grid
    DiscountedEVinterp=interp1(a1_grid,DiscountedEV,a1prime_grid);

    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod_e(ReturnFn, n_z, n_e, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

        %Calc the max and it's index
        [~,maxindex1]=max(entireRHS_ii,[],1);

        % Just keep the 'midpoint' version of maxindex1 [as GI]
        midpoints_jj(1,:,level1ii,:,:,:)=maxindex1;

        % Attempt for improved version
        maxgap=squeeze(max(max(max(max(maxindex1(1,:,2:end,:,:,:)-maxindex1(1,:,1:end-1,:,:,:),[],6),[],5),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,:,ii,:,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_z-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_z-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod_e(ReturnFn, n_z, n_e, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3);
                aprimez=aprimeindexes+N_a1*a2ind+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprimez,[(maxgap(ii)+1),N_a2,1,N_a2,N_z,N_e])); % autoexpand level1iidiff(ii) in 3rd-dim
                [~,maxindex]=max(entireRHS_ii,[],1);
                midpoints_jj(1,:,curraindex,:,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(1,:,ii,:,:,:);
                midpoints_jj(1,:,curraindex,:,:,:)=repelem(loweredge,1,1,length(curraindex),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is 1-by-n_a2-by-n_a1-by-n_a2-by-n_z-by-n_e
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
        % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod_e(ReturnFn,n_z,n_e,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid,z_gridvals_J(:,:,jj),e_gridvals_J(:,:,jj),  ReturnFnParamsVec,2);
        aprimez=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprimez),[n2long*N_a2,N_a,N_z,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);
        V(:,:,:,jj)=shiftdim(Vtempii,1);
        Policy(1,:,:,:,jj)=midpoints_jj(maxindexL2a2+N_a2*a12ind+N_a2*N_a*zind+N_a2*N_a*N_z*eind); % a1prime midpoint
        Policy(2,:,:,:,jj)=maxindexL2a2; % a2prime
        Policy(3,:,:,:,jj)=maxindexL2a1; % a1primeL2ind

        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind + n2long*N_a2*N_a*N_z*eind;
        linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind + n2long*N_a2*N_a*N_z*eind;
        isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        PolicyL2flag(1,:,:,:,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod_e(ReturnFn, n_z, special_n_e, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1);

            entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

            %Calc the max and it's index
            [~,maxindex1]=max(entireRHS_ii,[],1);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoints_jj(1,:,level1ii,:,:)=maxindex1;

            % Attempt for improved version
            maxgap=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_z
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_z
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod_e(ReturnFn, n_z, special_n_e, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3);
                    aprimez=aprimeindexes+N_a1*a2ind+N_a*zBind;
                    entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprimez,[(maxgap(ii)+1),N_a2,1,N_a2,N_z])); % autoexpand level1iidiff(ii) in 3rd-dim
                    [~,maxindex]=max(entireRHS_ii,[],1);
                    midpoints_jj(1,:,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(1,:,ii,:,:);
                    midpoints_jj(1,:,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1);
                end
            end

            % Turn this into the 'midpoint'
            midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is 1-by-n_a2-by-n_a1-by-n_a2-by-n_z
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
            % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod_e(ReturnFn,n_z,special_n_e,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid,z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
            aprimez=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprimez),[n2long*N_a2,N_a,N_z]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);
            V(:,:,e_c,jj)=shiftdim(Vtempii,1);
            Policy(1,:,:,e_c,jj)=midpoints_jj(maxindexL2a2+N_a2*a12ind+N_a2*N_a*zind); % a1prime midpoint
            Policy(2,:,:,e_c,jj)=maxindexL2a2; % a2prime
            Policy(3,:,:,e_c,jj)=maxindexL2a1; % a1primeL2ind

            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
            linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
            isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            PolicyL2flag(1,:,:,e_c,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_z=DiscountedEV(:,:,1,1,z_c);
            DiscountedEVinterp_z=DiscountedEVinterp(:,:,1,1,z_c);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                % n-Monotonicity
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod_e(ReturnFn, special_n_z, special_n_e, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_val, e_val, ReturnFnParamsVec,1);

                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z;

                %Calc the max and it's index
                [~,maxindex1]=max(entireRHS_ii,[],1);

                % Just keep the 'midpoint' version of maxindex1 [as GI]
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
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod_e(ReturnFn, special_n_z, special_n_e, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, e_val, ReturnFnParamsVec,3);
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
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod_e(ReturnFn,special_n_z,special_n_e,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid,z_val, e_val, ReturnFnParamsVec,2);
                aprime=a1primeindexes+N_a1fine*a2ind;
                entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp_z(aprime),[n2long*N_a2,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                maxindexL2a1=rem(maxindexL2-1,n2long)+1;
                maxindexL2a2=ceil(maxindexL2/n2long);
                V(:,z_c,e_c,jj)=shiftdim(Vtempii,1);
                Policy(1,:,z_c,e_c,jj)=midpoints_jj(maxindexL2a2+N_a2*a12ind); % a1prime midpoint
                Policy(2,:,z_c,e_c,jj)=maxindexL2a2; % a2prime
                Policy(3,:,z_c,e_c,jj)=maxindexL2a1; % a1primeL2ind

                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
                linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
                isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                PolicyL2flag(1,:,z_c,e_c,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
            end
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

Policy=[Policy;PolicyL2flag];



end
