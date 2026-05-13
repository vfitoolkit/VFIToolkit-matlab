function [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_DC2B_GI_nod_raw(Vnext,n_a,n_z, a_grid, z_gridvals,pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% DC2B: two endogenous states, divide-and-conquer on the first endo state, but not on the second endo state

N_a=prod(n_a);
N_z=prod(n_z);

n_a1=n_a(1);
n_a2=n_a(2:end);
N_a1=n_a1;
N_a2=prod(n_a2);
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% Preallocate
if vfoptions.lowmemory==0
    midpoints=zeros(1,N_a2,N_a1,N_a2,N_z,'gpuArray');
elseif vfoptions.lowmemory==1 % loops over z
    midpoints=zeros(1,N_a2,N_a1,N_a2,'gpuArray');
    special_n_z=ones(1,length(n_z));
end


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

% precompute
a2ind=gpuArray(0:1:N_a2-1); % already includes -1
zind=shiftdim(gpuArray(0:1:N_z-1),-1); % already includes -1
zBind=shiftdim(gpuArray(0:1:N_z-1),-3); % already includes -1

a12ind=repmat(gpuArray(0:1:N_a1-1),1,N_a2)+N_a1*repelem(gpuArray(0:1:N_a2-1),1,N_a1);

V=zeros(N_a,N_z,'gpuArray');
Policy=zeros(3,N_a,N_z,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%%

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

EV=Vnext.*shiftdim(pi_z',-1);
EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
EV=sum(EV,2); % sum over z', leaving a singular second dimension

DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_a1,N_a2,1,1,N_z]);  % autoexpand (a,z)
% Interpolate EV over aprime_grid
DiscountedEVinterp=interp1(a1_grid,DiscountedEV,a1prime_grid);

if vfoptions.lowmemory==0

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, n_z, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals, ReturnFnParamsVec,1);

    entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

    %Calc the max and it's index
    [~,maxindex1]=max(entireRHS_ii,[],1);

    % Just keep the 'midpoint' vesion of maxindex1 [as GI]
    midpoints(1,:,level1ii,:,:)=maxindex1;

    % Attempt for improved version
    maxgap=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_z
            a1primeindexes=loweredge+(0:1:maxgap(ii))';
            % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, n_z, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals, ReturnFnParamsVec,3);
            aprimez=a1primeindexes+N_a1*a2ind+N_a*zBind;
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprimez,[(maxgap(ii)+1),N_a2,1,N_a2,N_z])); % autoexpand level1iidiff(ii) in 3rd-dim
            [~,maxindex]=max(entireRHS_ii,[],1);
            midpoints(1,:,curraindex,:,:)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(1,:,ii,:,:);
            midpoints(1,:,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1);
        end
    end

    % Turn this into the 'midpoint'
    midpoints=max(min(midpoints,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is 1-by-n_a2-by-n_a1-by-n_a2-by-n_z
    a1primeindexes=(midpoints+(midpoints-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
    % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn,n_z,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid,z_gridvals, ReturnFnParamsVec,2);
    aprimez=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprimez),[n2long*N_a2,N_a,N_z]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    maxindexL2a1=rem(maxindexL2-1,n2long)+1;
    maxindexL2a2=ceil(maxindexL2/n2long);
    V=shiftdim(Vtempii,1);
    Policy(1,:,:)=midpoints(maxindexL2a2+N_a2*a12ind+N_a2*N_a*zind); % a1prime midpoint
    Policy(2,:,:)=maxindexL2a2; % a2prime
    Policy(3,:,:)=maxindexL2a1; % a1primeL2ind

elseif vfoptions.lowmemory==1

    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        DiscountedEV_z=DiscountedEV(:,:,1,1,z_c);
        DiscountedEVinterp_z=DiscountedEVinterp(:,:,1,1,z_c);

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, special_n_z, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_val, ReturnFnParamsVec,1);

        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z;

        %Calc the max and it's index
        [~,maxindex1]=max(entireRHS_ii,[],1);

        % Just keep the 'midpoint' vesion of maxindex1 [as GI]
        midpoints(1,:,level1ii,:)=maxindex1;

        % Attempt for improved version
        maxgap=squeeze(max(max(maxindex1(1,:,2:end,:)-maxindex1(1,:,1:end-1,:),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,:,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-n_a2-by-1-by-n_a2
                a1primeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, special_n_z, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val, ReturnFnParamsVec,3);
                aprime=a1primeindexes+N_a1*a2ind;
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z(reshape(aprime,[(maxgap(ii)+1),N_a2,1,N_a2])); % autoexpand level1iidiff(ii) in 3rd-dim
                [~,maxindex]=max(entireRHS_ii,[],1);
                midpoints(1,:,curraindex,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(1,:,ii,:);
                midpoints(1,:,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoints=max(min(midpoints,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is 1-by-n_a2-by-n_a1-by-n_a2
        a1primeindexes=(midpoints+(midpoints-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
        % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, special_n_z,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid,z_val, ReturnFnParamsVec,2);
        aprime=a1primeindexes+N_a1fine*a2ind;
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp_z(aprime),[n2long*N_a2,N_a]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);
        V(:,z_c)=shiftdim(Vtempii,1);
        Policy(1,:,z_c)=midpoints(maxindexL2a2+N_a2*a12ind); % a1prime midpoint
        Policy(2,:,z_c)=maxindexL2a2; % a2prime
        Policy(3,:,z_c)=maxindexL2a1; % a1primeL2ind
    end
end



%% Currently Policy(1,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.

% So Policy(3,:,:)-n2short-2 is 0 for the current midpoint

% Following does the exact same calc as without TPath
fineindex1=(n2short+1)*(Policy(1,:,:)-1)+1 +(Policy(3,:,:)-n2short-2);
L1a=ceil((fineindex1-1)/(n2short+1))-1;
L1=max(L1a,0)+1; % lower grid point index
L2=fineindex1-(L1-1)*(n2short+1); % L2 index
Policy(1,:,:)=L1;
Policy(3,:,:)=L2;

end
