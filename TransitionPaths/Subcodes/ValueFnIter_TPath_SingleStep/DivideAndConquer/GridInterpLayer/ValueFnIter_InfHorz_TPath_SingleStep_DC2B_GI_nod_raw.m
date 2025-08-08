function [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_DC2B_GI_nod_raw(Vnext,n_a,n_z, a_grid, z_gridvals,pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% DC2B: two endogenous states, divide-and-conquer on the first endo state, but not on the second endo state

N_a=prod(n_a);
N_z=prod(n_z);

N_a1=n_a(1);
N_a2=n_a(2);
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
% vfoptions.level1n=5;
level1ii=round(linspace(1,N_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:n_a(1),a_grid(1:n_a(1)),linspace(1,n_a(1),n_a(1)+(n_a(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

V=zeros(N_a,N_z,'gpuArray');
Policy=zeros(3,N_a,N_z,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%%

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

if vfoptions.lowmemory==0
    EV=Vnext.*shiftdim(pi_z',-1);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    % Interpolate EV over aprime_grid
    EVinterp=reshape(interp1(a1_grid,reshape(EV,[N_a1,N_a2,N_z]),a1prime_grid),[N_a1prime*N_a2,N_z]);

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, n_z, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals, ReturnFnParamsVec,1);

    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV,[N_a1,N_a2,1,1,N_z]); % autoexpand (a,z)

    % Calc the max and it's index: a1prime conditional on a2prime
    [~,maxindex1]=max(entireRHS_ii,[],1);
    midpoints(1,:,level1ii,:,:)=maxindex1;

    % Attempt for improved version
    maxgap=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
    for ii=1:(vfoptions.level1n-1)
        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
        if maxgap(ii)>0
            loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is 1-by-n_a2-by-1-by-n_a1-by-n_a2-by-n_z
            aprimeindexes=loweredge+(0:1:maxgap(ii))';
            % aprime possibilities are maxgap(ii)+1-n_a2-by-1-by-n_a2-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, n_z, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals, ReturnFnParamsVec,1);
            aprimez=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1)+N_a1*(0:1:N_a2-1)+N_a*shiftdim((0:1:N_z-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV(reshape(aprimez,[(maxgap(ii)+1),N_a2,level1iidiff(ii),N_a2,N_z]));
            [~,maxindex]=max(entireRHS_ii,[],1);
            midpoints(1,:,curraindex,:,:)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(1,:,ii,:,:);
            midpoints(1,:,curraindex,:,:)=maxindexfix+(loweredge-1); % THIS MIGHT BE INCORRECT??
        end
    end

    % Turn this into the 'midpoint'
    midpoints=max(min(midpoints,n_a(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is 1-by-n_a2-by-n_a1-by-n_a2-by-n_z
    a1primeindexes=(midpoints+(midpoints-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
    % a1prime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, n_z, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_gridvals, ReturnFnParamsVec,2);
    aprimez=a1primeindexes+N_a1prime*(0:1:N_a2-1)+N_a1prime*N_a2*shiftdim((0:1:N_z-1),-3);
    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimez(:)),[n2long*N_a2,N_a,N_z]);
    [V,maxindex2]=max(entireRHS_ii,[],1);
    V=shiftdim(V,1);
    % midpoint has a1 midpoint, maxindex2 has L2index for a1 with index for a2
    maxindexL2=shiftdim(rem(maxindex2-1,n2long)+1,1);
    maxindexa2prime=shiftdim(ceil(maxindex2/n2long),1);
    maxindexa1prime=midpoints(maxindexa2prime+N_a2*((0:1:N_a-1)'+N_a*(0:1:N_z-1)));
    Policy(1,:,:)=maxindexa1prime; % midpoint for a1
    Policy(2,:,:)=maxindexa2prime; % a2
    Policy(3,:,:)=maxindexL2; % a1prime L2index

elseif vfoptions.lowmemory==1
    error('not yet implemented vfoptions.lowmemory==1 with DC2B and GI in InfHorz TPath')


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
