function [V,Policy2]=ValueFnIter_InfHorz_TPath_SingleStep_DC2B_GI_raw(Vnext,n_d,n_a,n_z, d_grid, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% DC2B: two endogenous states, divide-and-conquer on the first endo state, but not on the second endo state

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

N_a1=n_a(1);
N_a2=n_a(2);
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

%%
d_gridvals=CreateGridvals(n_d,d_grid,1);

% Preallocate
if vfoptions.lowmemory==0
    midpoints=zeros(N_d,1,N_a2,N_a1,N_a2,N_z,'gpuArray');
elseif vfoptions.lowmemory==1 % loops over z
    midpoints=zeros(N_d,1,N_a2,N_a1,N_a2,'gpuArray');
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

% For debugging, uncomment next two lines, with this 'aprime_grid' you
% should get exact same value fn as without interpolation (as it doesn't
% really interpolate, it just repeats points)
% aprime_grid=repelem(a_grid,1+n2short,1);
% aprime_grid=aprime_grid(1:(N_a+(N_a-1)*n2short));

V=zeros(N_a,N_z,'gpuArray');
Policy=zeros(4,N_a,N_z,'gpuArray'); % first dim: d,a1,a2,a1L2

%%
% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

if vfoptions.lowmemory==0

    EV=Vnext.*shiftdim(pi_z',-1);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension
    entireEV=repelem(shiftdim(EV,-1),N_d,1,1,1); % [d,aprime,1,z]

    % Interpolate EV over aprime_grid
    EVinterp=reshape(interp1(a1_grid,reshape(EV,[N_a1,N_a2,N_z]),a1prime_grid),[N_a1prime*N_a2,N_z]);
    entireEVinterp=repmat(shiftdim(EVinterp,-1),N_d,1,1,1); % [d,aprime,1,z]
    
    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_Par2(ReturnFn, n_d, n_z, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals, ReturnFnParamsVec,1);

    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEV,[N_d,N_a1,N_a2,1,1,N_z]); % move a2prime into same dimension as (a1,a2), so second dimension is solely a1prime

    % First, we want a1prime conditional on (d,1,a2prime,a1,a2,z)
    [~,maxindex1]=max(entireRHS_ii,[],2);
    midpoints(:,1,:,level1ii,:,:)=maxindex1;

    % Attempt for improved version
    maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1-by-n_a2-by-1-by-n_a2-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_Par2(ReturnFn, n_d, n_z, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals, ReturnFnParamsVec,3);
            daprimez=(1:1:N_d)'+N_d*repelem(a1primeindexes-1,1,1,1,level1iidiff(ii),1,1)+N_d*N_a1*shiftdim((0:1:N_a2-1),-1)+N_d*N_a*shiftdim((0:1:N_z-1),-4); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV(reshape(daprimez,[N_d,(maxgap(ii)+1),N_a2,level1iidiff(ii),N_a2,N_z]));
            [~,maxindex]=max(entireRHS_ii,[],2);
            midpoints(:,1,:,curraindex,:,:)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(:,1,:,ii,:,:);
            midpoints(:,1,:,curraindex,:,:)=maxindexfix+(loweredge-1); % THIS MIGHT BE INCORRECT??
        end
    end
    
    % Turn this into the 'midpoint'
    midpoints=max(min(midpoints,n_a(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2-by-n_z
    a1primeindexes=(midpoints+(midpoints-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
    % a1prime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_Par2(ReturnFn, n_d, n_z, d_gridvals, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_gridvals, ReturnFnParamsVec,2);
    daprimez=gpuArray(1:1:N_d)'+N_d*(a1primeindexes-1)+N_d*N_a1prime*shiftdim((0:1:N_a2-1),-1)+N_d*N_a1prime*N_a2*shiftdim((0:1:N_z-1),-4);
    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEVinterp(daprimez(:)),[N_d*n2long*N_a2,N_a,N_z]);
    [V,maxindex2]=max(entireRHS_ii,[],1);
    V=shiftdim(V,1);
    % midpoint has a1 midpoint, maxindex2 has d, L2index for a1 and index for a2
    d_ind=shiftdim(rem(maxindex2-1,N_d)+1,1);
    maxindex2p2=shiftdim(ceil(maxindex2/N_d),1);
    maxindexL2=rem(maxindex2p2-1,n2long)+1;
    maxindexa2prime=ceil(maxindex2p2/n2long);
    maxindexa1prime=midpoints(d_ind+N_d*(maxindexa2prime-1)+N_d*N_a2*(0:1:N_a-1)'+N_d*N_a2*N_a*(0:1:N_z-1));
    Policy(1,:,:)=d_ind; % midpoint for a1
    Policy(2,:,:)=maxindexa1prime; % midpoint for a1
    Policy(3,:,:)=maxindexa2prime; % a2
    Policy(4,:,:)=maxindexL2; % a1prime L2index
    
elseif vfoptions.lowmemory==1
    error('not yet implemented vfoptions.lowmemory==1 with DC2B and GI in InfHorz TPath')

end


%% Currently Policy(2,:) is the midpoint, and Policy(4,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(4,:)
% counting 0:nshort+1 up from this.

% So Policy(3,:,:)-n2short-2 is 0 for the current midpoint

% Following does the exact same calc as without TPath
fineindex1=(n2short+1)*(Policy(2,:,:)-1)+1 +(Policy(4,:,:)-n2short-2);
L1a=ceil((fineindex1-1)/(n2short+1))-1;
L1=max(L1a,0)+1; % lower grid point index
L2=fineindex1-(L1-1)*(n2short+1); % L2 index
Policy(2,:,:)=L1;
Policy(4,:,:)=L2;


%% Policy in transition paths
l_d=length(n_d);
Policy2=zeros(l_d+3,N_a,N_z);
% sort d variables
Policy2(1,:,:)=rem(Policy(1,:,:)-1,n_d(1))+1;
if l_d>1
    if l_d>2
        for ii=2:l_d-1
            Policy2(ii,:,:)=rem(ceil(Policy(1,:,:)/prod(n_d(1:ii-1)))-1,n_d(ii))+1;
        end
    end
    Policy2(l_d,:,:)=ceil(Policy(1,:,:)/prod(n_d(1:l_d-1)));
end
% rest are already in right shape
Policy2(l_d+1:end,:,:)=Policy(2:4,:,:);


end
