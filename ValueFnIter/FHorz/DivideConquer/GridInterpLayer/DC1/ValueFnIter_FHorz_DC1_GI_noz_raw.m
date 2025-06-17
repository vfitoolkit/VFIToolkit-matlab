function [V,Policy]=ValueFnIter_FHorz_DC1_GI_noz_raw(n_d,n_a,N_j, d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% divide-and-conquer for length(n_a)==1

N_d=prod(n_d);
N_a=prod(n_a);

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(3,N_a,N_j,'gpuArray'); % first dim indexes the optimal choice for aprime and aprime2 (in GI layer)

%%
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
d_gridvals=CreateGridvals(n_d,d_grid,1);

% Preallocate
midpoints_jj=zeros(N_d,1,N_a,'gpuArray');

aind=0:1:N_a-1; % already includes -1

% n-Monotonicity
% vfoptions.level1n=5;
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
% n2aprime=length(aprime_grid);

% For debugging, uncomment next two lines, with this 'aprime_grid' you
% should get exact same value fn as without interpolation (as it doesn't
% really interpolate, it just repeats points)
% aprime_grid=repelem(a_grid,1+n2short,1);
% aprime_grid=aprime_grid(1:(N_a+(N_a-1)*n2short));

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);

    % First, we want aprime conditional on (d,1,a,z)
    [~,maxindex1]=max(ReturnMatrix_ii,[],2);

    % Just keep the 'midpoint' vesion of maxindex1 [as GI]
    midpoints_jj(:,1,level1ii)=maxindex1;

    % Attempt for improved version
    maxindex1=squeeze(maxindex1);
    maxgap=max(maxindex1(:,2:end)-maxindex1(:,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,ii),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,3);
            [~,maxindex]=max(ReturnMatrix_ii,[],2);
            midpoints_jj(:,1,curraindex,:)=shiftdim(maxindex+(loweredge-1),1);
        else
            loweredge=maxindex1(:,ii);
            midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1);
        end
    end

    % Turn this into the 'midpoint'
    midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is n_d-1-by-n_a
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
    % aprime possibilities are n_d-by-n2long-by-n_a
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn,n_d, d_gridvals,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec,2);
    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
    V(:,N_j)=shiftdim(Vtempii,1);
    d_ind=rem(maxindexL2-1,N_d)+1;
    allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a
    Policy(1,:,N_j)=d_ind; % d
    Policy(2,:,N_j)=shiftdim(squeeze(midpoints_jj(allind)),-1); % midpoint
    Policy(3,:,N_j)=shiftdim(maxindexL2,-1); % aprimeL2ind
    
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,1]);    % First, switch V_Jplus1 into Kron form
    entireEV=repmat(V_Jplus1',N_d,1); % [d,aprime]

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);

    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV; % (d,aprime,a)

    % First, we want aprime conditional on (d,1,a,z)
    [~,maxindex1]=max(entireRHS_ii,[],2);

    % Just keep the 'midpoint' vesion of maxindex1 [as GI]
    midpoints_jj(:,1,level1ii)=maxindex1;

    % Attempt for improved version
    maxindex1=squeeze(maxindex1);
    maxgap=max(maxindex1(:,2:end)-maxindex1(:,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,ii),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1
            aprimeindexes=loweredge+(0:1:maxgap(ii)); 
            % aprime possibilities are n_d-by-maxgap(ii)+1
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(loweredge+(0:1:maxgap(ii))), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,3);
            daprime=(repmat(1:1:N_d,1,maxgap(ii)+1))'+N_d*repelem(aprimeindexes-1,1,level1iidiff(ii)); % all the d, with the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEV(daprime),[N_d,(maxgap(ii)+1),level1iidiff(ii)]);
            [~,maxindex]=max(entireRHS_ii,[],2);
            midpoints_jj(:,1,curraindex)=shiftdim(maxindex+(loweredge-1),1);
        else
            loweredge=maxindex1(:,ii);
            midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1);
        end
    end

    % Turn this into the 'midpoint'
    midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is n_d-1-by-n_a
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
    % aprime possibilities are n_d-by-n2long-by-n_a
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn,n_d,d_gridvals,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec,2);
    daprimez=(1:1:N_d)'+N_d*(aprimeindexes-1);
    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEVinterp(daprimez(:)),[N_d*n2long,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V(:,N_j)=shiftdim(Vtempii,1);
    d_ind=rem(maxindexL2-1,N_d)+1;
    allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a
    Policy(1,:,N_j)=d_ind; % d
    Policy(2,:,N_j)=shiftdim(squeeze(midpoints_jj(allind)),-1); % midpoint
    Policy(3,:,N_j)=shiftdim(maxindexL2,-1); % aprimeL2ind

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
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    VKronNext_j=V(:,jj+1);
    entireEV=repmat(VKronNext_j',N_d,1); % [d,aprime]

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);
    
    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV; % (d,aprime,a)

    % First, we want aprime conditional on (d,1,a,z)
    [~,maxindex1]=max(entireRHS_ii,[],2);

    % Just keep the 'midpoint' version of maxindex1 [as GI]
    midpoints_jj(:,1,level1ii)=maxindex1;
    
    % Attempt for improved version
    maxindex1=squeeze(maxindex1);
    maxgap=max(maxindex1(:,2:end)-maxindex1(:,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,ii),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1
            aprimeindexes=loweredge+(0:1:maxgap(ii)); 
            % aprime possibilities are n_d-by-maxgap(ii)+1
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,3);
            daprime=(repmat(1:1:N_d,1,maxgap(ii)+1))'+N_d*repelem(aprimeindexes(:)-1,1,level1iidiff(ii)); % all the d, with the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEV(daprime),[N_d,(maxgap(ii)+1),level1iidiff(ii)]);
            [~,maxindex]=max(entireRHS_ii,[],2);
            midpoints_jj(:,1,curraindex)=shiftdim(maxindex+(loweredge-1),1);
        else
            loweredge=maxindex1(:,ii);
            midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1);
        end
    end

    % Turn this into the 'midpoint'
    midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is n_d-1-by-n_a
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
    % aprime possibilities are n_d-by-n2long-by-n_a
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn,n_d,d_gridvals,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec,2);
    daprimez=(1:1:N_d)'+N_d*(aprimeindexes-1);
    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEVinterp(daprimez(:)),[N_d*n2long,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V(:,jj)=shiftdim(Vtempii,1);
    d_ind=rem(maxindexL2-1,N_d)+1;
    allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a
    Policy(1,:,jj)=d_ind; % d
    Policy(2,:,jj)=shiftdim(squeeze(midpoints_jj(allind)),-1); % midpoint
    Policy(3,:,jj)=shiftdim(maxindexL2,-1); % aprimeL2ind

end

% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:)=Policy(2,:,:)-adjust; % lower grid point
Policy(3,:,:)=adjust.*Policy(3,:,:)+(1-adjust).*(Policy(3,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=Policy(1,:,:)+N_d*(Policy(2,:,:)-1)+N_d*N_a*(Policy(3,:,:)-1);

end
