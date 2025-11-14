function [V, Policy]=ValueFnIter_FHorz_DC2B_GI2B_noz_raw(n_d,n_a, N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% divide-and-conquer in the first endo state

N_d=prod(n_d);
N_a=prod(n_a);

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(4,N_a,N_j,'gpuArray'); % first dim is (d,a1prime midpoint,a2prime,a1prime L2)

%%
n_a1=n_a(1);
n_a2=n_a(2:end);
N_a1=n_a1;
N_a2=n_a2;
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% preallocate
midpoints_jj=zeros(N_d,1,N_a2,N_a1,N_a2,'gpuArray');

% n-Monotonicity
% vfoptions.level1n=[21,21];
level1ii=round(linspace(1,n_a(1),vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);
% aprime_grid=[a1prime_grid; a2_grid];

% For debugging, uncomment next two lines, with this 'aprime_grid' you
% should get exact same value fn as without interpolation (as it doesn't
% really interpolate, it just repeats points)
% aprime_grid=repelem(a_grid,1+n2short,1);
% aprime_grid=aprime_grid(1:(N_a+(N_a-1)*n2short));

% precompute
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-1); % already includes -1

a12ind=repmat(gpuArray(0:1:N_a1-1),1,N_a2)+N_a1*repelem(gpuArray(0:1:N_a2-1),1,N_a1);

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_noz_Par2(ReturnFn, n_d, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, ReturnFnParamsVec, 1);

    % First, we want a1prime conditional on (d,1,a2prime,a)
    [~,maxindex1]=max(ReturnMatrix_ii,[],2);

    % Just keep the 'midpoint' vesion of maxindex1 [as GI]
    midpoints_jj(:,1,:,level1ii,:)=maxindex1;
    
    % Attempt for improved version
    maxgap=squeeze(max(max(max(maxindex1(:,1,:,2:end,:)-maxindex1(:,1,:,1:end-1,:),[],5),[],3),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,:,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1-by-n_a2-by-n_a1-by-n_a2
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1-by-n_a2-by-n_a1-by-n_a2
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_noz_Par2(ReturnFn, n_d, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, ReturnFnParamsVec,3);
            [~,maxindex]=max(ReturnMatrix_ii,[],1);
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
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_noz_Par2(ReturnFn,n_d,d_gridvals,a1prime_grid(a1primeindexes),a2_grid,a1_grid,a2_grid,ReturnFnParamsVec,2);
    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
    maxindexL2d=rem(maxindexL2-1,N_d)+1;
    maxindexL2a=ceil(maxindexL2/N_d);
    maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
    maxindexL2a2=ceil(maxindexL2a/n2long);
    V(:,N_j)=shiftdim(Vtempii,1);
    Policy(1,:,N_j)=maxindexL2d; % d
    Policy(2,:,N_j)=midpoints_jj(maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind); % a1prime midpoint
    Policy(3,:,N_j)=maxindexL2a2; % a2prime
    Policy(4,:,N_j)=maxindexL2a1; % a1primeL2ind

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=reshape(vfoptions.V_Jplus1,[N_a,1]); % Using V_Jplus1
    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_a1,N_a2]);
    % Interpolate EV over aprime_grid
    DiscountedEVinterp=interp1(a1_grid,DiscountedEV,a1prime_grid);
    DiscountedEV=shiftdim(DiscountedEV,-1); % will autoexand d in 1st-dim
    DiscountedEVinterp=shiftdim(DiscountedEVinterp,-1); % will autoexand d in 1st-dim

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_noz_Par2(ReturnFn, n_d, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, ReturnFnParamsVec, 1);

    entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

    % First, we want a1prime conditional on (d,1,a2prime,a,z)
    [~,maxindex1]=max(entireRHS_ii,[],2);
    
    % Just keep the 'midpoint' vesion of maxindex1 [as GI]
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
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_noz_Par2(ReturnFn, n_d, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, ReturnFnParamsVec,3);
            aprime=a1primeindexes+N_a1*a2ind;
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprime,[N_d,(maxgap(ii)+1),N_a2,level1iidiff(ii),N_a2])); % autoexpand level1iidiff(ii) in 4th-dim
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
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_noz_Par2(ReturnFn,n_d,d_gridvals,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid,ReturnFnParamsVec,2);
    aprime=a1primeindexes+N_a1fine*a2ind;
    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprime),[N_d*n2long*N_a2,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    maxindexL2d=rem(maxindexL2-1,N_d)+1;
    maxindexL2a=ceil(maxindexL2/N_d);
    maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
    maxindexL2a2=ceil(maxindexL2a/n2long);
    V(:,N_j)=shiftdim(Vtempii,1);
    Policy(1,:,N_j)=maxindexL2d; % d
    Policy(2,:,N_j)=midpoints_jj(maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind); % a1prime midpoint
    Policy(3,:,N_j)=maxindexL2a2; % a2prime
    Policy(4,:,N_j)=maxindexL2a1; % a1primeL2ind

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
    
    DiscountedEV=DiscountFactorParamsVec*reshape(V(:,jj+1),[N_a1,N_a2]); % will autoexand d in 1st-dim
    % Interpolate EV over aprime_grid
    DiscountedEVinterp=interp1(a1_grid,DiscountedEV,a1prime_grid);
    DiscountedEV=shiftdim(DiscountedEV,-1); % will autoexand d in 1st-dim
    DiscountedEVinterp=shiftdim(DiscountedEVinterp,-1); % will autoexand d in 1st-dim

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_noz_Par2(ReturnFn, n_d, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, ReturnFnParamsVec, 1);

    entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

    % First, we want a1prime conditional on (d,1,a2prime,a,z)
    [~,maxindex1]=max(entireRHS_ii,[],2);
    
    % Just keep the 'midpoint' vesion of maxindex1 [as GI]
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
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_noz_Par2(ReturnFn, n_d, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, ReturnFnParamsVec,3);
            aprime=a1primeindexes+N_a1*a2ind;
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprime,[N_d,(maxgap(ii)+1),N_a2,1,N_a2])); % autoexpand level1iidiff(ii) in 4th-dim
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
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_noz_Par2(ReturnFn,n_d,d_gridvals,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid,ReturnFnParamsVec,2);
    aprime=a1primeindexes+N_a1fine*a2ind;
    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprime),[N_d*n2long*N_a2,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    maxindexL2d=rem(maxindexL2-1,N_d)+1;
    maxindexL2a=ceil(maxindexL2/N_d);
    maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
    maxindexL2a2=ceil(maxindexL2a/n2long);
    V(:,jj)=shiftdim(Vtempii,1);
    Policy(1,:,jj)=maxindexL2d; % d
    Policy(2,:,jj)=midpoints_jj(maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind); % a1prime midpoint
    Policy(3,:,jj)=maxindexL2a2; % a2prime
    Policy(4,:,jj)=maxindexL2a1; % a1primeL2ind
end




%% Currently Policy(2,:) is the midpoint, and Policy(4,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(4,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(4,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:)=Policy(2,:,:)-adjust; % lower grid point
Policy(4,:,:)=adjust.*Policy(4,:,:)+(1-adjust).*(Policy(4,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=Policy(1,:,:)+N_d*(Policy(2,:,:)-1)+N_d*N_a1*(Policy(3,:,:)-1)+N_d*N_a1*N_a2*(Policy(4,:,:)-1);



end