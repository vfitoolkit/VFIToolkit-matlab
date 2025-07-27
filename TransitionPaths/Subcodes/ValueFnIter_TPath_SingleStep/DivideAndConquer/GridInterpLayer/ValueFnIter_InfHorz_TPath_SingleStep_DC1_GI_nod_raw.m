function [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_GI_nod_raw(Vnext,n_a,n_z, a_grid, z_gridvals,pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_a=prod(n_a);
N_z=prod(n_z);

% Preallocate
if vfoptions.lowmemory==0
    midpoints=zeros(1,N_a,N_z,'gpuArray');
elseif vfoptions.lowmemory==1 % loops over z
    midpoints=zeros(1,N_a,'gpuArray');
    special_n_z=ones(1,length(n_z));
end

% n-Monotonicity
% vfoptions.level1n=5;
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

% For debugging, uncomment next two lines, with this 'aprime_grid' you
% should get exact same value fn as without interpolation (as it doesn't
% really interpolate, it just repeats points)
% aprime_grid=repelem(a_grid,1+n2short,1);
% aprime_grid=aprime_grid(1:(N_a+(N_a-1)*n2short));

Policy=zeros(2,N_a,N_z,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

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
    EVinterp=interp1(a_grid,EV,aprime_grid);

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid, a_grid(level1ii), z_gridvals, ReturnFnParamsVec,1);

    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV;

    %Calc the max and it's index
    [~,maxindex1]=max(entireRHS_ii,[],1);

    % Just keep the 'midpoint' vesion of maxindex1 [as GI]
    midpoints(1,level1ii,:)=maxindex1;

    % Attempt for improved version
    maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is 1-by-1-by-n_z
            aprimeindexes=loweredge+(0:1:maxgap(ii))';
            % aprime possibilities are maxgap(ii)+1-by-1-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals, ReturnFnParamsVec,2);
            aprimez=repelem(aprimeindexes,1,level1iidiff(ii),1)+N_a*shiftdim((0:1:N_z-1),-1); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV(aprimez);
            [~,maxindex]=max(entireRHS_ii,[],1);
            midpoints(1,curraindex,:)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(1,ii,:);
            midpoints(1,curraindex,:)=repelem(loweredge,1,length(curraindex),1);
        end
    end

    % Turn this into the 'midpoint'
    midpoints=max(min(midpoints,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is 1-by-n_a-by-n_z
    aprimeindexes=(midpoints+(midpoints-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
    % aprime possibilities are n_d-by-n2long-by-n_a-by-n_z
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,n_z,aprime_grid(aprimeindexes),a_grid,z_gridvals,ReturnFnParamsVec,2);
    aprimez=aprimeindexes+n2aprime*shiftdim((0:1:N_z-1),-1);
    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimez(:)),[n2long,N_a,N_z]);
    [V,maxindexL2]=max(entireRHS_ii,[],1);
    V=shiftdim(V,1);
    Policy(1,:,:)=shiftdim(squeeze(midpoints),-1); % midpoint
    Policy(2,:,:)=shiftdim(maxindexL2,-1); % aprimeL2ind

elseif vfoptions.lowmemory==1
    V=zeros(N_a,N_z,'gpuArray'); % preallocate

    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        
        EV_z=Vnext.*shiftdim(pi_z(z_c,:)',-1);
        EV_z(isnan(EV_z))=0; % multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2); % sum over z', leaving a singular second dimension

        % Interpolate EV over aprime_grid
        EVinterp_z=interp1(a_grid,EV_z,aprime_grid);

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_z, a_grid, a_grid(level1ii), z_val, ReturnFnParamsVec,1);

        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV_z;

        % Calc the max and it's index
        [~,maxindex1]=max(entireRHS_ii,[],1);

        % Just keep the 'midpoint' vesion of maxindex1 [as GI]
        midpoints(1,level1ii)=maxindex1;

        % Attempt for improved version
        maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-1
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are maxgap(ii)+1-by-1
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_z, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, ReturnFnParamsVec,2);
                aprimez=repelem(aprimeindexes,1,level1iidiff(ii),1); % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV_z(aprimez);
                [~,maxindex]=max(entireRHS_ii,[],1);
                midpoints(1,curraindex)=shiftdim(maxindex+(loweredge-1),1);
            else
                loweredge=maxindex1(1,ii);
                midpoints(1,curraindex)=repelem(loweredge,length(curraindex),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoints=max(min(midpoints,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is 1-by-n_a
        aprimeindexes=(midpoints+(midpoints-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a
        ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,special_n_z,aprime_grid(aprimeindexes),a_grid,z_val,ReturnFnParamsVec,2);
        % aprime=aprimeindexes;
        entireRHS_ii_z=ReturnMatrix_ii_z+DiscountFactorParamsVec*reshape(EVinterp_z(aprimeindexes(:)),[n2long,N_a]);
        [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
        V(:,z_c)=shiftdim(Vtempii,1);
        Policy(1,:,z_c)=shiftdim(squeeze(midpoints),-1); % midpoint
        Policy(2,:,z_c)=shiftdim(maxindexL2,-1); % aprimeL2ind
    end
end

%% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(1,:,:,:)=Policy(1,:,:,:)-adjust; % lower grid point
Policy(2,:,:,:)=adjust.*Policy(2,:,:,:)+(1-adjust).*(Policy(2,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)


end
