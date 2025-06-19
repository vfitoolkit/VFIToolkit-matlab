function [V,Policy]=ValueFnIter_FHorz_SemiExo_GI_noz_raw(n_d1,n_d2,n_a,n_semiz,N_j, d1_grid, d2_grid, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

n_d=[n_d1,n_d2];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=N_d1*N_d2;
N_a=prod(n_a);
N_semiz=prod(n_semiz);

V=zeros(N_a,N_semiz,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d,d2,aprime seperately
Policy=zeros(4,N_a,N_semiz,N_j,'gpuArray'); % first dim indexes the optimal choice for d1,d2,aprime and aprime2 (in GI layer)

%%
d1_grid=gpuArray(d1_grid);
d2_grid=gpuArray(d2_grid);
a_grid=gpuArray(a_grid);

special_n_d=[n_d1,ones(1,length(n_d2))];

d_gridvals=CreateGridvals(n_d,[d1_grid; d2_grid],1);

d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_d2,length(n_d1)+length(n_d2)]),[1,3,2]); % version to use when looping over d2

if vfoptions.lowmemory>0
    special_n_semiz=ones(1,length(n_semiz));
end

aind=0:1:N_a-1; % already includes -1
semizind=shiftdim((0:1:N_semiz-1),-1); % already includes -1

% Preallocate
V_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');
midpoint_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');

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

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_semiz, d_gridvals, a_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        % Treat standard problem as just being the first layer
        [~,maxindex]=max(ReturnMatrix,[],2);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a-by-n_semiz
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_semiz
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,n_semiz,d_gridvals,aprime_grid(aprimeindexes),a_grid,semiz_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*semizind; % midpoint is n_d-by-1-by-n_a-by-n_semiz
        Policy(1,:,:,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1); % d1
        Policy(2,:,:,N_j)=shiftdim(ceil(d_ind/N_d1),-1); % d2
        Policy(3,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
        Policy(4,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind

    elseif vfoptions.lowmemory==1

        for z_c=1:N_semiz
            z_val=semiz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_semiz, d_gridvals, a_grid, z_val, ReturnFnParamsVec,1);
            % Treat standard problem as just being the first layer
            [~,maxindex]=max(ReturnMatrix_z,[],2);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,special_n_semiz,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_val,ReturnFnParamsVec,2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V(:,z_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a
            Policy(1,:,z_c,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1); % d1
            Policy(2,:,z_c,N_j)=shiftdim(ceil(d_ind/N_d1),-1); % d2
            Policy(3,:,z_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
            Policy(4,:,z_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
        end

    end
else
    % Using V_Jplus1
    EV=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j);
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            EV_d2=EV.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension
            
            % Interpolate EV over aprime_grid
            EVinterp=interp1(a_grid,EV_d2,aprime_grid);
            entireEVinterp=repelem(EVinterp,N_d1,1,1);

            ReturnMatrix_d2=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, special_n_d, n_a, n_semiz, d12c_gridvals, a_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
            entireRHS=ReturnMatrix_d2+DiscountFactorParamsVec*shiftdim(EV_d2,-1);
            % Treat standard problem as just being the first layer
            [~,maxindex]=max(entireRHS,[],2);

            % Turn maxindex into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-by-1-by-n_a-by-n_semiz
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a-by-n_semiz
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, special_n_d, n_semiz, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            daprimez=(1:1:N_d1)'+N_d1*(aprimeindexes-1)+N_d1*n2aprime*shiftdim((0:1:N_bothz-1),-2); % the current aprime
            entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(entireEVinterp(daprimez(:)),[N_d1*n2long,N_a,N_semiz]);
            [Vtemp,maxindex]=max(entireRHS_ii,[],1);
            
            V_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,d2_c)=shiftdim(maxindex,1);

            d1_ind=rem(maxindex-1,N_d1)+1;
            allind=d1_ind+N_d1*aind+N_d1*N_a*semizind; % loweredge is n_d-by-1-by-n_a-by-n_semiz
            midpoint_ford2_jj(:,:,d2_c)=squeeze(midpoint(allind));
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
        V(:,:,N_j)=V_jj;
        Policy(2,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
        Policy(1,:,:,N_j)=shiftdim(rem(d1aprimeL2_ind-1,N_d1)+1,-1); % d1
        Policy(4,:,:,N_j)=shiftdim(ceil(d1aprimeL2_ind/N_d1),-1); % aprimeL2ind
        Policy(3,:,:,N_j)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]); % midpoint
        
    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j);
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);

                %Calc the condl expectation term (except beta), which depends on z but not on control variables
                EV_d2z=EV.*(ones(N_a,1,'gpuArray')*pi_semiz(z_c,:));
                EV_d2z(isnan(EV_d2z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_d2z=sum(EV_d2z,2);

                % Interpolate EV over aprime_grid
                EVinterp_z=interp1(a_grid,EV_d2z,aprime_grid);
                entireEVinterp_z=repelem(EVinterp_z,N_d1,1,1);

                ReturnMatrix_d2z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, special_n_d, n_a, special_n_semiz, d12c_gridvals, a_grid, z_val, ReturnFnParamsVec,1);
                entireRHS_z=ReturnMatrix_d2z+DiscountFactorParamsVec*shiftdim(EV_d2z,-1);
                % Treat standard problem as just being the first layer
                [~,maxindex]=max(entireRHS_z,[],2);

                % Turn maxindex into the 'midpoint'
                midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-by-1-by-n_a
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, z_val, ReturnFnParamsVec,2);
                daprime=(1:1:N_d1)'+N_d1*(aprimeindexes-1); % the current aprime
                entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(entireEVinterp_z(daprime(:)),[N_d1*n2long,N_a]);
                [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                V_ford2_jj(:,z_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,z_c,d2_c)=shiftdim(maxindex,1);

                d1_ind=rem(maxindex-1,N_d1)+1;
                allind=d1_ind+N_d1*aind; % loweredge is n_d-by-1-by-n_a
                midpoint_ford2_jj(:,z_c,d2_c)=squeeze(midpoint(allind));
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
        V(:,:,N_j)=V_jj;
        Policy(2,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
        Policy(1,:,:,N_j)=shiftdim(rem(d1aprimeL2_ind-1,N_d1)+1,-1); % d1
        Policy(4,:,:,N_j)=shiftdim(ceil(d1aprimeL2_ind/N_d1),-1); % aprimeL2ind
        Policy(3,:,:,N_j)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]); % midpoint

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
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    EV=V(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            pi_semiz=pi_semiz_J(:,:,d2_c,jj);
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            EV_d2=EV.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            % Interpolate EV over aprime_grid
            EVinterp=interp1(a_grid,EV_d2,aprime_grid);
            entireEVinterp=repelem(EVinterp,N_d1,1,1);

            ReturnMatrix_d2=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, special_n_d, n_a, n_semiz, d12c_gridvals, a_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,1);
            entireRHS=ReturnMatrix_d2+DiscountFactorParamsVec*shiftdim(EV_d2,-1);
            % Treat standard problem as just being the first layer
            [~,maxindex]=max(entireRHS,[],2);

            % Turn maxindex into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-by-1-by-n_a-by-n_semiz
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a-by-n_semiz
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, special_n_d, n_semiz, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
            daprimez=(1:1:N_d1)'+N_d1*(aprimeindexes-1)+N_d1*n2aprime*shiftdim((0:1:N_semiz-1),-2); % the current aprime
            entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(entireEVinterp(daprimez(:)),[N_d1*n2long,N_a,N_semiz]);
            [Vtemp,maxindex]=max(entireRHS_ii,[],1);
            
            V_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,d2_c)=shiftdim(maxindex,1);

            d1_ind=rem(maxindex-1,N_d1)+1;
            allind=d1_ind+N_d1*aind+N_d1*N_a*semizind; % loweredge is n_d-by-1-by-n_a-by-n_semiz
            midpoint_ford2_jj(:,:,d2_c)=squeeze(midpoint(allind));
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
        V(:,:,jj)=V_jj;
        Policy(2,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
        Policy(1,:,:,jj)=shiftdim(rem(d1aprimeL2_ind-1,N_d1)+1,-1); % d1
        Policy(4,:,:,jj)=shiftdim(ceil(d1aprimeL2_ind/N_d1),-1); % aprimeL2ind
        Policy(3,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]); % midpoint

    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            pi_semiz=pi_semiz_J(:,:,d2_c,jj);
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);

                %Calc the condl expectation term (except beta), which depends on z but not on control variables
                EV_d2z=EV.*(ones(N_a,1,'gpuArray')*pi_semiz(z_c,:));
                EV_d2z(isnan(EV_d2z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_d2z=sum(EV_d2z,2);

                % Interpolate EV over aprime_grid
                EVinterp_z=interp1(a_grid,EV_d2z,aprime_grid);
                entireEVinterp_z=repelem(EVinterp_z,N_d1,1,1);

                ReturnMatrix_d2z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, special_n_d, n_a, special_n_semiz, d12c_gridvals, a_grid, z_val, ReturnFnParamsVec,1);
                entireRHS_z=ReturnMatrix_d2z+DiscountFactorParamsVec*shiftdim(EV_d2z,-1);
                % Treat standard problem as just being the first layer
                [~,maxindex]=max(entireRHS_z,[],2);

                % Turn maxindex into the 'midpoint'
                midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-by-1-by-n_a
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, z_val, ReturnFnParamsVec,2);
                daprime=(1:1:N_d1)'+N_d1*(aprimeindexes-1); % the current aprime
                entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(entireEVinterp_z(daprime(:)),[N_d1*n2long,N_a]);
                [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                V_ford2_jj(:,z_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,z_c,d2_c)=shiftdim(maxindex,1);

                d1_ind=rem(maxindex-1,N_d1)+1;
                allind=d1_ind+N_d1*aind; % loweredge is n_d-by-1-by-n_a
                midpoint_ford2_jj(:,z_c,d2_c)=squeeze(midpoint(allind));
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
        V(:,:,jj)=V_jj;
        Policy(2,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
        Policy(1,:,:,jj)=shiftdim(rem(d1aprimeL2_ind-1,N_d1)+1,-1); % d1
        Policy(4,:,:,jj)=shiftdim(ceil(d1aprimeL2_ind/N_d1),-1); % aprimeL2ind
        Policy(3,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]); % midpoint
        
    end
end



%% Currently Policy(3,:) is the midpoint, and Policy(4,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(3,:) to 'lower grid point' and then have Policy(4,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(4,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(3,:,:,:,:)=Policy(3,:,:,:,:)-adjust; % lower grid point
Policy(4,:,:,:,:)=adjust.*Policy(4,:,:,:,:)+(1-adjust).*(Policy(4,:,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=squeeze(Policy(1,:,:,:,:)+N_d1*(Policy(2,:,:,:,:)-1)+N_d*(Policy(3,:,:,:,:)-1)+N_d*N_a*(Policy(4,:,:,:,:)-1));


end
