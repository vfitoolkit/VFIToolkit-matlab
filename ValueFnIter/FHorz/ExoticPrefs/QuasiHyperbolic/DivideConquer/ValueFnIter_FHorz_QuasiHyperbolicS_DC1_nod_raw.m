function [Vunderbar,Policy,Vhat]=ValueFnIter_FHorz_QuasiHyperbolicS_DC1_nod_raw(n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Sophisticated quasi-hyperbolic discounting variant of ValueFnIter_FHorz_DC1_nod_raw.
% No d (decision) variables. Uses divide-and-conquer on a (GPU, parallel==2 only).
%
% DiscountFactorParamNames is the standard discount factor beta
% vfoptions.QHadditionaldiscount gives the name of beta_0, the additional discount factor parameter
%
% The 'Sophisticated' quasi-hyperbolic solution takes into account the time-inconsistent behaviour of their future self.
% Let Vunderbar_j be the exponential discounting value fn of the time-inconsistent policy function.
% Vhat_j = u_t + beta_0*beta*E[Vunderbar_{j+1}]
% Vunderbar_j = Vhat_j + (beta - beta_0*beta)*EV_at_optimal_aprime

N_a=prod(n_a);
N_z=prod(n_z);

Vhat=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray');  % optimal aprime index

%%
if vfoptions.lowmemory==0
    loweredgesize=[1,1,N_z];
elseif vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
end

zind=shiftdim(gpuArray((0:1:N_z-1)),-1);  % 1-by-N_z

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% j=N_j  (terminal period)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % No discounting at terminal period.
    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        [Vtempii,maxindex1]=max(ReturnMatrix_ii,[],1);
        Vhat(level1ii,:,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,:,N_j)=shiftdim(maxindex1,1);
        maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Vhat(curraindex,:,N_j)=shiftdim(Vtempii,1);
                Policy(curraindex,:,N_j)=shiftdim(maxindex+loweredge-1,1);
            else
                loweredge=maxindex1(1,ii,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, reshape(a_grid(loweredge),loweredgesize), a_grid(curraindex), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                Vhat(curraindex,:,N_j)=shiftdim(ReturnMatrix_ii,1);
                Policy(curraindex,:,N_j)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
            end
        end
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_z, a_grid, a_grid(level1ii), z_val, ReturnFnParamsVec,1);
            [Vtempii,maxindex1]=max(ReturnMatrix_ii,[],1);
            Vhat(level1ii,z_c,N_j)=shiftdim(Vtempii,1);
            Policy(level1ii,z_c,N_j)=shiftdim(maxindex1,1);
            maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_z, a_grid(aprimeindexes), a_grid(curraindex), z_val, ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    Vhat(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    Policy(curraindex,z_c,N_j)=shiftdim(maxindex+loweredge-1,1);
                else
                    loweredge=maxindex1(1,ii);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_z, a_grid(loweredge), a_grid(curraindex), z_val, ReturnFnParamsVec,2);
                    Vhat(curraindex,z_c,N_j)=shiftdim(ReturnMatrix_ii,1);
                    Policy(curraindex,z_c,N_j)=loweredge;
                end
            end
        end
    end

    Vunderbar=Vhat;

else
    % Using V_Jplus1 (should be Vunderbar for sophisticated)
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=reshape(vfoptions.V_Jplus1,[N_a,N_z]);
    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);   % N_a-by-1-by-N_z

    Vunderbar=zeros(N_a,N_z,N_j,'gpuArray');

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        % --- Vhat search (beta0*beta) ---
        entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
        [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
        Vhat(level1ii,:,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,:,N_j)=shiftdim(maxindex1,1);
        maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                aprimez=aprimeindexes+N_a*zind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV(aprimez),[(maxgap(ii)+1),1,N_z]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat(curraindex,:,N_j)=shiftdim(Vtempii,1);
                Policy(curraindex,:,N_j)=shiftdim(maxindex+loweredge-1,1);
            else
                loweredge=maxindex1(1,ii,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, reshape(a_grid(loweredge),loweredgesize), a_grid(curraindex), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                aprimez=loweredge+N_a*zind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV(aprimez),[1,1,N_z]);
                Vhat(curraindex,:,N_j)=shiftdim(entireRHS_ii,1);
                Policy(curraindex,:,N_j)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
            end
        end
        % Vunderbar = Vhat + (beta - beta0*beta)*EV_at_optimal_aprime
        aprime_ind=Policy(:,:,N_j);   % N_a-by-N_z
        EV_at_policy=EV(aprime_ind+N_a*zind);
        Vunderbar(:,:,N_j)=Vhat(:,:,N_j)+(beta-beta0beta)*EV_at_policy;

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            EV_z=EV(:,:,z_c);

            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_z, a_grid, a_grid(level1ii), z_val, ReturnFnParamsVec,1);

            % --- Vhat search (beta0*beta) ---
            entireRHS_ii=ReturnMatrix_ii+beta0beta*EV_z;
            [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
            Vhat(level1ii,z_c,N_j)=shiftdim(Vtempii,1);
            Policy(level1ii,z_c,N_j)=shiftdim(maxindex1,1);
            maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_z, a_grid(aprimeindexes), a_grid(curraindex), z_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*EV_z(aprimeindexes);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vhat(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    Policy(curraindex,z_c,N_j)=shiftdim(maxindex+loweredge-1,1);
                else
                    loweredge=maxindex1(1,ii);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_z, a_grid(loweredge), a_grid(curraindex), z_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*EV_z(loweredge);
                    Vhat(curraindex,z_c,N_j)=shiftdim(entireRHS_ii,1);
                    Policy(curraindex,z_c,N_j)=loweredge;
                end
            end
            % Vunderbar per z
            aprime_ind_z=Policy(:,z_c,N_j);
            EV_at_policy_z=EV_z(aprime_ind_z);
            Vunderbar(:,z_c,N_j)=Vhat(:,z_c,N_j)+(beta-beta0beta)*EV_at_policy_z;
        end
    end
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    EVsource=Vunderbar(:,:,jj+1);
    EV=EVsource.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);   % N_a-by-1-by-N_z

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid, a_grid(level1ii), z_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
        [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
        Vhat(level1ii,:,jj)=shiftdim(Vtempii,1);
        Policy(level1ii,:,jj)=shiftdim(maxindex1,1);
        maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                aprimez=aprimeindexes+N_a*zind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV(aprimez),[(maxgap(ii)+1),1,N_z]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat(curraindex,:,jj)=shiftdim(Vtempii,1);
                Policy(curraindex,:,jj)=shiftdim(maxindex+loweredge-1,1);
            else
                loweredge=maxindex1(1,ii,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, reshape(a_grid(loweredge),loweredgesize), a_grid(curraindex), z_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                aprimez=loweredge+N_a*zind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV(aprimez),[1,1,N_z]);
                Vhat(curraindex,:,jj)=shiftdim(entireRHS_ii,1);
                Policy(curraindex,:,jj)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
            end
        end
        aprime_ind=Policy(:,:,jj);
        EV_at_policy=EV(aprime_ind+N_a*zind);
        Vunderbar(:,:,jj)=Vhat(:,:,jj)+(beta-beta0beta)*EV_at_policy;

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            EV_z=EV(:,:,z_c);

            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_z, a_grid, a_grid(level1ii), z_val, ReturnFnParamsVec,1);

            entireRHS_ii=ReturnMatrix_ii+beta0beta*EV_z;
            [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
            Vhat(level1ii,z_c,jj)=shiftdim(Vtempii,1);
            Policy(level1ii,z_c,jj)=shiftdim(maxindex1,1);
            maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_z, a_grid(aprimeindexes), a_grid(curraindex), z_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*EV_z(aprimeindexes);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vhat(curraindex,z_c,jj)=shiftdim(Vtempii,1);
                    Policy(curraindex,z_c,jj)=shiftdim(maxindex+loweredge-1,1);
                else
                    loweredge=maxindex1(1,ii);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_z, a_grid(loweredge), a_grid(curraindex), z_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*EV_z(loweredge);
                    Vhat(curraindex,z_c,jj)=shiftdim(entireRHS_ii,1);
                    Policy(curraindex,z_c,jj)=loweredge;
                end
            end
            aprime_ind_z=Policy(:,z_c,jj);
            EV_at_policy_z=EV_z(aprime_ind_z);
            Vunderbar(:,z_c,jj)=Vhat(:,z_c,jj)+(beta-beta0beta)*EV_at_policy_z;
        end
    end
end

end
