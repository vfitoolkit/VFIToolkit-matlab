function [V,Policy]=ValueFnIter_Case2_FHorz_QuasiHyperbolic_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

eval('fieldexists_ExogShockFn=1;vfoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;vfoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); %indexes the optimal choice for d given rest of dimensions a,z

%%
if vfoptions.lowmemory>0 || Case2_Type==11
    special_n_z=ones(1,length(n_z));
    z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
end
if vfoptions.lowmemory>1
    special_n_a=ones(1,length(n_a));
    a_gridvals=CreateGridvals(n_a,a_grid,1); % The 1 at end indicates want output in form of matrix.
end

%%

%% j=N_j

if vfoptions.verbose==1
    sprintf('Age j is currently %i \n',N_j)
end


% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if fieldexists_ExogShockFn==1
    if fieldexists_ExogShockFnParamNames==1
        ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,N_j);
        ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
        for ii=1:length(ExogShockFnParamsVec)
            ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
        end
        [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
        z_grid=gpuArray(z_grid); pi_z=gpuArray(z_grid);
    else
        [z_grid,pi_z]=vfoptions.ExogShockFn(N_j);
        z_grid=gpuArray(z_grid); pi_z=gpuArray(z_grid);
    end
end

if vfoptions.lowmemory==0
    
    %if vfoptions.returnmatrix==2 % GPU
    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,:,N_j)=Vtemp;
    Policy(:,:,N_j)=maxindex;
    
elseif vfoptions.lowmemory==1
    
    %if vfoptions.returnmatrix==2 % GPU
    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_grid, a_grid, z_val, ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
        V(:,z_c,N_j)=Vtemp;
        Policy(:,z_c,N_j)=maxindex;
    end
    
elseif vfoptions.lowmemory==2
    
    %if vfoptions.returnmatrix==2 % GPU
    for a_c=1:N_a
        a_val=a_gridvals(a_c,:);
        ReturnMatrix_a=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, special_n_a, n_z, d_grid, a_val, z_grid, ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix_a,[],1);
        V(a_c,:,N_j)=Vtemp;
        Policy(a_c,:,N_j)=maxindex; 
    end
    
end

if strcmp(vfoptions.quasi_hyperbolic,'Naive')
    Vtilde=V;
else % strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
    Vunderbar=V;
end

%%
if Case2_Type==1 % phi_a'(d,a,z,z')
    if vfoptions.phiaprimedependsonage==0
        PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
        if vfoptions.lowmemory==0
            Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
        end
    end
    
    for reverse_j=1:N_j-1
        jj=N_j-reverse_j;
        
        if vfoptions.verbose==1
            sprintf('Age j is currently %i \n',jj)
        end
        
        % Create a vector containing all the return function parameters (in order)
        ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
        if length(DiscountFactorParamsVec)>2
            DiscountFactorParamsVec=[prod(DiscountFactorParamsVec(1:end-1));DiscountFactorParamsVec(end)];
        end
        beta=prod(DiscountFactorParamsVec(1:end-1)); % Discount factor between any two future periods
        beta0beta=prod(DiscountFactorParamsVec); % Discount factor between today and tomorrow.
        
        if fieldexists_ExogShockFn==1
            if fieldexists_ExogShockFnParamNames==1
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
                z_grid=gpuArray(z_grid); pi_z=gpuArray(z_grid);
            else
                [z_grid,pi_z]=vfoptions.ExogShockFn(jj);
                z_grid=gpuArray(z_grid); pi_z=gpuArray(z_grid);
            end
        end
        
        if strcmp(vfoptions.quasi_hyperbolic,'Naive')
            VKronNext_j=V(:,:,jj+1); % Use V (goes into the equation to determine V)
        else % strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
            VKronNext_j=Vunderbar(:,:,jj+1); % Use Vunderbar (goes into the equation to determine Vhat)
        end
    
        if vfoptions.lowmemory==0
            if vfoptions.phiaprimedependsonage==1
                PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
                Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
            end
            
            ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
            
            %        Phi_aprimeKron=Phi_aprimeKronFn_j(j);
            for z_c=1:N_z
                for a_c=1:N_a
                    entireEV_z=zeros(N_d,1);
                    for zprime_c=1:N_z
                        if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                            for d_c=1:N_d
                                entireEV_z(d_c)=entireEV_z(d_c)+VKronNext_j(Phi_aprimeMatrix(d_c,a_c,z_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                            end
                        end
                    end
                    
                    entireEV_z=kron(EV_z,ones(N_d,1));
            
                    if strcmp(vfoptions.quasi_hyperbolic,'Naive')
                        % For naive, we compue V which is the exponential
                        % discounter case, and then from this we get Vtilde and
                        % Policy (which is Policytilde) that correspond to the
                        % naive quasihyperbolic discounter
                        % First V
                        entireRHS=ReturnMatrix(:,a_c,z_c)+beta*entireEV_z; % Use the two-future-periods discount factor
                        [Vtemp,~]=max(entireRHS,[],1);
                        V(a_c,z_c,jj)=Vtemp;
                        % Now Vtilde and Policy
                        entireRHS=ReturnMatrix(:,a_c,z_c)+beta0beta*entireEV_z;
                        [Vtemp,maxindex]=max(entireRHS,[],1);
                        Vtilde(a_c,z_c,jj)=Vtemp; % Evaluate what would have done under exponential discounting
                        Policy(a_c,z_c,jj)=maxindex; % Use the policy from solving the problem of Vtilde
                    elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
                        % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
                        % and then we compute Vunderbar.
                        % First Vhat
                        entireRHS=ReturnMatrix(:,a_c,z_c)+beta0beta*entireEV_z;  % Use the today-to-tomorrow discount factor
                        [Vtemp,maxindex]=max(entireRHS,[],1);
                        V(a_c,z_c,jj)=Vtemp; % Note that this is Vhat when sophisticated
                        Policy(a_c,z_c,jj)=maxindex; % This is the policy from solving the problem of Vhat
                        % Now Vstar
                        entireRHS=ReturnMatrix(:,a_c,z_c)+beta*entireEV_z; % Use the two-future-periods discount factor
                        Vunderbar(a_c,z_c,jj)=entireRHS(maxindex); % Evaluate time-inconsistent policy using two-future-periods discount rate
                    end
                end
            end
        elseif vfoptions.lowmemory==1
            for z_c=1:N_z
                z_val=z_gridvals(z_c,:);
                
                if vfoptions.phiaprimedependsonage==1
                    PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
                end
                Phi_aprimeMatrix_z=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
                ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_grid, a_grid, z_val, ReturnFnParamsVec);
                for a_c=1:N_a
                    entireEV_z=zeros(N_d,1);
                    for zprime_c=1:N_z
                        if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                            for d_c=1:N_d
                                entireEV_z(d_c)=entireEV_z(d_c)+VKronNext_j(Phi_aprimeMatrix_z(d_c,a_c,1,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                            end
                        end
                    end
                    
                    if strcmp(vfoptions.quasi_hyperbolic,'Naive')
                        % For naive, we compue V which is the exponential
                        % discounter case, and then from this we get Vtilde and
                        % Policy (which is Policytilde) that correspond to the
                        % naive quasihyperbolic discounter
                        % First V
                        entireRHS=ReturnMatrix_z(:,a_c)+beta*entireEV_z; % Use the two-future-periods discount factor
                        [Vtemp,~]=max(entireRHS,[],1);
                        V(a_c,z_c,jj)=Vtemp;
                        % Now Vtilde and Policy
                        entireRHS=ReturnMatrix_z(:,a_c)+beta0beta*entireEV_z;
                        [Vtemp,maxindex]=max(entireRHS,[],1);
                        Vtilde(a_c,z_c,jj)=Vtemp; % Evaluate what would have done under exponential discounting
                        Policy(a_c,z_c,jj)=maxindex; % Use the policy from solving the problem of Vtilde
                    elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
                        % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
                        % and then we compute Vunderbar.
                        % First Vhat
                        entireRHS=ReturnMatrix_z(:,a_c)+beta0beta*entireEV_z;  % Use the today-to-tomorrow discount factor
                        [Vtemp,maxindex]=max(entireRHS,[],1);
                        V(a_c,z_c,jj)=Vtemp; % Note that this is Vhat when sophisticated
                        Policy(a_c,z_c,jj)=maxindex; % This is the policy from solving the problem of Vhat
                        % Now Vstar
                        entireRHS=ReturnMatrix_z(:,a_c)+beta*entireEV_z; % Use the two-future-periods discount factor
                        Vunderbar(a_c,z_c,jj)=entireRHS(maxindex); % Evaluate time-inconsistent policy using two-future-periods discount rate
                    end
                end
            end
%         elseif vfoptions.lowmemory==2 % NOT YET IMPLEMENTED
%             for a_c=1:N_a
%                 if vfoptions.phiaprimedependsonage==1
%                     PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
%                 end
%                 Phi_aprimeMatrix_a=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, special_n_a, n_z, d_grid, a_val, z_grid,PhiaprimeParamsVec);
%                 ReturnMatrix_a=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, special_n_a,n_z, d_grid, a_val, z_grid, ReturnFnParamsVec);
%                 for z_c=1:N_z
%                     entireEV_z=zeros(N_d,1);
%                     for zprime_c=1:N_z
%                         if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%                             for d_c=1:N_d
%                                 entireEV_z(d_c)=entireEV_z(d_c)+VKronNext_j(Phi_aprimeMatrix_a(d_c,1,z_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
%                             end
%                         end
%                     end
%                     entireRHS=ReturnMatrix_a(:,1,z_c)+beta*entireEV_z; %aprime by 1
%                     
%                     %calculate in order, the maximizing aprime indexes
%                     [Vtemp,maxindex]=max(entireRHS,[],1);
%                     V(a_c,z_c,jj)=Vtemp;
%                     Policy(a_c,z_c,jj)=maxindex;
%                 end
            end
        end
    end
end

if Case2_Type==11 % phi_a'(d,a,z')
    if vfoptions.phiaprimedependsonage==0
        PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
        if vfoptions.lowmemory==0
            Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
        end
    end
    
    for reverse_j=1:N_j-1
        jj=N_j-reverse_j;
        
        if vfoptions.verbose==1
            sprintf('Age j is currently %i \n',jj)
        end
        
        % Create a vector containing all the return function parameters (in order)
        ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
        if length(DiscountFactorParamsVec)>2
            DiscountFactorParamsVec=[prod(DiscountFactorParamsVec(1:end-1));DiscountFactorParamsVec(end)];
        end
        beta=prod(DiscountFactorParamsVec(1:end-1)); % Discount factor between any two future periods
        beta0beta=prod(DiscountFactorParamsVec); % Discount factor between today and tomorrow.

        
        if fieldexists_ExogShockFn==1
            if fieldexists_ExogShockFnParamNames==1
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
                z_grid=gpuArray(z_grid); pi_z=gpuArray(z_grid);
            else
                [z_grid,pi_z]=vfoptions.ExogShockFn(jj);
                z_grid=gpuArray(z_grid); pi_z=gpuArray(z_grid);
            end
        end
        
        if strcmp(vfoptions.quasi_hyperbolic,'Naive')
            VKronNext_j=V(:,:,jj+1); % Use V (goes into the equation to determine V)
        else % strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
            VKronNext_j=Vunderbar(:,:,jj+1); % Use Vunderbar (goes into the equation to determine Vhat)
        end
    
        if vfoptions.lowmemory==0 % CAN PROBABLY IMPROVE THIS (lowmemory=0 is just doing same as lowmemory=1 for Case2_Type=11)
            % Current Case2_Type=11: phi_a'(d,a,z') 
            if vfoptions.phiaprimedependsonage==1
                PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
            end
            Phi_aprimeMatrix_z=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec); %z_grid as doing all of zprime (note that it is independent of z as Case2_Type=11)
            Phi_aprimeMatrix_z=reshape(Phi_aprimeMatrix_z,[N_d*N_a*N_z,1]);
            aaaVnextj=kron(ones(N_d,1),VKronNext_j);
            for z_c=1:N_z
                z_val=z_gridvals(z_c,:); % Value of z (not of z')
                aaa=kron(pi_z(z_c,:),ones(N_d*N_a,1,'gpuArray'));
              
                zprime_ToMatchPhi=kron((1:1:N_z)',ones(N_d*N_a,1));
              
                ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_grid, a_grid, z_val, ReturnFnParamsVec);

                EV_z=aaaVnextj(Phi_aprimeMatrix_z+(N_d*N_a)*(zprime_ToMatchPhi-1)); %*aaa;
                
                EV_z=reshape(EV_z,[N_d*N_a,N_z]);
                EV_z=EV_z.*aaa;
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=reshape(sum(EV_z,2),[N_d,N_a]);
                
                
                if strcmp(vfoptions.quasi_hyperbolic,'Naive')
                    % For naive, we compue V which is the exponential
                    % discounter case, and then from this we get Vtilde and
                    % Policy (which is Policytilde) that correspond to the
                    % naive quasihyperbolic discounter
                    % First V
                    entireRHS=ReturnMatrix_z+beta*EV_z;  %d by a (by z)  % Use the two-future-periods discount factor
                    [Vtemp,~]=max(entireRHS,[],1);
                    V(:,z_c,jj)=Vtemp;
                    % Now Vtilde and Policy
                    entireRHS=ReturnMatrix_z+beta0beta*EV_z;
                    [Vtemp,maxindex]=max(entireRHS,[],1);
                    Vtilde(:,z_c,jj)=Vtemp; % Evaluate what would have done under exponential discounting
                    Policy(:,z_c,jj)=maxindex; % Use the policy from solving the problem of Vtilde
                elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
                    % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
                    % and then we compute Vunderbar.
                    % First Vhat
                    entireRHS=ReturnMatrix_z+beta0beta*EV_z;  % Use the today-to-tomorrow discount factor
                    [Vtemp,maxindex]=max(entireRHS,[],1);
                    V(:,z_c,jj)=Vtemp; % Note that this is Vhat when sophisticated
                    Policy(:,z_c,jj)=maxindex; % This is the policy from solving the problem of Vhat
                    % Now Vstar
                    entireRHS=ReturnMatrix_z+beta*EV_z; % Use the two-future-periods discount factor
                    Vunderbar(:,z_c,jj)=entireRHS(maxindex); % Evaluate time-inconsistent policy using two-future-periods discount rate
                end
            end
        elseif vfoptions.lowmemory==1
            % Current Case2_Type=11: phi_a'(d,a,z') 
            if vfoptions.phiaprimedependsonage==1
                PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
            end
            Phi_aprimeMatrix_z=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec); %z_grid as doing all of zprime (note that it is independent of z as Case2_Type=11)
            Phi_aprimeMatrix_z=reshape(Phi_aprimeMatrix_z,[N_d*N_a*N_z,1]);
            aaaVnextj=kron(ones(N_d,1),VKronNext_j);
            for z_c=1:N_z
                z_val=z_gridvals(z_c,:); % Value of z (not of z')
                aaa=kron(pi_z(z_c,:),ones(N_d*N_a,1,'gpuArray'));
                
                zprime_ToMatchPhi=kron((1:1:N_z)',ones(N_d*N_a,1));
                
                ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_grid, a_grid, z_val, ReturnFnParamsVec);

                EV_z=aaaVnextj(Phi_aprimeMatrix_z+(N_d*N_a)*(zprime_ToMatchPhi-1)); %*aaa;
                
                EV_z=reshape(EV_z,[N_d*N_a,N_z]);
                EV_z=EV_z.*aaa;
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=reshape(sum(EV_z,2),[N_d,N_a]);
                
                if strcmp(vfoptions.quasi_hyperbolic,'Naive')
                    % For naive, we compue V which is the exponential
                    % discounter case, and then from this we get Vtilde and
                    % Policy (which is Policytilde) that correspond to the
                    % naive quasihyperbolic discounter
                    % First V
                    entireRHS=ReturnMatrix_z+beta*EV_z;  %d by a (by z)  % Use the two-future-periods discount factor
                    [Vtemp,~]=max(entireRHS,[],1);
                    V(:,z_c,jj)=Vtemp;
                    % Now Vtilde and Policy
                    entireRHS=ReturnMatrix_z+beta0beta*EV_z;
                    [Vtemp,maxindex]=max(entireRHS,[],1);
                    Vtilde(:,z_c,jj)=Vtemp; % Evaluate what would have done under exponential discounting
                    Policy(:,z_c,jj)=maxindex; % Use the policy from solving the problem of Vtilde
                elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
                    % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
                    % and then we compute Vunderbar.
                    % First Vhat
                    entireRHS=ReturnMatrix_z+beta0beta*EV_z;  % Use the today-to-tomorrow discount factor
                    [Vtemp,maxindex]=max(entireRHS,[],1);
                    V(:,z_c,jj)=Vtemp; % Note that this is Vhat when sophisticated
                    Policy(:,z_c,jj)=maxindex; % This is the policy from solving the problem of Vhat
                    % Now Vstar
                    entireRHS=ReturnMatrix_z+beta*EV_z; % Use the two-future-periods discount factor
                    Vunderbar(:,z_c,jj)=entireRHS(maxindex); % Evaluate time-inconsistent policy using two-future-periods discount rate
                end
            end
%         elseif vfoptions.lowmemory==2 % NOT YET IMPLEMENTED
%             for a_c=1:N_a
%                 if vfoptions.phiaprimedependsonage==1
%                     PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
%                 end
%                 Phi_aprimeMatrix_a=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, special_n_a, n_z, d_grid, a_val, z_grid,PhiaprimeParamsVec);
%                 ReturnMatrix_a=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, special_n_a,n_z, d_grid, a_val, z_grid, ReturnFnParamsVec);
%                 for z_c=1:N_z
%                     entireEV_z=zeros(N_d,1);
%                     for zprime_c=1:N_z
%                         if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%                             for d_c=1:N_d
%                                 entireEV_z(d_c)=entireEV_z(d_c)+VKronNext_j(Phi_aprimeMatrix_a(d_c,1,z_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
%                             end
%                         end
%                     end
%                     entireRHS=ReturnMatrix_a(:,1,z_c)+beta*entireEV_z; %aprime by 1
%                     
%                     %calculate in order, the maximizing aprime indexes
%                     [Vtemp,maxindex]=max(entireRHS,[],1);
%                     V(a_c,z_c,jj)=Vtemp;
%                     Policy(a_c,z_c,jj)=maxindex;
%                 end
            end
        end
    end
end

if Case2_Type==12 % phi_a'(d,a,z)
    if vfoptions.phiaprimedependsonage==0
        PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
        if vfoptions.lowmemory==0
            Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
        end
    end
    
    aaa=kron(pi_z,ones(N_d,1,'gpuArray'));
    
    for reverse_j=1:N_j-1
        jj=N_j-reverse_j;
        
        if vfoptions.verbose==1
            sprintf('Age j is currently %i \n',jj)
        end
        
        % Create a vector containing all the return function parameters (in order)
        ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
        if length(DiscountFactorParamsVec)>2
            DiscountFactorParamsVec=[prod(DiscountFactorParamsVec(1:end-1));DiscountFactorParamsVec(end)];
        end
        beta=prod(DiscountFactorParamsVec(1:end-1)); % Discount factor between any two future periods
        beta0beta=prod(DiscountFactorParamsVec); % Discount factor between today and tomorrow.
        
        if fieldexists_ExogShockFn==1
            if fieldexists_ExogShockFnParamNames==1
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
                z_grid=gpuArray(z_grid); pi_z=gpuArray(z_grid);
            else
                [z_grid,pi_z]=vfoptions.ExogShockFn(jj);
                z_grid=gpuArray(z_grid); pi_z=gpuArray(z_grid);
            end
        end
        
        if strcmp(vfoptions.quasi_hyperbolic,'Naive')
            VKronNext_j=V(:,:,jj+1); % Use V (goes into the equation to determine V)
        else % strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
            VKronNext_j=Vunderbar(:,:,jj+1); % Use Vunderbar (goes into the equation to determine Vhat)
        end
    
        if vfoptions.lowmemory==0
            if vfoptions.phiaprimedependsonage==1
                PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
                Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
            end
            
            %if vfoptions.returnmatrix==2 % GPU
            ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);

            for z_c=1:N_z
                for a_c=1:N_a
                    EV_az=VKronNext_j(Phi_aprimeMatrix(:,a_c,z_c),:); %(d,z')
                    EV_az=EV_az.*aaa;
                    EV_az(isnan(EV_az))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                    EV_az=sum(EV_az,2); % reshape(sum(EV_az,2),[N_d,1,1]);
                    
                    if strcmp(vfoptions.quasi_hyperbolic,'Naive')
                        % For naive, we compue V which is the exponential
                        % discounter case, and then from this we get Vtilde and
                        % Policy (which is Policytilde) that correspond to the
                        % naive quasihyperbolic discounter
                        % First V
                        entireRHS=ReturnMatrix(:,a_c,z_c)+beta*EV_az; %aprime by 1 % Use the two-future-periods discount factor
                        [Vtemp,~]=max(entireRHS,[],1);
                        V(a_c,z_c,jj)=Vtemp;
                        % Now Vtilde and Policy
                        entireRHS=ReturnMatrix(:,a_c,z_c)+beta0beta*EV_az;
                        [Vtemp,maxindex]=max(entireRHS,[],1);
                        Vtilde(a_c,z_c,jj)=Vtemp; % Evaluate what would have done under exponential discounting
                        Policy(a_c,z_c,jj)=maxindex; % Use the policy from solving the problem of Vtilde
                    elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
                        % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
                        % and then we compute Vunderbar.
                        % First Vhat
                        entireRHS=ReturnMatrix(:,a_c,z_c)+beta0beta*EV_az;  % Use the today-to-tomorrow discount factor
                        [Vtemp,maxindex]=max(entireRHS,[],1);
                        V(a_c,z_c,jj)=Vtemp; % Note that this is Vhat when sophisticated
                        Policy(a_c,z_c,jj)=maxindex; % This is the policy from solving the problem of Vhat
                        % Now Vstar
                        entireRHS=ReturnMatrix(:,a_c,z_c)+beta*EV_az; % Use the two-future-periods discount factor
                        Vunderbar(a_c,z_c,jj)=entireRHS(maxindex); % Evaluate time-inconsistent policy using two-future-periods discount rate
                    end
                end
            end
            
        elseif vfoptions.lowmemory==1
            for z_c=1:N_z
                z_val=z_gridvals(z_c,:);
                
                if vfoptions.phiaprimedependsonage==1
                    PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
                    Phi_aprimeMatrix_z=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, special_n_z, d_grid, a_grid, z_val,PhiaprimeParamsVec);
                end
                ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_grid, a_grid, z_val, ReturnFnParamsVec);
                for a_c=1:N_a
                    entireEV_z=zeros(N_d,1,'gpuArray');
                    for zprime_c=1:N_z                        
                        if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                            for d_c=1:N_d
                                entireEV_z(d_c)=entireEV_z(d_c)+VKronNext_j(Phi_aprimeMatrix_z(d_c,a_c,1),zprime_c)*pi_z(z_c,zprime_c);
                            end
                        end
                    end
                    
                    
                    if strcmp(vfoptions.quasi_hyperbolic,'Naive')
                        % For naive, we compue V which is the exponential
                        % discounter case, and then from this we get Vtilde and
                        % Policy (which is Policytilde) that correspond to the
                        % naive quasihyperbolic discounter
                        % First V
                        entireRHS=ReturnMatrix_z(:,a_c)+beta*entireEV_z; %aprime by 1 % Use the two-future-periods discount factor
                        [Vtemp,~]=max(entireRHS,[],1);
                        V(a_c,z_c,jj)=Vtemp;
                        % Now Vtilde and Policy
                        entireRHS=ReturnMatrix_z(:,a_c)+beta0beta*entireEV_z;
                        [Vtemp,maxindex]=max(entireRHS,[],1);
                        Vtilde(a_c,z_c,jj)=Vtemp; % Evaluate what would have done under exponential discounting
                        Policy(a_c,z_c,jj)=maxindex; % Use the policy from solving the problem of Vtilde
                    elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
                        % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
                        % and then we compute Vunderbar.
                        % First Vhat
                        entireRHS=ReturnMatrix_z(:,a_c)+beta0beta*entireEV_z;  % Use the today-to-tomorrow discount factor
                        [Vtemp,maxindex]=max(entireRHS,[],1);
                        V(a_c,z_c,jj)=Vtemp; % Note that this is Vhat when sophisticated
                        Policy(a_c,z_c,jj)=maxindex; % This is the policy from solving the problem of Vhat
                        % Now Vstar
                        entireRHS=ReturnMatrix_z(:,a_c)+beta*entireEV_z; % Use the two-future-periods discount factor
                        Vunderbar(a_c,z_c,jj)=entireRHS(maxindex); % Evaluate time-inconsistent policy using two-future-periods discount rate
                    end
                end
            end
%         elseif vfoptions.lowmemory==2 % NOT YET IMPLEMENTED
%             for a_c=1:N_a
%                 if vfoptions.phiaprimedependsonage==1
%                     PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
%                 end
%                 Phi_aprimeMatrix_a=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, special_n_a, n_z, d_grid, a_val, z_grid,PhiaprimeParamsVec);
%                 ReturnMatrix_a=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, special_n_a,n_z, d_grid, a_val, z_grid, ReturnFnParamsVec);
%                 for z_c=1:N_z
%                     entireEV_z=zeros(N_d,1);
%                     for zprime_c=1:N_z
%                         if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%                             for d_c=1:N_d
%                                 entireEV_z(d_c)=entireEV_z(d_c)+VKronNext_j(Phi_aprimeMatrix_a(d_c,1,z_c),zprime_c)*pi_z(z_c,zprime_c);
%                             end
%                         end
%                     end
%                     entireRHS=ReturnMatrix_a(:,1,z_c)+beta*entireEV_z; %aprime by 1
%                     
%                     %calculate in order, the maximizing aprime indexes
%                     [Vtemp,maxindex]=max(entireRHS,[],1);
%                     V(a_c,z_c,jj)=Vtemp;
%                     Policy(a_c,z_c,jj)=maxindex;
%                 end
%             end
        end
    end
end

if Case2_Type==2  % phi_a'(d,z,z')
    
    aaa=kron(pi_z,ones(N_d,1,'gpuArray'));
    
    if vfoptions.phiaprimedependsonage==0
        PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
        if vfoptions.lowmemory==0
            Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
        end
    end
    
    for reverse_j=1:N_j-1
        jj=N_j-reverse_j;
        
        if vfoptions.verbose==1
            sprintf('Age j is currently %i \n',jj)
        end
        
        % Create a vector containing all the return function parameters (in order)
        ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
        if length(DiscountFactorParamsVec)>2
            DiscountFactorParamsVec=[prod(DiscountFactorParamsVec(1:end-1));DiscountFactorParamsVec(end)];
        end
        beta=prod(DiscountFactorParamsVec(1:end-1)); % Discount factor between any two future periods
        beta0beta=prod(DiscountFactorParamsVec); % Discount factor between today and tomorrow.

        
        if fieldexists_ExogShockFn==1
            if fieldexists_ExogShockFnParamNames==1
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
                z_grid=gpuArray(z_grid); pi_z=gpuArray(z_grid);
            else
                [z_grid,pi_z]=vfoptions.ExogShockFn(jj);
                z_grid=gpuArray(z_grid); pi_z=gpuArray(z_grid);
            end
        end
        
        if strcmp(vfoptions.quasi_hyperbolic,'Naive')
            VKronNext_j=V(:,:,jj+1); % Use V (goes into the equation to determine V)
        else % strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
            VKronNext_j=Vunderbar(:,:,jj+1); % Use Vunderbar (goes into the equation to determine Vhat)
        end
        
        if vfoptions.phiaprimedependsonage==1
            PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
            Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
        end
        
        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);

        EV=zeros(N_d*N_z,N_z,'gpuArray');
        for zprime_c=1:N_z
            EV(:,zprime_c)=VKronold(Phi_aprime(:,:,zprime_c),zprime_c); %(d,z')
        end
        EV=EV.*aaa;
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=reshape(sum(EV,2),[N_d,1,N_z]);
        
        for z_c=1:N_z % Can probably eliminate this loop and replace with a matrix multiplication operation thereby making it faster
            if strcmp(vfoptions.quasi_hyperbolic,'Naive')
                % For naive, we compue V which is the exponential
                % discounter case, and then from this we get Vtilde and
                % Policy (which is Policytilde) that correspond to the
                % naive quasihyperbolic discounter
                % First V
                entireRHS=ReturnMatrix(:,:,z_c)+beta*EV(:,z_c)*ones(1,N_a,1,'gpuArray'); %aprime by 1 % Use the two-future-periods discount factor
                [Vtemp,~]=max(entireRHS,[],1);
                V(:,z_c,jj)=Vtemp;
                % Now Vtilde and Policy
                entireRHS=ReturnMatrix(:,:,z_c)+beta0beta*EV(:,z_c)*ones(1,N_a,1,'gpuArray');
                [Vtemp,maxindex]=max(entireRHS,[],1);
                Vtilde(:,z_c,jj)=Vtemp; % Evaluate what would have done under exponential discounting
                Policy(:,z_c,jj)=maxindex; % Use the policy from solving the problem of Vtilde
            elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
                % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
                % and then we compute Vunderbar.
                % First Vhat
                entireRHS=ReturnMatrix(:,:,z_c)+beta0beta*EV(:,z_c)*ones(1,N_a,1,'gpuArray');  % Use the today-to-tomorrow discount factor
                [Vtemp,maxindex]=max(entireRHS,[],1);
                V(:,z_c,jj)=Vtemp; % Note that this is Vhat when sophisticated
                Policy(:,z_c,jj)=maxindex; % This is the policy from solving the problem of Vhat
                % Now Vstar
                entireRHS=ReturnMatrix(:,:,z_c)+beta*EV(:,z_c)*ones(1,N_a,1,'gpuArray'); % Use the two-future-periods discount factor
                Vunderbar(:,z_c,jj)=entireRHS(maxindex); % Evaluate time-inconsistent policy using two-future-periods discount rate
            end
        end
    end
end

if Case2_Type==3  % phi_a'(d,z')
    if vfoptions.phiaprimedependsonage==0
        PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
        Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
    end
    for reverse_j=1:N_j-1
        jj=N_j-reverse_j;
        
        % Create a vector containing all the return function parameters (in order)
        ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
        if length(DiscountFactorParamsVec)>2
            DiscountFactorParamsVec=[prod(DiscountFactorParamsVec(1:end-1));DiscountFactorParamsVec(end)];
        end
        beta=prod(DiscountFactorParamsVec(1:end-1)); % Discount factor between any two future periods
        beta0beta=prod(DiscountFactorParamsVec); % Discount factor between today and tomorrow.

        
        if vfoptions.phiaprimedependsonage==1
            PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
            Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
        end
        
        if fieldexists_ExogShockFn==1
            if fieldexists_ExogShockFnParamNames==1
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
                [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsVec);
                z_grid=gpuArray(z_grid); pi_z=gpuArray(z_grid);
            else
                [z_grid,pi_z]=vfoptions.ExogShockFn(jj);
                z_grid=gpuArray(z_grid); pi_z=gpuArray(z_grid);
            end
        end
        
        if strcmp(vfoptions.quasi_hyperbolic,'Naive')
            VKronNext_j=V(:,:,jj+1); % Use V (goes into the equation to determine V)
        else % strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
            VKronNext_j=Vunderbar(:,:,jj+1); % Use Vunderbar (goes into the equation to determine Vhat)
        end
        
        if vfoptions.lowmemory==0
            ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
            for z_c=1:N_z
                
                EV_z=zeros(N_d,1);
                for zprime_c=1:N_z
                    if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                        for d_c=1:N_d
                            EV_z(d_c)=EV_z(d_c)+(VKronNext_j(:,zprime_c).*Phi_aprimeMatrix(:,d_c,zprime_c))*pi_z(z_c,zprime_c);
                        end
                    end
                end
                
                if strcmp(vfoptions.quasi_hyperbolic,'Naive')
                    % For naive, we compue V which is the exponential
                    % discounter case, and then from this we get Vtilde and
                    % Policy (which is Policytilde) that correspond to the
                    % naive quasihyperbolic discounter
                    % First V
                    entireRHS_z=ReturnMatrix(:,:,z_c)+beta*EV_z*ones(1,N_a,1,'gpuArray'); %aprime by 1 % Use the two-future-periods discount factor
                    [Vtemp,~]=max(entireRHS_z,[],1);
                    V(:,z_c,jj)=Vtemp;
                    % Now Vtilde and Policy
                    entireRHS_z=ReturnMatrix(:,:,z_c)+beta0beta*EV_z*ones(1,N_a,1,'gpuArray');
                    [Vtemp,maxindex]=max(entireRHS_z,[],1);
                    Vtilde(:,z_c,jj)=Vtemp; % Evaluate what would have done under exponential discounting
                    Policy(:,z_c,jj)=maxindex; % Use the policy from solving the problem of Vtilde
                elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
                    % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
                    % and then we compute Vunderbar.
                    % First Vhat
                    entireRHS_z=ReturnMatrix(:,:,z_c)+beta0beta*EV_z*ones(1,N_a,1,'gpuArray');  % Use the today-to-tomorrow discount factor
                    [Vtemp,maxindex]=max(entireRHS_z,[],1);
                    V(:,z_c,jj)=Vtemp; % Note that this is Vhat when sophisticated
                    Policy(:,z_c,jj)=maxindex; % This is the policy from solving the problem of Vhat
                    % Now Vstar
                    entireRHS_z=ReturnMatrix(:,:,z_c)+beta*EV_z*ones(1,N_a,1,'gpuArray'); % Use the two-future-periods discount factor
                    Vunderbar(:,z_c,jj)=entireRHS_z(maxindex); % Evaluate time-inconsistent policy using two-future-periods discount rate
                end
            end
        elseif vfoptions.lowmemory==1
            for z_c=1:N_z
                z_val=z_gridvals(z_c,:);
                ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_grid, a_grid, z_val, ReturnFnParamsVec);
                EV_z=zeros(N_d,1);
                for zprime_c=1:N_z
                    if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                        for d_c=1:N_d
                            EV_z(d_c)=EV_z(d_c)+VKronNext_j(Phi_aprimeMatrix(d_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                        end
                    end
                end

                if strcmp(vfoptions.quasi_hyperbolic,'Naive')
                    % For naive, we compue V which is the exponential
                    % discounter case, and then from this we get Vtilde and
                    % Policy (which is Policytilde) that correspond to the
                    % naive quasihyperbolic discounter
                    % First V
                    entireRHS_z=ReturnMatrix_z+beta*EV_z*ones(1,N_a,1,'gpuArray'); %aprime by 1 % Use the two-future-periods discount factor
                    [Vtemp,~]=max(entireRHS_z,[],1);
                    V(:,z_c,jj)=Vtemp;
                    % Now Vtilde and Policy
                    entireRHS_z=ReturnMatrix_z+beta0beta*EV_z*ones(1,N_a,1,'gpuArray');
                    [Vtemp,maxindex]=max(entireRHS_z,[],1);
                    Vtilde(:,z_c,jj)=Vtemp; % Evaluate what would have done under exponential discounting
                    Policy(:,z_c,jj)=maxindex; % Use the policy from solving the problem of Vtilde
                elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
                    % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
                    % and then we compute Vunderbar.
                    % First Vhat
                    entireRHS_z=ReturnMatrix_z+beta0beta*EV_z*ones(1,N_a,1,'gpuArray');  % Use the today-to-tomorrow discount factor
                    [Vtemp,maxindex]=max(entireRHS_z,[],1);
                    V(:,z_c,jj)=Vtemp; % Note that this is Vhat when sophisticated
                    Policy(:,z_c,jj)=maxindex; % This is the policy from solving the problem of Vhat
                    % Now Vstar
                    entireRHS_z=ReturnMatrix_z+beta*EV_z*ones(1,N_a,1,'gpuArray'); % Use the two-future-periods discount factor
                    Vunderbar(:,z_c,jj)=entireRHS_z(maxindex); % Evaluate time-inconsistent policy using two-future-periods discount rate
                end
            end
%         elseif vfoptions.lowmemory==2 % NOT YET IMPLEMENTED
%             EV_z=zeros(N_d,1);
%             for z_c=1:N_Z
%                 for zprime_c=1:N_z
%                     if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%                         for d_c=1:N_d
%                             EV_z(d_c)=EV_z(d_c)+VKronNext_j(Phi_aprimeMatrix(d_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
%                         end
%                     end
%                 end
%                 for a_c=1:N_a
%                     a_val=a_gridvals(z_c,:);
%                     ReturnMatrix_az=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, special_n_a, special_n_z, d_grid, a_val, z_val, ReturnFnParamsVec);
%                     
%                     entireRHS=ReturnMatrix_az+beta*EV_z; %aprime by 1
%                     
%                     %calculate in order, the maximizing aprime indexes
%                     [Vtemp,maxindex]=max(entireRHS,[],1);
%                     V(a_c,z_c,jj)=Vtemp;
%                     Policy(a_c,z_c,jj)=maxindex;
%                 end
            end
        end
    end
end

if Case2_Type==4  % phi_a'(d,a)
    PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
    Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
    aaa=kron(pi_z,ones(N_d,1,'gpuArray'));
    
    for reverse_j=1:N_j-1
        jj=N_j-reverse_j;
        
        if fieldexists_ExogShockFn==1
            if fieldexists_ExogShockFnParamNames==1
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
                z_grid=gpuArray(z_grid); pi_z=gpuArray(z_grid);
            else
                [z_grid,pi_z]=vfoptions.ExogShockFn(jj);
                z_grid=gpuArray(z_grid); pi_z=gpuArray(z_grid);
            end
        end
        
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid,ReturnFnParamsVec);
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
        if length(DiscountFactorParamsVec)>2
            DiscountFactorParamsVec=[prod(DiscountFactorParamsVec(1:end-1));DiscountFactorParamsVec(end)];
        end
        beta=prod(DiscountFactorParamsVec(1:end-1)); % Discount factor between any two future periods
        beta0beta=prod(DiscountFactorParamsVec); % Discount factor between today and tomorrow.
        
        if strcmp(vfoptions.quasi_hyperbolic,'Naive')
            VKronNext_j=V(:,:,jj+1); % Use V (goes into the equation to determine V)
        else % strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
            VKronNext_j=Vunderbar(:,:,jj+1); % Use Vunderbar (goes into the equation to determine Vhat)
        end

        EV=zeros(N_d*N_z,N_z,'gpuArray');
        for zprime_c=1:N_z
            EV(:,zprime_c)=VKronNext_j(Phi_aprimeMatrix(:,zprime_c)*ones(1,N_z),zprime_c); %(d,z')
        end
        EV=EV.*aaa;
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=reshape(sum(EV,2),[N_d,1,N_z]);
        
        for z_c=1:N_z % Can probably eliminate this loop and replace with a matrix multiplication operation thereby making it faster
            if strcmp(vfoptions.quasi_hyperbolic,'Naive')
                % For naive, we compue V which is the exponential
                % discounter case, and then from this we get Vtilde and
                % Policy (which is Policytilde) that correspond to the
                % naive quasihyperbolic discounter
                % First V
                entireRHS=ReturnMatrix(:,:,z_c)+beta*EV(:,z_c)*ones(1,N_a,1,'gpuArray'); %aprime by 1 % Use the two-future-periods discount factor
                [Vtemp,~]=max(entireRHS,[],1);
                V(:,z_c,jj)=Vtemp;
                % Now Vtilde and Policy
                entireRHS=ReturnMatrix(:,:,z_c)+beta0beta*EV(:,z_c)*ones(1,N_a,1,'gpuArray');
                [Vtemp,maxindex]=max(entireRHS,[],1);
                Vtilde(:,z_c,jj)=Vtemp; % Evaluate what would have done under exponential discounting
                Policy(:,z_c,jj)=maxindex; % Use the policy from solving the problem of Vtilde
            elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
                % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
                % and then we compute Vunderbar.
                % First Vhat
                entireRHS=ReturnMatrix(:,:,z_c)+beta0beta*EV(:,z_c)*ones(1,N_a,1,'gpuArray');  % Use the today-to-tomorrow discount factor
                [Vtemp,maxindex]=max(entireRHS,[],1);
                V(:,z_c,jj)=Vtemp; % Note that this is Vhat when sophisticated
                Policy(:,z_c,jj)=maxindex; % This is the policy from solving the problem of Vhat
                % Now Vstar
                entireRHS=ReturnMatrix(:,:,z_c)+beta*EV(:,z_c)*ones(1,N_a,1,'gpuArray'); % Use the two-future-periods discount factor
                Vunderbar(:,z_c,jj)=entireRHS(maxindex); % Evaluate time-inconsistent policy using two-future-periods discount rate
            end
        end
    end
end

if Case2_Type==6 % phi_a'(d)
    PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
    Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
    aaa=kron(pi_z,ones(N_d,1,'gpuArray'));
    
    for reverse_j=1:N_j-1
        jj=N_j-reverse_j;
        
        if fieldexists_ExogShockFn==1
            if fieldexists_ExogShockFnParamNames==1
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
                z_grid=gpuArray(z_grid); pi_z=gpuArray(z_grid);
            else
                [z_grid,pi_z]=vfoptions.ExogShockFn(jj);
                z_grid=gpuArray(z_grid); pi_z=gpuArray(z_grid);
            end
        end
        
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid,ReturnFnParamsVec);
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
        if length(DiscountFactorParamsVec)>2
            DiscountFactorParamsVec=[prod(DiscountFactorParamsVec(1:end-1));DiscountFactorParamsVec(end)];
        end
        beta=prod(DiscountFactorParamsVec(1:end-1)); % Discount factor between any two future periods
        beta0beta=prod(DiscountFactorParamsVec); % Discount factor between today and tomorrow.
        
        if strcmp(vfoptions.quasi_hyperbolic,'Naive')
            VKronNext_j=V(:,:,jj+1); % Use V (goes into the equation to determine V)
        else % strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
            VKronNext_j=Vunderbar(:,:,jj+1); % Use Vunderbar (goes into the equation to determine Vhat)
        end

        
        EV=zeros(N_d*N_z,N_z,'gpuArray');
        for zprime_c=1:N_z
            EV(:,zprime_c)=VKronNext_j(Phi_aprimeMatrix*ones(1,N_z),zprime_c); %(d,z')
        end
        EV=EV.*aaa;
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=reshape(sum(EV,2),[N_d,1,N_z]);
        
        for z_c=1:N_z % Can probably eliminate this loop and replace with a matrix multiplication operation thereby making it faster
            if strcmp(vfoptions.quasi_hyperbolic,'Naive')
                % For naive, we compue V which is the exponential
                % discounter case, and then from this we get Vtilde and
                % Policy (which is Policytilde) that correspond to the
                % naive quasihyperbolic discounter
                % First V
                entireRHS=ReturnMatrix(:,:,z_c)+beta*EV(:,z_c)*ones(1,N_a,1,'gpuArray'); %aprime by 1 % Use the two-future-periods discount factor
                [Vtemp,~]=max(entireRHS,[],1);
                V(:,z_c,jj)=Vtemp;
                % Now Vtilde and Policy
                entireRHS=ReturnMatrix(:,:,z_c)+beta0beta*EV(:,z_c)*ones(1,N_a,1,'gpuArray');
                [Vtemp,maxindex]=max(entireRHS,[],1);
                Vtilde(:,z_c,jj)=Vtemp; % Evaluate what would have done under exponential discounting
                Policy(:,z_c,jj)=maxindex; % Use the policy from solving the problem of Vtilde
            elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
                % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
                % and then we compute Vunderbar.
                % First Vhat
                entireRHS=ReturnMatrix(:,:,z_c)+beta0beta*EV(:,z_c)*ones(1,N_a,1,'gpuArray');  % Use the today-to-tomorrow discount factor
                [Vtemp,maxindex]=max(entireRHS,[],1);
                V(:,z_c,jj)=Vtemp; % Note that this is Vhat when sophisticated
                Policy(:,z_c,jj)=maxindex; % This is the policy from solving the problem of Vhat
                % Now Vstar
                entireRHS=ReturnMatrix(:,:,z_c)+beta*EV(:,z_c)*ones(1,N_a,1,'gpuArray'); % Use the two-future-periods discount factor
                Vunderbar(:,z_c,jj)=entireRHS(maxindex); % Evaluate time-inconsistent policy using two-future-periods discount rate
            end
        end
    end
end

end