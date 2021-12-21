function [V,Policy]=ValueFnIter_Case2_FHorz_Dynasty_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); %indexes the optimal choice for d given rest of dimensions a,z

%%

eval('fieldexists_pi_z_J=1;vfoptions.pi_z_J;','fieldexists_pi_z_J=0;')
eval('fieldexists_ExogShockFn=1;vfoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;vfoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')

if vfoptions.lowmemory>0
    special_n_z=ones(1,length(n_z));
    z_gridvals=CreateGridvals(n_z,z_grid,1); 
end
if vfoptions.lowmemory>1
    special_n_a=ones(1,length(n_a));
    a_gridvals=CreateGridvals(n_a,a_grid,1); 
end

%%
Vold=zeros(N_a,N_z,N_j);
tempcounter=1;
currdist=Inf;
while currdist>vfoptions.tolerance
    %%
    
    
    if Case2_Type==1 % phi_a'(d,a,z,z')
        PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
        Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
        
        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;
            
            % Create a vector containing all the return function parameters (in order)
            ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
            DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
            
            if fieldexists_pi_z_J==1
                z_grid=vfoptions.z_grid_J(:,jj);
                pi_z=vfoptions.pi_z_J(:,:,jj);
            elseif fieldexists_ExogShockFn==1
                if fieldexists_ExogShockFnParamNames==1
                    ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for ii=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                    end
                    [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
                    z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
                else
                    [z_grid,pi_z]=vfoptions.ExogShockFn(jj);
                    z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
                end
            end
            
            if reverse_j==0 % So j==N_j
                VKronNext_j=V(:,:,1);
            else
                VKronNext_j=V(:,:,jj+1);
            end
            
            if vfoptions.lowmemory==0
                
                %if vfoptions.returnmatrix==2 % GPU
                ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
                %        FmatrixKron_j=reshape(FmatrixFn_j(j),[N_d,N_a,N_z]);
                %        Phi_aprimeKron=Phi_aprimeKronFn_j(j);
                for z_c=1:N_z
                    for a_c=1:N_a
                        RHSpart2=zeros(N_d,1);
                        for zprime_c=1:N_z
                            if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                                for d_c=1:N_d
                                    RHSpart2(d_c)=RHSpart2(d_c)+VKronNext_j(Phi_aprimeMatrix(d_c,a_c,z_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                                end
                            end
                        end
                        entireRHS=ReturnMatrix(:,a_c,z_c)+beta_j(jj)*RHSpart2; %aprime by 1
                        
                        %calculate in order, the maximizing aprime indexes
                        [V(a_c,z_c,jj),Policy(a_c,z_c,jj)]=max(entireRHS,[],1);
                    end
                end
            end
        end
    end
    
    if Case2_Type==2  % phi_a'(d,z,z')
        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;
            
            if fieldexists_pi_z_J==1
                z_grid=vfoptions.z_grid_J(:,jj);
                pi_z=vfoptions.pi_z_J(:,:,jj);
            elseif fieldexists_ExogShockFn==1
                if fieldexists_ExogShockFnParamNames==1
                    ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for ii=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                    end
                    [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
                    z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
                else
                    [z_grid,pi_z]=vfoptions.ExogShockFn(jj);
                    z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
                end
            end
            
            if reverse_j==0 % So j==N_j
                VKronNext_j=V(:,:,1);
            else
                VKronNext_j=V(:,:,jj+1);
            end
            FmatrixKron_j=reshape(FmatrixFn_j(jj),[N_d,N_a,N_z]);
            Phi_aprimeKron=Phi_aprimeKronFn_j(jj);
            for z_c=1:N_z
                RHSpart2=zeros(N_d,1);
                for zprime_c=1:N_z
                    if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                        for d_c=1:N_d
                            RHSpart2(d_c)=RHSpart2(d_c)+VKronNext_j(Phi_aprimeKron(d_c,z_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                        end
                    end
                end
                for a_c=1:N_a
                    entireRHS=FmatrixKron_j(:,a_c,z_c)+beta_j(jj)*RHSpart2; %aprime by 1
                    
                    %calculate in order, the maximizing aprime indexes
                    [V(a_c,z_c,jj),Policy(a_c,z_c,jj)]=max(entireRHS,[],1);
                end
            end
        end
    end
    
    
    if Case2_Type==3  % phi_a'(d,z')
        if vfoptions.phiaprimedependsonage==0
            PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
            Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
            for reverse_j=0:N_j-1
                jj=N_j-reverse_j;
                
                if fieldexists_pi_z_J==1
                    z_grid=vfoptions.z_grid_J(:,jj);
                    pi_z=vfoptions.pi_z_J(:,:,jj);
                elseif fieldexists_ExogShockFn==1
                    if fieldexists_ExogShockFnParamNames==1
                        ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
                        ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                        for ii=1:length(ExogShockFnParamsVec)
                            ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                        end
                        [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
                        z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
                    else
                        [z_grid,pi_z]=vfoptions.ExogShockFn(jj);
                        z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
                    end
                end
                
                if reverse_j==0 % So j==N_j
                    VKronNext_j=V(:,:,1);
                else
                    VKronNext_j=V(:,:,jj+1);
                end
                FmatrixKron_j=reshape(FmatrixFn_j(jj),[N_d,N_a,N_z]);
                Phi_aprimeKron=Phi_aprimeKronFn_j(jj);
                for z_c=1:N_z
                    RHSpart2=zeros(N_d,1);
                    for zprime_c=1:N_z
                        if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                            for d_c=1:N_d
                                RHSpart2(d_c)=RHSpart2(d_c)+VKronNext_j(Phi_aprimeKron(d_c),zprime_c)*pi_z(z_c,zprime_c);
                            end
                        end
                    end
                    for a_c=1:N_a
                        entireRHS=FmatrixKron_j(:,a_c,z_c)+beta_j(jj)*RHSpart2; %aprime by 1
                        
                        %calculate in order, the maximizing aprime indexes
                        [V(a_c,z_c,jj),Policy(a_c,z_c,jj)]=max(entireRHS,[],1);
                    end
                end
            end
        elseif vfoptions.phiaprimedependsonage==1
            for reverse_j=0:N_j-1
                jj=N_j-reverse_j;
                
                if fieldexists_pi_z_J==1
                    z_grid=vfoptions.z_grid_J(:,jj);
                    pi_z=vfoptions.pi_z_J(:,:,jj);
                elseif fieldexists_ExogShockFn==1
                    if fieldexists_ExogShockFnParamNames==1
                        ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
                        ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                        for ii=1:length(ExogShockFnParamsVec)
                            ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                        end
                        [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
                        z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
                    else
                        [z_grid,pi_z]=vfoptions.ExogShockFn(jj);
                        z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
                    end
                end
                
                PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
                Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
                
                if reverse_j==0 % So j==N_j
                    VKronNext_j=V(:,:,1);
                else
                    VKronNext_j=V(:,:,jj+1);
                end
                FmatrixKron_j=reshape(FmatrixFn_j(jj),[N_d,N_a,N_z]);
                Phi_aprimeKron=Phi_aprimeKronFn_j(jj);
                for z_c=1:N_z
                    RHSpart2=zeros(N_d,1);
                    for zprime_c=1:N_z
                        if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                            for d_c=1:N_d
                                RHSpart2(d_c)=RHSpart2(d_c)+VKronNext_j(Phi_aprimeKron(d_c),zprime_c)*pi_z(z_c,zprime_c);
                            end
                        end
                    end
                    for a_c=1:N_a
                        entireRHS=FmatrixKron_j(:,a_c,z_c)+beta_j(jj)*RHSpart2; %aprime by 1
                        
                        %calculate in order, the maximizing aprime indexes
                        [V(a_c,z_c,jj),Policy(a_c,z_c,jj)]=max(entireRHS,[],1);
                    end
                end
            end
        end
        %     for reverse_j=1:N_j-1
        %         j=N_j-reverse_j;
        %         VKronNext_j=V(:,:,j+1);
        %         FmatrixKron_j=reshape(FmatrixFn_j(j),[N_d,N_a,N_z]);
        %         Phi_aprimeKron=Phi_aprimeKronFn_j(j);
        %         for z_c=1:N_z
        %             RHSpart2=zeros(N_d,1);
        %             for zprime_c=1:N_z
        %                 if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        %                     for d_c=1:N_d
        %                         RHSpart2(d_c)=RHSpart2(d_c)+VKronNext_j(Phi_aprimeKron(d_c),zprime_c)*pi_z(z_c,zprime_c);
        %                     end
        %                 end
        %             end
        %             for a_c=1:N_a
        %                 entireRHS=FmatrixKron_j(:,a_c,z_c)+beta_j(j)*RHSpart2; %aprime by 1
        %
        %                 %calculate in order, the maximizing aprime indexes
        %                 [V(a_c,z_c,j),Policy(a_c,z_c,j)]=max(entireRHS,[],1);
        %             end
        %         end
        %     end
    end
    
    if Case2_Type==4  % phi_a'(d,a)
        PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
        Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
        aaa=kron(pi_z,ones(N_d,1,'gpuArray'));
        
        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;
            
            if fieldexists_pi_z_J==1
                z_grid=vfoptions.z_grid_J(:,jj);
                pi_z=vfoptions.pi_z_J(:,:,jj);
            elseif fieldexists_ExogShockFn==1
                if fieldexists_ExogShockFnParamNames==1
                    ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for ii=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                    end
                    [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
                    z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
                else
                    [z_grid,pi_z]=vfoptions.ExogShockFn(jj);
                    z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
                end
            end
            
            if reverse_j==0 % So j==N_j
                VKronNext_j=V(:,:,1);
            else
                VKronNext_j=V(:,:,jj+1);
            end
            
            ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid,ReturnFnParamsVec);
            EV=zeros(N_d*N_z,N_z,'gpuArray');
            for zprime_c=1:N_z
                EV(:,zprime_c)=VKronNext_j(Phi_aprimeMatrix(:,zprime_c)*ones(1,N_z),zprime_c); %(d,z')
            end
            EV=EV.*aaa;
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=reshape(sum(EV,2),[N_d,1,N_z]);
            
            for z_c=1:N_z % Can probably eliminate this loop and replace with a matrix multiplication operation thereby making it faster
                entireRHS=ReturnMatrix(:,:,z_c)+beta*EV(:,z_c)*ones(1,N_a,1,'gpuArray');
                
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS,[],1);
                V(:,z_c,jj)=Vtemp;
                Policy(:,z_c,jj)=maxindex;
            end
        end
    end
    
    if Case2_Type==5  % phi_a'(d,e')
        % In Case2_Type==5 it is no longer the case that Phi_aprime contains
        % the index of the relevant point. Instead it now contains the
        % probability of each point.
        
        if vfoptions.phiaprimedependsonage==0
            if vfoptions.phiaprimematrix==1
                Phi_aprimeMatrix_e=Phi_aprime;
            elseif vfoptions.phiaprimematrix==2
                disp('ERROR: COMBINATION OF Case2_Type==5 and vfoptions.phiaprimematrix==2 HAS NOT BEEN IMPLEMENTED')
                PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
                %     Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z,d_grid, a_grid, z_grid,PhiaprimeParamsVec);
                Phi_aprimeMatrix_e=CreatePhiaprimeMatrix_Case2_Disc_Par2_e(Phi_aprime, Case2_Type, n_d, n_a, n_z,d_grid, a_grid,e_grid, z_grid, PhiaprimeParamsVec);
            end
        end
        
        aaa=kron(pi_z,ones(N_d,1,'gpuArray'));
        
        %prob_e
        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;
            
            if fieldexists_pi_z_J==1
                z_grid=vfoptions.z_grid_J(:,jj);
                pi_z=vfoptions.pi_z_J(:,:,jj);
            elseif fieldexists_ExogShockFn==1
                if fieldexists_ExogShockFnParamNames==1
                    ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for ii=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                    end
                    [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
                    z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
                else
                    [z_grid,pi_z]=vfoptions.ExogShockFn(jj);
                    z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
                end
            end
            
            if reverse_j==0 % So j==N_j
                VKronNext_j=V(:,:,1);
            else
                VKronNext_j=V(:,:,jj+1);
            end
            
            if vfoptions.phiaprimedependsonage==1
                if vfoptions.phiaprimematrix==1
                    Phi_aprimeMatrix_e=Phi_aprime(:,:,jj);
                elseif vfoptions.phiaprimematrix==2
                    disp('ERROR: COMBINATION OF Case2_Type==5 and vfoptions.phiaprimematrix==2 HAS NOT BEEN IMPLEMENTED')
                    PhiaprimeParamsVec=[jj,CreateVectorFromParams(Parameters, PhiaprimeParamNames)];
                    %     Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z,d_grid, a_grid, z_grid,PhiaprimeParamsVec);
                    Phi_aprimeMatrix_e=CreatePhiaprimeMatrix_Case2_Disc_Par2_e(Phi_aprime, Case2_Type, n_d, n_a, n_z,d_grid, a_grid, z_grid,e_grid, PhiaprimeParamsVec);
                end
            end
            
            if vfoptions.lowmemory==0
                ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid,ReturnFnParamsVec);
                EV=zeros(N_d*N_z,N_z,'gpuArray');
                for zprime_c=1:N_z % This can likely be improved
                    EV(:,zprime_c)=VKronNext_j(Phi_aprimeMatrix(:)*ones(1,N_z),zprime_c); %(d,z')
                end
                EV=EV.*aaa;
                EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV=reshape(sum(EV,2),[N_d,1,N_z]);
                
                for z_c=1:N_z % Can probably eliminate this loop and replace with a matrix multiplication operation thereby making it faster
                    entireRHS=ReturnMatrix(:,:,z_c)+beta*EV(:,z_c)*ones(1,N_a,1,'gpuArray');
                    
                    %Calc the max and it's index
                    [Vtemp,maxindex]=max(entireRHS,[],1);
                    V(:,z_c,jj)=Vtemp;
                    Policy(:,z_c,jj)=maxindex;
                end
            elseif vfoptions.lowmemory==1
                
                EV=zeros(N_d*N_z,N_z,'gpuArray');
                for zprime_c=1:N_z % This can likely be improved
                    EV(:,zprime_c)=VKronNext_j(Phi_aprimeMatrix(:)*ones(1,N_z),zprime_c); %(d,z')
                end
                EV=EV.*aaa;
                EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV=reshape(sum(EV,2),[N_d,1,N_z]);
                
                for z_c=1:N_z % Can probably eliminate this loop and replace with a matrix multiplication operation thereby making it faster
                    z_val=z_gridvals(z_c,:);
                    ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_grid, a_grid, z_val, ReturnFnParamsVec);
                    
                    entireRHS=ReturnMatrix_z+beta*EV(:,z_c)*ones(1,N_a,1,'gpuArray');
                    
                    %Calc the max and it's index
                    [Vtemp,maxindex]=max(entireRHS,[],1);
                    V(:,z_c,jj)=Vtemp;
                    Policy(:,z_c,jj)=maxindex;
                end
            end
            
        end
    end
    
    Vdist=reshape(V-Vold,[N_a*N_z*N_j,1]); Vdist(isnan(Vdist))=0;
    currdist=max(abs(Vdist)); %IS THIS reshape() & max() FASTER THAN max(max()) WOULD BE?
    Vold=V;
    
    tempcounter=tempcounter+1;
    if vfoptions.verbose==1 && rem(tempcounter,10)==0
        fprintf('Value Fn Iteration: After %d steps, current distance is %8.2f \n', tempcounter, currdist);
    end
end


end