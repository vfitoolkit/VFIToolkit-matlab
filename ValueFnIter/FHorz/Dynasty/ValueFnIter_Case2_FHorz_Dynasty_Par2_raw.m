function [V,Policy]=ValueFnIter_Case2_FHorz_Dynasty_Par2_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

eval('fieldexists_ExogShockFn=1;vfoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;vfoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')

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
        if vfoptions.phiaprimedependsonage==0
            PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
            if vfoptions.lowmemory==0
                Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
            end
        end
        
        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;
            
            if vfoptions.verbose==1
                sprintf('Age j is currently %i \n',jj)
            end
            
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
                if vfoptions.phiaprimedependsonage==1
                    PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
                    Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
                end
                
                %if vfoptions.returnmatrix==2 % GPU
                ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
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
                        entireRHS=ReturnMatrix(:,a_c,z_c)+DiscountFactorParamsVec*RHSpart2; %aprime by 1
                        
                        %calculate in order, the maximizing aprime indexes
                        [V(a_c,z_c,jj),Policy(a_c,z_c,jj)]=max(entireRHS,[],1);
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
                        RHSpart2=zeros(N_d,1);
                        for zprime_c=1:N_z
                            if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                                for d_c=1:N_d
                                    RHSpart2(d_c)=RHSpart2(d_c)+VKronNext_j(Phi_aprimeMatrix_z(d_c,a_c,1,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                                end
                            end
                        end
                        entireRHS=ReturnMatrix_z(:,a_c)+DiscountFactorParamsVec*RHSpart2; %aprime by 1
                        
                        %calculate in order, the maximizing aprime indexes
                        [V(a_c,z_c,jj),Policy(a_c,z_c,jj)]=max(entireRHS,[],1);
                    end
                end
            elseif vfoptions.lowmemory==2
                for a_c=1:N_a
                    if vfoptions.phiaprimedependsonage==1
                        PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
                    end
                    Phi_aprimeMatrix_a=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, special_n_a, n_z, d_grid, a_val, z_grid,PhiaprimeParamsVec);
                    ReturnMatrix_a=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, special_n_a,n_z, d_grid, a_val, z_grid, ReturnFnParamsVec);
                    for z_c=1:N_z
                        RHSpart2=zeros(N_d,1);
                        for zprime_c=1:N_z
                            if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                                for d_c=1:N_d
                                    RHSpart2(d_c)=RHSpart2(d_c)+VKronNext_j(Phi_aprimeMatrix_a(d_c,1,z_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                                end
                            end
                        end
                        entireRHS=ReturnMatrix_a(:,1,z_c)+DiscountFactorParamsVec*RHSpart2; %aprime by 1
                        
                        %calculate in order, the maximizing aprime indexes
                        [V(a_c,z_c,jj),Policy(a_c,z_c,jj)]=max(entireRHS,[],1);
                    end
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
        
        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;
            
            if vfoptions.verbose==1
                sprintf('Age j is currently %i \n',jj)
            end
            
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
                if vfoptions.phiaprimedependsonage==1
                    PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
                end
                
                %if vfoptions.returnmatrix==2 % GPU
                ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
                %        FmatrixKron_j=reshape(FmatrixFn_j(j),[N_d,N_a,N_z]);
                %        Phi_aprimeKron=Phi_aprimeKronFn_j(j);
                for z_c=1:N_z
                    for a_c=1:N_a
                        RHSpart2=zeros(N_d,1,'gpuArray');
                        for zprime_c=1:N_z
                            z_val=z_gridvals(zprime_c,:);
                            Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, special_n_z, d_grid, a_grid, z_val,PhiaprimeParamsVec);
                            if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                                for d_c=1:N_d
                                    RHSpart2(d_c)=RHSpart2(d_c)+VKronNext_j(Phi_aprimeMatrix(d_c,a_c,z_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                                end
                            end
                        end
                        entireRHS=ReturnMatrix(:,a_c,z_c)+DiscountFactorParamsVec*RHSpart2; %aprime by 1
                        
                        %calculate in order, the maximizing aprime indexes
                        [V(a_c,z_c,jj),Policy(a_c,z_c,jj)]=max(entireRHS,[],1);
                    end
                end
            elseif vfoptions.lowmemory==1
                for z_c=1:N_z
                    z_val=z_gridvals(z_c,:);
                    
                    if vfoptions.phiaprimedependsonage==1
                        PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
                    end
                    ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_grid, a_grid, z_val, ReturnFnParamsVec);
                    for a_c=1:N_a
                        RHSpart2=zeros(N_d,1,'gpuArray');
                        for zprime_c=1:N_z
                            zprime_val=z_gridvals(zprime_c,:);
                            Phi_aprimeMatrix_z=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, special_n_z, d_grid, a_grid, zprime_val,PhiaprimeParamsVec);
                            
                            if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                                for d_c=1:N_d
                                    RHSpart2(d_c)=RHSpart2(d_c)+VKronNext_j(Phi_aprimeMatrix_z(d_c,a_c,1),zprime_c)*pi_z(z_c,zprime_c);
                                end
                            end
                        end
                        entireRHS=ReturnMatrix_z(:,a_c)+DiscountFactorParamsVec*RHSpart2; %aprime by 1
                        
                        %calculate in order, the maximizing aprime indexes
                        [V(a_c,z_c,jj),Policy(a_c,z_c,jj)]=max(entireRHS,[],1);
                    end
                end
            elseif vfoptions.lowmemory==2
                for a_c=1:N_a
                    if vfoptions.phiaprimedependsonage==1
                        PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
                    end
                    Phi_aprimeMatrix_a=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, special_n_a, n_z, d_grid, a_val, z_grid,PhiaprimeParamsVec);
                    ReturnMatrix_a=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, special_n_a,n_z, d_grid, a_val, z_grid, ReturnFnParamsVec);
                    for z_c=1:N_z
                        RHSpart2=zeros(N_d,1);
                        for zprime_c=1:N_z
                            if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                                for d_c=1:N_d
                                    RHSpart2(d_c)=RHSpart2(d_c)+VKronNext_j(Phi_aprimeMatrix_a(d_c,1,z_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                                end
                            end
                        end
                        entireRHS=ReturnMatrix_a(:,1,z_c)+DiscountFactorParamsVec*RHSpart2; %aprime by 1
                        
                        %calculate in order, the maximizing aprime indexes
                        [V(a_c,z_c,jj),Policy(a_c,z_c,jj)]=max(entireRHS,[],1);
                    end
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
        
        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;
            
            if vfoptions.verbose==1
                sprintf('Age j is currently %i \n',jj)
            end
            
            % Create a vector containing all the return function parameters (in order)
            ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
            DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
            DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
            
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
                        entireRHS=ReturnMatrix(:,a_c,z_c)+DiscountFactorParamsVec*EV_az; %aprime by 1
                        
                        %calculate in order, the maximizing aprime indexes
                        [V(a_c,z_c,jj),Policy(a_c,z_c,jj)]=max(entireRHS,[],1);
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
                        RHSpart2=zeros(N_d,1,'gpuArray');
                        for zprime_c=1:N_z
                            if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                                for d_c=1:N_d
                                    RHSpart2(d_c)=RHSpart2(d_c)+VKronNext_j(Phi_aprimeMatrix_z(d_c,a_c,1),zprime_c)*pi_z(z_c,zprime_c);
                                end
                            end
                        end
                        entireRHS=ReturnMatrix_z(:,a_c)+DiscountFactorParamsVec*RHSpart2; %aprime by 1
                        
                        %calculate in order, the maximizing aprime indexes
                        [V(a_c,z_c,jj),Policy(a_c,z_c,jj)]=max(entireRHS,[],1);
                    end
                end
            elseif vfoptions.lowmemory==2
                for a_c=1:N_a
                    if vfoptions.phiaprimedependsonage==1
                        PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
                    end
                    Phi_aprimeMatrix_a=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, special_n_a, n_z, d_grid, a_val, z_grid,PhiaprimeParamsVec);
                    ReturnMatrix_a=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, special_n_a,n_z, d_grid, a_val, z_grid, ReturnFnParamsVec);
                    for z_c=1:N_z
                        RHSpart2=zeros(N_d,1);
                        for zprime_c=1:N_z
                            if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                                for d_c=1:N_d
                                    RHSpart2(d_c)=RHSpart2(d_c)+VKronNext_j(Phi_aprimeMatrix_a(d_c,1,z_c),zprime_c)*pi_z(z_c,zprime_c);
                                end
                            end
                        end
                        entireRHS=ReturnMatrix_a(:,1,z_c)+DiscountFactorParamsVec*RHSpart2; %aprime by 1
                        
                        %calculate in order, the maximizing aprime indexes
                        [V(a_c,z_c,jj),Policy(a_c,z_c,jj)]=max(entireRHS,[],1);
                    end
                end
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
        
        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;
            
            if vfoptions.verbose==1
                sprintf('Age j is currently %i \n',jj)
            end
            
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
            
            if vfoptions.phiaprimedependsonage==1
                PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
                Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
            end
            
            %if vfoptions.returnmatrix==2 % GPU
            ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
            %         FmatrixKron_j=reshape(FmatrixFn_j(jj),[N_d,N_a,N_z]);
            %         Phi_aprimeKron=Phi_aprimeKronFn_j(jj);
            
            EV=zeros(N_d*N_z,N_z,'gpuArray');
            for zprime_c=1:N_z
                EV(:,zprime_c)=VKronold(Phi_aprime(:,:,zprime_c),zprime_c); %(d,z')
            end
            EV=EV.*aaa;
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=reshape(sum(EV,2),[N_d,1,N_z]);
            
            for z_c=1:N_z % Can probably eliminate this loop and replace with a matrix multiplication operation thereby making it faster
                entireRHS=ReturnMatrix(:,:,z_c)+prod(DiscountFactorParamsVec)*EV(:,z_c)*ones(1,N_a,1,'gpuArray');
                
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS,[],1);
                V(:,z_c,jj)=Vtemp;
                Policy(:,z_c,jj)=maxindex;
            end
        end
    end
    
    if Case2_Type==3  % phi_a'(d,z')
        if vfoptions.phiaprimedependsonage==0
            PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
            Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
        end
        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;
            
            % Create a vector containing all the return function parameters (in order)
            ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
            DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
            
            if vfoptions.phiaprimedependsonage==1
                PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
                Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
            end
            
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

                    entireRHS_z=ReturnMatrix(:,:,z_c)+DiscountFactorParamsVec*EV_z*ones(1,N_a,1);
                    
                    %Calc the max and it's index
                    [Vtemp,maxindex]=max(entireRHS_z,[],1);
                    V(:,z_c,jj)=Vtemp;
                    Policy(:,z_c,jj)=maxindex;
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
                    entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*EV_z*ones(1,N_a,1);
                    
                    %Calc the max and it's index
                    [Vtemp,maxindex]=max(entireRHS_z,[],1);
                    V(:,z_c,jj)=Vtemp;
                    Policy(:,z_c,jj)=maxindex;
                end
            elseif vfoptions.lowmemory==2
                EV_z=zeros(N_d,1);
                for z_c=1:N_Z
                    for zprime_c=1:N_z
                        if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                            for d_c=1:N_d
                                EV_z(d_c)=EV_z(d_c)+VKronNext_j(Phi_aprimeMatrix(d_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                            end
                        end
                    end
                    for a_c=1:N_a
                        a_val=a_gridvals(z_c,:);
                        ReturnMatrix_az=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, special_n_a, special_n_z, d_grid, a_val, z_val, ReturnFnParamsVec);
                        
                        entireRHS=ReturnMatrix_az+DiscountFactorParamsVec*EV_z; %aprime by 1
                        
                        %calculate in order, the maximizing aprime indexes
                        [V(a_c,z_c,jj),Policy(a_c,z_c,jj)]=max(entireRHS,[],1);
                    end
                end
            end
        end
    end
    
    if Case2_Type==4  % phi_a'(d,a)
        PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
        Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
        aaa=kron(pi_z,ones(N_d,1,'gpuArray'));
        
        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;
            
            if reverse_j==0 % So j==N_j
                VKronNext_j=V(:,:,1);
            else
                VKronNext_j=V(:,:,jj+1);
            end
            
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
    
    Vdist=reshape(V-Vold,[N_a*N_z*N_j,1]); Vdist(isnan(Vdist))=0;
    currdist=max(abs(Vdist)); %IS THIS reshape() & max() FASTER THAN max(max()) WOULD BE?
    Vold=V;
    
    tempcounter=tempcounter+1;
    if vfoptions.verbose==1 && rem(tempcounter,10)==0
        fprintf('Value Fn Iteration: After %d steps, current distance is %8.2f \n', tempcounter, currdist);
    end
end

end