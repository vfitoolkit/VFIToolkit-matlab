function [V,Policy]=ValueFnIter_Case2_FHorz_AgeDepGrids_Dynasty_Par2_raw(daz_gridstructure,N_j,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions)

V=struct();
Policy=struct(); %indexes the optimal choice for d given rest of dimensions a,z

fprintf('ValueFnIter_Case2_FHorz_AgeDependentGrids_Par2_Dynasty_raw() \n')

%% Precompute some things for speed. (precomputing these makes sense when using dynasty, normally is pointless for finite-horizon problems)
jstrN_j=daz_gridstructure.jstr{N_j}; % Note, all the other jstr are in daz_gridstructure.

ReturnFnParamsMatrix=zeros(N_j,length(ReturnFnParamNames));
DiscountFactorParamsMatrix=zeros(N_j,length(DiscountFactorParamNames));
for jj=1:N_j
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsMatrix(jj,:)=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsMatrix(jj,:)=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
end

%%
V.j001=zeros(daz_gridstructure.N_a.('j001'),daz_gridstructure.N_z.('j001')); % Initial guess for j==1 (dynasty means this is essentially the starting point)
tempcounter=1;
currdist=Inf;
while currdist>vfoptions.tolerance
    
    tempcounter
    [currdist, vfoptions.tolerance]
    % Make a three digit number out of jj=1. Needed so that the first time
    % codes gets Vnextj=V.(jstr); this will be the age 1 due to dynasty.
    jstr='j001';
    
    %%
    if Case2_Type==1 % phi_a'(d,a,z,z')
        if vfoptions.phiaprimedependsonage==0
            fprintf('ERROR: state dependent grids and Case2_Type==1 mean that you must have vfoptions.phiaprimedependsonage==1')
        end
        
        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;
            
            if vfoptions.verbose==1
                sprintf('Age j is currently %i \n',jj)
            end
            
            Vnextj=V.(jstr); % Note that it is important that this is done before updating 'jstr'
            
            % Create a vector containing all the return function parameters (in order)
            ReturnFnParamsVec=ReturnFnParamsMatrix(jj,:); %CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
            DiscountFactorParamsVec=DiscountFactorParamsMatrix(jj,:); %CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
            
            jstr=daz_gridstructure.jstr{jj};
            % Get the relevant grid and transition matrix
            if vfoptions.agedependentgrids(1)==1
                N_d=daz_gridstructure.N_d.(jstr(:));
                n_d=daz_gridstructure.n_d.(jstr(:));
                d_grid=daz_gridstructure.d_grid.(jstr(:));
            end
            if vfoptions.agedependentgrids(2)==1
                N_a=daz_gridstructure.N_a.(jstr(:));
                n_a=daz_gridstructure.n_a.(jstr(:));
                a_grid=daz_gridstructure.a_grid.(jstr(:));
                N_aprime=daz_gridstructure.N_aprime.(jstr(:));
                if vfoptions.lowmemory==2
                    special_n_a=ones(1,length(n_a));
                    a_gridvals=daz_gridstructure.a_gridvals.(jstr(:));
                end
            end
            if vfoptions.agedependentgrids(3)==1
                N_z=daz_gridstructure.N_z.(jstr(:));
                n_z=daz_gridstructure.n_z.(jstr(:));
                z_grid=daz_gridstructure.z_grid.(jstr(:));
                pi_z=daz_gridstructure.pi_z.(jstr(:));
                if vfoptions.lowmemory==1
                    special_n_z=ones(1,length(n_z));
                    z_gridvals=daz_gridstructure.z_gridvals.(jstr(:));
                end
            end
            
            V_j=zeros(N_a,N_z);
            Policy_j=zeros(N_a,N_z);
            
            if vfoptions.lowmemory==0
                if vfoptions.phiaprimedependsonage==1
                    PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
                    Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
                end
                
                ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
                for z_c=1:N_z
                    for a_c=1:N_a
                        RHSpart2=zeros(N_d,1);
                        for zprime_c=1:N_z
                            if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                                for d_c=1:N_d
                                    RHSpart2(d_c)=RHSpart2(d_c)+Vnextj(Phi_aprimeMatrix(d_c,a_c,z_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                                end
                            end
                        end
                        entireRHS=ReturnMatrix(:,a_c,z_c)+DiscountFactorParamsVec*RHSpart2; %aprime by 1
                        
                        %calculate in order, the maximizing aprime indexes
                        [V_j(a_c,z_c),Policy_j(a_c,z_c)]=max(entireRHS,[],1);
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
                                    RHSpart2(d_c)=RHSpart2(d_c)+Vnextj(Phi_aprimeMatrix_z(d_c,a_c,1,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                                end
                            end
                        end
                        entireRHS=ReturnMatrix_z(:,a_c)+DiscountFactorParamsVec*RHSpart2; %aprime by 1
                        
                        %calculate in order, the maximizing aprime indexes
                        [V_j(a_c,z_c),Policy_j(a_c,z_c)]=max(entireRHS,[],1);
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
                                    RHSpart2(d_c)=RHSpart2(d_c)+Vnextj(Phi_aprimeMatrix_a(d_c,1,z_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                                end
                            end
                        end
                        entireRHS=ReturnMatrix_a(:,1,z_c)+DiscountFactorParamsVec*RHSpart2; %aprime by 1
                        
                        %calculate in order, the maximizing aprime indexes
                        [V_j(a_c,z_c),Policy_j(a_c,z_c)]=max(entireRHS,[],1);
                    end
                end
            end
            
            V.(jstr)=V_j;
            Policy.(jstr)=Policy_j;
        end
    end
    %%
    if Case2_Type==11 % phi_a'(d,a,z')
        if vfoptions.phiaprimedependsonage==0
            fprintf('ERROR: state dependent grids and Case2_Type==11 mean that you must have vfoptions.phiaprimedependsonage==1')
        end
        
        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;
            
            if vfoptions.verbose==1
                sprintf('Age j is currently %i \n',jj)
            end
            
            Vnextj=V.(jstr); % Note that it is important that this is done before updating 'jstr'
            
            % Create a vector containing all the return function parameters (in order)
            ReturnFnParamsVec=ReturnFnParamsMatrix(jj,:); %CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
            DiscountFactorParamsVec=DiscountFactorParamsMatrix(jj,:); %CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
           
            jstr=daz_gridstructure.jstr{jj};
            % Get the relevant grid and transition matrix
            if vfoptions.agedependentgrids(1)==1
                N_d=daz_gridstructure.N_d.(jstr(:));
                n_d=daz_gridstructure.n_d.(jstr(:));
                d_grid=daz_gridstructure.d_grid.(jstr(:));
            end
            if vfoptions.agedependentgrids(2)==1
                N_a=daz_gridstructure.N_a.(jstr(:));
                n_a=daz_gridstructure.n_a.(jstr(:));
                a_grid=daz_gridstructure.a_grid.(jstr(:));
                N_aprime=daz_gridstructure.N_aprime.(jstr(:));
                if vfoptions.lowmemory==2
                    special_n_a=ones(1,length(n_a));
                    a_gridvals=daz_gridstructure.a_gridvals.(jstr(:));
%                     
%                     a_gridvals=zeros(N_a,length(n_a),'gpuArray');
%                     for i2=1:N_a
%                         sub=zeros(1,length(n_a));
%                         sub(1)=rem(i2-1,n_a(1))+1;
%                         for ii=2:length(n_a)-1
%                             sub(ii)=rem(ceil(i2/prod(n_a(1:ii-1)))-1,n_a(ii))+1;
%                         end
%                         sub(length(n_a))=ceil(i2/prod(n_a(1:length(n_a)-1)));
%                         
%                         if length(n_a)>1
%                             sub=sub+[0,cumsum(n_a(1:end-1))];
%                         end
%                         a_gridvals(i2,:)=a_grid(sub);
%                     end
                end
            end
            if vfoptions.agedependentgrids(3)==1
                N_z=daz_gridstructure.N_z.(jstr(:));
                N_zprime=daz_gridstructure.N_zprime.(jstr(:));
                n_z=daz_gridstructure.n_z.(jstr(:));
                z_grid=daz_gridstructure.z_grid.(jstr(:));
                pi_z=daz_gridstructure.pi_z.(jstr(:));
                if vfoptions.lowmemory==1
                    special_n_z=ones(1,length(n_z));
                    z_gridvals=daz_gridstructure.z_gridvals.(jstr(:));
%                     z_gridvals=zeros(N_z,length(n_z),'gpuArray');
%                     for i1=1:N_z
%                         sub=zeros(1,length(n_z));
%                         sub(1)=rem(i1-1,n_z(1))+1;
%                         for ii=2:length(n_z)-1
%                             sub(ii)=rem(ceil(i1/prod(n_z(1:ii-1)))-1,n_z(ii))+1;
%                         end
%                         sub(length(n_z))=ceil(i1/prod(n_z(1:length(n_z)-1)));
%                         
%                         if length(n_z)>1
%                             sub=sub+[0,cumsum(n_z(1:end-1))];
%                         end
%                         z_gridvals(i1,:)=z_grid(sub);
%                     end
                end
            end
            
            V_j=zeros(N_a,N_z);
            Policy_j=zeros(N_a,N_z);
            
            if vfoptions.lowmemory==0
                if vfoptions.phiaprimedependsonage==1
                    PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
                end
                
                %if vfoptions.returnmatrix==2 % GPU
                ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
                for z_c=1:N_z
                    for a_c=1:N_a
                        RHSpart2=zeros(N_d,1,'gpuArray');
                        for zprime_c=1:N_zprime
                            z_val=z_gridvals(zprime_c,:);
                            Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, special_n_z, d_grid, a_grid, z_val,PhiaprimeParamsVec);
                            if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                                for d_c=1:N_d
                                    RHSpart2(d_c)=RHSpart2(d_c)+Vnextj(Phi_aprimeMatrix(d_c,a_c,z_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                                end
                            end
                        end
                        entireRHS=ReturnMatrix(:,a_c,z_c)+DiscountFactorParamsVec*RHSpart2; %aprime by 1
                        
                        %calculate in order, the maximizing aprime indexes
                        [V_j(a_c,z_c),Policy_j(a_c,z_c)]=max(entireRHS,[],1);
                    end
                end
            elseif vfoptions.lowmemory==1
                % Current Case2_Type=11: phi_a'(d,a,z')
                if vfoptions.phiaprimedependsonage==1
                    PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
                end
                Phi_aprimeMatrix_z=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_zprime, d_grid, a_grid, zprime_grid,PhiaprimeParamsVec); %z_grid as doing all of zprime (note that it is independent of z as Case2_Type=11)
                Phi_aprimeMatrix_z=reshape(Phi_aprimeMatrix_z,[N_d*N_a*N_zprime,1]);
                for z_c=1:N_z
                    z_val=z_gridvals(z_c,:); % Value of z (not of z')
                    aaa=kron(pi_z(z_c,:),ones(N_d*N_a,1,'gpuArray'));
                    zprime_ToMatchPhi=kron((1:1:N_zprime)',ones(N_d*N_a,1));
                    
                    ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_grid, a_grid, z_val, ReturnFnParamsVec);
                    EV_z=Vnextj(Phi_aprimeMatrix_z+(N_aprime)*(zprime_ToMatchPhi-1));
                    
                    EV_z=reshape(EV_z,[N_d*N_a,N_zprime]);
                    EV_z=EV_z.*aaa;
                    EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                    EV_z=reshape(sum(EV_z,2),[N_d,N_a]);
                    
                    entireRHS=ReturnMatrix_z+DiscountFactorParamsVec*EV_z; %d by a (by z)
                    
                    %calculate in order, the maximizing aprime indexes
                    [V_j(:,z_c),Policy_j(:,z_c)]=max(entireRHS,[],1);
                end
            elseif vfoptions.lowmemory==2
                for a_c=1:N_a
                    if vfoptions.phiaprimedependsonage==1
                        PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
                    end
                    Phi_aprimeMatrix_a=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, special_n_a, n_zprime, d_grid, a_val, zprime_grid,PhiaprimeParamsVec);
                    ReturnMatrix_a=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, special_n_a,n_z, d_grid, a_val, z_grid, ReturnFnParamsVec);
                    for z_c=1:N_z
                        RHSpart2=zeros(N_d,1);
                        for zprime_c=1:N_zprime
                            if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                                for d_c=1:N_d
                                    RHSpart2(d_c)=RHSpart2(d_c)+Vnextj(Phi_aprimeMatrix_a(d_c,1,z_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                                end
                            end
                        end
                        entireRHS=ReturnMatrix_a(:,1,z_c)+DiscountFactorParamsVec*RHSpart2; %aprime by 1
                        
                        %calculate in order, the maximizing aprime indexes
                        [V_j(a_c,z_c),Policy_j(a_c,z_c)]=max(entireRHS,[],1);
                    end
                end
            end
            
            V.(jstr)=V_j;
            Policy.(jstr)=Policy_j;
        end
    end
    %%
    if Case2_Type==12 % phi_a'(d,a,z)
        if vfoptions.phiaprimedependsonage==0
            fprintf('ERROR: state dependent grids and Case2_Type==12 mean that you must have vfoptions.phiaprimedependsonage==1')
        end
        
        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;
            
            if vfoptions.verbose==1
                sprintf('Age j is currently %i \n',jj)
            end
            
            Vnextj=V.(jstr); % Note that it is important that this is done before updating 'jstr' (is next periods value fn and used to compute the expectations)
            
            % Create a vector containing all the return function parameters (in order)
            ReturnFnParamsVec=ReturnFnParamsMatrix(jj,:); %CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
            DiscountFactorParamsVec=DiscountFactorParamsMatrix(jj,:); %CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
            DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
            
            jstr=daz_gridstructure.jstr{jj};
            % Get the relevant grid and transition matrix
            if vfoptions.agedependentgrids(1)==1
                N_d=daz_gridstructure.N_d.(jstr(:));
                n_d=daz_gridstructure.n_d.(jstr(:));
                d_grid=daz_gridstructure.d_grid.(jstr(:));
            end
            if vfoptions.agedependentgrids(2)==1
                N_a=daz_gridstructure.N_a.(jstr(:));
                n_a=daz_gridstructure.n_a.(jstr(:));
                a_grid=daz_gridstructure.a_grid.(jstr(:));
                N_aprime=daz_gridstructure.N_aprime.(jstr(:));
                if vfoptions.lowmemory==2
                    special_n_a=ones(1,length(n_a));
                    a_gridvals=daz_gridstructure.a_gridvals.(jstr(:));
%                     
%                     a_gridvals=zeros(N_a,length(n_a),'gpuArray');
%                     for i2=1:N_a
%                         sub=zeros(1,length(n_a));
%                         sub(1)=rem(i2-1,n_a(1))+1;
%                         for ii=2:length(n_a)-1
%                             sub(ii)=rem(ceil(i2/prod(n_a(1:ii-1)))-1,n_a(ii))+1;
%                         end
%                         sub(length(n_a))=ceil(i2/prod(n_a(1:length(n_a)-1)));
%                         
%                         if length(n_a)>1
%                             sub=sub+[0,cumsum(n_a(1:end-1))];
%                         end
%                         a_gridvals(i2,:)=a_grid(sub);
%                     end
                end
            end
            if vfoptions.agedependentgrids(3)==1
                N_z=daz_gridstructure.N_z.(jstr(:));
                n_z=daz_gridstructure.n_z.(jstr(:));
                z_grid=daz_gridstructure.z_grid.(jstr(:));
                N_zprime=daz_gridstructure.N_zprime.(jstr(:));
                n_zprime=daz_gridstructure.n_zprime.(jstr(:));
                zprime_grid=daz_gridstructure.zprime_grid.(jstr(:));
                pi_z=daz_gridstructure.pi_z.(jstr(:));
                if vfoptions.lowmemory==1
                    special_n_z=ones(1,length(n_z));
                    z_gridvals=daz_gridstructure.z_gridvals.(jstr(:));
%                     z_gridvals=zeros(N_z,length(n_z),'gpuArray');
%                     for i1=1:N_z
%                         sub=zeros(1,length(n_z));
%                         sub(1)=rem(i1-1,n_z(1))+1;
%                         for ii=2:length(n_z)-1
%                             sub(ii)=rem(ceil(i1/prod(n_z(1:ii-1)))-1,n_z(ii))+1;
%                         end
%                         sub(length(n_z))=ceil(i1/prod(n_z(1:length(n_z)-1)));
%                         
%                         if length(n_z)>1
%                             sub=sub+[0,cumsum(n_z(1:end-1))];
%                         end
%                         z_gridvals(i1,:)=z_grid(sub);
%                     end
                end
            end
            
            % Predeclare a few variables.
            V_j=zeros(N_a,N_z,'gpuArray');
            Policy_j=zeros(N_a,N_z,'gpuArray');
            
            if vfoptions.lowmemory==0
                % Current Case2_Type=12: phi_a'(d,a,z)
                if vfoptions.phiaprimedependsonage==1
                    PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
                    Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
                end
                aaa=kron(pi_z,ones(N_d*N_a,1,'gpuArray')); % in the case that only the grids for 'a' change this could be skipped but seems unlikely enough that I have not coded for the possibility
                
                zprime_ToMatchPhi=kron((1:1:N_zprime)',ones(N_d*N_a*N_z,1));
                
                ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
                
                Phi_aprimeMatrix=reshape(Phi_aprimeMatrix,[N_d*N_a*N_z,1]);
                aaaPhi_aprimeMatrix=kron(Phi_aprimeMatrix,ones(N_zprime,1));
                
                EV=Vnextj(aaaPhi_aprimeMatrix+(N_aprime)*(zprime_ToMatchPhi-1)); %(d,z')
                EV=reshape(EV,[N_d*N_a*N_z,N_zprime]);
                EV=EV.*aaa;
                EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV=sum(EV,2); % reshape(sum(EV_az,2),[N_d,1,1]);
                EV=reshape(EV,[N_d,N_a,N_z]);
                entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; % d by a by z
                
                %calculate in order, the maximizing aprime indexes
                [V_j,Policy_j]=max(entireRHS,[],1);
                V_j=shiftdim(V_j,1);
                Policy_j=shiftdim(Policy_j,1);
                if vfoptions.dynasty_howards>0 % For Howards, store
                    a_ToMatchPolicy=kron((0:1:N_a-1)'*N_d,ones(N_z,1));
                    z_ToMatchPolicy=kron((0:1:N_z-1)'*N_d*N_a,ones(N_a,1));
                    Ftemp.(jstr(:))=ReturnMatrix(Policy_j(:),a_ToMatchPolicy,z_ToMatchPolicy);
                    Phi_of_Policy.(jstr(:))= Phi_aprimeMatrix(Policy_j(:)+a_ToMatchPolicy+z_ToMatchPolicy); % Phi_aprime is N_d*N_a*N_z, combine it with Policy (which tells d in terms of N_a*N_z) to get aprime in terms of N_a*N_z
                end
                
            elseif vfoptions.lowmemory==1
                % Current Case2_Type=12: phi_a'(d,a,z)
                if vfoptions.phiaprimedependsonage==1
                    PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj); % This line could be done outsize the z_c for loop
                end
                zprime_ToMatchPhi=kron((1:1:N_zprime)',ones(N_d*N_a,1));
                if vfoptions.dynasty_howards>0 % Predeclare some variables.
                    Ftemp_j=zeros(N_a,N_z,'gpuArray');
                    Phi_of_Policy_j=zeros(N_a,N_z,'gpuArray'); % Current Case2_Type=12: phi_a'(d,a,z). So Phi_of_Policy gives index of a' in terms of (a,z)
                end
                
                for z_c=1:N_z
                    % Vnextj is aprime-by-zprime
                    z_val=z_gridvals(z_c,:);
                    aaa=kron(pi_z(z_c,:),ones(N_d*N_a,1,'gpuArray'));
                    if vfoptions.phiaprimedependsonage==1
                        %                     PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj); % This line could be done outsize the z_c for loop
                        Phi_aprimeMatrix_z=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, special_n_z, d_grid, a_grid, z_val,PhiaprimeParamsVec);
                    end
                    Phi_aprimeMatrix_z=reshape(Phi_aprimeMatrix_z,[N_d*N_a,1]);
                    aaaPhi_aprimeMatrix_z=kron(ones(N_zprime,1),Phi_aprimeMatrix_z);
                    ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_grid, a_grid, z_val, ReturnFnParamsVec);
                                        
                    EV_z=Vnextj(aaaPhi_aprimeMatrix_z+N_aprime*(zprime_ToMatchPhi-1)); % (d,aprime,zprime) % SHOULD IT JUST BE N_aprime?
                    EV_z=reshape(EV_z,[N_d*N_a,N_zprime]);
                    EV_z=EV_z.*aaa;
                    EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                    EV_z=reshape(sum(EV_z,2),[N_d,N_a]);
                    
                    entireRHS=ReturnMatrix_z+DiscountFactorParamsVec*EV_z; % d by a (by z)
                    
                    %calculate in order, the maximizing aprime indexes
                    [V_j(:,z_c),Policy_j(:,z_c)]=max(entireRHS,[],1);
                    if vfoptions.dynasty_howards>0 % For Howards, store
                        a_ToMatchPolicy=(0:1:N_a-1)'*N_d;
%                         size(Policy_j(:,z_c))
%                         size(a_ToMatchPolicy)
%                         size(ReturnMatrix_z)
%                         size(Phi_aprimeMatrix_z)
%                         [z_c, N_d, N_a]
%                         [min(Policy_j(:,z_c)+a_ToMatchPolicy), max(Policy_j(:,z_c)+a_ToMatchPolicy)]
%                         prod(isfinite(Policy_j(:,z_c)+a_ToMatchPolicy))
%                         if N_d==1
%                             Ftemp_j(:,z_c)=ReturnMatrix_z(Policy_j(:,z_c)+a_ToMatchPolicy);
%                         else
%                             size(ReturnMatrix_z(Policy_j(:,z_c)+a_ToMatchPolicy))
%                             size(Ftemp_j)
%                         end
                        Ftemp_j(:,z_c)=ReturnMatrix_z(Policy_j(:,z_c)+a_ToMatchPolicy);
                        Phi_of_Policy_j(:,z_c)= Phi_aprimeMatrix_z(Policy_j(:,z_c)+a_ToMatchPolicy); % Phi_aprime_z is N_d*N_a, combine it with Policy_ (which tells d in terms of N_a) to get aprime in terms of N_a
                    end
                end
                if vfoptions.dynasty_howards>0
                    Ftemp.(jstr(:))=Ftemp_j;
                    Phi_of_Policy.(jstr(:))=Phi_of_Policy_j;
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
                                    RHSpart2(d_c)=RHSpart2(d_c)+Vnextj(Phi_aprimeMatrix_a(d_c,1,z_c),zprime_c)*pi_z(z_c,zprime_c);
                                end
                            end
                        end
                        entireRHS=ReturnMatrix_a(:,1,z_c)+DiscountFactorParamsVec*RHSpart2; %aprime by 1
                        
                        %calculate in order, the maximizing aprime indexes
                        [V_j(a_c,z_c),Policy_j(a_c,z_c)]=max(entireRHS,[],1);
                        if vfoptions.dynasty_howards>0
                            Ftemp_j(a_c,z_c)=ReturnMatrix_a(Policy_j(a_c,z_c),1,z_c);
                            Phi_of_Policy_j(a_c,z_c)=Phi_aprimematrix_a(Policy_j(a_c,z_c),1,z_c);
                        end
                    end
                end
                if vfoptions.dynasty_howards>0
                    Ftemp.(jstr(:))=Ftemp_j;
                    Phi_of_Policy.(jstr(:))=Phi_of_Policy_j;
                end
            end
            
            V.(jstr)=V_j;
            Policy.(jstr)=Policy_j;
        end
    end
    
    if Case2_Type==2  % phi_a'(d,z,z')
        
        if vfoptions.phiaprimedependsonage==0
            fprintf('ERROR: state dependent grids and Case2_Type==2 mean that you must have vfoptions.phiaprimedependsonage==1')
        end
        
        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;
            
            if vfoptions.verbose==1
                sprintf('Age j is currently %i \n',jj)
            end
            
            Vnextj=V.(jstr); % Note that it is important that this is done before updating 'jstr'
            
            % Create a vector containing all the return function parameters (in order)
            ReturnFnParamsVec=ReturnFnParamsMatrix(jj,:); %CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
            DiscountFactorParamsVec=DiscountFactorParamsMatrix(jj,:); %CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
            
            jstr=daz_gridstructure.jstr{jj};
            % Get the relevant grid and transition matrix
            if vfoptions.agedependentgrids(1)==1
                N_d=daz_gridstructure.N_d.(jstr(:));
                n_d=daz_gridstructure.n_d.(jstr(:));
                d_grid=daz_gridstructure.d_grid.(jstr(:));
            end
            if vfoptions.agedependentgrids(2)==1
                N_a=daz_gridstructure.N_a.(jstr(:));
                N_aprime=daz_gridstructure.N_aprime.(jstr(:));
                n_a=daz_gridstructure.n_a.(jstr(:));
                a_grid=daz_gridstructure.a_grid.(jstr(:));
                if vfoptions.lowmemory==2
                    special_n_a=ones(1,length(n_a));
                    a_gridvals=daz_gridstructure.a_gridvals.(jstr(:));
                    
%                     a_gridvals=zeros(N_a,length(n_a),'gpuArray');
%                     for i2=1:N_a
%                         sub=zeros(1,length(n_a));
%                         sub(1)=rem(i2-1,n_a(1))+1;
%                         for ii=2:length(n_a)-1
%                             sub(ii)=rem(ceil(i2/prod(n_a(1:ii-1)))-1,n_a(ii))+1;
%                         end
%                         sub(length(n_a))=ceil(i2/prod(n_a(1:length(n_a)-1)));
%                         
%                         if length(n_a)>1
%                             sub=sub+[0,cumsum(n_a(1:end-1))];
%                         end
%                         a_gridvals(i2,:)=a_grid(sub);
%                     end
                end
            end
            if vfoptions.agedependentgrids(3)==1
                N_z=daz_gridstructure.N_z.(jstr(:));
                n_z=daz_gridstructure.n_z.(jstr(:));
                z_grid=daz_gridstructure.z_grid.(jstr(:));
                pi_z=daz_gridstructure.pi_z.(jstr(:));
                if vfoptions.lowmemory==1
                    special_n_z=ones(1,length(n_z));
                    z_gridvals=daz_gridstructure.z_gridvals.(jstr(:));
%                     z_gridvals=zeros(N_z,length(n_z),'gpuArray');
%                     for i1=1:N_z
%                         sub=zeros(1,length(n_z));
%                         sub(1)=rem(i1-1,n_z(1))+1;
%                         for ii=2:length(n_z)-1
%                             sub(ii)=rem(ceil(i1/prod(n_z(1:ii-1)))-1,n_z(ii))+1;
%                         end
%                         sub(length(n_z))=ceil(i1/prod(n_z(1:length(n_z)-1)));
%                         
%                         if length(n_z)>1
%                             sub=sub+[0,cumsum(n_z(1:end-1))];
%                         end
%                         z_gridvals(i1,:)=z_grid(sub);
%                     end
                end
            end
            aaa=kron(pi_z,ones(N_d,1,'gpuArray')); % in the case that only the grids for 'a' change this could be skipped but seems unlikely enough that I have not coded for the possibility
            
            V_j=zeros(N_a,N_z);
            Policy_j=zeros(N_a,N_z);
            
            if vfoptions.phiaprimedependsonage==1
                PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
                Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
            end
            
            ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
            
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
                V_j(:,z_c)=Vtemp;
                Policy_j(:,z_c)=maxindex;
            end
            
            V.(jstr)=V_j;
            Policy.(jstr)=Policy_j;
        end
    end
    
    if Case2_Type==3  % phi_a'(d,z')
        if vfoptions.phiaprimedependsonage==0
            fprintf('ERROR: state dependent grids and Case2_Type==3 mean that you must have vfoptions.phiaprimedependsonage==1')
        end
        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;
            
            Vnextj=V.(jstr); % Note that it is important that this is done before updating 'jstr'
            
            % Create a vector containing all the return function parameters (in order)
            ReturnFnParamsVec=ReturnFnParamsMatrix(jj,:); %CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
            DiscountFactorParamsVec=DiscountFactorParamsMatrix(jj,:); %CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);            
            
            jstr=daz_gridstructure.jstr{jj};
            % Get the relevant grid and transition matrix
            if vfoptions.agedependentgrids(1)==1
                N_d=daz_gridstructure.N_d.(jstr(:));
                n_d=daz_gridstructure.n_d.(jstr(:));
                d_grid=daz_gridstructure.d_grid.(jstr(:));
            end
            if vfoptions.agedependentgrids(2)==1
                N_a=daz_gridstructure.N_a.(jstr(:));
                N_aprime=daz_gridstructure.N_aprime.(jstr(:));
                n_a=daz_gridstructure.n_a.(jstr(:));
                a_grid=daz_gridstructure.a_grid.(jstr(:));
                if vfoptions.lowmemory==2
                    special_n_a=ones(1,length(n_a));
                    a_gridvals=daz_gridstructure.a_gridvals.(jstr(:));
%                     a_gridvals=zeros(N_a,length(n_a),'gpuArray');
%                     for i2=1:N_a
%                         sub=zeros(1,length(n_a));
%                         sub(1)=rem(i2-1,n_a(1))+1;
%                         for ii=2:length(n_a)-1
%                             sub(ii)=rem(ceil(i2/prod(n_a(1:ii-1)))-1,n_a(ii))+1;
%                         end
%                         sub(length(n_a))=ceil(i2/prod(n_a(1:length(n_a)-1)));
%                         
%                         if length(n_a)>1
%                             sub=sub+[0,cumsum(n_a(1:end-1))];
%                         end
%                         a_gridvals(i2,:)=a_grid(sub);
%                     end
                end
            end
            if vfoptions.agedependentgrids(3)==1
                N_z=daz_gridstructure.N_z.(jstr(:));
                n_z=daz_gridstructure.n_z.(jstr(:));
                z_grid=daz_gridstructure.z_grid.(jstr(:));
                pi_z=daz_gridstructure.pi_z.(jstr(:));
                if vfoptions.lowmemory==1
                    special_n_z=ones(1,length(n_z));
                    z_gridvals=daz_gridstructure.z_gridvals.(jstr(:));
%                     z_gridvals=zeros(N_z,length(n_z),'gpuArray');
%                     for i1=1:N_z
%                         sub=zeros(1,length(n_z));
%                         sub(1)=rem(i1-1,n_z(1))+1;
%                         for ii=2:length(n_z)-1
%                             sub(ii)=rem(ceil(i1/prod(n_z(1:ii-1)))-1,n_z(ii))+1;
%                         end
%                         sub(length(n_z))=ceil(i1/prod(n_z(1:length(n_z)-1)));
%                         
%                         if length(n_z)>1
%                             sub=sub+[0,cumsum(n_z(1:end-1))];
%                         end
%                         z_gridvals(i1,:)=z_grid(sub);
%                     end
                end
            end
            
            V_j=zeros(N_a,N_z);
            Policy_j=zeros(N_a,N_z);
            
            if vfoptions.phiaprimedependsonage==1
                PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
                Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
            end
            
            if vfoptions.lowmemory==0
                ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
                for z_c=1:N_z
                    
                    EV_z=zeros(N_d,1);
                    for zprime_c=1:N_z
                        if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                            for d_c=1:N_d
                                EV_z(d_c)=EV_z(d_c)+(Vnextj(:,zprime_c).*Phi_aprimeMatrix(:,d_c,zprime_c))*pi_z(z_c,zprime_c);
                            end
                        end
                    end
                    
                    entireRHS_z=ReturnMatrix(:,:,z_c)+DiscountFactorParamsVec*EV_z*ones(1,N_a,1);
                    
                    %Calc the max and it's index
                    [Vtemp,maxindex]=max(entireRHS_z,[],1);
                    V_j(:,z_c)=Vtemp;
                    Policy_j(:,z_c)=maxindex;
                end
            elseif vfoptions.lowmemory==1
                for z_c=1:N_z
                    z_val=z_gridvals(z_c,:);
                    ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_grid, a_grid, z_val, ReturnFnParamsVec);
                    EV_z=zeros(N_d,1);
                    for zprime_c=1:N_z
                        if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                            for d_c=1:N_d
                                EV_z(d_c)=EV_z(d_c)+Vnextj(Phi_aprimeMatrix(d_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                            end
                        end
                    end
                    entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*EV_z*ones(1,N_a,1);
                    
                    %Calc the max and it's index
                    [Vtemp,maxindex]=max(entireRHS_z,[],1);
                    V_j(:,z_c)=Vtemp;
                    Policy_j(:,z_c)=maxindex;
                end
            elseif vfoptions.lowmemory==2
                EV_z=zeros(N_d,1);
                for z_c=1:N_Z
                    for zprime_c=1:N_z
                        if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                            for d_c=1:N_d
                                EV_z(d_c)=EV_z(d_c)+Vnextj(Phi_aprimeMatrix(d_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                            end
                        end
                    end
                    for a_c=1:N_a
                        a_val=a_gridvals(z_c,:);
                        ReturnMatrix_az=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, special_n_a, special_n_z, d_grid, a_val, z_val, ReturnFnParamsVec);
                        
                        entireRHS=ReturnMatrix_az+DiscountFactorParamsVec*EV_z; %aprime by 1
                        
                        %calculate in order, the maximizing aprime indexes
                        [V_j(a_c,z_c),Policy_j(a_c,z_c)]=max(entireRHS,[],1);
%                         % For howards improvement algorithm we want to keep
%                         Ftemp(a_c,z_c)=ReturnMatrix_az(Policy_j(a_c,z_c));
                    end
                end
            end
            
            V.(jstr)=V_j;
            Policy.(jstr)=Policy_j;
        end
    end
    
    if tempcounter>=5 % I simply assume you won't converge on the first try when using dynasty
        % No need to check convergence for the whole value function, if the
        % 'oldest', N_j, has converged then necessarily so have all the others.
        % Since I anyway want this jj=N_j for storing Vold (where only the
        % N_j value function needs to be stored, it is done just prior to
        % this if statement)
        N_a=daz_gridstructure.N_a.(jstrN_j);
        N_z=daz_gridstructure.N_z.(jstrN_j);
        Vdist=reshape(V.(jstrN_j)-Vold,[N_a*N_z,1]); Vdist(isnan(Vdist))=0;
        currdist=max(abs(Vdist)); %IS THIS reshape() & max() FASTER THAN max(max()) WOULD BE?
        
        if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.dynasty_maxhowards % Howards iteration method, adapted for dynasty
            % isfinite(currdist): ensures do not contaminate value function with -Infs
            % tempcounter>=2: just because there is probably not much point
            % tempcounter<vfoptions.dynasty_maxhowards: to guarantee that Howards cannot break convergence
            % currdist>(vfoptions.tolerance*10): to ensure eventual solution is not driven by Howards
            
            V_j=V.j001; % the first thing will be using N_j, and so will want to base it on the dynasty
            
            if Case2_Type==12  % phi_a'(d,a,z)
                % Ftemp and Phi_of_Policy have been saved (from every j)
                if vfoptions.lowmemory==0
                    for howards_c=1:vfoptions.dynasty_howards
                        for reverse_j=0:N_j-1
                            jj=N_j-reverse_j;
                            jstr=daz_gridstructure.jstr{jj};
                            
                            N_a=daz_gridstructure.N_a.(jstr(:));
                            N_aprime=daz_gridstructure.N_aprime.(jstr(:));
                            N_z=daz_gridstructure.N_z.(jstr(:));
                            N_zprime=daz_gridstructure.N_zprime.(jstr(:));
                            pi_z=daz_gridstructure.pi_z.(jstr(:));
                            DiscountFactorParamsVec=DiscountFactorParamsMatrix(jj,:); %CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
                            DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
                            
                            aaa=kron(pi_z,ones(N_a,1,'gpuArray'));
                            aaaPhi_of_Policy=kron(Phi_of_Policy.(jstr(:)),ones(N_zprime,1));
%                             Phi_aprimeMatrix=reshape(Phi_aprimeMatrix,[N_d*N_a*N_z,1]);
%                             aaaPhi_aprimeMatrix=kron(Phi_aprimeMatrix,ones(N_zprime,1));
                            zprime_ToMatchPhiOfPolicy=kron((1:1:N_zprime)',ones(N_a*N_z,1));

                            EV=V_j(aaaPhi_of_Policy+(N_aprime)*(zprime_ToMatchPhiOfPolicy-1)); % (1,z')
%                             EV=V_j(aaaPhi_aprimeMatrix+(N_aprime)*(zprime_ToMatchPhi-1)); %(d,z')
                            EV=reshape(EV,[N_a*N_a,N_zprime]);
                            EV=EV.*aaa;
%                             EV=reshape(EV,[N_d*N_a*N_z,N_zprime]);
%                             EV=EV.*aaa;
%                             EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%                             EV=sum(EV,2); % reshape(sum(EV_az,2),[N_d,1,1]);
%                             EV=reshape(EV,[N_d,N_a,N_z]);
                            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                            EV=sum(EV,2);
                            EV=reshape(EV,[N_a,N_z]);
                            V_j=Ftemp.(jstr(:))+DiscountFactorParamsVec*EV; % d by a by z
                            if howards_c==vfoptions.dynasty_howards % If on last iteration, then store them.
                                V.(jstr(:))=V_j;
                            end
                        end
                    end
                elseif vfoptions.lowmemory==1
                    for howards_c=1:vfoptions.dynasty_howards
                        for reverse_j=0:N_j-1
                            jj=N_j-reverse_j;
                            jstr=daz_gridstructure.jstr{jj};
                            
                            N_a=daz_gridstructure.N_a.(jstr(:));
                            N_aprime=daz_gridstructure.N_aprime.(jstr(:));
                            N_z=daz_gridstructure.N_z.(jstr(:));
                            N_zprime=daz_gridstructure.N_zprime.(jstr(:));
                            pi_z=daz_gridstructure.pi_z.(jstr(:));
                            z_gridvals=daz_gridstructure.z_gridvals.(jstr(:));
                            DiscountFactorParamsVec=DiscountFactorParamsMatrix(jj,:); %CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
                            DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
                            Phi_of_Policy_j=Phi_of_Policy.(jstr(:));
                            Ftemp_j=Ftemp.(jstr(:));
                            V_jplus1=V_j;
                            V_j=zeros(N_a,N_z,'gpuArray'); % Am going to overwrite, but need this to be the right size.
                            
                            zprime_ToMatchPhi=kron((1:1:N_zprime)',ones(N_a,1));
                            for z_c=1:N_z
                                % Vnextj is aprime-by-zprime
                                z_val=z_gridvals(z_c,:);
                                Phi_of_Policy_z=Phi_of_Policy_j(:,z_c);
                                Ftemp_z=Ftemp_j(:,z_c);
                                aaa=kron(pi_z(z_c,:),ones(N_a,1,'gpuArray'));
                                aaaPhi_aprimeMatrix_z=kron(ones(N_zprime,1),Phi_of_Policy_z);
                                
                                EV_z=V_jplus1(aaaPhi_aprimeMatrix_z+N_aprime*(zprime_ToMatchPhi-1)); % (d,aaprime,zprime) % SHOULD IT JUST BE N_aprime?
                                EV_z=reshape(EV_z,[N_a,N_zprime]);
                                EV_z=EV_z.*aaa;
                                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%                                 EV_z=reshape(sum(EV_z,2),[N_d,N_a]);
                                EV_z=sum(EV_z,2);
                                
                                V_j(:,z_c)=Ftemp_z+DiscountFactorParamsVec*EV_z; % d by a
                                
                            end
                            if howards_c==vfoptions.dynasty_howards % If on last iteration, then store them.
                                jstr=daz_gridstructure.jstr{jj};
                                V.(jstr(:))=V_j;
                            end
                        end
                    end
                elseif lowmemory==2
                    fprintf('WARNING: Have not yet implemented combo of Howards>0 and lowmemory=2 for Dynasty in Case2_FHorz_AgeDepGrids')
                    dbstack
                end
                
            end

        end
    end
    Vold=V.(jstrN_j); % Only the final period one is ever needed
    
    tempcounter=tempcounter+1;
    if vfoptions.verbose==1 && rem(tempcounter,10)==0
        fprintf('Value Fn Iteration: After %d steps, current distance is %8.2f \n', tempcounter, currdist);
    end
end

end
