function [VKron,PolicyIndexesKron]=ValueFnIter_Case2_FHorz_raw(n_d, n_a, N_z, N_j, pi_z, Phi_aprimeKronFn_j, Case2_Type, beta_j, FmatrixFn_j)
    
N_d=prod(n_d); %Had to make this change to allow for custom Case5 for Imai & Keane (2004).
N_a=prod(n_a);

%disp('Starting Value Fn Iteration')
VKron=zeros(N_a,N_z,N_j);
PolicyIndexesKron=zeros(N_a,N_z,N_j); %indexes the optimal choice for d given rest of dimensions a,z

if Case2_Type==5
    FmatrixKron_j=reshape(FmatrixFn_j(N_j+19),[N_d,N_a,N_z]);
    for z_c=1:N_z
        for a_c=1:N_a
            [VKron(a_c,z_c,N_j),PolicyIndexesKron(a_c,z_c,N_j)]=max(FmatrixKron_j(:,a_c,z_c),[],1);
        end
    end    
else
    FmatrixKron_j=reshape(FmatrixFn_j(N_j),[N_d,N_a,N_z]);
    for z_c=1:N_z
        for a_c=1:N_a
            [VKron(a_c,z_c,N_j),PolicyIndexesKron(a_c,z_c,N_j)]=max(FmatrixKron_j(:,a_c,z_c),[],1);
        end
    end
end

if Case2_Type==1
    for reverse_j=1:N_j-1
        j=N_j-reverse_j;
        VKronNext_j=VKron(:,:,j+1);
        FmatrixKron_j=reshape(FmatrixFn_j(j),[N_d,N_a,N_z]);
        Phi_aprimeKron=Phi_aprimeKronFn_j(j);
        for z_c=1:N_z
            for a_c=1:N_a
                RHSpart2=zeros(N_d,1);
                for zprime_c=1:N_z
                    if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                        for d_c=1:N_d
                            RHSpart2(d_c)=RHSpart2(d_c)+VKronNext_j(Phi_aprimeKron(d_c,a_c,z_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                        end
                    end
                end
                entireRHS=FmatrixKron_j(:,a_c,z_c)+beta_j(j)*RHSpart2; %aprime by 1
                
                %calculate in order, the maximizing aprime indexes
                [VKron(a_c,z_c,j),PolicyIndexesKron(a_c,z_c,j)]=max(entireRHS,[],1);
            end
        end
    end
end

if Case2_Type==2
    for reverse_j=1:N_j-1
        j=N_j-reverse_j;
        VKronNext_j=VKron(:,:,j+1);
        FmatrixKron_j=reshape(FmatrixFn_j(j),[N_d,N_a,N_z]);
        Phi_aprimeKron=Phi_aprimeKronFn_j(j);
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
                entireRHS=FmatrixKron_j(:,a_c,z_c)+beta_j(j)*RHSpart2; %aprime by 1
                
                %calculate in order, the maximizing aprime indexes
                [VKron(a_c,z_c,j),PolicyIndexesKron(a_c,z_c,j)]=max(entireRHS,[],1);
            end
        end
    end
end


if Case2_Type==3
    for reverse_j=1:N_j-1
        j=N_j-reverse_j;
        VKronNext_j=VKron(:,:,j+1);
        FmatrixKron_j=reshape(FmatrixFn_j(j),[N_d,N_a,N_z]);
        Phi_aprimeKron=Phi_aprimeKronFn_j(j);
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
                entireRHS=FmatrixKron_j(:,a_c,z_c)+beta_j(j)*RHSpart2; %aprime by 1
                
                %calculate in order, the maximizing aprime indexes
                [VKron(a_c,z_c,j),PolicyIndexesKron(a_c,z_c,j)]=max(entireRHS,[],1);
            end
        end
    end
end

if Case2_Type==4
    for reverse_j=1:N_j-1
        j=N_j-reverse_j;
        VKronNext_j=VKron(:,:,j+1);
        FmatrixKron_j=reshape(FmatrixFn_j(j),[N_d,N_a,N_z]);
        Phi_aprimeKron=Phi_aprimeKronFn_j(j);
        for z_c=1:N_z
            RHSpart2=zeros(N_d,1);
            for zprime_c=1:N_z
                if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                    for d_c=1:N_d
                        for a_c=1:N_a
                            RHSpart2(d_c)=RHSpart2(d_c)+VKronNext_j(Phi_aprimeKron(d_c,a_c),zprime_c)*pi_z(z_c,zprime_c);
                        end
                    end
                end
            end
            for a_c=1:N_a
                entireRHS=FmatrixKron_j(:,a_c,z_c)+beta_j(j)*RHSpart2; %aprime by 1
                
                %calculate in order, the maximizing aprime indexes
                [VKron(a_c,z_c,j),PolicyIndexesKron(a_c,z_c,j)]=max(entireRHS,[],1);
            end
        end
    end
end

if Case2_Type==5 %Custom Case2_Type written just for Imai & Keane (2004).
    for reverse_j=1:N_j-1
        j=N_j-reverse_j;
        VKronNext_j=VKron(:,:,j+1);
        FmatrixKron_j=reshape(FmatrixFn_j(j+19),[N_d,N_a,N_z]);
        Phi_aprimeKron=Phi_aprimeKronFn_j(j+19); %Size is [n_a(2),n_d(1),n_a(1)], or in notation used in codes for Imai & Keane (2004), [hprime, l,a]. Unlike all other Case2's this one does not give a specific value for hprime.
                                              %rather it gives a probability distn over future hprimes.
        for z_c=1:N_z
            RHSpart2=zeros(N_d,1);
            for zprime_c=1:N_z
                if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                    for l_c=1:n_d(1) %hours worked, l
                        for h_c=1:n_a(2) %human capital, h
                            for hprime_c=1:n_a(2) %hprime
                                if Phi_aprimeKron(hprime_c,l_c,h_c)>0 %This will help avoid -Inf*0 problems
                                    for aprime_c=1:n_a(1)
                                        ahprime_c=sub2ind_homemade(n_a,[aprime_c,hprime_c]);
                                        RHSpart2(l_c)=RHSpart2(l_c)+Phi_aprimeKron(hprime_c,l_c,h_c)*VKronNext_j(ahprime_c,zprime_c)*pi_z(z_c,zprime_c);
                                    end
                                end
                            end
                        end
                    end
                end
            end
            for a_c=1:N_a
                entireRHS=FmatrixKron_j(:,a_c,z_c)+beta_j(j+19)*RHSpart2; %aprime by 1
                
                %calculate in order, the maximizing aprime indexes
                [VKron(a_c,z_c,j),PolicyIndexesKron(a_c,z_c,j)]=max(entireRHS,[],1);
            end
        end
    end
end

end