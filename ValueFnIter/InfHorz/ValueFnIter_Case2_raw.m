function [VKron,PolicyIndexesKron]=ValueFnIter_Case2_raw(VKron, n_d, n_a, n_z, pi_z, beta, Fmatrix, Phi_aprime, Case2_Type, Howards, Verbose, Tolerance)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

PolicyIndexesKron=zeros(N_a,N_z); %indexes the optimal choice for d given rest of dimensions a,z
tempcounter=1; currdist=Inf;

if Case2_Type==1
    while currdist>Tolerance
        
        VKronold=VKron;
        
        for z_c=1:N_z
            for a_c=1:N_a
                %first calc the second half of the RHS (except beta)
                RHSpart2=zeros(N_d,1);
                for zprime_c=1:N_z
                    if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                        %                     Phi_of_d=Phi_aprimeKron(:,a_c,z_c,zprime_c);
                        %                     RHSpart2=RHSpart2+VKronold(Phi_of_d,zprime_c)*pi_z(z_c,zprime_c);
                        for d_c=1:N_d
                            RHSpart2(d_c)=RHSpart2(d_c)+VKronold(Phi_aprime(d_c,a_c,z_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                        end
                    end
                end
                entireRHS=Fmatrix(:,a_c,z_c)+beta*RHSpart2; %d by 1
                
                %then maximizing d indexes
                [VKron(a_c,z_c),PolicyIndexesKron(a_c,z_c)]=max(entireRHS,[],1);
            end
        end
        
        VKrondist=reshape(VKron-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
        currdist=max(abs(VKrondist));
        
        if isfinite(currdist) %Use Howards Policy Fn Iteration Improvement
            for Howards_counter=1:Howards
                VKrontemp=VKron;
                for a_c=1:N_a
                    for z_c=1:N_z
                        %VKron(a_c,z_c)=FmatrixKron(PolicyIndexesKron(a_c,z_c),a_c,z_c);
                        %for zprime_c=1:N_z %Note: There is probably some better way to do this with matrix algebra that saves looping over zprime_c
                        %(ie. more like the Howards improvement algorithm code implemented in Case 1)
                        %The difficulty is just getting the indexes for VKrontemp right
                        temp=0;
                        for zprime_c=1:N_z
                            if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                                temp=temp+VKrontemp(Phi_aprime(PolicyIndexesKron(a_c,z_c),a_c,z_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                            end
                        end
                        VKron(a_c,z_c)=Fmatrix(PolicyIndexesKron(a_c,z_c),a_c,z_c)+beta*temp;
                        %VKron(a_c,z_c)=VKron(a_c,z_c)+beta*VKrontemp(Phi_aprimeKron(PolicyIndexesKron(a_c,z_c),a_c,z_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                        %end
                    end
                end
            end
        end
        
        if rem(tempcounter,100)==0
            disp(tempcounter)
            disp(currdist)
        end
        
        tempcounter=tempcounter+1;
    end
end


if Case2_Type==2
    while currdist>Tolerance
        
        VKronold=VKron;
        
        for z_c=1:N_z
            %Calc the condl expectation (except beta)
            RHSpart2=zeros(N_d,1);
            for zprime_c=1:N_z
                if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                    for d_c=1:N_d
                        RHSpart2(d_c)=RHSpart2(d_c)+VKronold(Phi_aprime(d_c,z_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                    end
                end
            end
            
            for a_c=1:N_a
                entireRHS=Fmatrix(:,a_c,z_c)+beta*RHSpart2; %d by 1
                
                %then maximizing d indexes
                [VKron(a_c,z_c),PolicyIndexesKron(a_c,z_c)]=max(entireRHS,[],1);
            end
        end
        
        VKrondist=reshape(VKron-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
        currdist=max(abs(VKrondist));
        
        if isfinite(currdist) %Use Howards Policy Fn Iteration Improvement
            for Howards_counter=1:Howards
                VKrontemp=VKron;
                for z_c=1:N_z
                    for a_c=1:N_a
                        %VKron(a_c,z_c)=FmatrixKron(PolicyIndexesKron(a_c,z_c),a_c,z_c);
                        %for zprime_c=1:N_z %Note: There is probably some better way to do this with matrix algebra that saves looping over zprime_c
                        %(ie. more like the Howards improvement algorithm code implemented in Case 1)
                        %The difficulty is just getting the indexes for VKrontemp right
                        temp=0;
                        for zprime_c=1:N_z
                            if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                                temp=temp+VKrontemp(Phi_aprime(PolicyIndexesKron(a_c,z_c),z_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                            end
                        end
                        VKron(a_c,z_c)=Fmatrix(PolicyIndexesKron(a_c,z_c),a_c,z_c)+beta*temp;
                        %VKron(a_c,z_c)=VKron(a_c,z_c)+beta*VKrontemp(Phi_aprimeKron(PolicyIndexesKron(a_c,z_c),a_c,z_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                        %end
                    end
                end
            end
        end
        
        if rem(tempcounter,100)==0
            disp(tempcounter)
            disp(currdist)
        end
        
        tempcounter=tempcounter+1;
    end
end



if Case2_Type==3
    while currdist>Tolerance
        
        VKronold=VKron;
        
        for z_c=1:N_z
            %Calc the condl expectation (except beta)
            RHSpart2=zeros(N_d,1);
            for zprime_c=1:N_z
                if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                    for d_c=1:N_d
                        RHSpart2(d_c)=RHSpart2(d_c)+VKronold(Phi_aprime(d_c),zprime_c)*pi_z(z_c,zprime_c);
                    end
                end
            end
            
            for a_c=1:N_a
                entireRHS=Fmatrix(:,a_c,z_c)+beta*RHSpart2; %d by 1
                
                %then maximizing d indexes
                [VKron(a_c,z_c),PolicyIndexesKron(a_c,z_c)]=max(entireRHS,[],1);
            end
        end
        
        VKrondist=reshape(VKron-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
        currdist=max(abs(VKrondist));
        
        if isfinite(currdist) %Use Howards Policy Fn Iteration Improvement
            for Howards_counter=1:Howards
                VKrontemp=VKron;
                for z_c=1:N_z
                    for a_c=1:N_a
                        %VKron(a_c,z_c)=FmatrixKron(PolicyIndexesKron(a_c,z_c),a_c,z_c);
                        %for zprime_c=1:N_z %Note: There is probably some better way to do this with matrix algebra that saves looping over zprime_c
                        %(ie. more like the Howards improvement algorithm code implemented in Case 1)
                        %The difficulty is just getting the indexes for VKrontemp right
                        temp=0;
                        for zprime_c=1:N_z
                            if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                                temp=temp+VKrontemp(Phi_aprime(PolicyIndexesKron(a_c,z_c)),zprime_c)*pi_z(z_c,zprime_c);
                            end
                        end
                        VKron(a_c,z_c)=Fmatrix(PolicyIndexesKron(a_c,z_c),a_c,z_c)+beta*temp;
                        %VKron(a_c,z_c)=VKron(a_c,z_c)+beta*VKrontemp(Phi_aprimeKron(PolicyIndexesKron(a_c,z_c),a_c,z_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                        %end
                    end
                end
            end
        end
        
        if rem(tempcounter,100)==0
            disp(tempcounter)
            disp(currdist)
        end
        
        tempcounter=tempcounter+1;
    end
end




    
end