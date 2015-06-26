function [VKron,PolicyIndexesKron]=ValueFnIter_Case2_Par2_raw(VKron, n_d, n_a, n_z, pi_z, beta, ReturnMatrix, Phi_aprime, Case2_Type, Howards, Verbose, Tolerance)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

PolicyIndexesKron=zeros(N_a,N_z,'gpuArray'); %indexes the optimal choice for d given rest of dimensions a,z
Ftemp=zeros(N_a,N_z,'gpuArray');

% bbb=reshape(pi_z,[1,N_z*N_z]);
% %bbb=reshape(shiftdim(pi_z,-1),[1,N_z*N_z]);
% ccc=kron(ones(N_a,1,'gpuArray'),bbb);
% aaa=reshape(ccc,[N_a*N_z,N_z]);
aaa=kron(ones(N_a,1,'gpuArray'),pi_z);

tempcounter=1; currdist=Inf;

if Case2_Type==1
    disp('ERROR: Case2_Type==1 has not yet been implemented for GPU')
%     while currdist>Tolerance
%         
%         VKronold=VKron;
%         
%         for z_c=1:N_z
%             for a_c=1:N_a
%                 %first calc the second half of the RHS (except beta)
%                 RHSpart2=zeros(N_d,1);
%                 for zprime_c=1:N_z
%                     if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%                         %                     Phi_of_d=Phi_aprime(:,a_c,z_c,zprime_c);
%                         %                     RHSpart2=RHSpart2+VKronold(Phi_of_d,zprime_c)*pi_z(z_c,zprime_c);
%                         for d_c=1:N_d
%                             RHSpart2(d_c)=RHSpart2(d_c)+VKronold(Phi_aprime(d_c,a_c,z_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
%                         end
%                     end
%                 end
%                 entireRHS=Fmatrix(:,a_c,z_c)+beta*RHSpart2; %d by 1
%                 
%                 %then maximizing d indexes
%                 [VKron(a_c,z_c),PolicyIndexesKron(a_c,z_c)]=max(entireRHS,[],1);
%             end
%         end
%         
%         for z_c=1:N_z
%             ReturnMatrix_z=ReturnMatrix(:,:,z_c);
%             Phi_aprime_z=Phi_aprime(:,:,z_c,:);
%             Calc the condl expectation term (except beta), which depends on z but
%             not on control variables
%             EV_z=VKronold.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
%             EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%             EV_z=sum(EV_z,2);
%             EV_z=VKronold.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
%             EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%             EV_z=sum(EV_z,2);
%             
%             entireEV_z=kron(EV_z,ones(N_d,1));
%             entireRHS=ReturnMatrix_z+beta*entireEV_z*ones(1,N_a,1);
%             
%             Calc the max and it's index
%             [Vtemp,maxindex]=max(entireRHS,[],1);
%             VKron(:,z_c)=Vtemp;
%             PolicyIndexes(:,z_c)=maxindex;
%             
%             tempmaxindex=maxindex+(0:1:N_a-1)*(N_d*N_a);
%             Ftemp(:,z_c)=ReturnMatrix_z(tempmaxindex);
%         end
%         
%         VKrondist=reshape(VKron-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
%         currdist=max(abs(VKrondist));
%         
%         if isfinite(currdist) %Use Howards Policy Fn Iteration Improvement
%             for Howards_counter=1:Howards
%                 VKrontemp=VKron;
%                 for a_c=1:N_a
%                     for z_c=1:N_z
%                         %VKron(a_c,z_c)=FmatrixKron(PolicyIndexesKron(a_c,z_c),a_c,z_c);
%                         %for zprime_c=1:N_z %Note: There is probably some better way to do this with matrix algebra that saves looping over zprime_c
%                         %(ie. more like the Howards improvement algorithm code implemented in Case 1)
%                         %The difficulty is just getting the indexes for VKrontemp right
%                         temp=0;
%                         for zprime_c=1:N_z
%                             if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%                                 temp=temp+VKrontemp(Phi_aprime(PolicyIndexesKron(a_c,z_c),a_c,z_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
%                             end
%                         end
%                         VKron(a_c,z_c)=ReturnMatrix(PolicyIndexesKron(a_c,z_c),a_c,z_c)+beta*temp;
%                         %VKron(a_c,z_c)=VKron(a_c,z_c)+beta*VKrontemp(Phi_aprimeKron(PolicyIndexesKron(a_c,z_c),a_c,z_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
%                         %end
%                     end
%                 end
%             end
%         end
%         
%         if rem(tempcounter,100)==0
%             disp(tempcounter)
%             disp(currdist)
%         end
%         
%         tempcounter=tempcounter+1;
%     end
end


if Case2_Type==2
    
    Phi_of_Policy=zeros(N_a,N_z,N_z,'gpuArray'); %a'(a,z',z)
%     Phi_of_Policy2=zeros(N_a,N_z,N_z,'gpuArray'); %a'(a,z',z)
    % SHOULD IT BE pi_z OR pi_z' IN aaa ????? (NEED TO CHECK THIS)
    aaa=kron(ones(N_a,1,'gpuArray'),pi_z); % Part of one of the Howards Implemantations
    
    while currdist>Tolerance
        
        VKronold=VKron;
        
%         tic;
        for z_c=1:N_z
            ReturnMatrix_z=ReturnMatrix(:,:,z_c);
            Phi_aprime_z=Phi_aprime(:,:,z_c); %Case2_Type==2: phi(d,z,z')
            %Calc the condl expectation term (except beta), which depends on z
            EV_z=VKronold(Phi_aprime_z).*(ones(N_d,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            entireRHS=ReturnMatrix_z+beta*EV_z*ones(1,N_a,1,'gpuArray');
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);
            VKron(:,z_c)=Vtemp;
            PolicyIndexesKron(:,z_c)=maxindex;
            
            tempmaxindex=maxindex+(0:1:N_a-1)*(N_d);
            Ftemp(:,z_c)=ReturnMatrix_z(tempmaxindex);
            
            % Phi_of_Policy2(:,:,z_c)=Phi_aprime_z(maxindex,:);
        end
%         time1=toc;
        
        VKrondist=reshape(VKron-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
        currdist=max(abs(VKrondist));

        % I HAVE DISABLED HOWARDS AS FOR SOME REASON IT DOESN'T SEEM TO BE WORKING CORRECTLY. 
        % I CANNOT FIGURE OUT WHY BUT EVKrontemp TERM APPEARS TO CALCULATE INCORRECTLY. 
        % MAYBE EVEN JUST ROUNDING ERRORS???    
%         tic;
        if isfinite(currdist) %Use Howards Policy Fn Iteration Improvement
            for z_c=1:N_z
                Phi_of_Policy(:,:,z_c)=Phi_aprime(PolicyIndexesKron(:,z_c),:,z_c);
            end
            % %             Phi_of_Policy=Phi_of_Policy2; % They are equal, use
            % %                     % whichever is fastest! appears to be largely a deadheat, have just gone with Phi_of_Policy
            
%             Htimes=[0,0,0];
            for Howards_counter=1:Howards
                % Version 3 of the implementation appears to be fastest
                
%                 % Howards Version 1
%                 tic;
%                 VKrontemp=VKron;
%                 EVKrontemp=zeros(N_a,N_z,'gpuArray'); % Can move this outside most of the loops
%                 for z_c=1:N_z
%                     temp1=reshape(VKrontemp(Phi_of_Policy(:,:,z_c),z_c),[N_a,N_z]).*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
%                     temp1(isnan(temp1))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%                     VKron(:,z_c)=Ftemp(:,z_c)+beta*sum(temp1,2);
%                 end
%                 Htimes(1)=Htimes(1)+toc;
%                 
%                 % Howards Version 2
%                 tic;
%                 EVKrontemp=zeros(N_a,N_z,'gpuArray'); % Can move this outside most of the loops
%                 for z_c=1:N_z
%                     temp1=reshape(VKron(Phi_of_Policy(:,:,z_c),z_c),[N_a,N_z]).*kron(ones(N_a,1,'gpuArray'),pi_z(z_c,:));
%                     temp1(isnan(temp1))=0;
%                     EVKrontemp(:,z_c)=sum(temp1,2);
%                 end
%                 VKron=Ftemp+beta*EVKrontemp;
%                 Htimes(2)=Htimes(2)+toc;
                
                % Howards Version 3
%                 tic;
                EVKrontemp=zeros(N_a*N_z,N_z,'gpuArray'); % Can move this outside most of the loops
                for z_c=1:N_z
                    EVKrontemp(:,z_c)=VKron(Phi_of_Policy(:,:,z_c),z_c);
                end
                EVKrontemp=EVKrontemp.*aaa;
                EVKrontemp(isnan(EVKrontemp))=0;
                EVKrontemp=reshape(sum(reshape(EVKrontemp,[N_a,N_z,N_z]),2),[N_a,N_z]);
                VKron=Ftemp+beta*EVKrontemp;
%                 Htimes(3)=Htimes(3)+toc;
                
            end
        end
%         Htimes
%         time2=toc;
  
        if rem(tempcounter,100)==0
            disp(tempcounter)
            disp(currdist)
%             fprintf('times: %2.8f, %2.8f \n',time1,time2)
        end
        
        tempcounter=tempcounter+1;
    end
end



if Case2_Type==3
    disp('ERROR: Case2_Type==3 has not yet been implemented for GPU')
%     Phi_of_Policy=zeros(N_a,N_z,N_z,'gpuArray'); %a'(a,z',z)
% 
%     while currdist>Tolerance
%         
%         VKronold=VKron;
%         
%         for z_c=1:N_z
%             %Calc the condl expectation (except beta)
%             RHSpart2=zeros(N_d,1);
%             for zprime_c=1:N_z
%                 if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%                     for d_c=1:N_d
%                         RHSpart2(d_c)=RHSpart2(d_c)+VKronold(Phi_aprime(d_c),zprime_c)*pi_z(z_c,zprime_c);
%                     end
%                 end
%             end
%             
%             for a_c=1:N_a
%                 entireRHS=ReturnMatrix(:,a_c,z_c)+beta*RHSpart2; %d by 1
%                 
%                 %then maximizing d indexes
%                 [VKron(a_c,z_c),PolicyIndexesKron(a_c,z_c)]=max(entireRHS,[],1);
%             end
%         end
%         
%         VKrondist=reshape(VKron-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
%         currdist=max(abs(VKrondist));
%         
%         if isfinite(currdist) %Use Howards Policy Fn Iteration Improvement
%             for Howards_counter=1:Howards
%                 VKrontemp=VKron;
%                 for z_c=1:N_z
%                     for a_c=1:N_a
%                         %VKron(a_c,z_c)=FmatrixKron(PolicyIndexesKron(a_c,z_c),a_c,z_c);
%                         %for zprime_c=1:N_z %Note: There is probably some better way to do this with matrix algebra that saves looping over zprime_c
%                         %(ie. more like the Howards improvement algorithm code implemented in Case 1)
%                         %The difficulty is just getting the indexes for VKrontemp right
%                         temp=0;
%                         for zprime_c=1:N_z
%                             if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%                                 temp=temp+VKrontemp(Phi_aprime(PolicyIndexesKron(a_c,z_c)),zprime_c)*pi_z(z_c,zprime_c);
%                             end
%                         end
%                         VKron(a_c,z_c)=ReturnMatrix(PolicyIndexesKron(a_c,z_c),a_c,z_c)+beta*temp;
%                         %VKron(a_c,z_c)=VKron(a_c,z_c)+beta*VKrontemp(Phi_aprimeKron(PolicyIndexesKron(a_c,z_c),a_c,z_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
%                         %end
%                     end
%                 end
%             end
%         end
%         
%         if rem(tempcounter,100)==0
%             disp(tempcounter)
%             disp(currdist)
%         end
%         
%         tempcounter=tempcounter+1;
%     end
end




    
end