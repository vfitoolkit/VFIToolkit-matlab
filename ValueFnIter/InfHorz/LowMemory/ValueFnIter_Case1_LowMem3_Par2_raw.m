function [VKron, Policy]=ValueFnIter_Case1_LowMem3_Par2_raw(VKron, n_d,n_a,n_zsp, d_grid,a_grid,z_grid, pi_szp, beta, ReturnFn, ReturnFnParams, Howards,Howards2,Tolerance,lowmemorydimensions) %Verbose,
% The code works on the assumption that the lowmemorydimensions are the 'first' dimensions (s & z, but not p; this subfn was developed specifically for the Hayek method)

N_d=prod(n_d);
N_a=prod(n_a);
N_szp=prod(n_zsp);
l_szp=length(n_zsp);

% n_z1=n_z(lowmemorydimensions);
% n_z2=n_z(setdiff(1:1:N_z,lowmemorydimensions));
% N_z1=prod(n_z1);
% N_z2=prod(n_z2);
% z1_grid=z_grid(1:sum(n_z1));
% z2_grid=z_grid(sum(n_z1)+1:end);
n_sz=n_zsp(lowmemorydimensions);
n_p=n_zsp(setdiff(1:1:l_szp,lowmemorydimensions));
N_sz=prod(n_sz);
N_p=prod(n_p);
l_sz=length(n_sz);
sz_grid=z_grid(1:sum(n_sz));
p_grid=z_grid(sum(n_sz)+1:end);

PolicyIndexes=zeros(N_a,N_szp,'gpuArray');

Ftemp=zeros(N_a,N_szp,'gpuArray');

bbb=reshape(shiftdim(pi_szp,-1),[1,N_szp*N_szp]);
ccc=kron(ones(N_a,1,'gpuArray'),bbb);
aaa=reshape(ccc,[N_a*N_szp,N_szp]);

%%
% % Get pi_szp_p from pi_szp
% pi_szp_p=reshape(pi_szp,[N_szp,N_sz,N_p]);
% pi_szp_p=sum(pi_szp_p,2);
% pi_szp_p=permute(pi_szp_p,[1,3,2]); % Would this be faster as just pi_szp_p=reshape(pi_szp_p,[N_szp,Np]) ?

pi_p_szp_sz=reshape(pi_szp,[N_sz,N_p,N_szp]); % sz, p to next period szp
pi_p_szp_sz=permute(pi_p_szp_sz,[2,3,1]); % p to next period szp, third dimensions is today's sz

%%
sz_gridvals=CreateGridvals(n_sz,sz_grid,1); % 1 is to create sz_gridvals as matrix

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance
    VKronold=VKron;
    
    
    for sz_c=1:N_sz
        
        szvals=sz_gridvals(sz_c,:);
        szvalspgrid=[szvals';p_grid];
        
        ReturnMatrix_sz=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn,n_d, n_a, [ones(1,l_sz),n_p],d_grid, a_grid, szvalspgrid,ReturnFnParams);
        pi_p_szp_given_sz=pi_p_szp_sz(:,:,sz_c);
        for p_c=1:N_p
            ReturnMatrix_szp=ReturnMatrix_sz(:,:,p_c);
            %Calc the condl expectation term (except beta), which depends on z but not on control variables
            %         EV_sz=VKronold.*(ones(N_a,1,'gpuArray')*pi_z(sz_c,:));
            EV_szp=VKronold.*(ones(N_a,1,'gpuArray')*pi_p_szp_given_sz(p_c,:));
            EV_szp(isnan(EV_szp))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_szp=sum(EV_szp,2);
            
            entireEV_szp=kron(EV_szp,ones(N_d,1));
            entireRHS=ReturnMatrix_szp+beta*entireEV_szp*ones(1,N_a,1);
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);
            
            szp_c=sub2ind_homemade([N_sz,N_p],[sz_c,p_c]);
            
            VKron(:,szp_c)=Vtemp;
            PolicyIndexes(:,szp_c)=maxindex;
            
            tempmaxindex=maxindex+(0:1:N_a-1)*(N_d*N_a);
            Ftemp(:,szp_c)=ReturnMatrix_szp(tempmaxindex);
        end
    end

    VKrondist=reshape(VKron-VKronold,[N_a*N_szp,1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist)); %IS THIS reshape() & max() FASTER THAN max(max()) WOULD BE?

    if isfinite(currdist) && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        for Howards_counter=1:Howards
            VKrontemp=VKron;
            
            VKrontemp(~isfinite(VKrontemp))=0;
            
            EVKrontemp=VKrontemp(ceil(PolicyIndexes/N_d),:);
            EVKrontemp=EVKrontemp.*aaa;
%             EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_szp]);
            VKron=Ftemp+beta*EVKrontemp;
        end
    end
    
%     if Verbose==1
%         if rem(tempcounter,100)==0
%             disp(tempcounter)
%             disp(currdist)
%         end
%         tempcounter=tempcounter+1;
%     end
    tempcounter=tempcounter+1;
end

Policy=zeros(2,N_a,N_szp,'gpuArray'); %NOTE: this is not actually in Kron form
Policy(1,:,:)=shiftdim(rem(PolicyIndexes-1,N_d)+1,-1);
Policy(2,:,:)=shiftdim(ceil(PolicyIndexes/N_d),-1);

end