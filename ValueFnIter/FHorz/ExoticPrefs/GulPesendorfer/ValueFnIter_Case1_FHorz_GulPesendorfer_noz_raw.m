function [V,Policy2]=ValueFnIter_Case1_FHorz_GulPesendorfer_noz_raw(n_d,n_a,N_j, d_grid, a_grid, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);

if vfoptions.lowmemory>0
    special_n_a=ones(1,length(n_a));
    a_gridvals=CreateGridvals(n_a,a_grid,1); % The 1 at end indicates want output in form of matrix.
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, n_d, n_a, d_grid, a_grid, ReturnFnParamsVec,0);

        TemptationMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(TemptationFn, n_d, n_a, d_grid, a_grid, TemptationFnParamsVec,0);
        MostTempting=max(TemptationMatrix,[],1);
        entireRHS=ReturnMatrix+TemptationMatrix-ones(N_d*N_a,1).*MostTempting;

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,N_j)=Vtemp;
        Policy(:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1
        %if vfoptions.returnmatrix==2 % GPU
        for a_c=1:N_a
            a_val=a_gridvals(a_c,:);
            ReturnMatrix_a=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, n_d, special_n_a, d_grid, a_val, ReturnFnParamsVec,0);

            TemptationMatrix_a=CreateReturnFnMatrix_Case1_Disc_noz_Par2(TemptationFn, n_d, special_n_a, d_grid, a_val, TemptationFnParamsVec,0);
            MostTempting_a=max(TemptationMatrix_a,[],1);
            entireRHS_a=ReturnMatrix_a+TemptationMatrix_a-ones(N_d*N_a,1).*MostTempting_a;

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_a,[],1);
            V(a_c,N_j)=Vtemp;
            Policy(a_c,N_j)=maxindex;
        end
    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,1]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    if vfoptions.lowmemory==0
        
        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, n_d, n_a, d_grid, a_grid, ReturnFnParamsVec,0);
        % (d,aprime,a)
        
        entireEV=kron(V_Jplus1,ones(N_d,1));

        TemptationMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(TemptationFn, n_d, n_a, d_grid, a_grid, TemptationFnParamsVec,0);
        MostTempting=max(TemptationMatrix,[],1);
        entireRHS=ReturnMatrix+TemptationMatrix-ones(N_d*N_a,1).*MostTempting+DiscountFactorParamsVec*entireEV*ones(1,N_a,1); % aprime-by-a
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,N_j)=Vtemp;
        Policy(:,N_j)=maxindex;
        
    elseif vfoptions.lowmemory==1
       
       for a_c=1:N_a
           a_val=a_gridvals(a_c,:);
           ReturnMatrix_a=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, n_d, special_n_a, d_grid, a_val, ReturnFnParamsVec,0);
           
           TemptationMatrix_a=CreateReturnFnMatrix_Case1_Disc_noz_Par2(TemptationFn, n_d, special_n_a, d_grid, a_val, TemptationFnParamsVec,0);
           MostTempting_a=max(TemptationMatrix_a,[],1);
           entireRHS_a=ReturnMatrix_a+TemptationMatrix_a-ones(N_d*N_a,1).*MostTempting_a+DiscountFactorParamsVec*V_Jplus1; % aprime-by-1
           
           %Calc the max and it's index
           [Vtemp,maxindex]=max(entireRHS_a,[],1);
           V(a_c,N_j)=Vtemp;
           Policy(a_c,N_j)=maxindex;
        end
        
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
    TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames, jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    VKronNext_j=V(:,jj+1);
    
    if vfoptions.lowmemory==0
        
        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, n_d, n_a, d_grid, a_grid, ReturnFnParamsVec,0);
        % (d,aprime,a)
        
        entireEV=kron(VKronNext_j,ones(N_d,1));

        TemptationMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(TemptationFn, n_d, n_a, d_grid, a_grid, TemptationFnParamsVec,0);
        MostTempting=max(TemptationMatrix,[],1);
        entireRHS=ReturnMatrix+TemptationMatrix-ones(N_d*N_a,1).*MostTempting+DiscountFactorParamsVec*entireEV*ones(1,N_a,1); % aprime-by-a
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,jj)=Vtemp;
        Policy(:,jj)=maxindex;
        
    elseif vfoptions.lowmemory==1
       
       for a_c=1:N_a
           a_val=a_gridvals(a_c,:);
           ReturnMatrix_a=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, n_d, special_n_a, d_grid, a_val, ReturnFnParamsVec,0);
           
           TemptationMatrix_a=CreateReturnFnMatrix_Case1_Disc_noz_Par2(TemptationFn, n_d, special_n_a, d_grid, a_val, TemptationFnParamsVec,0);
           MostTempting_a=max(TemptationMatrix_a,[],1);
           entireRHS_a=ReturnMatrix_a+TemptationMatrix_a-ones(N_d*N_a,1).*MostTempting_a+DiscountFactorParamsVec*VKronNext_j; % aprime-by-1
           
           %Calc the max and it's index
           [Vtemp,maxindex]=max(entireRHS_a,[],1);
           V(a_c,jj)=Vtemp;
           Policy(a_c,jj)=maxindex;
        end
        
    end
end

%%
Policy2=zeros(2,N_a,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
Policy2(1,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:)=shiftdim(ceil(Policy/N_d),-1);

end