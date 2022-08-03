function [V,Policy]=ValueFnIter_Case1_EndoType_TPath_SingleStep_Refine2_raw(Vnext,n_d,n_endostate,n_endotype,n_z,d_grid, endostate_grid, endotype_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
%% Refinement: calculate ReturnMatrix and 'remove' the d dimension

l_d=length(n_d);
N_endostate=prod(n_endostate);
N_endotype=prod(n_endotype);
N_z=prod(n_z);

% lowmemory acts differently for 'Refinement2'. It is only used for the return function.

ptsperlayer=vfoptions.refine_pts;
nlayers=vfoptions.refine_iter;

if rem(ptsperlayer,2)~=1
    error('You must use an odd number of points for layer')
end
n_d_layer=ones(1,l_d)*ptsperlayer;
ptsbetween=(ptsperlayer-3)/2;
ptseitherside=(ptsperlayer-1)/2;

cum_n_d=cumsum(n_d);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

%% For the multi-grid as part of refinement 2 we will want
d_grid1=d_grid(1:cum_n_d(1));
if l_d>1
    d_grid2=d_grid((1+cum_n_d(1)):cum_n_d(2));
    if l_d>2
        d_grid3=d_grid((1+cum_n_d(2)):cum_n_d(3));
        if l_d>3
            d_grid4=d_grid(1+cum_n_d(3):cum_n_d(4));
        end
    end
end


%%
if vfoptions.lowmemory==0
    
    if n_d(1)==0 % Nothing to do
        ReturnMatrix=zeros(n_endostate,N_endotype,n_endostate,N_endotype,N_z);
        for e_c=1:N_endotype
            ReturnFnParamsVec_e=[endotype_grid(e_c),ReturnFnParamsVec];
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_endostate, n_z, d_grid, endostate_grid, z_grid, ReturnFnParamsVec_e,1);
            ReturnMatrix_e=reshape(ReturnMatrix_e,[n_endostate,1,n_endostate,1,N_z]);
            for eprime_c=1:N_endotype
                ReturnMatrix(:,eprime_c,:,e_c,:)=ReturnMatrix_e;
            end
        end
    else % n_d(1)>0
        ReturnMatrix=zeros(n_endostate,N_endotype,n_endostate,N_endotype,N_z);
        dstar=zeros(n_endostate,N_endotype,n_endostate,N_endotype,N_z);

        for e_c=1:N_endotype
            ReturnFnParamsVec_e=[endotype_grid(e_c),ReturnFnParamsVec];
                        
            d_grid_layer_index=zeros(1,ptsperlayer*l_d);
            d_grid_layer=zeros(ptsperlayer*l_d,1);
            % Do layer 1
            d_grid_layer_endpointindex=reshape([1,1+cumsum(n_d(1:end-1)), cumsum(n_d)],[l_d,2]);
            temp=linspace(d_grid_layer_endpointindex(1,1),d_grid_layer_endpointindex(1,2),ptsperlayer)';
                        
            d_grid_layer_index(1:ptsperlayer)=temp;
            d_grid_layer(1:ptsperlayer)=d_grid(temp);
            for ii=2:l_d
                temp=linspace(d_grid_layer_endpointindex(ii,1),d_grid_layer_endpointindex(ii,2),ptsperlayer)';
                d_grid_layer_index(1+(ii-1)*ptsperlayer:ii*ptsperlayer)=temp;
                d_grid_layer(1+(ii-1)*ptsperlayer:ii*ptsperlayer)=d_grid(temp);
            end
            
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d_layer, n_endostate, n_z, d_grid_layer, endostate_grid, z_grid, ReturnFnParamsVec_e,1);
            % For refinement, now we solve for d*(aprime,a,z) that maximizes the ReturnFn
            [~,dstar_e]=max(ReturnMatrix_e,[],1); % We only want to keep ReturnMatrix values for the last layer
            %                 ReturnMatrix=shiftdim(ReturnMatrix,1);
            
            
            % Do layer 2
            %% NOTE: I AM NOT YET CONFIDENT OF THE *((ptsbetween+1)^(nlayers-1))
            if l_d==1
                dstar_e(dstar_e==1)=2; % If at the end, put it one point inside
                dstar_e(dstar_e==ptsperlayer)=ptsperlayer-1; % If at the end, put it one point inside
                d_grid_layer_center=dstar_e+(dstar_e-1)*((ptsbetween)^(nlayers-1)); % The center points
                d_grid_layer=d_grid_layer_center+(0:1:ptsperlayer-1)'*((ptsbetween+1)^(nlayers-2)); % Note: first dimension of d_grid_layer was just 1
                d1_grid_layer=reshape(d_grid(d_grid_layer),[ptsperlayer,n_endostate,n_endostate,n_z]);
                ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2_refineld1(ReturnFn, d1_grid_layer, n_endostate, n_z, endostate_grid, z_grid, ReturnFnParamsVec_e,1);
            elseif l_d==2
                temp=rem(dstar_e-1,ptsperlayer)+1; % dstar in the first dimension
                temp(temp==1)=2; % If at the end, put it one point inside
                temp(temp==ptsperlayer)=ptsperlayer-1; % If at the end, put it one point inside
                d_grid_layer_center=shiftdim(temp,-1)+shiftdim(temp-1,-1)*(ptsbetween^(nlayers-1)); % The center points
                d_grid_layer_index1=d_grid_layer_center +((-ptseitherside:1:ptseitherside))'*((ptsbetween+1)^(nlayers-2)); % (ptsperlayer,1,n_endostate,n_endostate,N_z)
                
                temp=ceil(dstar_e/ptsperlayer);  % dstar in the second dimension
                temp(temp==1)=2; % If at the end, put it one point inside
                temp(temp==ptsperlayer)=ptsperlayer-1;  % If at the end, put it one point inside
                d_grid_layer_center=shiftdim(temp,-1)+shiftdim(temp-1,-1)*(ptsbetween^(nlayers-1)); % The center points
                d_grid_layer_index2=d_grid_layer_center +((-ptseitherside:1:ptseitherside))*((ptsbetween+1)^(nlayers-2)); % (1,ptsperlayer,n_endostate,n_endostate,N_z)
                
                d1_grid_layer=reshape(d_grid1(d_grid_layer_index1),[ptsperlayer,1,n_endostate,n_endostate,n_z]); % note: before reshape is [ptsperlayer,1,n_endostate,n_endostate,N_z]
                d2_grid_layer=reshape(d_grid2(d_grid_layer_index2),[1,ptsperlayer,n_endostate,n_endostate,n_z]); % note: before reshape is [1,ptsperlayer,n_endostate,n_endostate,N_z]
                
                ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2_refineld2(ReturnFn, d1_grid_layer,d2_grid_layer, n_d_layer, n_endostate, n_z, endostate_grid, z_grid, ReturnFnParamsVec_e);
            elseif l_d==3
                temp=rem(dstar_e-1,ptsperlayer)+1; % dstar in the first dimension
                temp(temp==1)=2; % If at the end, put it one point inside
                temp(temp==ptsperlayer)=ptsperlayer-1; % If at the end, put it one point inside
                d_grid_layer_center=shiftdim(temp,-1)+shiftdim(temp-1,-1)*(ptsbetween^(nlayers-1)); % The center points
                d_grid_layer_index1=d_grid_layer_center +((-ptseitherside:1:ptseitherside))'*((ptsbetween+1)^(nlayers-2)); % (ptsperlayer,1,n_endostate,n_endostate,N_z)
                
                temp=rem(ceil(dstar_e/ptsperlayer)-1,ptsperlayer)+1;  % dstar in the second dimension
                temp(temp==1)=2; % If at the end, put it one point inside
                temp(temp==ptsperlayer)=ptsperlayer-1;  % If at the end, put it one point inside
                d_grid_layer_center=shiftdim(temp,-1)+shiftdim(temp-1,-1)*(ptsbetween^(nlayers-1)); % The center points
                d_grid_layer_index2=d_grid_layer_center +((-ptseitherside:1:ptseitherside))*((ptsbetween+1)^(nlayers-2)); % (1,ptsperlayer,n_endostate,n_endostate,N_z)

                temp=ceil(dstar_e/ptsperlayer^2);  % dstar in the second dimension
                temp(temp==1)=2; % If at the end, put it one point inside
                temp(temp==ptsperlayer)=ptsperlayer-1;  % If at the end, put it one point inside
                d_grid_layer_center=shiftdim(temp,-1)+shiftdim(temp-1,-1)*(ptsbetween^(nlayers-1)); % The center points
                d_grid_layer_index3=d_grid_layer_center +((-ptseitherside:1:ptseitherside))*((ptsbetween+1)^(nlayers-2)); % (1,ptsperlayer,n_endostate,n_endostate,N_z)

                d1_grid_layer=reshape(d_grid1(d_grid_layer_index1),[ptsperlayer,1,1,n_endostate,n_endostate,n_z]); % note: before reshape is [ptsperlayer,1,n_endostate,n_endostate,N_z]
                d2_grid_layer=reshape(d_grid2(d_grid_layer_index2),[1,ptsperlayer,1,n_endostate,n_endostate,n_z]); % note: before reshape is [1,ptsperlayer,n_endostate,n_endostate,N_z]
                d3_grid_layer=reshape(d_grid3(d_grid_layer_index3),[1,1,ptsperlayer,n_endostate,n_endostate,n_z]); % note: before reshape is [1,ptsperlayer,n_endostate,n_endostate,N_z]
                
                ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2_refineld3(ReturnFn, d1_grid_layer,d2_grid_layer,d3_grid_layer, n_d_layer, n_endostate, n_z, endostate_grid, z_grid, ReturnFnParamsVec_e);
            elseif l_d==4
                temp=rem(dstar_e-1,ptsperlayer)+1; % dstar in the first dimension
                temp(temp==1)=2; % If at the end, put it one point inside
                temp(temp==ptsperlayer)=ptsperlayer-1; % If at the end, put it one point inside
                d_grid_layer_center=shiftdim(temp,-1)+shiftdim(temp-1,-1)*(ptsbetween^(nlayers-1)); % The center points
                d_grid_layer_index1=d_grid_layer_center +((-ptseitherside:1:ptseitherside))'*((ptsbetween+1)^(nlayers-2)); % (ptsperlayer,1,n_endostate,n_endostate,N_z)
                
                temp=rem(ceil(dstar_e/ptsperlayer)-1,ptsperlayer)+1;  % dstar in the second dimension
                temp(temp==1)=2; % If at the end, put it one point inside
                temp(temp==ptsperlayer)=ptsperlayer-1;  % If at the end, put it one point inside
                d_grid_layer_center=shiftdim(temp,-1)+shiftdim(temp-1,-1)*(ptsbetween^(nlayers-1)); % The center points
                d_grid_layer_index2=d_grid_layer_center +((-ptseitherside:1:ptseitherside))*((ptsbetween+1)^(nlayers-2)); % (1,ptsperlayer,n_endostate,n_endostate,N_z)

                temp=rem(ceil(dstar_e/ptsperlayer^2)-1,ptsperlayer)+1;  % dstar in the third dimension
                temp(temp==1)=2; % If at the end, put it one point inside
                temp(temp==ptsperlayer)=ptsperlayer-1;  % If at the end, put it one point inside
                d_grid_layer_center=shiftdim(temp,-1)+shiftdim(temp-1,-1)*(ptsbetween^(nlayers-1)); % The center points
                d_grid_layer_index3=d_grid_layer_center +((-ptseitherside:1:ptseitherside))*((ptsbetween+1)^(nlayers-2)); % (1,ptsperlayer,n_endostate,n_endostate,N_z)

                temp=ceil(dstar_e/ptsperlayer^3);  % dstar in the fourth dimension
                temp(temp==1)=2; % If at the end, put it one point inside
                temp(temp==ptsperlayer)=ptsperlayer-1;  % If at the end, put it one point inside
                d_grid_layer_center=shiftdim(temp,-1)+shiftdim(temp-1,-1)*(ptsbetween^(nlayers-1)); % The center points
                d_grid_layer_index4=d_grid_layer_center +((-ptseitherside:1:ptseitherside))*((ptsbetween+1)^(nlayers-2)); % (1,ptsperlayer,n_endostate,n_endostate,N_z)

                d1_grid_layer=reshape(d_grid1(d_grid_layer_index1),[ptsperlayer,1,1,1,n_endostate,n_endostate,n_z]); % note: before reshape is [ptsperlayer,1,n_endostate,n_endostate,N_z]
                d2_grid_layer=reshape(d_grid2(d_grid_layer_index2),[1,ptsperlayer,1,1,n_endostate,n_endostate,n_z]); % note: before reshape is [1,ptsperlayer,n_endostate,n_endostate,N_z]
                d3_grid_layer=reshape(d_grid3(d_grid_layer_index3),[1,1,ptsperlayer,1,n_endostate,n_endostate,n_z]); % note: before reshape is [1,ptsperlayer,n_endostate,n_endostate,N_z]
                d4_grid_layer=reshape(d_grid4(d_grid_layer_index4),[1,1,1,ptsperlayer,n_endostate,n_endostate,n_z]); % note: before reshape is [1,ptsperlayer,n_endostate,n_endostate,N_z]
                
                ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2_refineld4(ReturnFn, d1_grid_layer,d2_grid_layer,d3_grid_layer,d4_grid_layer, n_d_layer, n_endostate, n_z, endostate_grid, z_grid, ReturnFnParamsVec_e);
            end
            [ReturnMatrix_e,dstar_e]=max(ReturnMatrix_e,[],1); % We only want to keep ReturnMatrix values for the last layer
            ReturnMatrix_e=shiftdim(ReturnMatrix_e,1);
            
            
            ReturnMatrix_e=reshape(ReturnMatrix_e,[n_endostate,1,n_endostate,1,N_z]);
            dstar_e=reshape(dstar_e,[n_endostate,1,n_endostate,1,N_z]);
            for eprime_c=1:N_endotype
                ReturnMatrix(:,eprime_c,:,e_c,:)=ReturnMatrix_e;
                dstar(:,eprime_c,:,e_c,:)=dstar_e;
            end
        end
        
    end
    
elseif vfoptions.lowmemory==1 % Loop over z
    
    z_gridvals_trans=CreateGridvals(n_z,z_grid,1)';
    n_z_temp=ones(1,length(n_z));

    if n_d(1)==0 % Nothing to do in terms of the refinement
        ReturnMatrix=zeros(n_endostate,N_endotype,n_endostate,N_endotype,N_z);
        for z_c=1:N_z
            z_val=z_gridvals_trans(:,z_c);
            for e_c=1:N_endotype
                ReturnFnParamsVec_e=[endotype_grid(e_c),ReturnFnParamsVec];
                ReturnMatrix_ez=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_endostate, n_z_temp, d_grid, endostate_grid, z_val, ReturnFnParamsVec_e,1);
                ReturnMatrix_ez=reshape(ReturnMatrix_ez,[n_endostate,1,n_endostate,1,1]);
                for eprime_c=1:N_endotype
                    ReturnMatrix(:,eprime_c,:,e_c,z_c)=ReturnMatrix_ez;
                end
            end
        end
    else % n_d(1)>0
        ReturnMatrix=zeros(n_endostate,N_endotype,n_endostate,N_endotype,N_z);
        dstar=zeros(n_endostate,N_endotype,n_endostate,N_endotype,N_z);
        
        for e_c=1:N_endotype
            ReturnFnParamsVec_e=[endotype_grid(e_c),ReturnFnParamsVec];
            
            d_grid_layer_index=zeros(1,ptsperlayer*l_d);
            d_grid_layer=zeros(ptsperlayer*l_d,1);

            % Do layer 1 (part of layer 1 is outside loop over z, and part is inside loop over z)
            d_grid_layer_endpointindex=reshape([1,1+cumsum(n_d(1:end-1)), cumsum(n_d)],[l_d,2]);
            temp=linspace(d_grid_layer_endpointindex(1,1),d_grid_layer_endpointindex(1,2),ptsperlayer)';
            
            d_grid_layer_index(1:ptsperlayer)=temp;
            d_grid_layer(1:ptsperlayer)=d_grid(temp);
            for ii=2:l_d
                temp=linspace(d_grid_layer_endpointindex(ii,1),d_grid_layer_endpointindex(ii,2),ptsperlayer)';
                d_grid_layer_index(1+(ii-1)*ptsperlayer:ii*ptsperlayer)=temp;
                d_grid_layer(1+(ii-1)*ptsperlayer:ii*ptsperlayer)=d_grid(temp);
            end
            
            for z_c=1:N_z
                z_val=z_gridvals_trans(:,z_c);
                ReturnMatrix_ez=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d_layer, n_endostate, n_z_temp, d_grid_layer, endostate_grid, z_val, ReturnFnParamsVec_e,1);
                % For refinement, now we solve for d*(aprime,a,z) that maximizes the ReturnFn
                [~,dstar_ez]=max(ReturnMatrix_ez,[],1); % We only want to keep ReturnMatrix values for the last layer
                %                 ReturnMatrix=shiftdim(ReturnMatrix,1);
                
                
                % Do layer 2
                %% NOTE: I AM NOT YET CONFIDENT OF THE *((ptsbetween+1)^(nlayers-1))
                if l_d==1
                    dstar_ez(dstar_ez==1)=2; % If at the end, put it one point inside
                    dstar_ez(dstar_ez==ptsperlayer)=ptsperlayer-1; % If at the end, put it one point inside
                    d_grid_layer_center=dstar_ez+(dstar_ez-1)*((ptsbetween)^(nlayers-1)); % The center points
                    d_grid_layer=d_grid_layer_center+(0:1:ptsperlayer-1)'*((ptsbetween+1)^(nlayers-2)); % Note: first dimension of d_grid_layer was just 1
                    d1_grid_layer=reshape(d_grid(d_grid_layer),[ptsperlayer,n_endostate,n_endostate]);
                    ReturnMatrix_ez=CreateReturnFnMatrix_Case1_Disc_Par2_refineld1(ReturnFn, d1_grid_layer, n_endostate, n_z_temp, endostate_grid, z_val, ReturnFnParamsVec_e,1);
                elseif l_d==2
                    temp=rem(dstar_ez-1,ptsperlayer)+1; % dstar in the first dimension
                    temp(temp==1)=2; % If at the end, put it one point inside
                    temp(temp==ptsperlayer)=ptsperlayer-1; % If at the end, put it one point inside
                    d_grid_layer_center=shiftdim(temp,-1)+shiftdim(temp-1,-1)*(ptsbetween^(nlayers-1)); % The center points
                    d_grid_layer_index1=d_grid_layer_center +((-ptseitherside:1:ptseitherside))'*((ptsbetween+1)^(nlayers-2)); % (ptsperlayer,1,n_endostate,n_endostate,N_z)
                    
                    temp=ceil(dstar_ez/ptsperlayer);  % dstar in the second dimension
                    temp(temp==1)=2; % If at the end, put it one point inside
                    temp(temp==ptsperlayer)=ptsperlayer-1;  % If at the end, put it one point inside
                    d_grid_layer_center=shiftdim(temp,-1)+shiftdim(temp-1,-1)*(ptsbetween^(nlayers-1)); % The center points
                    d_grid_layer_index2=d_grid_layer_center +((-ptseitherside:1:ptseitherside))*((ptsbetween+1)^(nlayers-2)); % (1,ptsperlayer,n_endostate,n_endostate,N_z)
                    
                    d1_grid_layer=reshape(d_grid1(d_grid_layer_index1),[ptsperlayer,1,n_endostate,n_endostate]); % note: before reshape is [ptsperlayer,1,n_endostate,n_endostate,N_z]
                    d2_grid_layer_z=reshape(d_grid2(d_grid_layer_index2),[1,ptsperlayer,n_endostate,n_endostate]); % note: before reshape is [1,ptsperlayer,n_endostate,n_endostate,N_z]
                    
                    ReturnMatrix_ez=CreateReturnFnMatrix_Case1_Disc_Par2_refineld2(ReturnFn, d1_grid_layer,d2_grid_layer_z, n_d_layer, n_endostate, n_z_temp, endostate_grid, z_val, ReturnFnParamsVec_e);
                elseif l_d==3
                    temp=rem(dstar_ez-1,ptsperlayer)+1; % dstar in the first dimension
                    temp(temp==1)=2; % If at the end, put it one point inside
                    temp(temp==ptsperlayer)=ptsperlayer-1; % If at the end, put it one point inside
                    d_grid_layer_center=shiftdim(temp,-1)+shiftdim(temp-1,-1)*(ptsbetween^(nlayers-1)); % The center points
                    d_grid_layer_index1=d_grid_layer_center +((-ptseitherside:1:ptseitherside))'*((ptsbetween+1)^(nlayers-2)); % (ptsperlayer,1,n_endostate,n_endostate,N_z)
                    
                    temp=rem(ceil(dstar_ez/ptsperlayer)-1,ptsperlayer)+1;  % dstar in the second dimension
                    temp(temp==1)=2; % If at the end, put it one point inside
                    temp(temp==ptsperlayer)=ptsperlayer-1;  % If at the end, put it one point inside
                    d_grid_layer_center=shiftdim(temp,-1)+shiftdim(temp-1,-1)*(ptsbetween^(nlayers-1)); % The center points
                    d_grid_layer_index2=d_grid_layer_center +((-ptseitherside:1:ptseitherside))*((ptsbetween+1)^(nlayers-2)); % (1,ptsperlayer,n_endostate,n_endostate,N_z)
                    
                    temp=ceil(dstar_ez/ptsperlayer^2);  % dstar in the second dimension
                    temp(temp==1)=2; % If at the end, put it one point inside
                    temp(temp==ptsperlayer)=ptsperlayer-1;  % If at the end, put it one point inside
                    d_grid_layer_center=shiftdim(temp,-1)+shiftdim(temp-1,-1)*(ptsbetween^(nlayers-1)); % The center points
                    d_grid_layer_index3=d_grid_layer_center +((-ptseitherside:1:ptseitherside))*((ptsbetween+1)^(nlayers-2)); % (1,ptsperlayer,n_endostate,n_endostate,N_z)
                    
                    d1_grid_layer=reshape(d_grid1(d_grid_layer_index1),[ptsperlayer,1,1,n_endostate,n_endostate]); % note: before reshape is [ptsperlayer,1,n_endostate,n_endostate,N_z]
                    d2_grid_layer=reshape(d_grid2(d_grid_layer_index2),[1,ptsperlayer,1,n_endostate,n_endostate]); % note: before reshape is [1,ptsperlayer,n_endostate,n_endostate,N_z]
                    d3_grid_layer=reshape(d_grid3(d_grid_layer_index3),[1,1,ptsperlayer,n_endostate,n_endostate]); % note: before reshape is [1,ptsperlayer,n_endostate,n_endostate,N_z]
                    
                    ReturnMatrix_ez=CreateReturnFnMatrix_Case1_Disc_Par2_refineld3(ReturnFn, d1_grid_layer,d2_grid_layer,d3_grid_layer, n_d_layer, n_endostate, n_z, endostate_grid, z_grid, ReturnFnParamsVec_e);
                elseif l_d==4
                    temp=rem(dstar_ez-1,ptsperlayer)+1; % dstar in the first dimension
                    temp(temp==1)=2; % If at the end, put it one point inside
                    temp(temp==ptsperlayer)=ptsperlayer-1; % If at the end, put it one point inside
                    d_grid_layer_center=shiftdim(temp,-1)+shiftdim(temp-1,-1)*(ptsbetween^(nlayers-1)); % The center points
                    d_grid_layer_index1=d_grid_layer_center +((-ptseitherside:1:ptseitherside))'*((ptsbetween+1)^(nlayers-2)); % (ptsperlayer,1,n_endostate,n_endostate,N_z)
                    
                    temp=rem(ceil(dstar_ez/ptsperlayer)-1,ptsperlayer)+1;  % dstar in the second dimension
                    temp(temp==1)=2; % If at the end, put it one point inside
                    temp(temp==ptsperlayer)=ptsperlayer-1;  % If at the end, put it one point inside
                    d_grid_layer_center=shiftdim(temp,-1)+shiftdim(temp-1,-1)*(ptsbetween^(nlayers-1)); % The center points
                    d_grid_layer_index2=d_grid_layer_center +((-ptseitherside:1:ptseitherside))*((ptsbetween+1)^(nlayers-2)); % (1,ptsperlayer,n_endostate,n_endostate,N_z)
                    
                    temp=rem(ceil(dstar_ez/ptsperlayer^2)-1,ptsperlayer)+1;  % dstar in the third dimension
                    temp(temp==1)=2; % If at the end, put it one point inside
                    temp(temp==ptsperlayer)=ptsperlayer-1;  % If at the end, put it one point inside
                    d_grid_layer_center=shiftdim(temp,-1)+shiftdim(temp-1,-1)*(ptsbetween^(nlayers-1)); % The center points
                    d_grid_layer_index3=d_grid_layer_center +((-ptseitherside:1:ptseitherside))*((ptsbetween+1)^(nlayers-2)); % (1,ptsperlayer,n_endostate,n_endostate,N_z)
                    
                    temp=ceil(dstar_ez/ptsperlayer^3);  % dstar in the fourth dimension
                    temp(temp==1)=2; % If at the end, put it one point inside
                    temp(temp==ptsperlayer)=ptsperlayer-1;  % If at the end, put it one point inside
                    d_grid_layer_center=shiftdim(temp,-1)+shiftdim(temp-1,-1)*(ptsbetween^(nlayers-1)); % The center points
                    d_grid_layer_index4=d_grid_layer_center +((-ptseitherside:1:ptseitherside))*((ptsbetween+1)^(nlayers-2)); % (1,ptsperlayer,n_endostate,n_endostate,N_z)
                    
                    d1_grid_layer=reshape(d_grid1(d_grid_layer_index1),[ptsperlayer,1,1,1,n_endostate,n_endostate]); % note: before reshape is [ptsperlayer,1,n_endostate,n_endostate,N_z]
                    d2_grid_layer=reshape(d_grid2(d_grid_layer_index2),[1,ptsperlayer,1,1,n_endostate,n_endostate]); % note: before reshape is [1,ptsperlayer,n_endostate,n_endostate,N_z]
                    d3_grid_layer=reshape(d_grid3(d_grid_layer_index3),[1,1,ptsperlayer,1,n_endostate,n_endostate]); % note: before reshape is [1,ptsperlayer,n_endostate,n_endostate,N_z]
                    d4_grid_layer=reshape(d_grid4(d_grid_layer_index4),[1,1,1,ptsperlayer,n_endostate,n_endostate]); % note: before reshape is [1,ptsperlayer,n_endostate,n_endostate,N_z]
                    
                    ReturnMatrix_ez=CreateReturnFnMatrix_Case1_Disc_Par2_refineld4(ReturnFn, d1_grid_layer,d2_grid_layer,d3_grid_layer,d4_grid_layer, n_d_layer, n_endostate, n_z, endostate_grid, z_grid, ReturnFnParamsVec_e);
                end
                [ReturnMatrix_ez,dstar_ez]=max(ReturnMatrix_ez,[],1); % We only want to keep ReturnMatrix values for the last layer
                ReturnMatrix_ez=shiftdim(ReturnMatrix_ez,1);
                
                
                ReturnMatrix_ez=reshape(ReturnMatrix_ez,[n_endostate,1,n_endostate,1]);
                dstar_ez=reshape(dstar_ez,[n_endostate,1,n_endostate,1]);
                for eprime_c=1:N_endotype
                    ReturnMatrix(:,eprime_c,:,e_c,z_c)=ReturnMatrix_ez;
                    dstar(:,eprime_c,:,e_c,z_c)=dstar_ez;
                end
            end
        end
    end

else
    dbstack
    error('Currently endotype only works for lowmemory=0 and lowmemory=1')    
end



%%
if vfoptions.verbose==1
    time=toc;
    fprintf('Time to create return fn matrix: %8.4f \n', time)
    fprintf('Starting Value Function \n')
    tic;
end

%%
% Now we can just put endogenous state into the value and policy
% function calculations as if it is just another endogenous state
% variable ('a' variable)
n_a=[n_endostate,n_endotype];
N_a=prod(n_a);

V=zeros(N_a,N_z,'gpuArray');
Policy_a=zeros(N_a,N_z,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

ReturnMatrix=reshape(ReturnMatrix,[N_a,N_a,N_z]);

%%
for z_c=1:N_z
    ReturnMatrix_z=ReturnMatrix(:,:,z_c);
    
    %Calc the condl expectation term (except beta), which depends on z but
    %not on control variables
    EV_z=Vnext.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
    EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV_z=sum(EV_z,2);
    
    entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*EV_z*ones(1,N_a,1);
    
    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS_z,[],1);
    V(:,z_c)=Vtemp;
    Policy_a(:,z_c)=maxindex;
end

%%
% For refinement, add d to Policy
% Policy is currently
if n_d(1)>0
    Policy=zeros(2,N_a,N_z,'gpuArray');
    Policy(2,:,:)=shiftdim(Policy_a,-1);
    temppolicyindex=reshape(Policy_a,[1,N_a*N_z])+(0:1:N_a*N_z-1)*N_a;
    Policy(1,:,:)=reshape(dstar(temppolicyindex),[N_a,N_z]);
else
    Policy=Policy_a;
end

end