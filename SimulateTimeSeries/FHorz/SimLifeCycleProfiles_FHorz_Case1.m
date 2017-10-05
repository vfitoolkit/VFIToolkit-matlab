function SimLifeCycleProfiles=SimLifeCycleProfiles_FHorz_Case1(InitialDist,Policy,ValuesFns,ValuesFnsParamNames,Parameters,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,pi_z, simoptions)
% Simulates a panel based on PolicyIndexes of 'numbersims' agents of length
% 'simperiods' beginning from randomly drawn InitialDist. Then based on
% this it computes life-cycle profiles for the 'VaulesFns' and reports
% mean, median, min, 19 intermediate ventiles, and max. (you can change from
% ventiles using simperiods.lifecyclepercentiles)
%
% InitialDist can be inputed as over the finite time-horizon (j), or
% without a time-horizon in which case it is assumed to be an InitialDist
% for time j=1. (So InitialDist is either n_a-by-n_z-by-n_j, or n_a-by-n_z)

N_a=prod(n_a);
N_z=prod(n_z);
N_d=prod(n_d);

%% Check which simoptions have been declared, set all others to defaults 
if exist('simoptions','var')==1
    %Check simoptions for missing fields, if there are some fill them with the defaults
    if isfield(simoptions,'parallel')==0
        simoptions.parallel=2;
    end
    if isfield(simoptions,'verbose')==0
        simoptions.verbose=0;
    end
    if isfield(simoptions,'simperiods')==0
        simoptions.simperiods=N_j;
    end
    if isfield(simoptions,'numbersims')==0
        simoptions.numbersims=10^4; % Given that aim is to calculate ventiles of life-cycle profiles 10^4 seems appropriate
    end
    if isfield(simoptions,'lifecyclepercentiles')==0
        simoptions.lifecyclepercentiles=20; % by default gives ventiles
    end
else
    %If simoptions is not given, just use all the defaults
    simoptions.parallel=2;
    simoptions.verbose=0;
    simoptions.simperiods=N_j;
    simoptions.numbersims=10^4; % Given that aim is to calculate ventiles of life-cycle profiles 10^4 seems appropriate
    simoptions.lifecyclepercentiles=20; % by default gives ventiles
end

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

if simoptions.parallel~=2
    d_grid=gather(d_grid);
    a_grid=gather(a_grid);
    z_grid=gather(z_grid);
end

tic;
PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z, N_j,simoptions); % This is actually being created inside SimPanelIndexes already
toc

% if simoptions.parallel~=2
%     PolicyIndexesKron=gather(PolicyIndexesKron);
% end

tic;
% Because we want life-cycle profiles we only use the part of InitialDist that is how agents appear 'at birth' (in j=1).
% (InitialDist is either n_a-by-n_z-by-n_j, or n_a-by-n_z)
if numel(InitialDist)==N_a*N_z % Has just been given for age j=1
    SimPanelIndexes=SimPanelIndexes_FHorz_Case1(InitialDist,PolicyIndexesKron,n_d,n_a,n_z,N_j,pi_z, simoptions);
else
    InitialDist=reshape(InitialDist,[N_a*N_z,N_j]);
    SimPanelIndexes=SimPanelIndexes_FHorz_Case1(InitialDist(:,1),PolicyIndexesKron,n_d,n_a,n_z,N_j,pi_z, simoptions); % Use only j=1: InitialDist(:,1)
end
toc

% Move everything to cpu for what remains.
d_grid=gather(d_grid);
a_grid=gather(a_grid);
z_grid=gather(z_grid);
PolicyIndexesKron=gather(PolicyIndexesKron);

SimPanelValues=zeros(length(ValuesFns), simoptions.simperiods, simoptions.numbersims);


d_val=zeros(1,l_d);
aprime_val=zeros(1,l_a);
a_val=zeros(1,l_a);
z_val=zeros(1,l_z);
% d_ind=zeros(1,num_d_vars); aprime_ind=zeros(1,l_a);

%% Precompute the gridvals vectors.
z_gridvals=-Inf*ones(N_z,l_z);
for i1=1:N_z
    sub=zeros(1,l_z);
    sub(1)=rem(i1-1,n_z(1))+1;
    for ii=2:l_z-1
        sub(ii)=rem(ceil(i1/prod(n_z(1:ii-1)))-1,n_z(ii))+1;
    end
    sub(l_z)=ceil(i1/prod(n_z(1:l_z-1)));
    
    if l_z>1
        sub=sub+[0,cumsum(n_z(1:end-1))];
    end
    z_gridvals(i1,:)=z_grid(sub);
end
a_gridvals=-Inf*ones(N_a,l_a);
for i2=1:N_a
    sub=zeros(1,l_a);
    sub(1)=rem(i2-1,n_a(1)+1);
    for ii=2:l_a-1
        sub(ii)=rem(ceil(i2/prod(n_a(1:ii-1)))-1,n_a(ii))+1;
    end
    sub(l_a)=ceil(i2/prod(n_a(1:l_a-1)));
    
    if l_a>1
        sub=sub+[0,cumsum(n_a(1:end-1))];
    end
    a_gridvals(i2,:)=a_grid(sub);
end

%%
SimPanelValues_ii=zeros(length(ValuesFns),simoptions.simperiods);
%% For sure the following could be made faster by parallelizing some stuff.
for ii=1:simoptions.numbersims
    SimPanel_ii=SimPanelIndexes(:,:,ii);
    for t=1:simoptions.simperiods
        a_sub=SimPanel_ii(1:l_a,t);
%         for jj1=1:l_a
%             if jj1==1
%                 a_val(jj1)=a_grid(a_sub(jj1));
%             else
%                 a_val(jj1)=a_grid(a_sub(jj1)+sum(n_a(1:jj1-1)));
%             end
%         end
        a_ind=sub2ind_homemade(n_a,a_sub);
        
        z_sub=SimPanel_ii((l_a+1):(l_a+l_z),t);
%         for jj2=1:l_z
%             if jj2==1
%                 z_val(jj2)=z_grid(z_sub(jj2));
%             else
%                 z_val(jj2)=z_grid(z_sub(jj2)+sum(n_z(1:jj2-1)));
%             end
%         end
        z_ind=sub2ind_homemade(n_z,z_sub);
        
        j_ind=SimPanel_ii(end,t);
        
        a_val=a_gridvals(a_ind,:);
        z_val=z_gridvals(z_ind,:);
        
        if l_d==0
            aprime_ind=PolicyIndexesKron(a_ind,z_ind,t); % Given dependence on t I suspect precomputing this as aprime_gridvals and d_gridvals would not be worthwhile
            aprime_sub=ind2sub_homemade(n_a,aprime_ind);
        else
            temp=PolicyIndexesKron(:,a_ind,z_ind,t);
            d_ind=temp(1); aprime_ind=temp(2);
            d_sub=ind2sub_homemade(n_a,d_ind);
            aprime_sub=ind2sub_homemade(n_a,aprime_ind);
            for kk1=1:l_d
                if kk1==1
                    d_val(kk1)=d_grid(d_sub(kk1));
                else
                    d_val(kk1)=d_grid(d_sub(kk1)+sum(n_d(1:kk1-1)));
                end
            end
        end
        for kk2=1:l_a
            if kk2==1
                aprime_val(kk2)=a_grid(aprime_sub(kk2));
            else
                aprime_val(kk2)=a_grid(aprime_sub(kk2)+sum(n_a(1:kk2-1)));
            end
        end
        
        if l_d==0
            for vv=1:length(ValuesFns)
                if isempty(ValuesFnsParamNames(vv).Names)  % check for 'SSvalueParamNames={}'
                    tempv=[aprime_val,a_val,z_val];
                    tempcell=cell(1,length(tempv));
                    for temp_c=1:length(tempv)
                        tempcell{temp_c}=tempv(temp_c);
                    end
                else
                    ValuesFnParamsVec=CreateVectorFromParams(Parameters,ValuesFnsParamNames(vv).Names,j_ind);
                    tempv=[aprime_val,a_val,z_val,ValuesFnParamsVec];
                    tempcell=cell(1,length(tempv));
                    for temp_c=1:length(tempv)
                        tempcell{temp_c}=tempv(temp_c);
                    end
                end
                SimPanelValues_ii(vv,t)=ValuesFns{vv}(tempcell{:});
                %SimPanelValues_ii(vv,t)=ValuesFns{vv}(aprime_val,a_val,z_val,ValuesFnParamsVec);
            end
        else
            for vv=1:length(ValuesFns)
                if isempty(ValuesFnsParamNames(vv).Names)  % check for 'SSvalueParamNames={}'
                    tempv=[d_val,aprime_val,a_val,z_val];
                    tempcell=cell(1,length(tempv));
                    for temp_c=1:length(tempv)
                        tempcell{temp_c}=tempv(temp_c);
                    end
                else
                    ValuesFnParamsVec=CreateVectorFromParams(Parameters,ValuesFnsParamNames(vv).Names,j_ind);
                    tempv=[d_val,aprime_val,a_val,z_val,ValuesFnParamsVec];
                    tempcell=cell(1,length(tempv));
                    for temp_c=1:length(tempv)
                        tempcell{temp_c}=tempv(temp_c);
                    end
                end
                SimPanelValues_ii(vv,t)=ValuesFns{vv}(tempcell{:});
                %SimPanelValues_ii(vv,t)=ValuesFns{vv}(d_val,aprime_val,a_val,z_val,ValuesFnParamsVec);
            end
%             for vv=1:length(ValuesFns)
%                 if isempty(ValuesFnsParamNames(vv).Names)  % check for 'SSvalueParamNames={}'
%                     SimPanelValues_ii(vv,t)=ValuesFns{vv}(d_val,aprime_val,a_val,z_val);
%                 else
%                     ValuesFnParamsVec=CreateVectorFromParams(Parameters,ValuesFnsParamNames(vv).Names,j_ind);
%                     SimPanelValues_ii(vv,t)=ValuesFns{vv}(d_val,aprime_val,a_val,z_val,ValuesFnParamsVec);
%                 end
%             end
        end
    end
    SimPanelValues(:,:,ii)=SimPanelValues_ii;
end

% Now get the life-cycle profiles by looking at (age-conditional) cross-sections of this panel data.
if simoptions.lifecyclepercentiles>0
    SimLifeCycleProfiles=zeros(length(ValuesFns), simoptions.simperiods, simoptions.lifecyclepercentiles+3); % Mean, median, min, 19 intermediate ventiles, and max.
    prctilegrid=(0:1/simoptions.lifecyclepercentiles:1)*100;
    for ii=1:length(ValuesFns)
        for jj=1:simoptions.simperiods
            temp=SimPanelValues(ii,jj,:);
            SimLifeCycleProfiles(ii,jj,1)=mean(temp);
            SimLifeCycleProfiles(ii,jj,2)=median(temp);
            SimLifeCycleProfiles(ii,jj,3:end)=prctile(temp,prctilegrid);
        end
    end
else
    SimLifeCycleProfiles=zeros(length(ValuesFns), simoptions.simperiods, 2); % Mean, median
    for ii=1:length(ValuesFns)
        for jj=1:simoptions.simperiods
            temp=SimPanelValues(ii,jj,:);
            SimLifeCycleProfiles(ii,jj,1)=mean(temp);
            SimLifeCycleProfiles(ii,jj,2)=median(temp);
        end
    end
end


