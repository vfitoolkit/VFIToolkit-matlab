function Path_mewj=UpdateAgeWeights(initial_mewj,Path_sj,Path_n,Path_immigrantmewj,Path_immigrationrate,Path_emmigrantmewj,Path_emmigrationrate, Path_fertilityrate)
% Important note: n is NOT the population growth rate, it is the rate of growth of the population of age j=1
%
% immigrationrate is immigrants as a percentage of the population
% emmigrationrate is emmigrants as a percentage of the population
% immigrantmewj is (unit mass) of age weights of immigrant population
% emmigrantmewj is (unit mass) of age weights of emmigrant population

if exist('fertilityrate','var')
    if ~isempty(Path_n)
        fprintf('ERROR: Cannot use population growth rate n and fertilityrate in same model \n')
        dbstack
        return
    end
    % Fertility rate is number of births as a percentage of population (last period)
    % I use this to calculate the growth rate of population of age j=1 so can then just us it.
    Path_n=zeros(1,T);
    for tt=2:n
        Path_n=(fertilityrate+sum(sj(1:end-1,tt).*initial_mewj(1:end-1)))-1;
    end
    % Note: 1+n=newborns+survivors  (because mass of population is 1)
    % Note: n in period t=1 is irrelevant as we already have mewj for period 1 (it is an initial condition)
end

%% Check input sizes
N_j=length(initial_mewj);
T=length(Path_n);

if ~exist('Path_immigrantmewj','var')
    Path_immigrantmewj=nan(N_j,T);
end
if ~exist('Path_immigrationrate','var')
    Path_immigrationrate=nan(1,T);
end
if ~exist('Path_emmigrantmewj','var')
    Path_emmigrantmewj=nan(N_j,T);
end
if ~exist('Path_emmigrationrate','var')
    Path_emmigrationrate=nan(1,T);
end
% If they were input, but are not a path
if size(Path_immigrantmewj)==[N_j,1]
    Path_immigrantmewj=Path_immigrantmewj.*ones(1,T);
elseif size(Path_immigrantmewj)==[1,N_j]
    Path_immigrantmewj=Path_immigrantmewj'.*ones(1,T);
end
if size(Path_emmigrantmewj)==[N_j,1]
    Path_emmigrantmewj=Path_emmigrantmewj.*ones(1,T);
elseif size(Path_emmigrantmewj)==[1,N_j]
    Path_emmigrantmewj=Path_emmigrantmewj'.*ones(1,T);
end
if length(Path_immigrationrate)==1
    Path_immigrationrate=Path_immigrationrate*ones(1,T);
end
if length(Path_emmigrationrate)==1
    Path_emmigrationrate=Path_emmigrationrate*ones(1,T);
end

if size(initial_mewj)==[1,N_j]
    initial_mewj=initial_mewj';
end

if size(Path_sj)==[T,N_j]
    Path_sj=Path_sj';
end
if ~all(size(Path_sj)==[N_j,T])
    error('The path on conditional survival probability should be N_j-by-T')
end

%%
Path_mewj=zeros(N_j,T);
Path_mewj(:,1)=initial_mewj;
for tt=2:T
    Path_mewj(:,tt)=UpdateAgeWeights_raw(Path_mewj(:,tt-1),Path_sj(:,tt-1),Path_n(tt),Path_immigrantmewj(:,tt),Path_immigrationrate(tt),Path_emmigrantmewj(:,tt),Path_emmigrationrate(tt));
end

end





