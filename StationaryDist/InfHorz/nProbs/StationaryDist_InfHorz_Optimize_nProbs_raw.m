function [StationaryDist,total_zeros_created,counter_at_max_a2]=StationaryDist_InfHorz_Optimize_nProbs_raw(StationaryDist, n_a1,n_a2,N_z_input, counter,epsilon,total_zeros_created,counter_at_max_a2, simoptions)

epsilon_round=10;

% For grid interpolation, N_a2 arrives as zero.  We simplify the implementation
% by treating this N_a1x1 grid as an 1xN_a1 grid (with N_a1 spelled N_a2 below).
if n_a2==0
    N_a1=1;
    n_a2=n_a1;
else
    N_a1=max(prod(n_a1),1);
end
N_a2=prod(n_a2);
if isfield(simoptions, 'a_grid')
    a2_grid_T=gather(double(simoptions.a_grid(sum(n_a1)+1:end)))';
else
    a2_grid_T=1:N_a2;
end

% Remember whether we want to return [N_a*N_z,1] or [N_a,N_z]
StationaryDist_size=size(StationaryDist);

if N_z_input==0
    N_z=1;
else
    N_z=N_z_input;
    if StationaryDist_size(2)==1 && N_z>1
        % For our purposes, we need to unmix N_z from N_a
        StationaryDist=reshape(StationaryDist,[N_a1*N_a2,N_z]);
    end
end

new_zeros_created=zeros(1,N_z);

% For large N_z, this loop can be changed to `parfor` for greater CPU parallelism
for z_c=1:N_z
    % When N_z=1, the index z_c is only every 1, which does nothing
    StationaryDist_row=round(reshape(StationaryDist(:,z_c),[N_a1,N_a2]),epsilon_round);

    row_prob_sum=full(sum(StationaryDist_row,'all'));
    if row_prob_sum==0 || all(arrayfun(@(r) nnz(StationaryDist_row(r,:)), 1:size(StationaryDist_row,1))<3)
        % Sometimes nobody chooses the path less taken
        continue
    end

    [rows,~]=find(StationaryDist_row~=0);
    for row=unique(rows')
        % Process agents' ExpAssets row by row (i.e., each N_a1 asset mixture)

        % Find and join up runs that are reasonably close together
        [~,ea_all_idx,all_vals]=find(StationaryDist_row(row,:));
        p=find(diff(ea_all_idx)>3); % p columns are start and end of mostly consecutive elements
        runs=[ea_all_idx(1),ea_all_idx(p+1);ea_all_idx(p),ea_all_idx(end)];

        for ridx=1:size(runs,2)
            if runs(2,ridx)-runs(1,ridx)<2
                % Don't bother with short runs
                continue
            end
            this_run=runs(1,ridx):runs(2,ridx);
            vals=zeros(1,length(this_run));
            ea_this_run=ea_all_idx(ismember(ea_all_idx,this_run))-this_run(1)+1;
            vals(ea_this_run)=all_vals(ismember(ea_all_idx,this_run));
            run_prob_sum=sum(vals);

            % Attempt to consolidate min and max values to the middle
            % Don't take credit for zeros we created to make runs longer
            starting_zeros=sum(vals==0);
            cidx=length(this_run);

            % See if we can collapse this system down to two or three basis elements
            % gridvals and the sums are indexed into `this_run`, not values of `this_run`
            gridvals=vals.*a2_grid_T(this_run);
            lower_sums=cumsum(gridvals,'forward');
            lower_sums=[vals(1)*(a2_grid_T(this_run(1))~=0),lower_sums(2:end)./a2_grid_T(this_run(2:end))];
            lower_sums(isinf(lower_sums))=lower_sums(circshift(isinf(lower_sums),-1));
            upper_sums=cumsum(gridvals,'reverse');
            upper_sums=[upper_sums(1:end-1)./a2_grid_T(this_run(1:end-1)),vals(end)*(a2_grid_T(this_run(end))~=0)];
            upper_sums(isinf(upper_sums))=upper_sums(circshift(isinf(upper_sums),1)); % grids should have only a single zero value
            % Where lower_sums(1:end-1)+upper_sums(2:end)<=run_prob_sum we have room to redistribute probabilities
            valid_crossover=run_prob_sum-(lower_sums(1:end-1)+upper_sums(2:end))>=0;
            lower_idx=find(valid_crossover,1,'first');
            if isempty(lower_idx)
                % Maybe we need a span of 3...it happens
                valid_crossover=run_prob_sum-(lower_sums(1:end-2)+upper_sums(3:end))>=0;
                lower_idx=find(valid_crossover,1,'first');
                assert(~isempty(lower_idx));
                upper_idx=lower_idx+2;
            else
                upper_idx=lower_idx+1;
            end
            SystemOfEquations=[a2_grid_T(this_run(lower_idx:upper_idx));ones(1,upper_idx-lower_idx+1)];
            GoalValues=[sum(gridvals); run_prob_sum];
            new_vals=linsolve(SystemOfEquations,GoalValues);
            if any(new_vals<0)
                % Maybe we need a span of 3...it happens
                lower_idx=lower_idx-(new_vals(2)<0);
                upper_idx=upper_idx+(new_vals(1)<0);
                SystemOfEquations=[a2_grid_T(this_run(lower_idx:upper_idx));ones(1,upper_idx-lower_idx+1)];
                new_vals=linsolve(SystemOfEquations,GoalValues);
                assert(all(new_vals>=0));
            end
            res_vec_mag = norm(GoalValues-SystemOfEquations*new_vals);
            new_vals=round(new_vals,epsilon_round)';
            if res_vec_mag==0 || (~isnan(res_vec_mag) && res_vec_mag/norm(new_vals)<=epsilon)
                % Put this valid redistribution into the Stationary Dist, finishing this run
                temp=sparse(row,this_run(lower_idx:upper_idx),new_vals,N_a1,N_a2);
                StationaryDist_row(row,this_run)=temp(row,this_run);
                new_zeros_created(z_c)=new_zeros_created(z_c)+cidx-2-starting_zeros;
            end
        end
    end
    StationaryDist(:,z_c)=reshape(StationaryDist_row,[N_a1*N_a2,1]);
end

temp=reshape(full(StationaryDist),[N_a1,N_a2,N_z]);
if ~isfinite(counter_at_max_a2) && any(temp(:,N_a2,:)~=0,'all')
    counter_at_max_a2=counter;
end

sum_new_zeros=sum(new_zeros_created);
total_zeros_created=total_zeros_created+sum_new_zeros;
if simoptions.verbose==2
    if sum_new_zeros || simoptions.verbose==2
        fprintf("Counter %3d: zeros created = %d \n", counter, sum_new_zeros);
    end
end

% Re-mix N_a and N_z if necessary
StationaryDist=reshape(StationaryDist,StationaryDist_size);


end
