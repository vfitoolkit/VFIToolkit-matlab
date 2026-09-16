function [StationaryDist_jj,total_zeros_created,jj_at_max_a2]=StationaryDist_FHorz_Optimize_nProbs_raw(StationaryDist_jj, n_a1,n_a2,N_z_input,jj, epsilon,total_zeros_created,jj_at_max_a2, simoptions)

% Dynamic precision rounding to prevent singlefp truncation
if isfield(simoptions, 'precision') && strcmp(simoptions.precision, 'single')
    epsilon_round = 6;
else
    epsilon_round = 10;
end

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

% Gather so we can round; it will be placed back into gpuArray by caller
StationaryDist_jj=gather(StationaryDist_jj);

% Remember whether we want to return [N_a*N_z,1], [N_a,N_z], or [N_a1, N_a2, N_z]
StationaryDist_jj_size=size(StationaryDist_jj);

if N_z_input==0
    N_z=1;
else
    N_z=N_z_input;
end

% FORCE RESHAPE to 2D [N_a1*N_a2, N_z] for the loop logic regardless of input shape
StationaryDist_jj = reshape(StationaryDist_jj, [N_a1*N_a2, N_z]);

new_zeros_created=zeros(1,N_z);

% For large N_z, this loop can be changed to `parfor` for greater CPU parallelism
for z_c=1:N_z
    % When N_z=1, the index z_c is only every 1, which does nothing
    StationaryDist_row_jj=round(reshape(StationaryDist_jj(:,z_c),[N_a1,N_a2]),epsilon_round);

    row_prob_sum=full(sum(StationaryDist_row_jj,'all'));

    % INSTANT VECTORIZED CHECK: Skip if the max non-zeros in any row is < 3
    if row_prob_sum==0 || max(sum(StationaryDist_row_jj > 0, 2)) < 3
        continue
    end

    [rows,~]=find(StationaryDist_row_jj~=0);
    for row=unique(rows')
        % Process agents' ExpAssets row by row (i.e., each N_a1 asset mixture)

        % Find and join up runs that are reasonably close together
        [~,ea_all_idx,all_vals_jj]=find(StationaryDist_row_jj(row,:));
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
            vals(ea_this_run)=all_vals_jj(ismember(ea_all_idx,this_run));
            run_prob_sum=sum(vals);

            % Attempt to consolidate min and max values to the middle
            % Don't take credit for zeros we created to make runs longer
            starting_zeros=sum(vals==0);
            cidx=length(this_run);

            % See if we can collapse this system down to two basis elements
            % gridvals and the sums are indexed into `this_run`, not values of `this_run`
            gridvals = vals .* a2_grid_T(this_run);

            % 1. Calculate the center of mass of this specific run
            target_mean = sum(gridvals) / run_prob_sum;

            % 2. Find the exact grid interval that bounds this mean
            lower_idx = find(a2_grid_T(this_run) <= target_mean, 1, 'last');

            % 3. Guard against exact hits on the upper boundary
            if lower_idx == length(this_run)
                lower_idx = lower_idx - 1;
            end
            upper_idx = lower_idx + 1;

            aL = a2_grid_T(this_run(lower_idx));
            aU = a2_grid_T(this_run(upper_idx));

            % 4. Direct algebraic solution to preserve 0th (mass) and 1st (mean) moments
            p_U = (sum(gridvals) - run_prob_sum * aL) / (aU - aL);
            p_L = run_prob_sum - p_U;

            new_vals = round([p_L, p_U], epsilon_round);

            % 5. Put this valid redistribution into the Stationary Dist, finishing this run
            temp = sparse(row, this_run(lower_idx:upper_idx), new_vals, N_a1, N_a2);
            StationaryDist_row_jj(row, this_run) = temp(row, this_run);

            % Tally the zeros we successfully created
            new_zeros_created(z_c) = new_zeros_created(z_c) + cidx - 2 - starting_zeros;
        end
    end
    StationaryDist_jj(:,z_c)=reshape(StationaryDist_row_jj,[N_a1*N_a2,1]);
end

temp=reshape(full(StationaryDist_jj),[N_a1,N_a2,N_z]);
if jj<jj_at_max_a2 && any(temp(:,N_a2,:)~=0,'all')
    jj_at_max_a2=jj;
end

sum_new_zeros=sum(new_zeros_created);
total_zeros_created=total_zeros_created+sum_new_zeros;
if simoptions.verbose==2
    if sum_new_zeros || simoptions.verbose==2
        fprintf("Age %3d: zeros created = %d \n", jj, sum_new_zeros);
    end
end

% Re-mix N_a and N_z if necessary
StationaryDist_jj=reshape(StationaryDist_jj,StationaryDist_jj_size);


end