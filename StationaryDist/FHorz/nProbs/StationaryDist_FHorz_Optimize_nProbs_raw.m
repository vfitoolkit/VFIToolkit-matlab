function [StationaryDist_jj,total_zeros_created,jj_at_max_a2]=StationaryDist_FHorz_Optimize_nProbs_raw(StationaryDist_jj, n_a1,n_a2,N_z_input,jj, epsilon,total_zeros_created,jj_at_max_a2, simoptions)

epsilon_round=10;

% For grid interpolation, N_a2 arrives as zero.  We simplify the implementation
% by treating this N_a1x1 grid as an 1xN_a1 grid (with N_a1 spelled N_a2 below).
if n_a2==0
    N_a1=1;
    n_a2=n_a1;
else
    N_a1=prod(n_a1);
end
N_a2=prod(n_a2);
if isfield(simoptions, 'a_grid')
    a2_grid_T=gather(double(simoptions.a_grid(sum(n_a1)+1:end)))';
else
    a2_grid_T=1:N_a2;
end

% Remember whether we want to return [N_a*N_z,1] or [N_a,N_z]
StationaryDist_jj_size=size(StationaryDist_jj);

if N_z_input==0
    N_z=1;
else
    N_z=N_z_input;
    if StationaryDist_jj_size(2)==1 && N_z>1
        % For our purposes, we need to unmix N_z from N_a
        StationaryDist_jj=reshape(StationaryDist_jj,[N_a1*N_a2,N_z]);
    end
end

new_zeros_created=zeros(1,N_z);

% For large N_z, this loop can be changed to `parfor` for greater CPU parallelism
for z_c=1:N_z
    % When N_z=1, the index z_c is only every 1, which does nothing
    StationaryDist_row_jj=round(reshape(StationaryDist_jj(:,z_c),[N_a1,N_a2]),epsilon_round);

    row_prob_sum=full(sum(StationaryDist_row_jj,'all'));
    if row_prob_sum==0 || all(arrayfun(@(r) nnz(StationaryDist_row_jj(r,:)), 1:size(StationaryDist_row_jj,1))<3)
        % Sometimes nobody chooses the path less taken
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

            zero_created=false;
            cidx=length(this_run);
            zero_candidate=zeros(1,cidx);
            nonzeros=true(1,cidx);

            if cidx>3
                % Just once, try to zero out both max and min in one shot
                if any(a2_grid_T(this_run)==0)
                    SystemOfEquations=[this_run;ones(1,cidx);[1,zeros(1,cidx-1)];[zeros(1,cidx-1),1]];
                    GoalValues=[sum(vals.*this_run); run_prob_sum; 0; 0];
                    new_vals=linsolve(SystemOfEquations,GoalValues);
                    res_vec_mag = norm(GoalValues-SystemOfEquations*new_vals);
                else
                    SystemOfEquations=[a2_grid_T(this_run);ones(1,cidx);[1,zeros(1,cidx-1)];[zeros(1,cidx-1),1]];
                    GoalValues=[sum(vals.*a2_grid_T(this_run)); run_prob_sum; 0; 0];
                    new_vals=linsolve(SystemOfEquations,GoalValues);
                    res_vec_mag = norm(GoalValues-SystemOfEquations*new_vals);
                end
                new_vals=round(new_vals,epsilon_round)';
                if isnan(res_vec_mag) || res_vec_mag/norm(new_vals)>epsilon || all(abs(new_vals-vals)<epsilon) || any(new_vals<0)
                    % zero_candidate(cidx)=0;
                else
                    vals=new_vals;
                    if cidx<5
                        % Early out if we collapsed 4 values into 2
                        temp=sparse(row,this_run,vals,N_a1,N_a2);
                        StationaryDist_row_jj(row,this_run)=temp(row,this_run);
                        new_zeros_created(z_c)=new_zeros_created(z_c)+sum(vals==0)-starting_zeros;
                        continue
                    end
                    zero_created=true;
                    nonzeros(1)=false; nonzeros(cidx)=false;
                    zero_candidate(1)=1; zero_candidate(cidx)=1;
                    cidx=cidx-1;
                end
            end
            zero_candidate(cidx)=1;
            while nnz(vals)>1
                % Try to zero out largest indices, perhaps squeezing zero from middle
                if any(a2_grid_T(this_run(nonzeros))==0)
                    SystemOfEquations=[this_run(nonzeros);ones(1,nnz(nonzeros));zero_candidate(nonzeros)];
                    GoalValues=[sum(vals(nonzeros).*this_run(nonzeros)); run_prob_sum; 0];
                    new_vals=linsolve(SystemOfEquations,GoalValues);
                    res_vec_mag = norm(GoalValues-SystemOfEquations*new_vals);
                else
                    SystemOfEquations=[a2_grid_T(this_run(nonzeros));ones(1,nnz(nonzeros));zero_candidate(nonzeros)];
                    GoalValues=[sum(vals(nonzeros).*a2_grid_T(this_run(nonzeros))); run_prob_sum; 0];
                    new_vals=linsolve(SystemOfEquations,GoalValues);
                    res_vec_mag = norm(GoalValues-SystemOfEquations*new_vals);
                end
                new_vals=round(new_vals,epsilon_round)';
                if isnan(res_vec_mag) || res_vec_mag/norm(new_vals)>epsilon || all(abs(new_vals-vals(nonzeros))<epsilon) || any(new_vals<0)
                    zero_candidate(cidx)=0;
                    break
                end
                vals(nonzeros)=new_vals;
                nonzeros(cidx)=false;
                zero_created=true;
                cidx=cidx-1;
                zero_candidate(cidx)=1;
            end
            if zero_created
                zero_candidate(cidx)=0;
            end
            cidx=find(zero_candidate==0,1,'first');
            zero_candidate(cidx)=1;
            while nnz(vals)>1
                % Try to zero out least index, perhaps squeezing zero from middle
                if any(a2_grid_T(this_run(nonzeros))==0)
                    SystemOfEquations=[this_run(nonzeros);ones(1,nnz(nonzeros));zero_candidate(nonzeros)];
                    GoalValues=[sum(vals(nonzeros).*this_run(nonzeros)); run_prob_sum; 0];
                    new_vals=linsolve(SystemOfEquations,GoalValues);
                    res_vec_mag = norm(GoalValues-SystemOfEquations*new_vals);
                else
                    SystemOfEquations=[a2_grid_T(this_run(nonzeros));ones(1,nnz(nonzeros));zero_candidate(nonzeros)];
                    GoalValues=[sum(vals(nonzeros).*a2_grid_T(this_run(nonzeros))); run_prob_sum; 0];
                    new_vals=linsolve(SystemOfEquations,GoalValues);
                    res_vec_mag = norm(GoalValues-SystemOfEquations*new_vals);
                end
                new_vals=round(new_vals,epsilon_round)';
                if isnan(res_vec_mag) || res_vec_mag/norm(new_vals)>epsilon || all(abs(new_vals-vals(nonzeros))<epsilon) || any(new_vals<0)
                    break
                end
                vals(nonzeros)=new_vals;
                nonzeros(cidx)=false;
                cidx=cidx+1;
                zero_candidate(cidx)=1;
            end
            temp=sparse(row,this_run,vals,N_a1,N_a2);
            StationaryDist_row_jj(row,this_run)=temp(row,this_run);
            new_zeros_created(z_c)=new_zeros_created(z_c)+sum(vals==0)-starting_zeros;
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
