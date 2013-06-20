%----- Brutal force approach: train and test using CVX
% Please note that the data for cross validation is arranged as follows. 
% The best 5 of the first 20 teams(training teams) in each 
% .mat file is the first 5 players in these team matrices. The last 10
% teams(validatoin teams) are already randomized.
% Since it takes more than half an hour to do 3-fold cross validation using
% brutal-force approach on dataset 1, we seperated the 3-fold to 3 individual .mat 
% files and each .mat file will take around 10 minutes to train and test. 
% For data2, each .mat file will take around 70 seconds to train and test.


clear; close all;
% --Dataset 1. This is the data that contains 30 features that are quite informative
% load('data1_teams_crossVali_1.mat');
% load('data1_teams_crossVali_2.mat');
% load('data1_teams_crossVali_3.mat');

% --Dataset 2. This is the data that contains 18 features that are less informative
% For this dataset,training time for each .mat for data2 is around 70 seconds
load('data2_teams_crossVali_1.mat');
% load('data2_teams_crossVali_2.mat');
% load('data2_teams_crossVali_3.mat');

% --Please make sure when changing dataset, also change the number of features
% n = 30;     %number of features of each player for data1
n = 18;     %number of features of each player for data2

m = 20;     %number of training teams
C = 0.005;    

ground = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
% we made sure the first five players of the training teams matrix are the
% actural best five
optimal = [1, 2, 3, 4, 5];

comb = combntns(1 : 10, 5);    %10 choose 5
leng_comb = length(comb);

inter = combntns(1 : 5, 2);    %given a subset, number of pairs of players is 5 choose 2
leng_inter = length(inter);

% matrix containing feature map difference between the labeled optimal
% subset and each possible subset
Psi = zeros(leng_comb * m, n + n*n);      
temp_Psi = zeros(leng_comb, n + n*n);

optimal_Psi = zeros(m, n + n*n);

% matrix R which is used to mat epsilon to a leng_comb*m matrix
R = zeros(leng_comb * m, m);
temp_R = eye(m);

% matrix b for all delta, i.e. set loss
b = zeros(leng_comb * m, 1);
temp_delta = zeros(leng_comb, 1);

% matrix Lambda for tuning the trade-off between individual players and
% pairs of players. The first n rows are like eye matrix so we give full
% weight to the individual player features. The rest n*n row are like eye 
% matrix multiplying lambdas so we give lamda weight to the pair features.
lambda = n / n^2;
Lambda = eye(n + n * n);
Lambda(n+1 : end, :) = Lambda(n+1 : end, :) * lambda;

%----- construct matrix Psi and R
for i = 1 : m
    team = teams{i, 1};
    phi_inter = zeros(n * n, 1);
    
    %***** construct feature map optimal_Psi for labeled optimal subset of
    %***** this team
    for j = 1 : leng_inter
        phi_inter = phi_inter + vec(team(optimal(inter(j, 1)), :)' * team(optimal(inter(j, 2)), :));
    end
    optimal_Psi(i, :) = [sum(team(optimal, :)) - sum(team(setdiff(ground, optimal), :)), phi_inter'];
    
    %***** construct featrue map Psi for all possible subset of this team
    for j = 1 : leng_comb
        phi_inter = zeros(n * n, 1);
        this_subset = team(comb(j, :), :);
        for k = 1 : leng_inter
            phi_inter = phi_inter + vec(this_subset(inter(k, 1), :)' * this_subset(inter(k, 2), :));
        end
        this_Psi = [sum(team(comb(j, :), :)) - sum(team(setdiff(ground, comb(j, :)), :)), phi_inter'];
        this_Psi = this_Psi - optimal_Psi(i, :);
        temp_Psi(j, :) = this_Psi;
    end
    Psi((i-1) * leng_comb + 1 : i * leng_comb, :) = temp_Psi;
    
    %***** construct the matrix R which is used to mat epsilon to a leng_comb*m
    %***** matrix
    R((i-1) * leng_comb + 1 : i * leng_comb, :) = repmat(temp_R(i, :), leng_comb, 1);
end


%----- construct matrix b for all delta, i.e. set loss
% choice 1 of set loss: simple rate of overlap between labeled optimal set
% and possible subset
for i = 1 : leng_comb
    temp_delta(i) = length(intersect(comb(i, :), optimal)) / 5 - 1;
end

for i = 1 : m
    b((i-1) * leng_comb + 1 : i * leng_comb, :) = temp_delta;
end

% choice 2 of set loss : rate of overlap times the sum of feature map of 
% possible subset over the optimal subset's feature map
% for i = 1 : m
%     for j = 1 : leng_comb
%         temp_delta(j) = (length(intersect(comb(j, :), optimal)) / 5 - 1) *...
%             (norm(Psi((i-1)*leng_comb + j, :)) / norm(optimal_Psi(i, :)));
%     end
%     b((i-1) * leng_comb + 1 : i * leng_comb, :) = temp_delta;
% end
% 
% for i = 1 : m
%     b((i-1) * leng_comb + 1 : i * leng_comb, :) = temp_delta;
% end


%----- optimize using CVX
cvx_begin
    variables Omega(n + n*n) epsilon(m);
    minimize 1/2 * sum(Omega.* Omega) + C/m * sum(epsilon);
    subject to
        Psi * (Lambda * Omega) - R * epsilon <= b;
        epsilon >= 0;
cvx_end


%-------------------------------------------------%

%----- Validation
Tau = zeros(leng_comb, 1);
predicted = zeros(30 - m, 5);
accuracy = zeros(30 - m, 1);
temp_expectation = 0;

% for each test team, we calculate the accuracy
for i = m + 1 : 30
    team = teams{i, 1};
    
    % construct the feature map for each 5-player combination
    for j = 1 : leng_comb
        phi_inter = zeros(n * n, 1);
        this_subset = team(comb(j, :), :);
        for k = 1 : leng_inter
            phi_inter = phi_inter + vec(this_subset(inter(k, 1), :)' * this_subset(inter(k, 2), :));
        end
        this_Psi = [sum(team(comb(j, :), :)) - sum(team(setdiff(ground, comb(j, :)), :)), phi_inter'];
        
        % calculate the score of this 5-player combination using the weight
        % vector learned
        Tau(j) = this_Psi * Omega;
    end
    [max_Tau, max_idx] = max(Tau);
    predicted(i - 20, :) = comb(max_idx, :);
    
    % use the random number list that we maintained, we compute the true
    % optimal subset index for comparison
    [sortedRandlist, indx] = sort(randlist(i-20, :));
    true_optimal = indx(1 : 5);
    
    % compute accuracy
    accuracy(i - 20) = length(intersect(predicted(i - 20, :), true_optimal)) / 5;
    temp_expectation = temp_expectation + accuracy(i - 20);
    fprintf('The prediction accuracy for team %d is %f.\n', i, accuracy(i - 20));
end

fprintf('The expection is %f.\n', temp_expectation / 10);

% -------------------------------%
% calculate the cumulative field goal percentage, rebounds and points per
% game. These three cumulative figures are used to compare with randomly 
% selectd subsets to show that our predicted best 5 perform much better.
stats_indx = [3, 12, 18];  % index for FGP, rebounds and pts
cumu_stats = 0;
for i = m + 1 : 30
    team = teams{i, 1};
    team = team(predicted(i - 20, :), :);
    team = team(:, stats_indx);
    this_stats = sum(team);
    cumu_stats = cumu_stats + this_stats;
end
avg_stats = cumu_stats / 50;