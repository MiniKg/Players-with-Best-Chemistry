% construct .mat for cross validation
% since it takes more than half an hour to do 3-fold cross validation using
% brutal-force approach, we seperated the 3-fold to 3 individual .mat files
% and each .mat file will take 10 minutes to train and test.
clear; close all; clc;
% load('data1_team_players_ordered.mat');
load('data2_team_players_ordered.mat');

m = 20;     %number of training teams
each_fold = 10;    %number of each fold
teams = cell(30, 2);

num_test = 10;

%hold the rand perms for future use of recognizing the original labeled optimal subset
randlist = zeros(num_test, 10);   

for fold = 1 : 3
%     savefile = strcat('data1_teams_crossVali_',num2str(fold), '.mat');
    savefile = strcat('data2_teams_crossVali_',num2str(fold), '.mat');

    % arrange the data so that the validation set becomes the last ten
    % teams
    validation_set = team_players((fold-1) * each_fold + 1 : fold * each_fold, :);
    teams(1 : m, :) = [team_players(1 : (fold-1) * each_fold, :); ...
        team_players(fold * each_fold+1 : end, :)];
    teams(m + 1 : end, :) = validation_set;
    
    % randomize the validation set, i.e. the last 10 teams in a particular file
    for i = m + 1 : m + num_test
    team = teams{i, 1};
    rand = randperm(10);
    randlist(i-m, :) = rand;
    team = team(rand, :);
    teams{i, 1} = team;
    end
    
    save(savefile, 'teams', 'randlist');
end