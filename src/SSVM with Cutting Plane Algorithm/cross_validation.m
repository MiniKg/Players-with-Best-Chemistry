 clear;
 load dataset2
 
 % the parameter of struct svm
 % -w setting,which choose the solving algorithm affects the performance a
 % lot , we set' -w 4': 1-slack algorithm (dual) with constraint cache, which 
 %is the fatest, and produces higher accuracy
 settings = ' -c 1 -o 1 -v 1  -e 0.3 -w 4 ';
  X ={};
  Y ={};
  for i = 1:30
       X{i} = team_players{i,1};
       Y{i} = [ones(5,1); zeros(5,1)];
  end
 final_accu = [];
% 3 fold cross validation
for cv_index = 0:10:20
   	fprintf('cross valiation:%d \n', cv_index/10+1);
    trainsetX = {};
    trainsetY = {};
    trainsize = 1:30;
    testsize = 1:10 ;
    testsize = testsize + cv_index;
    trainsize(testsize) = [];
    jj = 0;
    for i = trainsize
      jj = jj  + 1; 
      trainsetX{jj} = X{i};
      trainsetY{jj} = [ones(5,1); zeros(5,1)];
    end
 
    testsetX = {};
    jj = 0;
    for i = trainsize
      jj = jj + 1;
      testsetX{jj} = team_players{i,1};
      testsetY{jj} = [ones(5,1); zeros(5,1)];
    end
    
    % call the struct svm train & testing method
    % return the prediction accuracy
    [rates w accuracy] =  nba_player_pair(trainsetX,trainsetY,testsetX, testsetY,settings);

    disp('Accuracy for each testing team:');
    disp(rates');
    disp('The average accuracy is: '); 
    accuracy  
    final_accu = [final_accu accuracy];
end 

fprintf('the total accuarcy is:%f \n',mean(final_accu));