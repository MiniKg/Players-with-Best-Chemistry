function[rates w accuracy] =  nba_player_pair(trainsetX,trainsetY,testsetX, testsetY,settings)
% STRUCTURE SVM for optimal subset of NBA players
% We have applied model of pairs of players

  % ------------------------------------------------------------------
  %                                                    Run SVM struct
  % ------------------------------------------------------------------
  [mm nn] = size(trainsetX{1});
 % setting parameters
  parm.patterns = trainsetX ;
  parm.labels = trainsetY ;
  parm.lossFn = @lossCB ;
  parm.constraintFn  = @constraintCB ;
  parm.featureFn = @featureCB ;
  parm.dimension = nn + nn*nn
  % let the para to be wordy
  parm.verbose = 0 ;
  % run the ssvm train
  % if an error occurs ,try to make sure the source code of SSVM tool box 
  % is compiled  properly 
  model = svm_struct_learn(settings, parm) ;
  w = model.w ;% w is the parameter of our structure svm

  % ------------------------------------------------------------------
  %                                                Perform Prediction
  %         yhat = argmax_y<w,psi(x,y)>
  %
  % ------------------------------------------------------------------
  
  %  creat a set of all possible y
  combos = combntns(1:10,5);
  Yset =[];
  for y =   combos'
    tmp = zeros(10,1);
    tmp(y) = 1;
    Yset = [Yset tmp];
  end
  
  size_test = length(testsetX);
  % in our data set, ther first 5 players are the optimal subset
  target = [ones(5,1); zeros(5,1)];% this is the optimal subset
  total = 0;
  correct = 0;
  % test one by one
  rates = [];
  three_stats = [0 0 0];
  stats_index = [3 12 18];
  for i = 1:size_test  
      ttt = testsetX{i};    
      index =randperm(10);
      %index = 1:10;
      ttt = ttt(index,:);  % ttt is one test set of x
      yyy = target(index); % yyy is the optimal subset, y
      yhat = ssv_predict(parm,w,ttt,Yset);
      correct = correct + sum(yyy == yhat);% players we predict sucessfully
      total = total + 10;
      rates = [rates sum(yyy == yhat)/ 10];
      op_index = find(yhat>0);
      rd_index = randperm(10);
      rd_index = rd_index(1:5);
      mean_stats = mean(ttt(op_index,:),1);
      three_stats = three_stats + mean_stats(stats_index);
  end
  
  three_stats = three_stats/size_test;
  accuracy = correct / total;
  
end


% ------------------------------------------------------------------
%                                               SVM struct callbacks
% ------------------------------------------------------------------


function delta = lossCB(param, y, ybar)
    a = sum(y.*ybar);
    b = sum((1-y).*(1-ybar));
    tmp = 5;
    if tmp <10-5+a-b
        tmp = 10-5+a-b;
    end
    delta = 1 - a/tmp;

end

%call back function of feature mapping

function psi = featureCB(param, x, y)
    % get the feature map
    pos_index = find(y>0);
    neg_index = find(y ==0);
    w = sum(x(pos_index,:),1) - sum(x(neg_index,:),1);
    n = length(w);
    % now calculate theta
    combos = combntns(1:10,2);
    theta = zeros(n*n,1);
    % try all pairs in optimal subset
    for index = combos'
        flag = 1;
        i = index(1);
        j = index(2);
       % skip pairs that are not all in optimal subset
        if y(i)*y(j) == 0
            continue;
        end
        % xi xj are player vectors of player i and j
        xi = x(i,:);
        xj = x(j,:);
        % get inner procudct between their features
        tmp = xi'*xj;
        tmp = tmp(:);
        theta = theta + flag*tmp;        
    end
    %lambda is the weight constant of features
    lambda = 30/900;
    
    psi =  [w  lambda*theta'] ; 
    % ATTENTION! if you want to use this tool box
    % please return a sparse colum of psi, NOT a row vector
    psi = sparse(psi');
    
 
end

%call back function of constain
function yhat = constraintCB(param, model, x, y)
% slack resaling: argmax_y delta(yi, y+ 1 + <psi(x,y1), w> 
    combos = combntns(1:10,5);
    %initialize the maxx value
    maxxxx = -999999;
    yhat = [0 0 0  0];
    yi = y;
    for i = combos'% try all possible subsets
        y1 = zeros(10,1);
        y1(i) = 1;
        %y y1 yi y   
        delta = lossCB(param, yi, y1);
        psi1 = featureCB(param,x,y1);
     %   psi  = featureCB(param,x,yi);
        final = delta*(1 + dot(psi1,model.w));% - dot(psi,model.w));
        if final > maxxxx
            maxxxx = final;
            yhat = y1;
        end
     % max   
    end
  % yhat 
 
end

%   perform prediction
%   yhat = argmax_y<w,psi(x,y)>
function yhat = ssv_predict(parm,w,x,Yset)
    max_val = -999999;
    yhat = [];
 
    for y1 = Yset%travel every colum, try all possible subsets
        psi = featureCB(parm,x,y1);
        prj = dot(w,psi);
        if prj > max_val
            max_val = prj;
            yhat = y1;
        end
    end
end