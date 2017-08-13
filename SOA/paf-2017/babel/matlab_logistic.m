% Load the data
ds = readtable('titanic-dataset.csv');

% Handle missing ages
ds.Age(isnan(ds.Age)) = nanmean(ds.Age);

% Handle categoricals
ds.Embarked = categorical(ds.Embarked);
t = dummyvar(categorical(ds.Sex));
ds.Sex = t(:,1);

% Split X & Y.
y = ds(:,'Survived');
x = ds(:,{'Pclass','Sex','Age','SibSp','Parch','Fare'});

% Create training matrix (all numeric)
x = table2array(x);
%x = horzcat(x,dummyvar(ds.Embarked));
y = table2array(y);

% Training & validation split
[trainInd,valInd] = divideblock(length(x),0.7,0.3);
x_train = x(trainInd,:);
y_train = y(trainInd,:);
x_val = x(valInd,:);
y_val = y(valInd,:);

% Fit the model
model = glmfit(x_train,y_train,'binomial','link','logit');

% Predict and calculate accuracy.
pred = glmval(model,x_val,'logit');
pred = round(pred);
acc = (pred == y_val);
sum(acc)/length(acc)
