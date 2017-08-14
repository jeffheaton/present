/* Read the CSV */

PROC IMPORT DBMS=csv OUT=train  REPLACE
  DATAFILE="/folders/myfolders/titanic-dataset.csv";
  GETNAMES=YES;
RUN;

/* Fill in missing ages with median */
PROC STDIZE DATA=train OUT=train
            METHOD=median reponly;
    VAR Age;
RUN;

/* Train/Validate split */
PROC SURVEYSELECT DATA=train outall OUT=train METHOD=srs SAMPRATE=0.7;
RUN;

DATA validate;
	SET train;
	IF selected = 0;
RUN;

DATA train;
	SET train;
	IF selected = 1;
RUN;

/* Fit the logit */
PROC LOGISTIC data=train outmodel=model descending;
  CLASS Sex / PARAM=ref ;
  CLASS Embarked / PARAM=ref ;
  MODEL Survived = Sex Age Pclass Parch SibSp Embarked;  
RUN;

/* Predict */
PROC LOGISTIC INMODEL=model;
	SCORE DATA=validate OUT=pred;
RUN;

/* Turn prediction probabilities into class values (threshold=.5) */
DATA pred;
    SET PRED(KEEP = PassengerId Survived P_1);
    pred_survived = ROUND(P_1);
RUN;

/* Evaluate */
proc freq data=pred; 
	tables Survived * pred_survived; 
run; 


