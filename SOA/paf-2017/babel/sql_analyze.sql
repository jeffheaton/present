select survived, pclass,name,decode(sex,'male',1,'female',0) as sex,
nvl(age,(select avg(age) from jth_titanic)) as age,sib_sp,fare,
DECODE(embarked,'S',1,0) as embarked_s,
DECODE(embarked,'C',1,0) as embarked_c,
DECODE(embarked,'Q',1,0) as embarked_q
from jth_titanic;
commit;

select a.pclass, sum(survived) / (select count(*) from jth_titanic b where a.pclass=b.pclass) 
from jth_titanic a
group by pclass order by 1;