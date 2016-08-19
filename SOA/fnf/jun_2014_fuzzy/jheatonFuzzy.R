# Using Fuzzy Logic in R
# Jeff Heaton, April 22, 2014
library(sets)

## Setup the universe, the range of values that we will process.
sets_options("universe", seq(from = 0, to = 40, by = 0.1))

## Setup the Linguistic Variables for BMI, A1C blood pressure & underwriter rating.
variables <-
  set(
      bmi = 
        fuzzy_partition(varnames =
                          c(under = 9.25,
                            fit = 21.75,
                            over = 27.5,
                            obese  = 35),
                        sd = 3.0),
      a1c =
        fuzzy_partition(varnames =
                          c(l = 4, n = 5.25, h = 7),
                        FUN = fuzzy_cone, radius = 5),
      rating =
        fuzzy_partition(varnames =
                          c(DC = 10, ST = 5, PF = 1),
                        FUN = fuzzy_cone, radius = 5),
      bp = 
        fuzzy_partition(varnames =
                          c(norm = 0,
                            pre = 10,
                            hyp = 20,
                            shyp = 30),
                        sd = 2.5)
  )

## set up rules
rules <-
  set(
    fuzzy_rule(bmi %is% under || bmi %is% obese  || a1c %is% l,
               rating %is% DC),
    fuzzy_rule(bmi %is% over || a1c %is% n || bp %is% pre,
               rating %is% ST),
    fuzzy_rule(bmi %is% fit && a1c %is% n && bp %is% norm,
               rating %is% PF)
  )
## combine to a system
system <- fuzzy_system(variables, rules)
print(system)
plot(system) ## plots variables
## do inference
fi <- fuzzy_inference(system, list(bmi = 29, a1c=5, bp=20))
## plot resulting fuzzy set
plot(fi)
## defuzzify
gset_defuzzify(fi, "centroid")
## reset universe
sets_options("universe", NULL)

