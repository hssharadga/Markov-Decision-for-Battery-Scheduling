# Markov-Decision-for-Battery-Scheduling

Authors: Hussein Sharadga, Golbon Zakeri, Arash Khojaste


Paper: Scheduling Battery Systems Under Load Uncertianty Using Markov Decision Process 



Pricing **policy A** has  full details and notations so you might start with it.

Pricing **policies A-C** consumes about 15 minutes for Gurobi solver to return the solution.

Pricing **policy D** is for peak shaving which consumes a lot of time (about 9 hours).



Pricing **policy D** is for 6 peak thresholds:  consumes about 9 hours.

Pricing **policy D2** is for 11 peak thresholds:  consumes about 3 days  (working hours; for example 12 hours  day).



Pricing **policy A PV - policy C PV ** is after including the PV energy: 1000 Panels are used to meet some portion of  the School demand.  



data.pkl is the Solution for Policy D. 

data1.pkl is the Solution for Policy D2. 

data2.pkl is the Solution for 21 peak thresholds. 


#For Solution loading:
```
import pprint, pickle
pkl_file = open('data.pkl', 'rb')
y_ = pickle.load(pkl_file)
pkl_file.close()
```

Should you face an issue running the codes, please feel free to drop a LinkedIn message (https://www.linkedin.com/in/hussein-sharadga/).
