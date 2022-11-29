# -*- coding: utf-8 -*-
"""
Created on Sat Nov  17 03:14:34 2022

@author: Hussein Sharadga
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import random


# Time Count
import timeit
start = timeit.default_timer()


##########################
# A: Import School Results
##########################


school=pd.read_csv('school.csv')
school_test=school[329088-96*365+96*4-1:329088-2]# Load profile for one year, starts on January 1, 2010 at 12:00 AM (00:00 in 24 hr style)
school_test=school_test[0:360*24*4] # The data is recorded every 15 minutes, thus 4*24 points for one day and 360 is assumed to be the 12 months  (We know that 12 months is 365 days)
school_test=np.array(school_test)
school_test=np.reshape(school_test,(360*96))
# Averaging: 15 mins to hourly
school_test=np.reshape(school_test,(360*24,4))
school_test=np.mean(school_test,axis=1)         # Hourly demnad for 360 days

# One year for training and one year for test
x=329088-96*365+96*4-1
school_train=school[x-96*365-1:x-2]
school_train=school_train[1*96:-(4)*96+1]
school_train=np.array(school_train)
school_train=np.reshape(school_train,(360*96))
# Averaging: 15 mins to hourly
school_train=np.reshape(school_train,(360*24,4))
school_train=np.mean(school_train,axis=1)   

# Work days only: remove the weekends
school_train_work=[]
school_test_work=[]
for i in range(51):
    school_train_work=np.concatenate((school_train_work,school_train[24*7*i:24*7*i+24*5])) # 5 work days
    school_test_work=np.concatenate((school_test_work,school_test[24*7*i:24*7*i+24*5])) 
    
# plt.plot(school_test_work[0:24*7])
# plt.plot(school_train_work[0:24*7])
# plt.plot(school_test[0:24*7])
# plt.plot(school_train[0:24*7])


##########################
# B: Quantiles Fitting
##########################

# Sample
# plt.plot(school_train_work[0:24*14])

quantile=np.linspace(10, 100,num=10)

nn=255 # number of work day in a year

t=np.linspace(1, 24*nn,num=24*nn)


nn_T=1       # T period [number  of days]
# nn_T=20*3  ~ 3 month
w=2*np.pi/(24*nn_T)   # w=f= 2pi/T  T is the time required to finish one wave (step here is hours so the time unit is hour not second)

                      # f=1/T [period/s]  but period =2pi rad   thus  f=1/T [2pi rad /s] = 2pi/T
                      # while T is supposed to be in second it will be hour becasue the time step here is one hour
                      # or w=f=1/T but we take cos(2pi nwt);  here I am taking cos(nwt)
               
       
               
n=100   # number of Foureir terms oe degree
     

# cos/sin matrix          
matrix1=np.zeros((len(t),n))
matrix2=np.zeros((len(t),n))
for i in range (n):
    matrix1[:,i]=np.cos((i+1)*w*t);  # % cos matrix
    matrix2[:,i]=np.sin((i+1)*w*t); # % sin matrix

    
# Fourier quantile regression
demand_quantile=np.zeros((9,24))
for i in range(9):
    
    T=1-0.1*(i+1) # quantile (beta)


    m=gp.Model()
    muu=m.addVar(vtype='C',lb=-GRB.INFINITY,  name='muu')  
    A=m.addVars(n, lb=-GRB.INFINITY, vtype='C', name='A')
    B=m.addVars(n, lb=-GRB.INFINITY, vtype='C', name='B')
    C=m.addVars(len(t), lb=-GRB.INFINITY, vtype='C', name='C')
    D=m.addVars(len(t), lb=-GRB.INFINITY, vtype='C', name='D')
    

    # add auxiliary variables for max function
    auxvarpos=m.addVars(len(t), lb=-GRB.INFINITY, vtype='C', name='auxvarpos')
    auxvarneg=m.addVars(len(t), lb=-GRB.INFINITY, vtype='C', name='auxvarneg')
    maxobj1=m.addVars(len(t), lb=0, vtype='C', name="maxobj1")
    maxobj2=m.addVars(len(t), lb=0, vtype='C', name="maxobj2")
    
    # add auxiliary equality constraints
    m.addConstrs((auxvarpos[i] ==  school_train_work[i]-muu-C[i]-D[i]) for i in range(len(t)))
    m.addConstrs((auxvarneg[i] == -school_train_work[i]+muu+C[i]+D[i]) for i in range(len(t)))
    
    # add constraints maxobj1 = max(auxvarpos,0), maxobj2 = max(auxvarneg,0)
    m.addConstrs((maxobj1[i] == gp.max_(auxvarpos[i], constant=0) for i in range(len(t))))
    m.addConstrs((maxobj2[i] == gp.max_(auxvarneg[i], constant=0) for i in range(len(t))))
    
    obj1=gp.quicksum( maxobj1[i] for i in range(len(t)))
    obj2=gp.quicksum( maxobj2[i] for i in range(len(t)))
     
    m.setObjective((T*obj1+(1-T)*obj2)/len(t)) 
    
    
    # Wrong version:
    # obj1=gp.quicksum(T*np.max((school_train_work[i]-muu-C[i]-D[i]),0)         for i in range(len(t))) 
    # obj2=gp.quicksum((1-T)*np.max(-1*(school_train_work[i]-muu-C[i]-D[i]),0)  for i in range(len(t)))  
    # m.setObjective(obj1+obj2) 
    
    
    m.addConstrs( C[i]== gp.quicksum(A[k]*matrix1[i][k] for k in range (n)) for i in range (len(t))) 
    m.addConstrs( D[i]== gp.quicksum(B[k]*matrix2[i][k] for k in range (n)) for i in range (len(t))) 
    
    m.optimize()
    
    
    # Validation of Fourier quantile regression
    
    plt.plot(t[1:24*nn],school_train_work[1:24*nn],'k-', label='Actual')
    xx=[muu.x+C[i].x+D[i].x for i in range(len(t))] 
    plt.plot(t[1:24*nn],xx[1:24*nn],'r-', label='Fitted')
    plt.legend(loc='upper right')
    plt.xlabel("Time [hrs]")
    plt.ylabel("Demand [kWh]")
    plt.show()
    
    
    nn=7
    plt.plot(t[1:24*nn],school_train_work[1:24*nn],'k-', label='Actual')
    xx=[muu.x+C[i].x+D[i].x for i in range(len(t))] 
    plt.plot(t[1:24*nn],xx[1:24*nn],'r-', label='Fitted')
    plt.legend(loc='upper right')
    plt.xlabel("Time [hrs]")
    plt.ylabel("Demand [kWh]")
    plt.show()
    
    
    nn=nn_T
    plt.plot(t[1:24*nn],school_train_work[1:24*nn],'k-', label='Actual')
    xx=[muu.x+C[i].x+D[i].x for i in range(len(t))] 
    plt.plot(t[1:24*nn],xx[1:24*nn],'r-', label='Fitted')
    plt.legend(loc='upper right')
    plt.xlabel("Time [hrs]")
    plt.ylabel("Demand [kWh]")
    plt.show()
    
    # nn=10
    # plt.plot(t[24*nn:24*(nn+7)],school_train_work[24*nn:24*(nn+7)],'k-', label='Actual')
    # xx=[muu.x+C[i].x+D[i].x for i in range(len(t))] 
    # plt.plot(t[24*nn:24*(nn+7)],xx[24*nn:24*(nn+7)],'r-', label='Fitted')
    # plt.legend(loc='upper right')
    # plt.xlabel("Time [hrs]")
    # plt.ylabel("Demand [kWh]")
    # plt.show()
    
    
    # Storing the quantile 
    demand_quantile[i,:]=xx[0:24*nn]



# plot demand quantiles
for i in range(9):
    if i==0:
        plt.plot(demand_quantile[i,:],label='0.9')
    else:
        if i==8:
            plt.plot(demand_quantile[i,:],label='0.1')
        else:
            plt.plot(demand_quantile[i,:])
            
plt.xlabel("Time [hrs]")
plt.ylabel("Demand [kWh]")   
plt.legend() 
plt.show()    




#########################################################
# C: Porbabilty Transistion Matrix for demand quantiles
#########################################################

# Determing the quantile at every steps
qunatiles=np.zeros((255,24)) # 255 days and 24 hours
for i in range (255):
    for j in range (24):
        point=school_train_work[i*24+j]          # every step in the year
        qunatiles_at_point=demand_quantile[:,j]  # qunatiles at that time step of the day
        
        # Find the quantile crossponding to the point
        x=point>qunatiles_at_point
        y=np.where(x==True)
        yy=np.where(x==False)
        if len(yy[0])==9:
            y=9
        else: 
            y=y[0][0]
            if y==0:
                y=1
        quant=1-0.1*y
        
        qunatiles[i,j]=np.round(quant,1)
        # print(np.round(quant,1))
        
        
 # Test code       
# point=600
# x=point>demand_quantile[:,12]

# y=np.where(x==True)
# yy=np.where(x==False)

# if len(yy[0])==9:
#     y=9
# else: 
#     y=y[0][0]
#     if y==0:
#         y=1

# quant=1-0.1*y
# print(quant)
        
        
 

# Porbabilty Transistion Matrix
qunatiles=np.reshape(qunatiles,(1,255*24)) 

def transition_matrix(transitions):
    n = 9 #number of states

    transitions=[ int(transitions[0][i]*10-1) for i in range(len(qunatiles[0])) ] # 0.1 will be 0, 0.2 will be 1, 0.3 will be 2
    
    M = np.zeros((n,n))   # Porbabilty Transistion Matrix

    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1

    #now convert to probabilities:
    M = M/M.sum(axis=1, keepdims=True)
    return M


M = transition_matrix(qunatiles)   # Porbabilty Transistion Matrix
for row in M: print(' '.join(f'{x:.2f}' for x in row))
mm=np.round(M,2)                   # Porbabilty Transistion Matrix



# Plot the Porbabilty Transistion Matrix
fig, ax = plt.subplots()
min_val, max_val = 0.1, 0.9
intersection_matrix = mm
ax.matshow(intersection_matrix, cmap=plt.cm.Blues)
alpha=['0.1', '0.2','0.3', '0.4','0.5', '0.6','0.7', '0.8','0.9']
ax.set_xticklabels(['']+alpha)
ax.set_yticklabels(['']+alpha)
for i in range(9):
    for j in range(9):
        c = intersection_matrix[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
        
        
     
        
####################################   
# D Electrcity price
####################################


tt=np.linspace(1, 24,num=24)

# Trail demand 
demand= np.array([0, 50, 50, 50, 90, 90, 90, 90, 100, 100,100,100,100, 90,90, 50, 50, 50, 50, 40,30, 0, 0, 0]*10)
demand=np.reshape(demand,(10,24))

# Prices
a=[0.08]*7
b=[0.05]*8
c=[0.12]*6
d=[0.08]*3
cost=np.concatenate((a,b,c,d))


# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(tt,
        demand[0],
        color="red", 
        marker="")
# set x-axis label
ax.set_xlabel("time", fontsize = 14)
# set y-axis label
ax.set_ylabel("demand",
              color="red",
              fontsize=14)

# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(tt, cost,color="blue",marker="o")
ax2.set_ylabel("Price [$/kWh]",color="blue",fontsize=14)
plt.show()

# save the plot as a file
# fig.savefig('two_different_y_axis_for_single_python_plot_with_twinx.jpg',
#             format='jpeg',
#             dpi=100,
#             bbox_inches='tight')


#############
# E: Cost
#############
Cb=500 # Battery
Peak=[0,50,100,150,200,250,300,350,400,450,500]
C_=np.zeros((9,21,24,11,11))   # Cost [t, Soc, demnad_quantile, Alpha,peak]  that what it is supposed to be
                            # Cost [demnad_quantile, Alpha, t, Soc,peak]  that what it is right now
eta=0.92
for k in range (0,9):               # demnad_quantile
    Alpha=np.linspace(-1, 1,num=21)
    jj=-1                    
    for alpha in Alpha:
        alpha=np.round(alpha,1)
        jj=jj+1     # Counter 
        
        
        
        for j in range(0,11):  # SOC
            
            
            penalty=0
            SOC=j*0.1
            Es=(SOC+alpha)*Cb
            
            if Es>Cb:
                penalty=1500
            if Es<0:
                penalty=1500
            # C_[k,jj,i-1,j]=np.round(cost[i-1]*(max(demand_quantile[k][i-1]+max(alpha*eta,alpha/eta)*Cb,0)) +penalty,1)
            
            grid_energy_vector_day=[]
            for i in range (1,25):      # time
                
                grid_energy=max(demand_quantile[k][i-1]+max(alpha*eta,alpha/eta)*Cb,0)
                grid_energy_vector_day=np.append(grid_energy_vector_day,grid_energy)
                # if grid_energy>=200:
                #     cost=0.12
                # else:
                #     cost=0.05
                
                cost=0.05 
                
                if i==24:
                    for o in range(11):
                        C_[k,jj,i-1,j,o]=np.round(cost*grid_energy+penalty,1) +np.round(max(Peak[o],grid_energy)*7,1)
                else:
                    for o in range(11):
                        C_[k,jj,i-1,j,o]=np.round(cost*grid_energy+penalty,1) 





############
# F: MDP             
############

# Porbabilty Transistion Matrix for time
t_p = [[0 for j in range(24)] for i in range(24)]
t_p[23][0] = 1
for j in range(0,23):
  t_p[j][j+1] = 1
#print(t_p)



# Porbabilty Transistion Matrix for SOC
SOC_p = np.zeros((21,11,11))

# discharging
k=-1
for j in range(11):
    k=k+1
    for i in range(11-k):
        SOC_p[j][i][0]=1
    
    for ii in range(k):
        SOC_p[j][11-k+ii][ii+1]=1

# charging
k=-1
for j in range(11,21):
    k=k+1
    for i in range(11-k-1):
        SOC_p[j][i][i+1+k]=1
    
    
    for ii in range(k+1):
        SOC_p[j][11-k-1+ii][10]=1



# Porbabilty Transistion Matrix for Peak

peak_p=np.zeros((24,9,21,11,11))

for i in range(23):
    for q in range(9):
        for k in range(21):
            
            for p in range(11):
                
                grid_energy=max(demand_quantile[q][i]+max(Alpha[k]*eta,Alpha[k]/eta)*Cb,0)
                
                recent_peak=Peak[p]
                
                current_peak=max(recent_peak,grid_energy)
                
                if current_peak>=500:
                    u=10
                elif current_peak>=450:
                    u=9
                elif current_peak>=400:
                    u=8                             
                elif current_peak>=350:
                    u=7
                elif current_peak>=300:
                    u=6                 
                elif current_peak>=250:
                    u=5                             
                elif current_peak>=200:
                    u=4
                elif current_peak>=150:
                    u=3                   
                elif current_peak>=100:
                    u=2
                elif current_peak>=50:
                    u=1  
                else:
                    u=0
                
                
                
                peak_p[i][q][k][p][u]=1
                
                
for q in range (9):
    for  k in range (21):
        for p in range(6):
            peak_p[23][q][k][p][0]=1 # we reset at the begining of the day (end of the day)
            
        


# for j in range(9):
#     for k in range (21):
#         print(peak_p[23][j][k])


stop = timeit.default_timer()
print('Time in mins: ', (stop - start)/60)  


# MDP 
start = timeit.default_timer()

m=gp.Model()
# m.params.nonConvex=2

y=m.addVars(24, 11, 9, 21, 11, vtype='C',lb=0, ub=1, name='y')



obj=gp.quicksum(C_[q,k,i,j,o]*y[i,j,q,k,o] for i in  range(24) for j in range (11) for q in range (9) for k in range (21)  for o in range (11))

m.setObjective(obj) # the defult is to MINIMIZE


m.addConstr(gp.quicksum(y[i,j,q,k,o] for i in  range(24) for j in range (11) for q in range (9) for k in range (21)  for o in range (11))==1) 
# m.addConstr((gp.quicksum(y[i,j,q,k,o] for i in  range(24) for j in range (11) for q in range (9) for k in range (21)  for o in range (6))<=1.02) 
# m.addConstr((gp.quicksum(y[i,j,q,k] for i in  range(24) for j in range (11) for q in range (9) for k in range (21) )) <= 1) 

# m.addConstrs((gp.quicksum(y[ii,jj,qq,k] for k in range (21)) - (gp.quicksum(y[i,j,q,k]*(mm[q][qq]*SOC_p[k][j][jj]*t_p[i][ii]) for k in range (21) for i in  range(24) for j in range (11) for q in range (9))) ==0 for ii in  range(24) for jj in range (11) for qq in range (9)), name='const')
# m.addConstrs((gp.quicksum(y[ii,jj,qq,k] for k in range (21)) - (gp.quicksum(y[i,j,q,k]*(mm[q][qq]*SOC_p[k][j][jj]*t_p[i][ii]) for k in range (21) for i in  range(24) for j in range (11) for q in range (9))) >=-0.0001 for ii in  range(24) for jj in range (11) for qq in range (9)), name='const')

m.addConstrs(gp.quicksum(y[ii,jj,qq,k,oo] for k in range (21)) - gp.quicksum(y[i,j,q,k,o]*(mm[q][qq]*SOC_p[k][j][jj]*t_p[i][ii]*peak_p[i][q][k][o][oo]) for k in range (21) for i in  range(24) for j in range (11) for q in range (9) for o in range (11))<=0  for ii in  range(24) for jj in range (11) for qq in range (9) for oo in range (11))

m.optimize()


stop = timeit.default_timer()
print('Time in mins: ', (stop - start)/60)   

# m.printAttr('X')



# sum=0
# for ii in  range(24):
#     for jj in range (11):
#         for qq in range (9):
#             for k in range (21):
#                 for i in  range(24):
#                     for j in range (11):
#                         for q in range (9):
#                             sum=sum+mm[q][qq]*SOC_p[k][j][jj]*t_p[i][ii]




# MDP Validation for constraint (1)
sum=0
for i in  range(24):
    for j in range (11):
        for q in range (9):
            for k in range (21):
                for o in range(11):
                    sum=sum+y[i,j,q,k,o].x
                              
 
                
 

# MDP Validation for constraint (2) 
start = timeit.default_timer()
sum1=0
sum2=0
cons=[]
highest_violation=0
i=-1
for ii in  range(24):
    for jj in range (11):
        for qq in range (9):
            for oo in range(11):
                i=i+1
                sum1=0
                sum2=0
                
        
                # x=
                p1=[y[ii,jj,qq,k,o].x for k in range (21)]
                # sum1=sum1+np.sum(p1)
                sum1=np.sum(p1)

                p2= [ y[i,j,q,k,o].x*(mm[q][qq]*SOC_p[k][j][jj]*t_p[i][ii]*peak_p[i][q][k][o][oo]) for k in range (21) for i in  range(24) for j in range (11) for q in range (9) for o in range(11)]
                sum2=sum2+np.sum(p2)  
                cons=(sum1-sum2)
                # print(sum1)
                # print(sum2)
                # print(cons)
                violation=abs(sum1-sum2)
                if violation>highest_violation:
                    highest_violation=violation
                    print(sum1)
                    print(sum2)
                    print(cons)
                    print('-------------')
                    
                
                 
            
stop = timeit.default_timer()
print('Time in mins: ', (stop - start)/60) 





# Check: One descion for one status 
for i in  range(24):
    for j in range (11):
        for q in range (9):
            for o in range(11):
            
                sum=0
                for k in range (21):
                    if y[i,j,q,k,o].x>0:
                        sum=sum+1
                        
                if (sum>1):
                    print("probabilistic decision")
                    print(i,j,q,o)




y_=np.ones((24,11,9,11))*10
# store the best descion

for i in  range(24):
    for j in range (11):
        for q in range (9):
            for o in range(11):
            
                for k in range (21):
                    if y[i,j,q,k,o].x!=0:
                        y_[i][j][q][o]=k
            
# y_[8][5][0]=5           




# Testing

Cost_year=0
Cost_year_before=0
demnad_after=np.zeros(6120)
soc_vector=np.zeros(6120)
Cb=500
Es=Cb
soc_old=1
ii=-1
j=10 # initial battery status

o=0 # initial peak status
for jj in range(255):    # one year
    o=0 # initial peak status        # rest every days
    grid_energy_vector_day=[]
    grid_energy_vector_day_after=[]
    for i in range (24):
        ii=ii+1
        
        
        
        # demnad quantile

        point=school_test_work[jj*24+i]          # every step in the year
        
        grid_energy_vector_day=np.append(grid_energy_vector_day,point)
        
        qunatiles_at_point=demand_quantile[:,i]  # qunatiles at that time step of the day
        # Find the quantile crossponding to the point
        x=point>qunatiles_at_point
        yy=np.where(x==True)
        yyy=np.where(x==False)
        if len(yyy[0])==9:
            yy=9
        else: 
            yy=yy[0][0]
            if yy==0:
                yy=1
        quant=1-0.1*yy
        
        quant=np.round(quant,1)
        
        quant_=np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        q=np.where(quant_==quant)
        q=q[0][0]
    
        # battery decision
        
        
        
        # probabilistic decision at state [8][5][0]
        # y_[8][5][0]=5 
        if (i==5 and j==2 and q==0):
            # y[8,5,0,5]=0.002687467800245069
            # y[8,5,0,6]=0.0006865982270690567
            numberList = [8, 10]
            alpha=random.choices(numberList, weights=(45.2, 54.8), k=1)
            
            
            
        
            
    
        alpha=y_[i][j][q][o]
        alpha=int(alpha)
        alpha=Alpha[alpha]
        alpha=np.round(alpha,1)
        
        grid_energy_after=max(point+max(alpha*eta,alpha/eta)*Cb,0)
        

        grid_energy_vector_day_after=np.append(grid_energy_vector_day_after,grid_energy_after)
        
        if i==23:
            cost=0.05 
            
            current_cost_before=cost*point+max(grid_energy_vector_day)*7
            
            current_cost=cost*grid_energy_after+max(grid_energy_vector_day_after)*7
            
        else:
            cost=0.05  
            current_cost_before=cost*point
            
            current_cost=cost*(grid_energy_after)
                    
        
     
        
        
        
        # if max(point+max(alpha*eta,alpha/eta)*Cb,0)>=200:  # max(point+max(alpha*eta,alpha/eta)*Cb,0)   is the demand
        #     cost=0.12
        # else:
        #     cost=0.05
            
        # current_cost=cost*(point+alpha*Cb)
        
        
        demnad_after[ii]=point+max(alpha*eta,alpha/eta)*Cb
        
        Cost_year_before=Cost_year_before+current_cost_before
        Cost_year=Cost_year+current_cost
    
        # Update Battery status
        soc=np.round(soc_old+alpha,1)
        
        if soc<0:
            soc=0
            #alpha=soc_old
        if soc>1:
            soc=1
            #alpha=1-soc_old
        
        soc_=np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        soc_vector[ii]=soc
        j=np.where(soc_==soc)
        
        j=j[0][0]
        soc_old=soc
        
        
        # updating the peak value
        Peak_current=max(Peak[o],demnad_after[ii])
        
        if Peak_current>=500:
            o=10
        elif Peak_current>=450:
            o=9
        elif Peak_current>=400:
            o=8                             
        elif Peak_current>=350:
            o=7

        if Peak_current>=300:
            o=6
        elif Peak_current>=250:
            o=5
        elif Peak_current>=200:
            o=4                             
        elif Peak_current>=150:
            o=3
            
        elif Peak_current>=100:
            o=2
        elif Peak_current>=50:
            o=1     
        else:
            o=0
        
        
        # print(j)
    
    
print('Cost_year_before = $ ', int(Cost_year_before))
print('Cost_year_after = $ ', int(Cost_year))    
print('Saving = $' , int(Cost_year_before-Cost_year))    
    





# Plot results for one day
jj=45
fig,ax = plt.subplots()
ax.plot(school_test_work[24*jj:24*jj+24],color="green",label='demand before')

ax.plot(demnad_after[24*jj:24*jj+24], color="black", label='demand after')
ax.set_ylabel("demand [kW]",
              color="black",
              fontsize=14)

ax.set_xlabel("Time [hrs]", fontsize = 14)
major_ticks = np.arange(0, 23, 2)
ax.set_xticks(major_ticks)
ax.grid()

plt.legend() 
ax2=ax.twinx()
# ax2.plot(cost,"b--")

ax2.set_ylabel("Price [$/kWh]",color="blue",fontsize=14)
# plt.plot(Cb*soc_vector[24*jj:24*jj+24])
plt.legend()
plt.grid()
plt.show() 


# saving=np.sum(cost*school_test_work[24*jj:24*jj+24])-np.sum(cost*demnad_after[24*jj:24*jj+24])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.plot(Cb*soc_vector[24*jj:24*jj+24],'r-')

major_ticks = np.arange(0, 23, 2)
ax.set_xticks(major_ticks)
ax.set_xlabel("Time [hrs]", fontsize = 14)
ax.set_ylabel("Es [kWh]",
              color="black",
              fontsize=14)
ax.grid()








# Ideal Case


# penaly on charging/discharging and  no refund for negative Eg
start = timeit.default_timer()

Cb=500
Es0=Cb

# Schduling the battery over 255 days
start = timeit.default_timer()
m=gp.Model()

u=m.addVars(24*255, vtype='C',lb=-1, ub=1, name='y')
Es=m.addVars(24*255, vtype='C',lb=0, ub=Cb, name='y')
Eg=m.addVars(24*255, vtype='C',lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y')
Eg_pos=m.addVars(24*255, vtype='C',lb=0, ub=GRB.INFINITY, name='y')

Eb=m.addVars(24*255, vtype='C',lb=-Cb, ub=Cb, name='y')

Peaks=m.addVars(255, vtype='C',lb=0, ub=1000, name='y')

aux=m.addVars(24*255, vtype='C',lb=-Cb, ub=Cb, name='y')
aux2=m.addVars(24*255, vtype='C',lb=-Cb, ub=Cb, name='y')

obj=gp.quicksum(cost*Eg_pos[i] for i in  range(24*255) )+gp.quicksum(Peaks[i] for i in range(255))*7
m.setObjective(obj) # the defult is to MINIMIZE

# for j in range (255):
#     m.addConstr(Peaks[i]==gp.max_(Eg[24*j+i] for i in  range(24)))
    
    
m.addConstrs(Peaks[j]==gp.max_(Eg[24*j+i] for i in  range(24)) for j in range(255) )



m.addConstrs(aux[i]==u[i]*eta for i in  range(24*255))
m.addConstrs(aux2[i]==u[i]/eta for i in  range(24*255))


m.addConstrs((Eg_pos[i]==gp.max_(Eg[i], constant=0)) for i in  range(24*255))

m.addConstrs(Eb[i]==gp.max_(aux[i], aux2[i]) for i in  range(24*255))

m.addConstrs(Eg[i]==school_test_work[i]+Eb[i]*Cb for i in  range(24*255) ) # 

m.addConstr(Es[0]==Es0+Cb*u[0] )

m.addConstrs(Es[i]==Es[i-1]+Cb*u[i] for i in  range(1, 24*255)  )

m.optimize()

stop = timeit.default_timer()
print('Time in mins: ', (stop - start)/60) 











# Plot results for one day (ideal)
jj=45  # Day index out of 255 work days in this year

fig,ax = plt.subplots()
ax.plot(school_test_work[24*jj:24*jj+24],color="green",label='demand before') # drawstyle='steps-pre'


Egg=[Eg[i].x for i in range(6120)]
ax.plot(Egg[24*jj:24*jj+24], color="black", label='demand after') # , drawstyle='steps-pre'
ax.set_ylabel("demand [kW]",
              color="black",
              fontsize=14)

ax.set_xlabel("Time [hrs]", fontsize = 14)
major_ticks = np.arange(0, 23, 2)
ax.set_xticks(major_ticks)
ax.grid()

plt.legend() 
# ax2=ax.twinx()
# ax2.plot(cost,"b--", drawstyle='steps-pre')

# ax2.set_ylabel("Price [$/kWh]",color="blue",fontsize=14)
# plt.plot(Cb*soc_vector[24*jj:24*jj+24])
# plt.legend()
# plt.grid()
plt.show() 



# saving=np.sum(cost*school_test_work[24*jj:24*jj+24])-np.sum(cost*np.maximum(demnad_after[24*jj:24*jj+24],0))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
uu=[u[i].x for i in range(6120)]
ax.plot(Cb*soc_vector[24*jj:24*jj+24],'r-') # , drawstyle='steps-pre'

major_ticks = np.arange(0, 23, 2)
ax.set_xticks(major_ticks)
ax.set_xlabel("Time [hrs]", fontsize = 14)
ax.set_ylabel("Es [kWh]",
              color="black",
              fontsize=14)
ax.grid()


# print('Saving for this day = $ ', int(saving))












# # Saving MDP policy

# import pickle

# output = open('data.pkl', 'wb')
# pickle.dump(y_, output)
# output.close()



# # Loading
# import pprint, pickle

# pkl_file = open('data.pkl', 'rb')

# y_ = pickle.load(pkl_file)
# pprint.pprint(y_)

# pkl_file.close()
