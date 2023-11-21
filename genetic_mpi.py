#%%
# -*- coding: utf-8 -*-
from mpi4py import MPI
import math
import random
from itertools import permutations


nb_sol = 50
generations = 80
mutation_rate = 5
nb_cls = 4

class Point:
    def __init__(self,x):
        self.data = x
        
    def distance(self,p):
        d = 0.
        for i in range(len(p.data)):
            delta = self.data[i]-p.data[i]
            d += delta*delta
        d = math.sqrt(d)
        return d

class Solution:
    def __init__(self,n):
        self.data = []
        self.sumdist = 0.
        self.nb_class = n

    def quality(self):
        self.sumdist = 0.
        for i in range(len(self.data)):
            for j in range(i+1,len(self.data)):
                if(self.data[i] == self.data[j]):
                    self.sumdist += points[i].distance(points[j])            
        return self.sumdist

    def random_solution(self,ncls,nb_points):
        self.data.clear()
        for i in range(nb_points):
            self.data.append(random.randint(0,ncls-1))

    def set_data(self,data):
        self.data = data

    def mutation(self,rate): #revenir pour voir si l'alea serait mieux
        self.quality()
        val = self.sumdist
        for iter in range(rate):
            i = random.choice(range(len(self.data)))
            for k in range(self.nb_class):
                bck = self.data[i]
                self.data[i] = k
                self.quality()
                if(self.sumdist < val):
                    val = self.sumdist
                else:
                   self.data[i] = bck

    def crossover(self,data2,p):
        besti = 0 
        bestv = 0 
        for i in range(len(p)):  
            data = []
            val = 0
            for j in range(len(data2)):
                data.append(p[i][data2[j]])
            for j in range(len(data2)):
                if(self.data[j] == data[j]):
                    val += 1
            if(val > bestv):
                bestv=val
                besti=i
            data.clear()
        # print("bestv = ",bestv)
        for j in range(len(data2)):
            if(random.choice([0,1]) == 0):
                self.data[j] = p[besti][data2[j]]
                
    def valid(self):
        for s in range(len(self.data)):
            if(self.data[s] > self.nb_class):
                print(s," data = ",self.data[s])
                return False
        return True
    def display(self):
        print(self.data)

class Population:
    def __init__(self):
        self.sols = []
        
    def addSolution(self,s):
        self.sols.append(s)
    
    def getData(self,i):
        if(i>= 0 and i<len(self.sols)):
            return self.sols[i].data
        else: 
            return None

    def setData(self,i,d):
        if(i>= 0 and i<len(self.sols)):
            self.sols[i].data = d
            
    def sort(self):
        for s in self.sols:
            s.quality()
        cont = True
        while cont:
            cont = False
            for i in range(len(self.sols)-1):
                if(self.sols[i].sumdist > self.sols[i+1].sumdist):
                    self.sols[i],self.sols[i+1] = self.sols[i+1],self.sols[i]
                    cont = True

    def randomGoodData(self):
        pos = random.randint(0,len(self.sols)//2)
        return self.sols[pos].data

    def randomBadData(self):
        c = len(self.sols)//2
        pos = random.randint(0,c)+c
        if(pos >= len(self.sols)): 
            pos=c
        return self.sols[pos].data

    def replaceRandomBadData(self,d):
        c = len(self.sols)//2
        pos = random.randint(0,c)+c
        if(pos >= len(self.sols)): 
            pos=c
        self.sols[pos].data = d
        self.sols[pos].quality()
        
    def bestQuality(self):
        return self.sols[0].sumdist

    def bestData(self):
        return self.sols[0].data

    def valid(self):
        v = True
        for s in range(len(self.sols)):
            if(not self.sols[s].valid()):
                print(s," sol not valid")
                v = False
        return v

    def display(self):        
        for s in range(len(self.sols)):
            print(s," sol : ")
            self.sols[s].display()

perm_cls = list(permutations(list(range(nb_cls))))

points = []
points.append(Point([5.,3.]))
points.append(Point([2.,4.]))
points.append(Point([0.,1.]))
points.append(Point([8.,6.]))
points.append(Point([7.,3.]))
points.append(Point([2.,8.]))
points.append(Point([1.,6.]))
points.append(Point([2.,12.]))
points.append(Point([4.,10.]))
points.append(Point([5.,13.]))
points.append(Point([7.,2.]))
points.append(Point([15.,3.]))
points.append(Point([18.,20.]))
points.append(Point([20.,16.]))
points.append(Point([13.,6.]))
points.append(Point([9.,13.]))
points.append(Point([4.,9.]))
points.append(Point([1.,4.]))
points.append(Point([0.,7.]))
points.append(Point([6.,3.]))
nb_points = len(points)

tmp_sol = Solution(nb_cls)

#%%
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if(rank == 0):
    population = Population()
    for i in range(nb_sol):
        s = Solution(nb_cls)
        s.random_solution(nb_cls,nb_points)
        s.quality()
        population.addSolution(s)
    population.sort()

    print("Initial solution value : ",population.bestQuality())

    for pid in range(1,size):
        data1 = population.randomGoodData()
        data2 = population.randomBadData()
        comm.send(data1,dest=pid,tag=1)
        comm.send(data2,dest=pid,tag=2)
        
for iter in range(1,generations):
    if(rank == 0):
        for pid in range(1,size):
            data = comm.recv(source=pid,tag=1)
            population.replaceRandomBadData(data)
            population.sort()
            data1 = population.randomGoodData()
            data2 = population.randomBadData()
            comm.send(data1,dest=pid,tag=1)
            comm.send(data2,dest=pid,tag=2)
        print("It ",iter," solution value : ",population.bestQuality())
    else:
        data1 = comm.recv(source=0,tag=1)
        tmp_sol.set_data(data1)
        data2 = comm.recv(source=0,tag=2)
        tmp_sol.crossover(data2,perm_cls)
        tmp_sol.quality()
        # print('crossover {}'.format(rank),tmp_sol.sumdist)
        tmp_sol.mutation(mutation_rate)
        # print('mutation {}'.format(rank),tmp_sol.sumdist)
        comm.send(tmp_sol.data,dest=0,tag=1)

#finalisation
if(rank == 0):
    for pid in range(1,size):
        data = comm.recv(source=pid,tag=1)
        population.replaceRandomBadData(data)
        population.sort()
    print("Final solution value : ",population.bestQuality())
    print("Final solution : ",population.bestData())
else:
    data1 = comm.recv(source=0,tag=1)
    tmp_sol.set_data(data1)
    data2 = comm.recv(source=0,tag=2)
    tmp_sol.crossover(data2,perm_cls)
    tmp_sol.mutation(mutation_rate)
    comm.send(tmp_sol.data,dest=0,tag=1)

print("Complete ",rank)



# %%
