import math#imports
import random
import matplotlib.pyplot as plt
import statistics

def RandomMeanAndVarianceUniformDist(N=100,y=[0,10]):#N is length of list, y is min and max values of the list
    values = []
    for i in range(N):
        a = random.uniform(y[0],y[1])
        values.append(a)
    mean = sum(values)/N
    TheSum = []
    for i in range(N):
        TheSum.append((values[i]-mean)**2)
    variance = sum(TheSum)/(N-1)
    return [mean,variance]
    
#The mean being close to 5 every time is a good thing, variance is also good should be ((b-a)**2)/12

def RandomMeanAndVarianceNormalDist(N=100,mu=5,sigma=1):#N is length of list,mu is the mean, sigma is the standard deviation
    values = []
    for i in range(N):
        a = random.normalvariate(mu,sigma)
        values.append(a)
    mean = sum(values)/N
    TheSum = []
    for i in range(N):
        TheSum.append((values[i]-mean)**2)
    variance = sum(TheSum)/(N-1)
    return [mean,variance]#results make sense here, mean is close to mu and variance is close to sigma squared

def RandomMeanAndVarianceExponentialDist(N=100,lambd=0.2):#N is length of list, lambd is lambda (lambda is already used by python)
    values = []
    for i in range(N):
        a = random.expovariate(lambd)
        values.append(a)
    mean = sum(values)/N
    TheSum = []
    for i in range(N):
        TheSum.append((values[i]-mean)**2)
    variance = sum(TheSum)/(N-1)
    return [mean,variance]#both mean and variance look good here, should be 1/lambd for mean and 1/lambd**2 for variance, results show that


def MeanandVarianceCheckMean(function,N=100):
    values = []
    for i in range(N):
        values.append(function()[0])
    mean = sum(values)/N
    TheSum = []
    for i in range(N):
        TheSum.append((values[i]-mean)**2)
    variance = sum(TheSum)/(N-1)
    return[mean,variance]

def MeanandVarianceCheckVariance(function,N=100):
    values = []
    for i in range(N):
        values.append(function()[1])
    mean = sum(values)/N
    TheSum = []
    for i in range(N):
        TheSum.append((values[i]-mean)**2)
    variance = sum(TheSum)/(N-1)
    return[mean,variance]
    
def f(x):
    return x
def g(x):
    return 3*x**3
def example(x):
    return (math.cos(x)**2)
def test(x,y):
    return(3*(x**2)*y**3)
    
    
def SimpsonsOneThirdRule(a,b,function):
    answer = ((b-a)/6)*(function(a)+4*function((a+b)/2)+function(b))
    return answer
def SimpsonsCompositeRule(a,b,function,N):#N has to be even
    h = (b-a)/N
    TheSum1 = []
    for i in range(int((N/2)-1)):
        TheSum1.append(function(2*(a+(i+1)*h)))
    TheSum2 = []
    for i in range(int(N/2)):
        TheSum2.append(function(2*(a+(i+1)*h)-1))
    return (h/3)*(function(a)+4*sum(TheSum1)+2*sum(TheSum2)+function(b))

def TrapeziumRule2N(a,b,function):
    return (b-a)*(function(a)+function(b))/2
def TrapeziumRule(a,b,function,N):
    TheSum = []
    deltaxi = (b-a)/N
    for i in range(N):
        TheSum.append(((function(a+i*deltaxi)+function(a+(i+1)*deltaxi))/2)*deltaxi)
    return sum(TheSum)

def MonteCarloIntegration(a,b,function,N):
    values = []
    for i in range(N):
        values.append(random.uniform(a,b))
    functionvalues = []
    for i in range(N):
       functionvalues.append(function(values[i]))
    return ((b-a)/N)*sum(functionvalues)


def averager(a,b,function,N,method,n):
    values = []
    for i in range(n):
        values.append((1/2)*method(a,b,function,N))
    mean = sum(values)/n
    TheSum = []
    for i in range(n):
        TheSum.append((values[i]-mean)**2)
    variance = sum(TheSum)/(N-1)
    return[mean,variance]
        
def circle(x):
    if x[0]**2 + x[1]**2 < 1:
        return 1
    else:
        return 0
    
def sphere(x):
    if x[0]**2 + x[1]**2 + x[2]**2 < 1:
        return 1
    else:
        return 0
    
def foursphere(x):
    if x[0]**2 + x[1]**2 + x[2]**2 +x[3]**2 < 1:
        return 1
    else:
        return 0
    
def nsphere(x,n=5,r=1):
    a = 0
    for i in range(n):
        a += x[i]**2
    if a < r:
        return 1
    else:
        return 0
    
def nsphereanalytical(n,r=1):
    if n%2 == 0:
        return ((1/(math.factorial(n/2)))*(math.pi**(n/2))*(r**n))
    else:
        return ((2**n)*(math.factorial((n-1)/2)/math.factorial(n))*(math.pi**((n-1)/2))*(r**n))
    
def Rtoalpha(x,alpha=-1.75):
    r = (math.sqrt(x[0]**2+x[1]**2))
    if r < 1:
        return r**alpha
    else:
        return 0
    
    
def MultiVariableMonteCarlo(V,function,N,alpha=0):#V is a list of lists [[a,b],[b,c],etc] NV is the number of variables
    values = []
    for i in range(len(V)):
        values.append([])
        for j in range(N):
            values[i].append(random.uniform(V[i][0],V[i][1]))
    functionvalues = []
    for i in range(N):
        funcinput = []
        for j in range(len(V)):
            funcinput.append(values[j][i])
        functionvalues.append(function(funcinput))
    multiplier = 0
    for i in range(len(V)):
        multiplier = V[i][1]-V[i][0]
    multiplier = V[0][1]-V[0][0]
    for i in range(len(V)-1):
        multiplier = multiplier*(V[i+1][1]-V[i+1][0])
    multiplier = multiplier/N
    return multiplier*sum(functionvalues)


def MultiVariableMonteCarloSphere(N,NV):#V is a list of lists [[a,b],[b,c],etc] NV is the number of variables
    V = []
    for i in range(NV):
        V.append([-1,1])
    values = []
    for i in range(len(V)):
        values.append([])
        for j in range(N):
            values[i].append(random.uniform(V[i][0],V[i][1]))
    functionvalues = []
    for i in range(N):
        funcinput = []
        for j in range(len(V)):
            funcinput.append(values[j][i])
        functionvalues.append(nsphere(funcinput,NV))
    multiplier = 0
    for i in range(len(V)):
        multiplier = V[i][1]-V[i][0]
    multiplier = V[0][1]-V[0][0]
    for i in range(len(V)-1):
        multiplier = multiplier*(V[i+1][1]-V[i+1][0])
    multiplier = multiplier/N
    return multiplier*sum(functionvalues)

def plotter(N):
    allx = []
    for i in range(N):
        allx.append((-2/(N))*i)
    ally = []
    for i in range(N):
        ally.append(MultiVariableMonteCarlo([[-1,1],[-1,1]],Rtoalpha,1000,allx[i]))
    plt.plot(allx,ally)
    plt.xlabel('alpha')
    plt.ylabel('Result')
    
def errorcheck2(method,N):
    error = []
    amount = []
    for i in range(N):
        error.append(abs(method([[-1,1],[-1,1]],circle,i+1,2)-math.pi))
        amount.append(i)
    plt.plot(amount,error)
    plt.xlabel('N')
    plt.ylabel('Absoulte error')
        
def errorvariance2(method,N,M):
    error = []
    for i in range(M):
        error.append(method([[-1,1],[-1,1]],circle,N,2)-math.pi)
    mean = sum(error)/M
    TheSum = []
    for i in range(M):
        TheSum.append((error[i]-mean)**2)
    variance = sum(TheSum)/(M-1)
    median = statistics.median(error)
    return [mean,variance,median]