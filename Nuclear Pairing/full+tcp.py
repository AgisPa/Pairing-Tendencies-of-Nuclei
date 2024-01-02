import scipy
from scipy.misc import derivative
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import pandas as pd
from scipy.optimize import minimize
#Iterations for each nucleus. Corresponds to accuracy of Cv graph. Minimum of 5, suggested number ~50.
N =50
#Number of nulcei simulated. Up to 194.
numberelements= 19
data = pd.read_csv('20.csv')
pnzdata = pd.read_csv('pnz-a.csv')
e0find = pd.read_csv('e0.csv')
element = 0
elementindex = 0
maxindexe0 = pd.DataFrame.last_valid_index(e0find)
if maxindexe0<numberelements+1:
    print('not enough elements')
maxindexe0 = numberelements+1
maxindex = pd.DataFrame.last_valid_index(data)
massnumber = np.zeros(numberelements)
minfermi = np.zeros(maxindexe0)
for index in range(0, maxindexe0):
    br8 = int(e0find.iloc[index, 1])
    minfermi[index] = data.iloc[br8, 3]
kappa1 = np.zeros(numberelements)
neutrons = np.zeros(numberelements)
protons = np.zeros(numberelements)
tcn = np.zeros(numberelements)
pairing = np.zeros(maxindexe0)
delta = np.zeros(numberelements)
gap = np.zeros(numberelements)
Tfac = np.zeros(numberelements)
ptype=np.zeros(0)
name=np.zeros(0)
paircontour=[[0 for x in range(0,150)] for y in range(0,150)]
Tcontour = [[0 for x in range(0,150)] for y in range(0,150)]
PPcontour = [[0 for x in range(0,150)] for y in range(0,150)]
NNcontour = [[0 for x in range(0,150)] for y in range(0,150)]
allamodel = [[0 for x in range(0,N)] for y in range(0,maxindexe0)]
allbmodel = [[0 for x in range(0,N)] for y in range(0,maxindexe0)]
allgammamodel = [[0 for x in range(0,N)] for y in range(0,maxindexe0)]
allitamodel = [[0 for x in range(0,N)] for y in range(0,maxindexe0)]
alldeltamodel = [[0 for x in range(0,N)] for y in range(0,maxindexe0)]
while element<(numberelements):
    for index in range(elementindex, maxindex):
        trialne = False
        trialpr = False
        expenergy = np.zeros(maxindex)
        def boundpart(b):
            for index in range(elementindex,maxindex):
                expenergy[index] = ((2 * float(data.iloc[index, 5])) + 1) * math.exp(-b * float(data.iloc[index, 3]) / 1000)
                if index!=elementindex and float(data.iloc[index, 3])==0:
                    break
            return sum(expenergy)
        for index2 in range(0, 80):
            if int(data.iloc[index, 0]) == int(pnzdata.iloc[index2,0]):
                protons[element] = int(data.iloc[index, 0])
                neutrons[element] = int(data.iloc[index, 1])
                if protons[element]!=protons[element-1] and int(data.iloc[index, 3]) == 0:
                    trialpr = True
            if int(data.iloc[index, 1]) == int(pnzdata.iloc[index2,0]):
                neutrons[element] = int(data.iloc[index, 1])
                protons[element] = int(data.iloc[index, 0])
                if neutrons[element]!=neutrons[element-1] and int(data.iloc[index, 3]) == 0:
                    trialne = True
            if trialne or trialpr:
                break
        if trialne or trialpr:
            break
    elementindex = index
    temperature = np.zeros(N)
    energy = np.zeros(N)
    cv = np.zeros(N)
    e0 = 8*minfermi[element+1]/10000
    emax = 3000
    z3=np.zeros(N)
    z2 = np.zeros(N)
    z1 = np.zeros(N)
    derz1 = np.zeros(N)
    derz2 = np.zeros(N)
    derz = np.zeros(N)
    derz3 = np.zeros(N)
    z = np.zeros(N)
    neutron = int(neutrons[element])
    proton = int(protons[element])
    a2 = float(pnzdata.iloc[proton-9, 3])
    a1 = float(pnzdata.iloc[neutron-9, 4])
    pnz = pnzdata.iloc[neutron-9,2]+pnzdata.iloc[proton-9,1]
    massnumber[element] = proton + neutron
    ux = 2.5 + 150 / massnumber[element]
    if neutron>126:
        if neutron>155:
            Nval= 184-neutron
        else:
            Nval=neutron-126
    elif neutron>82:
        if neutron>104:
            Nval=126-neutron
        else:
            Nval=neutron-82
    elif neutron>50:
        if neutron>66:
            Nval=82-neutron
        else:
            Nval=neutron-50
    elif neutron>28:
        if neutron>39:
            Nval=50-neutron
        else:
            Nval=neutron-28
    elif neutron>20:
        if neutron>24:
            Nval=28-neutron
        else:
            Nval=neutron-20
    elif neutron>8:
        if neutron>14:
            Nval=20-neutron
        else:
            Nval=neutron-8
    if proton > 126:
        if proton > 155:
            Pval = 184 - proton
        else:
            Pval = proton - 126
    elif proton > 82:
        if proton > 104:
            Pval = 126 - proton
        else:
            Pval = proton - 82
    elif proton > 50:
        if proton > 66:
            Pval = 82 - proton
        else:
            Pval = proton - 50
    elif proton > 28:
        if proton > 39:
            Pval = 50 - proton
        else:
            Pval = proton - 28
    elif proton > 20:
        if proton > 24:
            Pval = 28 - proton
        else:
            Pval = proton - 20
    elif proton > 8:
        if proton > 14:
            Pval = 20 - proton
        else:
            Pval = proton - 8
    deltaon=0
    deltaop=0
    if neutron %2==0:
        deltaon = 41/massnumber[element]+0.94
    else:
        deltaon = 24/massnumber[element]+0.82
    if proton % 2==0:
        deltaop=59/massnumber[element]+1.11
    else:
        deltaop=25/massnumber[element]+0.75
    delta[element] = 31 / massnumber[element]
    gap[element] = (7.2 - 44 * pow((neutron - proton) / massnumber[element], 2)) / pow(massnumber[element], 1 / 3)
    if Nval!=0 and Pval!=0:
        Pfac = (Nval*Pval)/(Nval+Pval)
        Tfac[element]= (Nval*Pval)*delta[element]/((Nval+Pval)*gap[element])
    else:
        Pfac=0
    Tcontour[proton][neutron]=Tfac[element]
    pairing[element]=pnz
    if pnz>delta[element]:
        if Tfac[element]>0.85 and Nval>0 and Pval>0:
            Tcontour[proton][neutron]=1.1
            pairing[element]=pnz-delta[element]
            pnz=pnz-delta[element]
        else:
            Tcontour[proton][neutron]=0
    elif Tfac[element]>0.9:
        Tcontour[proton][neutron] = 1.1
        a1=a1-delta[element]
    else:
        Tcontour[proton][neutron]=0
    if deltaon<deltaop:
        if pairing[element]>0.9*deltaon and Nval>1:
            NNcontour[proton][neutron]=1.1
            pairing[element]=pairing[element]-deltaon
            if pairing[element] >0.9* deltaop and Pval>1:
                PPcontour[proton][neutron]=1.1
                pairing[element]=pairing[element]-deltaop
            else:
                PPcontour[proton][neutron]=0
        elif pairing[element] >0.9* deltaop and Pval>1 and Nval==0:
            PPcontour[proton][neutron] = 1.1
            pairing[element] = pairing[element] - deltaop
        else:
            NNcontour[proton][neutron]=0
            PPcontour[proton][neutron] = 0
    if deltaon>deltaop:
        if pairing[element]>0.9*deltaop and Pval>1:
            PPcontour[proton][neutron]=1.1
            pairing[element]=pairing[element]-deltaop
            if pairing[element]>0.9*deltaon and Nval>1:
                NNcontour[proton][neutron]=1.1
                pairing[element] = pairing[element] - deltaon
            else:
                NNcontour[proton][neutron]=0
        elif pairing[element]>0.9*deltaon and Nval>1 and Pval==0:
            NNcontour[proton][neutron] = 1.1
            pairing[element] = pairing[element] - deltaon
        else:
            NNcontour[proton][neutron]=0
            PPcontour[proton][neutron] = 0
    if Tcontour[proton][neutron]>1 or NNcontour[proton][neutron]>1 or PPcontour[proton][neutron]>1:
        paircontour[proton][neutron]=1.2
    else:
        paircontour[proton][neutron]=0
    print('Element: ', proton, '-', neutron, 'has P-N Pairing Factor:', round(Tfac[element], 2), 'P-N barrier:',
              round(delta[element], 2), 'Neutron Gap:', round(deltaon, 2), 'Valence N:', Nval, 'Proton Gap:',
              round(deltaop, 2), 'Valence P:', Pval, 'Available Energy:', round(pnz, 2),'with pairing:'," N-N:", NNcontour[proton][neutron],'-'," P-P:", PPcontour[proton][neutron],' P-N:', Tcontour[proton][neutron])
    if ((neutron in range(16, 32) or neutron in range(46, 54) or neutron in range(78, 86) or neutron in range(180,
                                                                                                              188)) and (
            proton in range(16, 32) or proton in range(46, 54) or proton in range(78, 86))):
        a = massnumber[element] * (0.00917 * (a1 + a2) + 0.142)
    elif ((neutron in range(16, 32) or neutron in range(46, 54) or neutron in range(78, 86) or neutron in range(180,
                                                                                                                188)) or (
                  proton in range(16, 32) or proton in range(46, 54) or proton in range(78, 86))):
        a = massnumber[element] * (0.00917 * (a1 + a2) + 0.131)
    else:
        a = massnumber[element] * (0.00917 * (a1 + a2) + 0.12)
    belln = np.zeros(N)
    cv1 = np.zeros(N)
    energy1 = np.zeros(N)
    energymodel = np.zeros(N)
    amodel = np.zeros(N)
    amodel[0] = -16.11
    bmodel = np.zeros(N)
    bmodel[0] = 20.21
    gammamodel = np.zeros(N)
    gammamodel[0] = 20.65
    itamodel = np.zeros(N)
    itamodel[0] = 48
    deltamodel = np.zeros(N)
    deltamodel[0] = 33
    allamodel[element][0] = -16.11
    allbmodel[element][0] = 20.21
    allgammamodel[element][0] = 20.65
    allitamodel[element][0] = 48
    alldeltamodel[element][0] = 33
    error = np.zeros(N)
    if neutron % 2 == 0 and proton % 2 == 0:
        faz = -1
    elif neutron % 2 != 0 and proton % 2 != 0:
        faz = 1
    else:
        faz = 0
    def radius(x):
        return 1.07 * (1 + 0.01 * x)
    def model(x, t):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        return x1 * massnumber[element] + x2 * pow(massnumber[element], 2 / 3) + (
                    x3 - x4 / pow(massnumber[element], 1 / 3)) * ((pow(proton - neutron, 2)) + 2 * (proton - neutron)) / \
               massnumber[element] + (pow(proton, 2)) * (
                           1 - 0.7636 / pow(proton, 2 / 3) - 2.99 * pow(radius(0), 2) / pow(
                       radius(t) * pow(massnumber[element], 1 / 3), 2)) / (
                           radius(t) * pow(massnumber[element], 1 / 3)) + x5 * faz / pow(massnumber[element], 3 / 4)
    initial = np.array([-16.11, 20.21, 20.65, 48, 33])
    energymodel[0] = model(initial, 0)
    temperature[0] = 0
    pnz=pairing[element]
    for j in range(1, N):
        bita = N / (5 * j)
        temperature[j] = 1 / bita
        z1[j] = boundpart(bita)
        z[j] = z[j] + z1[j]
        derz1[j] = scipy.misc.derivative(boundpart, bita, dx=1e-10)
        derz[j] = derz[j] + derz1[j]
        def exinte(E):
            return math.sqrt(np.pi) * math.exp(-bita * (E )) * math.exp(2 * math.sqrt(a * (E - pnz))) / (12 * pow(a, 1 / 4) * pow(E - pnz, 5 / 4))
        if e0 < ux and proton > 30 and j>10:
            e00 = ux + pnz - temperature[j] * math.log(temperature[j] * exinte(ux))
            def exinte2(E):
                return math.exp((E - e00) * bita) * bita
            if round(exinte(ux), 0) != round(exinte2(ux + pnz), 0):
                z3[j] = scipy.integrate.quad(exinte2, 0, ux + pnz)[0]
                def exintedb2(E):
                    return math.exp((E - e0) * bita) + (E - e0) * bita * math.exp((E - e0) * bita)
                derz3[j] = scipy.integrate.quad(exintedb2, 0, ux + pnz)[0]
                e0 == ux
                z[j] = z[j] + z3[j]
                derz[j] = derz[j] + derz3[j]
        z2[j] = scipy.integrate.quad(exinte, e0, emax)[0]
        z[j] = z[j] + z2[j]
        def exintedb(E):
            return -(E) * math.sqrt(np.pi) * math.exp(-bita * (E )) * math.exp(2 * math.sqrt(a * (E - pnz))) / (12 * pow(a, 1 / 4) * pow(E - pnz, 5 / 4))
        derz2[j] = scipy.integrate.quad(exintedb, e0, emax)[0]
        derz[j] = derz[j] + derz2[j]
        energy1[j] = -(derz1[j]) / (z1[j])
        cv1[j] = (-12 * energy1[j - 5] + 75 * energy1[j - 4] - 200 * energy1[j - 3] + 300 * energy1[j - 2] - 300 *energy1[j - 1] + 137 * energy1[j + 0]) / (60 * 1.0 * (3 / N) ** 1)
        energy[j] = -(derz[j]) / (z[j])
        energymodel[j] = energy[j] + model(initial, 0)
        def res(x):
            return pow(energymodel[j]- model(x, temperature[j]),2)
        def constrain1(x):
            x5 = x[4]
            return x5
        con1 = {'type': 'ineq', 'fun': constrain1}
        con = [con1]
        ba = (-17,0)
        bb = (19, 25)
        bgamma = (-20,35)
        bita = (-100, 70)
        bdelta = (0,45)
        bnds = (ba,bb,bgamma,bita,bdelta)
        sol = minimize(res, initial, method='SLSQP',bounds=bnds, constraints=con, tol= 1e-10)
        error[j] = sol.fun
        amodel[j] = sol.x[0]
        allamodel[element][j] = amodel[j] / maxindexe0 + allamodel[element-1][j]
        bmodel[j] = sol.x[1]
        allbmodel[element][j] = bmodel[j] / maxindexe0 + allbmodel[element - 1][j]
        gammamodel[j] = sol.x[2]
        allgammamodel[element][j] = gammamodel[j] / maxindexe0 + allgammamodel[element - 1][j]
        itamodel[j] = sol.x[3]
        allitamodel[element][j] = itamodel[j] / maxindexe0 + allitamodel[element - 1][j]
        deltamodel[j] = sol.x[4]
        alldeltamodel[element][j] = deltamodel[j] / maxindexe0 + alldeltamodel[element - 1][j]
        initial = [sol.x[0], sol.x[1], sol.x[2], sol.x[3], sol.x[4]]
        pf = 2*math.pi * pow( 3 * massnumber[element] / (8 * math.pi), 1/3)
        R = 1.25 * pow(massnumber[element], 1/3)
        L= 2* R
        S = 4 * math.pi * R**2
        V = 4 * math.pi * R**3 /3
        cv[j] = (3/4- (S*math.pi/(4*V*pf)) + (L*math.pi/(16*V*pf**2))) * (-12 * energy[j - 5] + 75 * energy[j - 4] - 200 * energy[j - 3] + 300 * energy[j - 2] - 300 * energy[j - 1] + 137 * energy[j + 0]) / (60 * 1.0 * (3 / N) ** 1)
    deg = cv[N - 1] / temperature[N - 1]
    def fit2(x):
        return deg * x
    for j in range(0, N):
        belln[j] = cv[j] - deg * temperature[j]
    maxbelln = np.amax(belln)
    for j in range(0, N):
        if maxbelln == belln[j]:
            br = j
    k1 = cv[br] / temperature[br]
    def fit1(x):
        return k1 * x
    kappa1[element] = k1-deg
    tcn[element] = temperature[br]
    if massnumber[element]==22:
        plt.subplot(2,2,1)
        plt.ylabel("Specific Heat",size= 22)
        plt.xlim(0,5)
        plt.ylim(0,np.amax(cv))
        plt.title("Na: Z=11, N=11",size= 28)
        plt.plot(temperature, fit1(temperature),'k--')
        plt.plot(temperature, cv)
    if massnumber[element]==48:
        plt.subplot(2, 2, 2)
        plt.xlim(0, 5)
        plt.ylim(0, np.amax(cv))
        plt.title("Ti: Z=22, N=26",size= 28)
        plt.plot(temperature, fit1(temperature),'k--')
        plt.plot(temperature, cv)
    if massnumber[element]==90:
        plt.subplot(2, 2, 3)
        plt.xlim(0, 5)
        plt.ylim(0, np.amax(cv))
        plt.title("Zr: Z=40, N=50",size= 28)
        plt.xlabel("Temperature", size=22)
        plt.ylabel("Specific Heat", size=22)
        plt.plot(temperature, fit1(temperature),'k--')
        plt.plot(temperature, cv)
    if massnumber[element]==140:
        plt.subplot(2, 2, 4)
        plt.xlim(0, 5)
        plt.ylim(0, np.amax(cv))
        plt.title("Ce: Z=58, N=82",size= 28)
        plt.xlabel("Temperature", size=22)
        plt.plot(temperature, fit1(temperature),'k--')
        plt.plot(temperature, cv)
    #print("The critical temperature for the element ", data.iloc[elementindex, 2], proton, neutron, "with P-factor",          Pfac, " is equal to",
    #      round(tcn[element], 2), "MeV")
    name=np.append(name,data.iloc[elementindex, 2])
    if Tcontour[proton][neutron]>1 and NNcontour[proton][neutron]>1:
        ptype=np.append(ptype,'P-N and N-N')
    elif Tcontour[proton][neutron]>1:
        ptype=np.append(ptype,'P-N')
    elif Tcontour[proton][neutron]>1 and PPcontour[proton][neutron]>1:
        ptype=np.append(ptype,'P-N and P-P')
    elif NNcontour[proton][neutron]>1:
        ptype=np.append(ptype,'N-N')
    elif NNcontour[proton][neutron] > 1 and PPcontour[proton][neutron] > 1:
        ptype=np.append(ptype,'N-N and P-P')
    elif PPcontour[proton][neutron]>1:
        ptype = np.append(ptype, 'P-P')
    else:
        ptype=np.append(ptype,'None')
    element = element + 1
plt.show()
shell= [0]
for i in range(1,140):
    if i==2 or i==8 or i==20 or i==28 or i==50 or i==82 or i==126:
        shell= np.append(i,shell)
    else:
        shell=np.append(" ",shell)
shell= shell[::-1]
plt.grid(True,which='both', linestyle='--', linewidth=0.2)
plt.axes().set_aspect('auto')
#plt.xticks(np.linspace(0,128,128), labels=shell)
#plt.yticks(np.linspace(0,84,84), labels=shell)
plt.xlabel('Neutrons', size= 14)
plt.ylabel('Protons', size= 14)
fig = plt.contourf(Tcontour, colors=('k','w'), levels=[0,1,2],marker='k^', extent=(0, 128, 0, 84),vmin=1,vmax=2,antialiased=False)
plt.title('P-N Interaction')
cbar = plt.colorbar(fig)
cbar.set_label("P-N pairs",size=14)
plt.xlim(0,128)
plt.ylim(0,84)
plt.show()
plt.grid(True,which='both', linestyle='--', linewidth=0.2)
plt.axes().set_aspect('auto')
#plt.xticks(np.linspace(0,128,128), labels=shell)
#plt.yticks(np.linspace(0,84,84), labels=shell)
plt.xlabel('Neutrons', size= 14)
plt.ylabel('Protons', size= 14)
fig = plt.contourf(NNcontour, colors=('k','w'), levels=[0,1,2],marker='k^', extent=(0, 128, 0, 84),vmin=1,vmax=2,antialiased=False)
plt.title('N-N Interaction')
cbar = plt.colorbar(fig)
cbar.set_label("NN pairs",size=14)
plt.xlim(0,128)
plt.ylim(0,84)
plt.show()
plt.grid(True,which='both', linestyle='--', linewidth=0.2)
plt.axes().set_aspect('auto')
#plt.xticks(np.linspace(0,128,128), labels=shell)
#plt.yticks(np.linspace(0,84,84), labels=shell)
plt.xlabel('Neutrons', size= 14)
plt.ylabel('Protons', size= 14)
fig = plt.contourf(PPcontour, colors=('k','w'), levels=[0,1,2],marker='k^', extent=(0, 128, 0, 84),vmin=1,vmax=2,antialiased=False)
plt.title('P-P Interaction')
cbar = plt.colorbar(fig)
cbar.set_label("PP pairs",size=14)
plt.xlim(0,128)
plt.ylim(0,84)
plt.show()
plt.grid(True,which='both', linestyle='--', linewidth=0.2)
plt.axes().set_aspect('auto')
#plt.xticks(np.linspace(0,128,128), labels=shell)
#plt.yticks(np.linspace(0,84,84), labels=shell)
plt.xlabel('Neutrons', size= 14)
plt.ylabel('Protons', size= 14)
fig = plt.contourf(paircontour, colors=('k','w'), levels=[0,1,2],marker='k^', extent=(0, 128, 0, 84),vmin=1,vmax=2,antialiased=False)
plt.title('Pair Interaction')
cbar = plt.colorbar(fig)
cbar.set_label("Pairs",size=14)
plt.xlim(0,128)
plt.ylim(0,84)
plt.show()
plt.imsave('name.png', Tcontour)
