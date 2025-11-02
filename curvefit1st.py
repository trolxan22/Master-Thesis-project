import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy 

df05 = pd.read_csv('vpotsolochiigual05.csv')
df09 = pd.read_csv('vpotsolochiigual09.csv')
df15 = pd.read_csv('vpotsolochiigual15.csv')
df26 = pd.read_csv('vpotsolochiigual26.csv')
df4 = pd.read_csv('vpotsolochiigual4.csv')
df5 = pd.read_csv('vpotsolochiigual5.csv')
df8 = pd.read_csv('vpotsolochiigual8.csv')
df10=pd.read_csv('vpotsolochiigual10.csv')

dfok05 = df05[df05['xi2']/df05['xi1']>0.0]
dfok09 = df09[df09['xi2']/df09['xi1']>0.0] 
dfok15 = df15[df15['xi2']/df15['xi1']>0.0]
dfok26 = df26[df26['xi2']/df26['xi1']>0.0]
dfok4 = df4[df4['xi2']/df4['xi1']>0.0]
dfok5 = df5[df5['xi2']/df5['xi1']>0.0]
dfok8 = df8[df8['xi2']/df8['xi1']>0.0]
dfok10 = df10[df10['xi2']/df10['xi1']>0.0]
plt.scatter(dfok05['xi1'], dfok05['xi2'], label=r'$\chi_{12}=0.5$', s=1)
plt.scatter(dfok09['xi1'], dfok09['xi2'], label=r'$\chi_{12}=0.9$', s=1)
plt.scatter(dfok15['xi1'], dfok15['xi2'], label=r'$\chi_{12}=1.5$',s=1)
plt.scatter(dfok26['xi1'], dfok26['xi2'], label=r'$\chi_{12}=2.6$', s=1)
plt.scatter(dfok4['xi1'], dfok4['xi2'], label=r'$\chi_{12}=4.0$', s=1)
plt.scatter(dfok5['xi1'], dfok5['xi2'], label=r'$\chi_{12}=5.0$', s=1)
plt.scatter(dfok8['xi1'], dfok8['xi2'], label=r'$\chi_{12}=8.0$', s=1)
plt.scatter(dfok10['xi1'], dfok10['xi2'], label=r'$\chi_{12}=10.0$', s=1)

plt.xlim(0,7)
plt.ylim(0,7)
plt.legend()
plt.show()



tol=1e-4

chi=0.5
df_0p5 = df05[np.abs(df05['chi12'] -chi)/chi <= tol]


chi=0.9
df_0p9 = df09[np.abs(df09['chi12'] -chi)/chi <= tol]

chi=1.5
df_1p5 = df15[np.abs(df15['chi12'] -chi)/chi <= tol]

chi=2.6
df_2p6 = df26[np.abs(df26['chi12'] -chi)/chi <= tol]

chi=4.0
df_40 = df4[np.abs(df4['chi12'] -chi)/chi <= tol]

chi=5.0
df_50 = df5[np.abs(df5['chi12'] -chi)/chi <= tol]

chi=8.0
df_80 = df8[np.abs(df8['chi12'] -chi)/chi <= tol]

chi=10.0
df_10 = df10[np.abs(df10['chi12'] -chi)/chi <= tol]


df0510 = pd.concat([df_0p5, df_1p5, df_2p6, df_40, df_50, df_80, df_10])

@np.vectorize
def xi1extrem(chi12):
    if(chi12<4): 
        xi1xtrem=chi12
    else:
        xi1xtrem=(2*np.sqrt(2)*np.sqrt(chi12-2))
    return xi1xtrem
@np.vectorize
def xi2extrem(chi12):
    xi2xtrem=(2*np.sqrt(2)*(chi12)**0.5)
    return xi2xtrem

df0510['xi1NORM']=df0510['xi1']/xi1extrem(df0510['chi12'])
df0510['xi2NORM']=df0510['xi2']/xi2extrem(df0510['chi12'])
df=df0510[(df0510['xi1']<=xi1extrem(df0510['chi12'])) & (df0510['xi2']<=xi2extrem(df0510['chi12']))]

chi = 0.5
tol = 1e-3

dft = df[np.abs(df['chi12']-chi)/chi <= tol]
plt.scatter(dft['xi1NORM'], dft['xi2NORM'], label=r'$\chi_{12}$=0.5', s=1)
#plt.plot(dft['xi1NORM'], dft['xi2NORM'], label=r'$\chi_{12}$=0.5')

chi = 0.9
dft = df[np.abs(df['chi12']-chi)/chi <= tol]
plt.scatter(dft['xi1NORM'], dft['xi2NORM'], label=r'$\chi_{12}$=0.9', s=1)
#plt.plot(xix2, xiy2, label=r'$\chi_{12}$=0.9')

chi = 1.5
dft = df[np.abs(df['chi12']-chi)/chi <= tol]
plt.scatter(dft['xi1NORM'], dft['xi2NORM'], label=r'$\chi_{12}=1.5$', s=1)

chi=2.6
dft = df[np.abs(df['chi12']-chi)/chi <= tol]
plt.scatter(dft['xi1NORM'], dft['xi2NORM'], label=r'$\chi_{12}=2.6$', s=1)


chi = 4.0
dft = df[np.abs(df['chi12']-chi)/chi <= tol]
plt.scatter(dft['xi1NORM'], dft['xi2NORM'], label=r'$\chi_{12}=4.0$', s=1)

chi = 5.0
dft = df[np.abs(df['chi12']-chi)/chi <= tol]
plt.scatter(dft['xi1NORM'], dft['xi2NORM'], label=r'$\chi_{12}=5.0$', s=1)

chi = 8.0
dft = df[np.abs(df['chi12']-chi)/chi <= tol]
plt.scatter(dft['xi1NORM'], dft['xi2NORM'], label=r'$\chi_{12}=8.0$', s=1)

chi = 10.0
dft = df[np.abs(df['chi12']-chi)/chi <= tol]
plt.scatter(dft['xi1NORM'], dft['xi2NORM'], label=r'$\chi_{12}=10.0$', s=1)


u = np.linspace(0, 1, 360)
plt.plot(u, np.sqrt(-u+1), label='raiz de ajuste')
plt.plot(u, -u+1, label='recta de ajuste')
#####AJUSTE###########
fsqrt = np.sqrt(-u+1)
fRECT = -u+1
cte = 0.5  #SI ES CERO, TOTALMENTE RECTA, SI ES UNO TOTALMENTE CURVA
fAJUS = cte*fsqrt + (1-cte)*fRECT
plt.plot(u, fAJUS, label='curva ajustada') 

##########################
plt.xlabel(r'$u$')
plt.ylabel(r'$f(u)$')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(loc="lower left", prop={'size':6})
plt.show()



chi = 0.5
tol = 1e-3

dft = df[np.abs(df['chi12']-chi)/chi <= tol]
plt.scatter(dft['xi1NORM'], dft['xi2NORM']-(1-dft['xi1NORM']), label=r'$\chi_{12}$=0.5', s=0.3)
#plt.plot(dft['xi1NORM'], dft['xi2NORM'], label=r'$\chi_{12}$=0.5')

chi = 0.9
dft = df[np.abs(df['chi12']-chi)/chi <= tol]
plt.scatter(dft['xi1NORM'], dft['xi2NORM']-(1-dft['xi1NORM']), label=r'$\chi_{12}$=0.9', s=0.3)
#plt.plot(xix2, xiy2, label=r'$\chi_{12}$=0.9')

chi = 1.5
dft = df[np.abs(df['chi12']-chi)/chi <= tol]
plt.scatter(dft['xi1NORM'], dft['xi2NORM']-(1-dft['xi1NORM']), label=r'$\chi_{12}=1.5$', s=0.3)

chi=2.6
dft = df[np.abs(df['chi12']-chi)/chi <= tol]
plt.scatter(dft['xi1NORM'], dft['xi2NORM']-(1-dft['xi1NORM']), label=r'$\chi_{12}=2.6$', s=0.3)


chi = 4.0
dft = df[np.abs(df['chi12']-chi)/chi <= tol]
plt.scatter(dft['xi1NORM'], dft['xi2NORM']-(1-dft['xi1NORM']), label=r'$\chi_{12}=4.0$', s=0.3)

chi = 5.0
dft = df[np.abs(df['chi12']-chi)/chi <= tol]
plt.scatter(dft['xi1NORM'], dft['xi2NORM']-(1-dft['xi1NORM']), label=r'$\chi_{12}=5.0$', s=0.3)

chi = 8.0
dft = df[np.abs(df['chi12']-chi)/chi <= tol]
#lt.scatter(dft['xi1NORM'], dft['xi2NORM']-(1-dft['xi1NORM']), label=r'$\chi_{12}=8.0$', s=0.3)

chi = 10.0
dft = df[np.abs(df['chi12']-chi)/chi <= tol]
#lt.scatter(dft['xi1NORM'], dft['xi2NORM']-(1-dft['xi1NORM']), label=r'$\chi_{12}=10.0$', s=0.3)


u = np.linspace(0, 1, 360)
#plt.plot(u, np.sqrt(-u+1), label='raiz de ajuste')
#plt.plot(u, -u+1, label='recta de ajuste')
#####AJUSTE###########
fsqrt = np.sqrt(-u+1)
fRECT = -u+1
cte = 0.5  #SI ES CERO, TOTALMENTE RECTA, SI ES UNO TOTALMENTE CURVA
fAJUS = cte*fsqrt + (1-cte)*fRECT
#plt.plot(u, fAJUS, label='curva ajustada') 

##########################
plt.xlabel(r'$u$')
plt.ylabel(r'$f(u)$')
plt.xlim(0, 1)
plt.ylim(-0.1, 0.5)
#lt.plot([0,1],[0,0])
#plt.legend(loc="lower left", prop={'size':6})
plt.show()


######AJUSTE4##########    
u = np.linspace(0,1,360)


@np.vectorize
def fitcurve(u, a1, a2, a3, a4, a5, a6, a7, a8, a9):
    an=a1+a2+a3+a4+a5+a6 +a7 + a8 + a9
    out = a1*u + a2*u**2 + a3*u**3 + a4*u**4 + a5*u**5 + a6*u**6 + a7*u**7 + a8*u**8 + a9*u**9 -an*u**10
    return out
tol=1e-3
chi=4
dft = df[np.abs(df['chi12']-chi)/chi <= tol]
x_data=dft['xi1NORM']
y_data=dft['xi2NORM']-(1-dft['xi1NORM'])

p0 = [1,0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.3, 0.2]
popt, pcov = scipy.optimize.curve_fit(fitcurve, x_data, y_data, p0=p0)
sigmaerr=np.log10(np.sqrt(np.mean((y_data-fitcurve(x_data, *popt))**2)))

a1_ajustado = popt[0]

a2_ajustado = popt[1]



print(f"Alpha ajustado: {alpha_ajustado}")
print(f"gamma0 ajustado:{gamma0_ajustado}")
print(f"error cuadratico:{sigmaerr}")

plt.plot(x_data, y_data, label='xis normalizados')
plt.plot(u, fitcurve(u, *popt), label=f'Ajuste con alpha = {alpha_ajustado:.2f}, Ajuste con gamma0 = {gamma0_ajustado:.2f}', color='red')

plt.legend(loc="lower left", prop={'size':7})
plt.xlim(0, 1)
plt.ylim(-0.2, 0.2)
plt.show()

plt.plot(x_data, np.log10(np.sqrt((y_data-fitcurve(x_data, *popt))**2)))
plt.show()


z = np.polyfit(x_data, y_data, 9)
fz=np.poly1d(z)
plt.plot(x_data, y_data, label='xis normalizados')

plt.plot(x_data, fz(x_data))

sigmaerr=np.log10(np.sqrt(np.mean((y_data-fz(x_data))**2)))
plt.show()
print(z)
print(sigmaerr)
np.sum(z)
