import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

def lagrange_polynomial(points):
    # x を [-1..1] の 0.01 刻みの配列として初期化。これに対応する配列 y を求めていく
    x = np.arange(0, 731, 0.1)
    # y を x と同じ大きさのゼロの配列として初期化しておく
    y = np.zeros(len(x))
    # ラグランジュ補間式に従って x から y を計算する
    # なお、標本点の x_n は points[n][0], y_n は points[n][1] としてとれる
    # 
    # 
    # （自分で書く）
    #make l
    
    lBunshi = np.ones((len(x),len(points)))
    lBunbo  = np.ones(len(points))

    for i in range(0,len(x)):
        for j in range(0, len(points)):
            for k in range(0,len(points)):
                if(j != k):
                    lBunshi[i][j] *= (x[i] - points[k][0])
    
    for i in range(0,len(points)):
        for j in range(0,len(points)):
            if(i != j):
                lBunbo[i] *= (points[i][0] - points[j][0])


                
    
    for i in range(0,len(y)):
        for j in range(0,len(points)):
            y[i] += (lBunshi[i][j]*points[j][1]/lBunbo[j])
    print()
    # 

    return(x, y)



def spline(points):
    # 標本点 points を x 座標の順でソートしておく
    points.sort()
    # 標本点の数 len(points) に対して、求める区間数 n をセット
    n = len(points) - 1
    if (n <= 0):
        return 0, 0
    # (4n, 4n) の係数行列を確保。デフォルト値を 0 で埋める
    A = np.zeros((4 * n, 4 * n))
    b = np.zeros(4 * n)
    # スプライン補間の条件に従い、標本点リストから計算して上記の A, b に値を埋めていく。
    #
    #
    # （自分で書く）

    A[2][0] = 6*points[0][0]
    A[2][1] = 2
    A[3][4*n - 4] = 6*points[n][0]
    A[3][4*n - 3] = 2
    for i in range (0,n):
        A[4*i][4*i]  = points[i][0]**3
        A[4*i][4*i + 1] = points[i][0]**2
        A[4*i][4*i + 2] = points[i][0]
        A[4*i][4*i + 3] = 1

        A[4*i + 1][4*i] = points[i+1][0]**3
        A[4*i + 1][4*i + 1] = points[i+1][0]**2
        A[4*i + 1][4*i + 2] = points[i+1][0]
        A[4*i + 1][4*i + 3] = 1

        if(i > 0):
            A[4*i + 2][4*i - 4] = 3*points[i][0]**2
            A[4*i + 2][4*i - 3] = 2*points[i][0]
            A[4*i + 2][4*i - 2] = 1

            A[4*i + 2][4*i] = -3*points[i][0]**2
            A[4*i + 2][4*i + 1] = -2*points[i][0]
            A[4*i + 2][4*i + 2] = -1

            A[4*i + 3][4*i - 4] = 6*points[i][0]
            A[4*i + 3][4*i - 3] = 2

            A[4*i + 3][4*i] = -6*points[i][0]
            A[4*i + 3][4*i + 1] = -2




        b[4*i] = points[i][1]
        b[4*i + 1] = points[i+1][1]
    #
    #

    # 上記の連立一次方程式を解く。結果は (a0, b0, c0, d0, ..., a_n-1, b_n-1, c_n-1, d_n-1) の長さ 4n の配列
    #
    sol = np.linalg.solve(A, b)
    x = np.arange(points[0][0], points[1][0], 0.01)
    y = sol[0]*(x**3) + sol[1]*(x**2) + sol[2]*x + sol[3]
    for i in range(1,n):
        xi = np.arange(points[i][0], points[i+1][0], 0.01)
        yi = sol[4*i]*(xi**3) + sol[4*i + 1]*(xi**2) + sol[4*i + 2]*(xi) + sol[4*i + 3]
        x = np.concatenate((x, xi))
        y = np.concatenate((y, yi))

    return(x,y)


file = "bikesharing.csv"
deli =  ","
rawData = np.loadtxt(file, delimiter =deli, skiprows=1)
N = np.shape(rawData)[0]
d = np.shape(rawData)[1]

rtemp = rawData[:,8]
rhum = rawData[:,10]
ratemp = rawData[:,9]
rwind = rawData[:,11]

avetemp = np.sum(rtemp)/N
avehum = np.sum(rhum)/N
aveatemp = np.sum(ratemp)/N
avewind = np.sum(rwind)/N

stdtemp = np.std(rtemp)
stdhum = np.std(rhum)
stdatemp = np.std(ratemp)
stdwind = np.std(rwind)

Temp = (rtemp - avetemp)/ stdtemp
Hum = (rhum - avehum) / stdhum
Atemp = (ratemp - aveatemp) / stdatemp
Wind = (rwind - avewind) / stdwind

d = 4
S = np.zeros((d,d))
Data = np.zeros((N,4))

for i in range(0,N):
    Data[i] = [Temp[i], Hum[i], Atemp[i], Wind[i]]

mu = np.array([avetemp, avehum, aveatemp, avewind])



S = np.cov(Data, rowvar = False, ddof = 1)

eigenvalues, eigenvectors = la.eig(S)

# 固有値を降順にソート
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]


#print("固有値：",eigenvalues)
#print("固有ベクトル：\n",eigenvectors)



#主成分を求める
projectedData = (Data)@eigenvectors
sumOfEVals = np.sum(eigenvalues)

I = np.zeros(N)
for i in range(0,N):
    I[i] = i

split5 = np.zeros((5,d))
for i in range(0,5):
    for j in range(0,d):
        split5[i,j] = projectedData[int((i/4)*(N-1)),j]

split11 = np.zeros((11,d))
for i in range(0,11):
    for j in range(0,d):
        split11[i,j] = projectedData[int((i/10)*(N-1)),j]

lag0 = np.zeros((5,2))
for i in range(0,5):
    lag0[i,0] = int((i/4)*(N-1))
    lag0[i,1] = split5[i,0]

lag1 = np.zeros((11,2))
for i in range(0,11):
    lag1[i,0] = int((i/10)*(N-1))
    lag1[i,1] = split11[i,0]

x0, y0 = lagrange_polynomial(lag0)
print(np.shape(split5))
plt.plot(x0,y0,color = "red", label = "lagrange")
plt.scatter(I,projectedData[:,0], s = 10, color = "blue", label = "actual points")
plt.scatter(lag0[:,0], lag0[:,1], color = "red", label = "sampling points")
plt.title("5-point Lagrange")
plt.legend()
plt.savefig("3_lagrange5p")
plt.show()


x1, y1 = lagrange_polynomial(lag1)
print(np.shape(split5))
plt.plot(x1,y1,color = "red", label = "lagrange")
plt.scatter(I,projectedData[:,0], s = 10, color = "blue", label = "actual points")
plt.scatter(lag1[:,0], lag1[:,1], color = "red", label = "sampling points")
plt.title("11-point Lagrange")
plt.legend()
plt.savefig("3_lagrange11p")
plt.show()


