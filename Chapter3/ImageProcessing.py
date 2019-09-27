import numpy as np 
import pandas as pd 
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
#读取文件数据
df = pd.read_csv('30SaverageSpeed12.csv',names=['t','g'])
 
fig=plt.figure(figsize=(19.2,10.8)) #关键步骤，将plt.figure()放在for循环前面。
 
for i in range(150,2874,4):
	Y0 = df.loc[0:i,'t']  #选取指定的数据
	X0 = df.loc[0:i,'g'] #选取指定的数据
 
	x0=np.array(Y0)
	y0=np.array(X0)	
 
 
	
# statsmodels.api
	from scipy.signal import savgol_filter
	zs1=savgol_filter(y0, 101, 1) # window size 51, polynomial order 3
 
 
	plt.xlim((-10,86400)) #设置坐标轴范围
	plt.ylim((-10,160))
 
	plt.plot(x0,y0) #画原曲线
 
	plt.plot(x0,zs1,lw=3,label='A') #画拟合曲线图
 
	plt.legend(prop={'family' : 'Times New Roman', 'size'   : 18}) #设置图例文字大小
 
	plt.axhline(y=86,color='k')#画水平线
 
 
	plt.yticks(fontproperties = 'Times New Roman', size = 18) #设置坐标轴刻度文字的尺寸
	plt.xticks(fontproperties = 'Times New Roman', size = 18)
 
	plt.title('30S', fontdict={'family' : 'Times New Roman', 'size'   : 18}) #图的标题字体格式
 
#设置坐标轴标体文字大小
	font2 = {'family':'Time New Roman','weight' : 'normal','size'   : 18,}
	plt.xlabel('time(s)',font2)
	plt.ylabel('speed(km/h)',font2)
 
##设置坐标轴粗细
	ax=plt.gca()
	ax.spines['bottom'].set_linewidth(2)
	ax.spines['left'].set_linewidth(2)
	ax.spines['right'].set_linewidth(2)
	ax.spines['top'].set_linewidth(2)
 
 
	fig.savefig('30S'+str(i)+'.png',dpi=100,pad_inches=0) #保存图片
	#plt.show()
	plt.clf()
