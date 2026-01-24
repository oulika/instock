import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pymysql

print(matplotlib.__version__)

connect = pymysql.connect(host='192.168.31.192', port=3306, user='root', passwd='wang521wei',database='stock_master'
                                                                                                      '')
cursor = connect.cursor()


cursor.execute('select mn_account_id from investment_portfolio where mn_account_id>40000 order by total_earnings desc limit 5')
alldata = cursor.fetchall()

for indexx,row in enumerate(alldata):
    mn_account_id = row[0]
    sql= f'select date ,total_yield from investment_portfolio_snapshot where mn_account_id={mn_account_id} and date>"2026-01-01"';
    cursor.execute(sql)
    allinfo=cursor.fetchall()

    for indexy,row1 in enumerate(allinfo):
        xarray = []
        yarray = []
        xarray.append(row1[0])
        yarray.append(row1[1])
        plt.plot(np.array(xarray), np.array(yarray), linestyle='-', marker='o', color='r',label=row[0])
        print(row1)
    #plt.subplot( 1, 1, 1)

plt.show()
