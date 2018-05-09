from django.shortcuts import render
from django.views.decorators import csrf
import MySQLdb
import re
import numpy
import math
import scipy.stats as st
from numpy import *
def query_data(request):
    ctx ={}
    if request.POST:
        q = request.POST['q']
        if q == 'anonymizer personal_information':
            from secureserver import anonymizer
            data, i = anonymizer.read_adult()
            ctx['rlt'] = anonymizer.get_result_one(data,INTUITIVE_ORDER=i)
            #print type(res),type(res[0])
            #for i in range(len(res)):
            #    res[i] = ','.join(res[i])
            #ctx['rlt'] = '\n'.join(res)
            return render(request, "query.html", ctx)
        ratioList = request.POST.getlist("selected_method")
        if not ratioList or len(ratioList) == 0:
            checkedRatio = 0
        else:
            checkedRatio = int(ratioList[0])
        db = MySQLdb.connect("localhost","root","root","secureserver" )
        cursor = db.cursor()
        cursor.execute(q)
        columns = cursor.description
        ctx['rlt'] = cursor.fetchall()
        if checkedRatio == 3: #number 3 means DF
            pass
            '''
            #q = str(q).strip().split(' ')
            eps = float(request.POST['eps'])
            #q = ' '.join(q[2:])
            ctx['rlt'] = list(ctx['rlt'])
            maxVal = max(ctx['rlt'])
            minVal = min(ctx['rlt'])
            from numpy.random import laplace
            for i in range(len(ctx['rlt'])):
                ctx['rlt'][i] = list(ctx['rlt'][i])
                ctx['rlt'][i][0] += laplace(0, float(maxVal[0]-minVal[0])/(eps/5))
            return render(request, "query.html", ctx)
            '''
            pass
        elif checkedRatio == 2: #number 2 means Base64
            #q = q.lstrip('base64')
            ctx['rlt'] = list(ctx['rlt'])
            import base64
            for i in range(len(ctx['rlt'])):
                ctx['rlt'][i] = list(ctx['rlt'][i])
                for j in range(len(ctx['rlt'][i])):
                    ctx['rlt'][i][j] = base64.encodestring(str(ctx['rlt'][i][j]))
            return render(request, "query.html", ctx)
        elif checkedRatio == 1: #number 1 means AES
            from Crypto.Cipher import AES
            from binascii import b2a_hex, a2b_hex
            #q = q.lstrip('AES ')
            obj = AES.new('This is a key123', AES.MODE_CBC, 'This is an IV456')
            ctx['rlt'] = list(ctx['rlt'])
            for i in range(len(ctx['rlt'])):
                ctx['rlt'][i] = list(ctx['rlt'][i])
                for j in range(len(ctx['rlt'][i])):
                    valStr = str(ctx['rlt'][i][j])
                    count = len(valStr)
                    add = 16 - (count % 16)
                    valStr = valStr + (' ' * add)
                    ctx['rlt'][i][j] = b2a_hex(obj.encrypt(valStr))
            return render(request, "query.html", ctx)
        db = MySQLdb.connect("localhost","root","root","secureserver" )
        cursor = db.cursor()
        cursor.execute(q)
        ctx['rlt'] = cursor.fetchall()

        targetRatio = 0
        targetList = request.POST.getlist("selected_results")
        if not targetList or len(targetList) == 0:
            targetRatio = 0
        else:
            targetRatio = int(targetList[0])
        print targetRatio
        if checkedRatio != 2 and checkedRatio != 1:
            if targetRatio == 1:
                if checkedRatio == 3:
                    eps = float(request.POST['eps'])
                    ctx['fingerprint'] = fingerprint(ctx['rlt'], True, eps)
                else:
                    ctx['fingerprint'] = fingerprint(ctx['rlt'], False, 0)
            elif targetRatio == 2:
                if checkedRatio == 4:
                    radius = float(request.POST['radius'])
                    ctx['clusters'] = clustering(ctx['rlt'], columns, True, radius)
                else:
                    ctx['clusters'] = clustering(ctx['rlt'], columns, False, 0)
                #for k in ctx['clusters']:
                #    print ctx['clusters'][k]
                #print ctx['clusters']
            elif targetRatio == 3:
                ctx['corr'] = correlation(ctx['rlt'], columns)
        ctx['rlt'] = ctx['rlt'][:5000]
    return render(request, "query.html", ctx)


def correlation(raw_data, columns):
    import pandas as pd, numpy as np, scipy
    result = [{columns[index][0]:column for index, column in enumerate(value)} for value in raw_data]
    df = pd.DataFrame(result)
    res = []
    print df.corr()
    
    '''
    for col1 in df.columns:
        tmp = []
        for col2 in df.columns:
            #tmp.append(np.corrcoef(df[col1], df[col2])[0, 1])
            tmp.append(scipy.stats.pearsonr(df[col1], df[col2]))
        res.append(tmp)
    print res
    '''
    return np.array(df.corr())

def fingerprint(raw_data, isDp, eps):
    data = {}
    for row in raw_data:
        deviceId = str(row[0])
        #high support for annbor
        #if deviceId not in ['90','10129','10131','10133','10134','10137','10154','10155','10158','10160','10177']:
        #    continue
        #high support for annbor 
        #if deviceId not in ['90','10129','10131','10133','10134','10137','10154','10155','10158','10160','10177']:
        #    continue
        #high support for vertical(road3)
        #if deviceId not in ['501','10131','10138','10140','10150','10164','10176','10574','10587','10612','10616','15101','17101','17102','17103']:
        #    continue
        #high support for straight (road4)
        if deviceId not in ['10146','13103','10610','10607','10587','10188','10146','502','10137','10161','10602','10605']:
            continue
        #high support for plymonth(road 2)
        #if deviceId not in ['10141','10207','10594','13103','13107','13108']:
        #    continue
        tripId = row[1]
        if deviceId not in data:
            data[deviceId] = {}
        if tripId not in data[deviceId]:
            data[deviceId][tripId]=[[],[],[],[],[]]
        data[deviceId][tripId][0].append(float(row[3]))  # 60
        data[deviceId][tripId][1].append(float(row[8]))  #72
        data[deviceId][tripId][2].append(float(row[20]))  # 39
        data[deviceId][tripId][3].append(float(row[21]))  #38
        data[deviceId][tripId][4].append(float(row[22])) #63
    '''
    #if isDp:
    #    from numpy.random import laplace
    #    for _id in data:
    #        for trip in data[_id]:
    #            for i in range(len(data[_id][trip])):
    #                maxVal = max(data[_id][trip][i])
    #                minVal = min(data[_id][trip][i])
    #                for j in range(len(data[_id][trip][i])):
    #                    data[_id][trip][i][j] += laplace(0, float(maxVal-minVal)/(eps/5))
    '''
    lookup={}
    '''######################
    	#index:
    	#0: Device id
    	#1: Trip id
    	#2: Time
    	#3: AccelPadal # 60 #32
    	#7: Ax #43
    	#8: Ay #72
    	#9: BounderyLeft #29
    	#10: BounderyRight  #41
    	#11: Brake #29
    	#20: Speed # 39
    	#21: TurnSignal # 38
    	#22: YawRate # 63
    	#####################
    '''
    dataX = []
    dataY = []
    for deviceId in data:
        for tripId in data[deviceId]:
            dataY.append(deviceId)
            tmpArr = []
            dataArr = data[deviceId][tripId]
            for i in range(len(dataArr)):
                narray=numpy.array(dataArr[i])
                N=len(dataArr[i])
                sum1=narray.sum()
                narray2=narray*narray
                sum2=narray2.sum()
                mean=sum1/N
                tmpArr.append(mean)
                tmpArr.append(math.sqrt(abs(sum2/N)))
                tmpArr.append(min(dataArr[i]))
                tmpArr.append(max(dataArr[i]))
                tmpArr.append(st.skew(narray))
                tmpArr.append(st.kurtosis(narray))
            dataX.append(tmpArr)
    from sklearn.metrics import classification_report
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cross_validation import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.25, random_state=11)
    lenOfRow = 0
    if x_test:
        lenOfRow = len(x_test[0])
    else:
        return [['Drive id','precision','recall','f1-score','support']]
    if isDp:
        from numpy.random import laplace
        print eps, type(eps)
        x_test = mat(x_test)
        for col in range(lenOfRow):
            maxVal = float(max(x_test[:,col]))
            minVal = float(min(x_test[:,col]))
            print maxVal, minVal,  col
            for row in range(len(x_test[:,col])):
                if maxVal - minVal < 0:
                    print maxVal,minVal, col
                    continue
                x_test[row,col] += laplace(0, float(maxVal - minVal)/eps)

    clf = RandomForestClassifier(n_estimators=200,criterion = 'entropy')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))
    res = []
    for row in classification_report(y_test, y_pred).split('\n'):
        tmp = []
        if 'precision' in row:
            tmp.append('Driver id')
            for col in row.split():
                if col:
                    tmp.append(col)
        elif 'total' in row:
            tmp.append('avg/total')
            for col in row.split()[3:]:
                if col:
                    tmp.append(col)
        else:
            for col in row.split():
                if col:
                    tmp.append(col)
        if tmp:
            res.append(tmp)
    return res 

def haversine(lon1, lat1, lon2, lat2):
    from math import radians, cos, sin, asin, sqrt
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])    
    dlon = lon2 - lon1   
    dlat = lat2 - lat1   
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2  
    c = 2 * asin(sqrt(a))   
    r = 6371
    return c * r * 1000  

def clustering(raw_data, columns, isObfuscated, radius):
    import pandas as pd, numpy as np, matplotlib.pyplot as plt, time, math, random
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn import metrics
    from geopy.distance import great_circle
    from shapely.geometry import MultiPoint
    kms_per_radian = 6371.0088
    epsilon = 0.1 / kms_per_radian
    result = [{columns[index][0]:column for index, column in enumerate(value)} for value in raw_data]
    df = pd.DataFrame(result)
    df1 = df[['Device']].drop_duplicates()
    print df1
    maxDistance = radius
    minDistance = radius
    res = {}
    for i, row in df1.iterrows():
        coords = df.loc[df['Device'] == row[0]].as_matrix(columns=['Latitude','Longitude']).astype(np.float)
        if isObfuscated:
            for i in range(len(coords)):
                angle = float(random.randint(0, 359))
                distance = float(random.randint(minDistance, maxDistance))
                coords[i][1] += float(distance * math.cos(3.141592654 * angle / 180)) / (1000 * kms_per_radian)
                coords[i][0] += float(distance * math.sin(3.141592654 * angle / 180)) / (1000 * kms_per_radian)
        db = KMeans(n_clusters=3, random_state=0).fit(np.radians(coords))
        #db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
        cluster_labels = db.labels_
        num_clusters = len(set(cluster_labels))
        clusters = pd.Series([coords[cluster_labels==n] for n in range(num_clusters)])
        def get_centermost_point(cluster):
            centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
            centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
            return tuple(centermost_point)
        centermost_points = clusters.map(get_centermost_point)
        res[row[0]] = []
        for point in centermost_points:
            print point[1], point[0]
            res[row[0]].append({'lat': point[0], 'lon': point[1], 'dis': haversine(point[1], point[0], -83.638979, 42.3239)})
    return res
 

def query_locations(request):
    ctx ={}
    if request.POST:
        q = request.POST['q']
        import pandas as pd, numpy as np, matplotlib.pyplot as plt, time
        from sklearn.cluster import DBSCAN
        from sklearn import metrics
        from geopy.distance import great_circle
        from shapely.geometry import MultiPoint
        kms_per_radian = 6371.0088
        #pro_info = pd.DataFrame(cursor.fetchall())
        db = MySQLdb.connect("localhost","root","root","secureserver")
        cursor = db.cursor()
        cursor.execute(q)
        columns = cursor.description
        result = [{columns[index][0]:column for index, column in enumerate(value)} for value in cursor.fetchall()]
        #ctx['rlt'] = [result[0].keys()]
        #return render(request, "query_locations.html", ctx)
        df = pd.DataFrame(result)
        coords = df.ix[:,[15,16]]
        coords = df.as_matrix(columns=['Latitude','Longitude']).astype(np.float)
        #ctx['rlt'] = coords #[result[0].keys()]
        #return render(request, "query_locations.html", ctx)
        epsilon = 0.1 / kms_per_radian
        db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
	cluster_labels = db.labels_
        num_clusters = len(set(cluster_labels))
        clusters = pd.Series([coords[cluster_labels==n] for n in range(num_clusters)])
        def get_centermost_point(cluster):
            centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
            centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
            return tuple(centermost_point)
        centermost_points = clusters.map(get_centermost_point)
        ctx['rlt'] = centermost_points
    return render(request, "query_locations.html", ctx)

def query_fingerprint(request):
    ctx ={}
    if request.POST:
        q = request.POST['q']
        db = MySQLdb.connect("localhost","root","root","secureserver")
        cursor = db.cursor()
        cursor.execute(q)
        #columns = cursor.description
        #result = [{columns[index][0]:column for index, column in enumerate(value)} for value in cursor.fetchall()]
        data = {}
        raw_data = cursor.fetchall()
        #print raw_data
        for row in raw_data:
            deviceId = str(row[0]) #re.sub("\D", "", row[0].strip())
            #high support for annbor
            if deviceId not in ['90','10129','10131','10133','10134','10137','10154','10155','10158','10160','10177']:
                continue
            #high support for vertical
            #if deviceId not in ['90','501','10131','10138','10140','10150','10164','10176','10574','10587','10612','10616','15101','17101','17102','17103']:
            #    continue
            #high support for straight
            #if deviceId not in ['10146','13103','10610','10607','10587','10188','10146','502','10137','10161','10602','10605']:
            #    continue
            #high support for plymonth
            #if deviceId not in ['10141','10207','10594','13103','13107','13108']:
            #    continue
            #high support for oneTry
            #if deviceId not in ['10141','10163','10188','10207','10567','10570','10594']:
            #    continue
            #high support for oneTry1
            #if deviceId not in ['10141','10163','10188','10207','10515','10567','10570','10594','10612','13101','13103','13105','13107','13108','13109']:
            #    continue
            #high support for oneTry0-7200
            #if deviceId not in ['10141','10163','10188','10207','10515','10567','10570','10594','10612','13101','13103','13105','13107','13108','13109']:
            #    continue

            tripId = row[1]
            if deviceId not in data:
                data[deviceId] = {}

            if tripId not in data[deviceId]:
                data[deviceId][tripId]=[[],[],[],[],[]]
            data[deviceId][tripId][0].append(float(row[3]))  # 60
            data[deviceId][tripId][1].append(float(row[8]))  #72
            data[deviceId][tripId][2].append(float(row[20]))  # 39
            data[deviceId][tripId][3].append(float(row[21]))  #38
            data[deviceId][tripId][4].append(float(row[22])) #63
            '''
            if not data[deviceId].has_key(tripId):
                data[deviceId][tripId]=[[]]
            data[deviceId][tripId][0].append(float(row[22]))  # 60
            '''
            #data[deviceId][tripId][0].append(float(row[11].strip())) # 29

    	#print data.keys()
    	lookup={}
    	######################
    	#index:
    	#0: Device id
    	#1: Trip id
    	#2: Time
    	#3: AccelPadal # 60 #32
    	#7: Ax #43
    	#8: Ay #72
    	#9: BounderyLeft #29
    	#10: BounderyRight  #41
    	#11: Brake #29
    	#20: Speed # 39
    	#21: TurnSignal # 38
    	#22: YawRate # 63
    	######################
    	dataX = []
    	dataY = []
    	#print type(data['10177'][0])
    	for deviceId in data:
       	    #if len(data[deviceId]) < 10:
            #	continue
            for tripId in data[deviceId]:
            	dataY.append(deviceId)
            	tmpArr = []
            	dataArr = data[deviceId][tripId]
            	for i in range(len(dataArr)):
                    narray=numpy.array(dataArr[i])
                    N=len(dataArr[i])
                    sum1=narray.sum()
                    narray2=narray*narray
                    sum2=narray2.sum()
                    mean=sum1/N
                    tmpArr.append(mean)
                    tmpArr.append(math.sqrt(abs(sum2/N)))
                    tmpArr.append(min(dataArr[i]))
                    tmpArr.append(max(dataArr[i]))
                    tmpArr.append(st.skew(narray))
                    tmpArr.append(st.kurtosis(narray))
                dataX.append(tmpArr)

	from sklearn.metrics import classification_report
    	from sklearn.ensemble import RandomForestClassifier
    	from sklearn.cross_validation import train_test_split
    	x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.75, random_state=11)
    	clf = RandomForestClassifier(n_estimators=200,criterion = 'entropy')
    	clf.fit(x_train, y_train)
    	y_pred = clf.predict(x_test)
    	print(classification_report(y_test, y_pred))

        ctx['rlt'] = classification_report(y_test, y_pred)
    return render(request, "query_fingerprint.html", ctx)
