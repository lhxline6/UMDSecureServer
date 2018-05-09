from django.shortcuts import render
from django.views.decorators import csrf
import MySQLdb
import re
import numpy
import math
import scipy.stats as st
from numpy import *
import random
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def query_new_data(request):
    ctx ={}
    if request.POST:
        q = request.POST['q']
        ratioList = request.POST.getlist("selected_method")
        if not ratioList or len(ratioList) == 0:
            checkedRatio = 0
        else:
            checkedRatio = int(ratioList[0])
        db = MySQLdb.connect("localhost","root","root","secureserver" )
        cursor = db.cursor()
        if q.strip()[:35] == 'select * from personal_information1':
            print q
            
            if checkedRatio == 3:
            	from secureserver import anonymizer
                data, i = anonymizer.read_adult()
                ctx['rlt'] = anonymizer.get_result_one(data,INTUITIVE_ORDER=i)
            else:
            	print q
            	cursor.execute(q)
                ctx['rlt'] = cursor.fetchall()
                print ctx['rlt']
            return render(request, "query-new.html", ctx)

        cursor.execute(q)
        columns = cursor.description
        ctx['rlt'] = list(cursor.fetchall())


        if checkedRatio == 2: #number 2 means Base64
            ctx['rlt'] = list(ctx['rlt'])
            import base64
            for i in range(len(ctx['rlt'])):
                ctx['rlt'][i] = list(ctx['rlt'][i])
                for j in range(len(ctx['rlt'][i])):
                    ctx['rlt'][i][j] = base64.encodestring(str(ctx['rlt'][i][j]))
            return render(request, "query-new.html", ctx)
        elif checkedRatio == 1: #number 1 means AES
            from Crypto.Cipher import AES
            from binascii import b2a_hex, a2b_hex
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
            return render(request, "query-new.html", ctx)
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

        if checkedRatio != 2 and checkedRatio != 1:
            if targetRatio == 1:
                if checkedRatio == 3: #fingerprinting attacks with Laplace noise
                    eps = float(request.POST['eps'])
                    ctx['fingerprint'] = fingerprint(ctx['rlt'], True, eps)
                elif checkedRatio == 5: #fingerprinting attacks with Frequently-changing Pseudonym
                    ctx['fingerprint'] = fingerprint_pesudo(ctx['rlt'], False, 0)
                else: #fingerprinting attack
                    ctx['fingerprint'] = fingerprint(ctx['rlt'], False, 0)
            elif targetRatio == 2:
                if checkedRatio == 4: #location inference with daily data and context-aware policy
                    radius = 0#float(request.POST['radius'])
                    ctx['clusters'] = clustering_respectively_context(ctx['rlt'], columns, True, radius)
                elif checkedRatio == 6:  #location inference with daily data and two-end policy
                    print 'asec'
                    ctx['clusters'] = clustering_respectively_portions(ctx['rlt'], columns, True, 0)
                else:
                    ctx['clusters'] = clustering_respectively(ctx['rlt'], columns, False, 0)
            elif targetRatio == 3:
                ctx['corr'] = correlation(ctx['rlt'], columns)
        ctx['rlt'] = list(ctx['rlt'][:5000])
        columns_name = []
        for col in columns:
        	columns_name.append(col[0])
        ctx['rlt'].insert(0, tuple(columns_name))
    return render(request, "query-new.html", ctx)


def correlation(raw_data, columns):
    import pandas as pd, numpy as np, scipy
    result = [{columns[index][0]:column for index, column in enumerate(value)} for value in raw_data]
    df = pd.DataFrame(result)
    res = []
    print df.corr()
    
    return np.array(df.corr())

#function to perform fingerprinting attacks
#isDp control whether adding differential privacy
#eps is a parameter in Laplace noise
def fingerprint(raw_data, isDp, eps):
    data = {}
    for row in raw_data:
        deviceId = str(row[0]).strip()
        #high support for annbor 
        if deviceId not in ['10129','10131','10133','10134','10137','10154','10155','10158','10160','10177']:
            continue
        #high support for vertical(road3)
        #if deviceId not in ['501','10131','10138','10140','10150','10164','10176','10574','10587','10612','10616','15101','17101','17102','17103']:
        #    continue
        #high support for straight (road4)
        #if deviceId not in ['10146','13103','10610','10607','10587','10188','10146','502','10137','10161','10602','10605']:
        #    continue
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
        #data[deviceId][tripId][0].append(float(row[11])) #63

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
    from sklearn.model_selection import train_test_split

    #x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.25)
    
    x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.1)
    '''
    #############################
    #Following code is for grid search for parameters
    #############################
    rf = RandomForestClassifier()
    param_grid = {
    'bootstrap': [True],
    'max_depth': [50],
    #'min_samples_leaf': [1, 3, 5],
    #'min_samples_split': [8, 10, 12],
    'n_estimators': [500]
    }
    print 'start search'
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5)
    print 'finish search'
    grid_search.fit(x_train, y_train)
    print 'best_params_', grid_search.best_params_
    best_grid = grid_search.best_estimator_
    print 'best_grid', best_grid
    #grid_accuracy = evaluate(best_grid, x_test, y_test)
    y_pred = best_grid.predict(x_test)
    print 'grid_accuracy'
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    '''
    
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

    clf = RandomForestClassifier(n_estimators=500,criterion = 'gini', max_depth=50)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
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

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))

#fuction to add frequently-changing pesudo
def fingerprint_pesudo(raw_data, isDp, eps):
    data = {}
    for row in raw_data:
        deviceId = str(row[0])
        #high support for annbor
        #if deviceId not in ['90','10129','10131','10133','10134','10137','10154','10155','10158','10160','10177']:
        #    continue
        #high support for annbor 
        if deviceId not in ['90','10129','10131','10133','10134','10137','10154','10155','10158','10160','10177']:
            continue
        #high support for vertical(road3)
        #if deviceId not in ['501','10131','10138','10140','10150','10164','10176','10574','10587','10612','10616','15101','17101','17102','17103']:
        #    continue
        #high support for straight (road4)
        #if deviceId not in ['10146','13103','10610','10607','10587','10188','10146','502','10137','10161','10602','10605']:
        #    continue
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
    lookup={}
    dataX = []
    dataY = []
    numSeg = 10
    x_train, x_test, y_train, y_test = [], [], [], []
    counts = [0 for _ in range(len(data.keys()))]
    for di, deviceId in enumerate(data):
        #print len(data[deviceId])
        for tripId in data[deviceId]:
            if counts[di] <= 0.75 * len(data[deviceId]):
                counts[di] += 1
                y_train.append(deviceId)
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
                #print len(tmpArr)
                x_train.append(tmpArr)
            else:
                dataArr = data[deviceId][tripId]
                flag = False
                for item in dataArr:
                    if len(item) < numSeg:
                        flag = True
                        break
                if flag:
                    continue
                tmpArr = [[] for _ in range(numSeg)]
                for i in range(len(dataArr)):
                    dataArrSeg = split(dataArr[i], numSeg)
                    for index, oneDataArrSeg in enumerate(dataArrSeg):
                        narray=numpy.array(oneDataArrSeg)
                        N=len(oneDataArrSeg)
                        sum1=narray.sum()
                        narray2=narray*narray
                        sum2=narray2.sum()
                        mean=sum1/N
                        tmpArr[index].append(mean)
                        tmpArr[index].append(math.sqrt(abs(sum2/N)))
                        tmpArr[index].append(min(oneDataArrSeg))
                        tmpArr[index].append(max(oneDataArrSeg))
                        tmpArr[index].append(st.skew(narray))
                        tmpArr[index].append(st.kurtosis(narray))
                x_test.extend(tmpArr)
                y_test.extend([deviceId for _ in range(numSeg)])
    from sklearn.metrics import classification_report
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cross_validation import train_test_split

    lenOfRow = 0
    if x_test:
        lenOfRow = len(x_test[0])
    else:
        return [['Drive id','precision','recall','f1-score','support']]
    if isDp:
        from numpy.random import laplace
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
    print(accuracy_score(y_test, y_pred))
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

#clustering using all data in raw_data
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
    df1 = df[['RxDevice']].drop_duplicates()
    print df1
    maxDistance = radius
    minDistance = radius
    res = {}
    lat, lng = 42.2990, -83.717
    for i, row in df1.iterrows():
        coords = df.loc[df['RxDevice'] == row[0]].as_matrix(columns=['Latitude','Longitude']).astype(np.float)
        if isObfuscated:
            newcoords = []
            for i in range(len(coords)):
                if haversine(coords[i][1], coords[i][0], lng, lat) > 1000:
                	newcoords.append(coords[i])

            coords = pd.DataFrame(newcoords)
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
        res = {}
        res['clusters'] = []

        minLat, minLng, minDis = 0.0, 0.0, float('inf')
        for point in centermost_points:
            print point[1], point[0]
            if haversine(point[1], point[0], lng, lat) < minDis:
            	minLat, minLng, minDis = point[0], point[1], haversine(point[1], point[0], lng, lat)
        res['clusters'].append({'lat': minLat, 'lon': minLng, 'dis': minDis})
        res['locations'] = coords
    return res

#clustering using each day's data in raw_data
def clustering_respectively(raw_data, columns, isObfuscated, radius):
    import pandas as pd, numpy as np, matplotlib.pyplot as plt, time, math, random
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn import metrics
    from geopy.distance import great_circle
    from shapely.geometry import MultiPoint
    kms_per_radian = 6371.0088
    epsilon = 0.1 / kms_per_radian
    lat,lng = 42.2990,-83.717

    result = [{columns[index][0]:column for index, column in enumerate(value)} for value in raw_data]
    df = pd.DataFrame(result)
    df1 = df[['RxDevice']].drop_duplicates()
    print df1
    maxDistance = radius
    minDistance = radius
    res = {}
    temp_res = {}
    total, totalnew = 0, 0
    for group in df.groupby(['DayNum']):
        for i, row in df1.iterrows():
            coords = group[1].loc[group[1]['RxDevice'] == row[0]].as_matrix(columns=['Latitude','Longitude']).astype(np.float)
            if isObfuscated:
                newcoords = []
                total += len(coords)
                for i in range(len(coords)):
                    if haversine(coords[i][1], coords[i][0], lng, lat) > radius:
                        newcoords.append(coords[i])
                coords = np.array(newcoords)
                totalnew += len(coords)
            if len(coords) < 3 or np.any(coords is None):
                continue
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
            res = {}
            res['clusters'] = []
            temp_res[group[0]] = 1000000
            for point in centermost_points:
                temp_res[group[0]] = min(temp_res[group[0]], haversine(point[1], point[0], lng, lat))
                print group[0], haversine(point[1], point[0], lng, lat)
                res['clusters'].append({'lat': point[0], 'lon': point[1], 'dis': haversine(point[1], point[0], lng, lat)})
            res['locations'] = coords
    
    temp_res = sorted(temp_res.items(), key=lambda x:x[0])
    avg_infer = []
    for item in temp_res:
    	print item[0],'\t',item[1]
    	avg_infer.append(item[1])
    res['result_daily_infer'] = float(sum(avg_infer)) / len(avg_infer)
    return res


from geopy.distance import great_circle
from shapely.geometry import MultiPoint
def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)
            
#clustering using each day's data in raw_data and perform context-aware policies
def clustering_respectively_context(raw_data, columns, isObfuscated, radius):
    import pandas as pd, numpy as np, matplotlib.pyplot as plt, time, math, random
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn import metrics

    kms_per_radian = 6371.0088
    epsilon = 0.1 / kms_per_radian
    lat,lng = 42.2990,-83.717

    result = [{columns[index][0]:column for index, column in enumerate(value)} for value in raw_data]
    df = pd.DataFrame(result)
    df1 = df[['RxDevice']].drop_duplicates()
    print df1
    maxDistance = radius
    minDistance = radius
    res = {}
    temp_res = {}
    total, totalnew = 0, 0
    print radius
    radii = [500, 1000, 1500, 2000, 2500, 3000]
    loss = []
    for radius in radii:
        for group in df.groupby(['DayNum']):
            if group[0] not in temp_res:
                temp_res[group[0]] = []
        
            for i, row in df1.iterrows():
                coords = group[1].loc[group[1]['RxDevice'] == row[0]].as_matrix(columns=['Latitude','Longitude']).astype(np.float)
                newcoords = []
                total += len(coords)
                for i in range(len(coords)):
                    if haversine(coords[i][1], coords[i][0], lng, lat) > radius:
                        newcoords.append(coords[i])
                coords = np.array(newcoords)
                totalnew += len(coords)
                if len(coords) < 3 or np.any(coords is None):
                    continue
                db = KMeans(n_clusters=3, random_state=0).fit(np.radians(coords))
                cluster_labels = db.labels_
                num_clusters = len(set(cluster_labels))
                clusters = pd.Series([coords[cluster_labels==n] for n in range(num_clusters)])
                centermost_points = clusters.map(get_centermost_point)
                res = {}
                res['clusters'] = []
                
                temp_res[group[0]].append(1000000)
                for point in centermost_points:
                    temp_res[group[0]][-1] = min(temp_res[group[0]][-1], haversine(point[1], point[0], lng, lat))
                    res['clusters'].append({'lat': point[0], 'lon': point[1], 'dis': haversine(point[1], point[0], lng, lat)})
                res['locations'] = coords
        loss.append(float(totalnew) / total)
    
    temp_res = temp_res.values()
    print temp_res

    distances = []
    for i in range(len(radii)):
    	cnt = 0.0
    	sum = 0.0
    	for x in temp_res:
            if len(x) != len(radii):
                continue
            cnt += 1
            sum += x[i]
        distances.append(str(sum/cnt))
    res['result_context'] = distances
    res['result_context_loss'] = [str(x) for x in loss]
    print '\t'.join(distances)
    print '\t'.join([str(x) for x in loss])
    return res

#clustering using each day's data in raw_data and perform two-ends policies
def clustering_respectively_portions(raw_data, columns, isObfuscated, radius):
    import pandas as pd, numpy as np, matplotlib.pyplot as plt, time, math, random
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn import metrics

    kms_per_radian = 6371.0088
    epsilon = 0.1 / kms_per_radian
    lat,lng = 42.2990,-83.717

    result = [{columns[index][0]:column for index, column in enumerate(value)} for value in raw_data]
    df = pd.DataFrame(result)
    df1 = df[['RxDevice']].drop_duplicates()
    print df1
    maxDistance = radius
    minDistance = radius
    res = {}
    temp_res = {}
    total, totalnew = 0, 0
    portions = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    loss = []
    for portion in portions:
        for group in df.groupby(['DayNum']):
            if group[0] not in temp_res:
                temp_res[group[0]] = []
        
            for i, row in df1.iterrows():
                coords = group[1].loc[group[1]['RxDevice'] == row[0]].as_matrix(columns=['Latitude','Longitude']).astype(np.float)
                total = len(coords)
                coords = coords[int(total * portion) : total - int(total * portion)]
                if len(coords) < 3 or np.any(coords is None):
                    continue
                db = KMeans(n_clusters=3, random_state=0).fit(np.radians(coords))
                cluster_labels = db.labels_
                num_clusters = len(set(cluster_labels))
                clusters = pd.Series([coords[cluster_labels==n] for n in range(num_clusters)])
                centermost_points = clusters.map(get_centermost_point)
                res = {}
                res['clusters'] = []
                
                temp_res[group[0]].append(1000000)
                for point in centermost_points:
                    temp_res[group[0]][-1] = min(temp_res[group[0]][-1], haversine(point[1], point[0], lng, lat))
                    res['clusters'].append({'lat': point[0], 'lon': point[1], 'dis': haversine(point[1], point[0], lng, lat)})
                res['locations'] = coords
        
    
    temp_res = temp_res.values()
    print temp_res
    distances = []
    for i in range(len(portions)):
    	#print i
    	cnt = 0.0
    	sum = 0.0
    	for x in temp_res:
            if len(x) != len(portions):
                continue
            cnt += 1
            sum += x[i]
        distances.append(str(sum/cnt))
    print '\t'.join(distances)
    res['result_twoend'] = distances
    return res


#clustering using each week's data in raw_data
def clustering_respectively_week(raw_data, columns, isObfuscated, radius):
    import pandas as pd, numpy as np, matplotlib.pyplot as plt, time, math, random
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn import metrics
    from geopy.distance import great_circle
    from shapely.geometry import MultiPoint
    kms_per_radian = 6371.0088
    epsilon = 0.1 / kms_per_radian
    lat,lng = 42.2990,-83.717


    result = [{columns[index][0]:column for index, column in enumerate(value)} for value in raw_data]
    df = pd.DataFrame(result)
    df1 = df[['RxDevice']].drop_duplicates()
    print df1
    maxDistance = radius
    minDistance = radius
    res = {}
    temp_res = {}
    total, totalnew = 0, 0
    weeks = [[42461, 42467], [42468, 42474], [42475, 42481], [42482, 42490]]
    for week in weeks:
        for i, row in df1.iterrows():
            coords = df.loc[(df['RxDevice'] == row[0]) & (df['DayNum'] >= week[0]) & (df['DayNum']<=week[1])].as_matrix(columns=['Latitude','Longitude']).astype(np.float)
            print week, len(coords)
            if isObfuscated:
                newcoords = []
                total += len(coords)
                for i in range(len(coords)):
                    if haversine(coords[i][1], coords[i][0], lng, lat) > radius:
                        newcoords.append(coords[i])
                coords = np.array(newcoords)
                totalnew += len(coords)
            if len(coords)==0 or np.any(coords is None):
                continue
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
            res = {}
            res['clusters'] = []
            temp_res[week[0]] = 1000000
            for point in centermost_points:
                #print point[1], point[0]
                temp_res[week[0]] = min(temp_res[week[0]], haversine(point[1], point[0], lng, lat))
                print week[0], haversine(point[1], point[0], lng, lat)
                res['clusters'].append({'lat': point[0], 'lon': point[1], 'dis': haversine(point[1], point[0], lng, lat)})
            res['locations'] = coords
    
    temp_res = sorted(temp_res.items(), key=lambda x:x[0])
    for item in temp_res:
    	print item[0],'\t',item[1]
    if isObfuscated:
        print float(totalnew) / total
    return res

