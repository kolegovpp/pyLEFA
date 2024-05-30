import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os
import random
import seaborn as sns
try:
    import gdal,ogr
except ModuleNotFoundError:
    from osgeo import gdal,ogr
    
from ca_plot import ca_plot

# getting data from averaging windows
def createSHPfromDictionary(outputGridfn, data_dict):
    # create output file
    outDriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outputGridfn):
        os.remove(outputGridfn)
    outDataSource = outDriver.CreateDataSource(outputGridfn)
    outLayer = outDataSource.CreateLayer(outputGridfn, geom_type=ogr.wkbPolygon)

    # create attribute fields
    for el in data_dict:
        # print(data_dict[el]);
        if el == 'id':
            outLayer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger));
        else:
            outLayer.CreateField(ogr.FieldDefn(el[0:8], ogr.OFTReal));

    # feature definition (needed to address attribute able data)
    featureDefn = outLayer.GetLayerDefn()

    feature_counter = 0;

    # create grid cells
    for idx in range(0, len(data_dict[el]), 1):
        # data_dict={"id":TAB_id,"X_left":TAB_X_left,"X_right":TAB_X_right,\
        #           "Y_top":TAB_Y_top,"Y_bottom":TAB_Y_bottom,base_filename:TAB_raster_value};
        # data_dict['X_left'][idx]
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(data_dict['X_left'][idx], data_dict['Y_top'][idx])
        ring.AddPoint(data_dict['X_right'][idx], data_dict['Y_top'][idx])
        ring.AddPoint(data_dict['X_right'][idx], data_dict['Y_bottom'][idx])
        ring.AddPoint(data_dict['X_left'][idx], data_dict['Y_bottom'][idx])
        ring.AddPoint(data_dict['X_left'][idx], data_dict['Y_top'][idx])
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

        # add new geom to layer
        outFeature = ogr.Feature(featureDefn)
        # outFeature.SetGeometry(poly)
        # outLayer.CreateFeature(outFeature)

        # Setting field data
        for el in data_dict:
            outFeature.SetField(el[0:8], float(
                data_dict[el][feature_counter]));  # conversion to FLOAT, cause GDAL dislike NUMPY ARRAY

        # creating of feature MUST be AFTER adding the data
        outFeature.SetGeometry(poly)
        outLayer.CreateFeature(outFeature)
        #
        # outFeature.SetField("id", 0);
        feature_counter = feature_counter + 1;

        outFeature = None

    # Save and close DataSources
    outDataSource = None

def get_image_classes(image):
    m,n=image.shape;
    km = KMeans(n_clusters=10)  #10 classes
    km.fit(image.flatten().reshape(-1,1))  #reshape(-1,1) - transpose of the single-dimension array
    km.predict(image.flatten().reshape(-1,1))
    image_classes = km.labels_.reshape(m,n);
    return image_classes


def kmeans_sort(img,do_reverse=0,n_clusters=2,elbow=0):

    shape=np.shape(img);
    data=img.flatten().reshape(-1,1);
    
    if elbow==1:
        #distortion of the method
        distortions = []
        K = range(1,10)
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(data)
            distortions.append(kmeanModel.inertia_)
        
        
        plt.figure(figsize=(16,8));
        plt.plot(K, distortions, 'bx-');
        plt.xlabel('k');
        plt.ylabel('Distortion');
        plt.title('The Elbow Method showing the optimal k');
        plt.savefig('elbow_plot.svg');
        plt.savefig('elbow_plot.png');
        plt.show();

    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    
    classes = kmeans.labels_
    
    mean_values=np.array([]);
    class_arr=np.arange(0,max(classes)+1);
    #arrange labels according to their values
    for label in class_arr:
        mean_values=np.append(mean_values,np.mean(data[classes==label]));        
    
    #sort values of array classes according to their labels
    if do_reverse==1:
        inds = mean_values[::-1].argsort();    
    else:
        inds = mean_values.argsort();
    new_classes=np.zeros(classes.shape,dtype=np.int8);
    for ind,new_val in zip(inds,range(0,len(inds))):
        new_classes[classes==ind]=new_val;
    
    new_classes=np.reshape(new_classes,shape);
    return new_classes; 


#делать красивую графику по книге Hoss Belyadi

threshold = 0.0



#open datatable
filename =  'datatable_l09_carlin_trend.csv'
filepath = os.path.join('..','..','csv',filename)

#dataframe
df = pd.read_csv(filepath)

#number of records
rec_number = len(df.index)
indeces_all = list(range(rec_number))

#train test split
train, test = train_test_split(df, test_size=0.3)

#keys in a table  'deposit_po', 'f3l9_density', 'l3l9_density', 'f3l9_density', 'l3l9_minkowski'
print(df.keys())

#copy df with predictors
#new_df = df[['f0_lc09den', 'l0_lc09den', 'f0_lc_09mi','l0_lc_09mi','deposit_po']].copy()
new_df = df[['f3l9_density', 'l3l9_density', 'f3l9_minkowski','l3l9_minkowski','deposit_po']].copy()
new_df = new_df.dropna()

#table without deposits
new_df2 = df[['f3l9_density', 'l3l9_density', 'f3l9_minkowski','l3l9_minkowski']].copy()
new_def2 = new_df2.dropna()

print('total number of samples=',len(new_df))
print('non-zero number of samples=',len(new_df[new_df['deposit_po']>0]))

#correlation between variables
print(df.corr(method='pearson'))

plt.figure()
plt.hist(new_df['deposit_po'])
plt.show()


plt.figure(figsize=(17,8))
sns.heatmap(new_df[new_df['deposit_po']>threshold].corr(method='pearson'), cmap='coolwarm', annot=True, linewidths=4, linecolor='black')
plt.savefig('heatmap.png')
plt.show()

# plt.figure(figsize=(17,8))
# sns.heatmap(new_df[new_df['deposit_po']>=0].corr(), cmap='coolwarm', annot=True, linewidths=4, linecolor='black')
# plt.savefig('heatmap2.png')
# plt.show()

#create correlation plots
plt.figure(figsize=(17,8))
sns.pairplot(new_df[new_df['deposit_po']>threshold], kind='scatter')
plt.savefig('pairplot.png')
plt.show()

# #create correlation plots
# plt.figure(figsize=(17,8))
# sns.pairplot(new_df[new_df['deposit_po']>=0], kind="scatter")
# plt.savefig('pairplot2.png')
# plt.show()

plt.plot(new_df[new_df['deposit_po']>threshold]['l3l9_density'],new_df[new_df['deposit_po']>threshold]['deposit_po'],'r+')
plt.xlabel('Плотность (совокупная длина) линейных элементов')
plt.ylabel('Плотность распределения рудных объектов')
plt.savefig('l3l9_density.png',dpi=300)
plt.savefig('l3l9_density.svg')
plt.show()

print(new_df[new_df['deposit_po']>threshold])


'''
K-means clustering

Задачаи по проекту:
    1) Источники вещества для объектов Карлинского типа (если есть границы,
                                        вынести на карту)
    2) Построить модель K-means кластеры
    3) PDPs


'''



from sklearn.cluster import KMeans
import numpy as np

from sklearn.preprocessing import StandardScaler



# X = np.array([[1, 2,3], [1, 4,3], [1, 0,5],
#                 [10, 2,8], [10, 4,16], [10, 0,34]])
# X = np.array([[1, 2], [1, 4], [1, 0],
#                [10, 2], [10, 4], [10, 0]])
#print(X)
#standardization before k-means
scaler = StandardScaler()
scaler.fit(new_df2)
new_df2_transform = scaler.transform(new_df2)

kmeans = KMeans(n_clusters=4, random_state=0, n_init=5,algorithm='auto')
kmeans.fit(new_df2_transform)
kmeans.labels_


distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(new_df2_transform)
    distortions.append(kmeanModel.inertia_)


plt.figure(figsize=(16,8));
plt.plot(K, distortions, 'bx-');
plt.xlabel('Число классов');
plt.ylabel('Искажение');
plt.title('Нахождение оптимального числа классов K-Means');
plt.savefig('elbow_plot.svg');
plt.savefig('elbow_plot.png');
plt.show();

#add k-means classes to dataframe
new_df2['kmeans_classes']=kmeans.labels_
df['kmeans_classes']=np.nan
df
for ind in new_df.index.values:
    df['kmeans_classes'].loc[ind] = new_df2['kmeans_classes'].loc[ind]

print(df)

df_dict = df.to_dict()

#output to shp file 
createSHPfromDictionary('new_df2.shp', df_dict)

#determine feature importance
#https://towardsdatascience.com/interpretable-k-means-clusters-feature-importances-7e516eeb8d3c 

from sklearn.ensemble import RandomForestClassifier

#df['Binary Cluster 0'] = df['Cluster'].map({0:1, 1:0, 2:0})
df['class0'] = df['kmeans_classes'].map({0:1, 1:0, 2:0, 3:0})
df['class1'] = df['kmeans_classes'].map({0:0, 1:1, 2:0, 3:0})
df['class2'] = df['kmeans_classes'].map({0:0, 1:0, 2:1, 3:0})
df['class3'] = df['kmeans_classes'].map({0:0, 1:0, 2:0, 3:1})

#copy df to new table df_stat
print(df['kmeans_classes'].value_counts())
print(df['class0'].value_counts())
df_stat = df.copy()
df_stat = df_stat.dropna()

#create tables for output table
dict_out = {'num':[],'predictors':[],'weights':[],'class kmeans':[]}

#classify features to get importancies 
kmeans_classes = ['class0','class1','class2','class3']
num = 0 
for cl in kmeans_classes:
    clf = RandomForestClassifier(random_state=1)
    #'f3l9_density', 'l3l9_density', 'f3l9_minkowski','l3l9_minkowski'
    predictor_list = ['f3l9_density', 'l3l9_density', 'f3l9_minkowski','l3l9_minkowski']
    clf.fit(df_stat[predictor_list].values, df_stat[cl].values)
    
    # Index sort the most important features
    sorted_feature_weight_idxes = np.argsort(clf.feature_importances_)[::-1] # Reverse sort
    
    print('class name=', cl)
    for ind in sorted_feature_weight_idxes:
        print(predictor_list[ind],clf.feature_importances_[ind])
        dict_out['num'].append(num)
        dict_out['predictors'].append(predictor_list[ind])
        dict_out['weights'].append(round(clf.feature_importances_[ind],4))
        dict_out['class kmeans'].append(cl)
        num += 1

#output weighted predictors of k-means classes
df_out = pd.DataFrame.from_dict(dict_out)
df_out.to_csv('predictors_classes.csv',sep=';',index=False)

#create C-A plots for every previctor to determine classes
ca_plot(df['l3l9_density'],'line density')
ca_plot(df['l3l9_minkowski'],'line mink')
ca_plot(df['f3l9_density'],'fault density')
ca_plot(df['f3l9_minkowski'],'fault mink')


