import pandas as pd
import numpy as np
import ee
import os
from sklearn.model_selection import train_test_split
import geopandas as gpd
from geopandas import GeoDataFrame


def getTargetLoss(startYear=2011,endYear=2020,treeCoverThreshold=30,maskToPrimaryForest=True,
                    forestChangeImage="UMD/hansen/global_forest_change_2020_v1_8",
                    primaryForestImage='UMD/GLAD/PRIMARY_HUMID_TROPICAL_FORESTS/v1/2001'):
                        
    forestChange = ee.Image(forestChangeImage)
    primaryForest = ee.Image(primaryForestImage)
    forestChangeValid = forestChange.select('datamask').eq(1)
    
    #select tree cover loss band and unmask with value 0 to represent no loss
    tclYear = forestChange.select(['lossyear']).unmask(0)

    #define mask of Hansen data years
    #get target years
    tclFiltered = (tclYear.gte(startYear - 2000)).And(tclYear.lte(endYear - 2000))
    #get early years
    earlyLoss = ((tclYear.lt(startYear - 2000)).And(tclYear.gt(0))).Or(tclYear.gt(endYear - 2000))

    #mask to tree cover >= tree cover threshold
    forestCoverValid = forestChange.select('treecover2000').gte(treeCoverThreshold)
  
    #get mask for tree cover >= tree cover threshold and valid forest change
    treeCoverLossValid = forestCoverValid.And(forestChangeValid)
    
    #if mask to primary forest is set, add mask
    if maskToPrimaryForest:
        treeCoverLossValid = treeCoverLossValid.And(primaryForest.eq(1))

    mask = treeCoverLossValid.And(earlyLoss.neq(1))
    
    return tclFiltered.mask(mask)
    
# projection_ee = dynamic_world_classifications_monthly.first().projection()
# projection = projection_ee.getInfo()
# crs = projection.get('crs')
# crsTransform = projection.get('transform')
def getRateOfClass(image,classValue,region,crs,crsTransform):
    classRate = image.eq(classValue).reduceRegion(reducer=ee.Reducer.mean(), 
                    geometry=region, crs=crs, crsTransform=crsTransform, bestEffort=True, maxPixels=1e13, tileScale=16)
    return classRate

def getStratifiedSampleBandPoints(image, region, bandName, **kwargs):
    """
    Function to perform stratitfied sampling of an image over a given region, using ee.Image.stratifiedSample(image, region, bandName, **kwargs)
    Args:
        image (ee.Image): an image to sample
        region (ee.Geometry): the geometry over which to sample
        bandName (String): the bandName to select for stratification
    Returns:
        An ee.FeatureCollection of sampled points along with coordinates
    """
    dargs = {
        'numPoints': 1000,
        'classBand': bandName,
        'region': region
    }
    dargs.update(kwargs)
    stratified_sample = image.stratifiedSample(**dargs)
    return stratified_sample
    
def splitTrainValidationTestStratified(featureCollection, stratifyColumn, trainSplit=0.6, testSplit=0.5,seed=30):
    negativePoints = featureCollection.filter(ee.Filter.eq(stratifyColumn,0))
    positivePoints = featureCollection.filter(ee.Filter.eq(stratifyColumn,1))
    
    #Assign random value between 0 and 1 to split into training, validation, and test sets
    negativePoints = negativePoints.randomColumn(columnName='TrainingSplit', seed=seed)
    positivePoints = positivePoints.randomColumn(columnName='TrainingSplit', seed=seed)
    
    #First split training set from the rest, taking 70% of the points for training
    #Roughly 60% training, 40% for validation + testing
    negativePointsTrain = negativePoints.filter(ee.Filter.lt('TrainingSplit', trainSplit))
    negativePointsNonTrain = negativePoints.filter(ee.Filter.gte('TrainingSplit', trainSplit))
    positivePointsTrain = positivePoints.filter(ee.Filter.lt('TrainingSplit', trainSplit))
    positivePointsNonTrain = positivePoints.filter(ee.Filter.gte('TrainingSplit', trainSplit))
    
    #Assign another random value between 0 and 1 to validation_and_test to split to validation and test sets
    negativePointsNonTrain = negativePointsNonTrain.randomColumn(columnName='ValidationSplit', seed=seed)
    positivePointsNonTrain = positivePointsNonTrain.randomColumn(columnName='ValidationSplit', seed=seed)
    
    #Of the 40% saved for validation + testing, half goes to validation and half goes to test
    #Meaning original sample will be 60% training, 20% validation, 20% testing
    negativePointsValidation = negativePointsNonTrain.filter(ee.Filter.lt('ValidationSplit', testSplit))
    negativePointsTest = negativePointsNonTrain.filter(ee.Filter.gte('ValidationSplit', testSplit))
    positivePointsValidation = positivePointsNonTrain.filter(ee.Filter.lt('ValidationSplit', testSplit))
    positivePointsTest = positivePointsNonTrain.filter(ee.Filter.gte('ValidationSplit', testSplit))
        
    trainingPoints = ee.FeatureCollection([negativePointsTrain, positivePointsTrain]).flatten()
    validationPoints = ee.FeatureCollection([negativePointsValidation, positivePointsValidation]).flatten()
    testPoints = ee.FeatureCollection([negativePointsTest, positivePointsTest]).flatten()
    
    return trainingPoints, validationPoints, testPoints
    
# def numberOfTestPoints(p0, sigma, alpha, beta):
# # ● Expected accuracy of the product (P0)
# # ● Precision of detecting differences from this accuracy (minimum detectable difference, δ)
# # ● Tolerance of Type I error (alpha, α)
# # ● Tolerance of Type II error (beta, ß)
#     n1 =
#
#
    
    
def rasterizeVector(featureCollection,propertyNames=[]):
    if propertyNames==[]:
        featureCollection = featureCollection.map(lambda x: x.set({'constant':1}))
        propertyNames = ['constant']
    rasterImage = featureCollection.reduceToImage(properties=propertyNames,reducer=ee.Reducer.first()).unmask(0)
    return rasterImage
    
def distanceToImage(image,maxDistance=50000):
    cost = ee.Image.constant(1)
  
    #calculate distance to, this is in a made up unit (not like meters) but it doens't matter because
    #we'll apply normalization to it later
    distanceImage = cost.cumulativeCost(source=image, maxDistance=maxDistance)
    return distanceImage.unmask(maxDistance)

    
    
def remove_bands(image,remove_list):
    band_names = image.bandNames()
    selectors = band_names.removeAll(kwargs.remove_list)
    return kwargs.image.select(selectors)






def compareRates(lossImage, predictionImage, region, computationScale):
    rateDifference = lossImage.subtract(predictionImage)
    rateDifferenceAverage = rateDifference.reduceRegion(reducer=ee.Reducer.mean().unweighted(), 
        geometry=region, 
        scale= computationScale, 
        crs= 'EPSG:4326',
        bestEffort=True, 
        maxPixels=1e13, 
        tileScale=16)
    rateDifferenceAverage = ee.Number(rateDifferenceAverage.values().get(0))

    differenceSquared = rateDifference.multiply(rateDifference)

    numerator = differenceSquared.reduceRegion(reducer=ee.Reducer.sum().unweighted(), 
        geometry=region, 
        scale= computationScale, 
        crs= 'EPSG:4326',
        bestEffort=True, 
        maxPixels=1e13, 
        tileScale=16)
    numerator = ee.Number(numerator.values().get(0))

    denominator = differenceSquared.reduceRegion(reducer=ee.Reducer.count(), 
        geometry=region, 
        scale= computationScale, 
        crs= 'EPSG:4326',
        bestEffort=True, 
        maxPixels=1e13, 
        tileScale=16)
    denominator = ee.Number(denominator.values().get(0))

    rmse = numerator.divide(denominator)
    rmse = rmse.sqrt()
    return ee.Feature(region,{'computationScale':computationScale, 
                            'averageRateDifference':rateDifferenceAverage,
                            'RMSE':rmse,'numerator':numerator,'denominator':denominator})












































