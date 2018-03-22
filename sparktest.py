from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, to_timestamp
from sparkts.datetimeindex import DayFrequency
from sparkts import datetimeindex, timeseriesrdd
from sparkts.models import ARIMA # NOTE: sparkts from pip does not have this, so I manually copied from upstream
from pyspark import SparkContext, SQLContext
from pyspark.mllib.linalg import DenseVector
import math

PROGRAM_FOLDER = "/home/ruixinbao/sparkplunge/small_prediction/"

def createDataFrame(sparkObj, filename):
    dataFrame = sparkObj.read.option("header", "true").csv(PROGRAM_FOLDER + filename)
    return dataFrame


def selectInfoFromDataFrame(dataFrame, companyname):
    dataInfo = dataFrame.select(dataFrame["Date"].alias(companyname + "Date"),
                                dataFrame["Close"].alias("close" + companyname.capitalize()))
    return dataInfo

def selectInfoAsNewNames(dataFrame, symbol):
    closeSymbol = "close" + symbol.capitalize()
    newNameDataFrame = dataFrame.select(dataFrame["date"], dataFrame[closeSymbol].alias("price")).withColumn("symbol", lit(symbol))
    # Reorder the columns with the newly selected names
    newNameDataFrame = newNameDataFrame.select(newNameDataFrame.date, newNameDataFrame.symbol, newNameDataFrame.price.alias("closingPrice"))
    return newNameDataFrame

def train_transform_func(vector):
    # Remap the value to elimiate NAN result for ease of calculation
    new_vec = DenseVector.toArray().map(lambda x : 0 if math.isnan(x) else x )
    arimaModel = ARIMA.fit_model(1, 0, 0, new_vec)
    forecasted = arimaModel.forecast(new_vec, 5) # 5 days for predict
    print(type(forecasted))
    exit(0)


def run_spark_application():
    # Creates session and spark context

    sc = SparkContext(appName="Stocks")
    spark = SQLContext.getOrCreate(sc)

    amazonDataFrame = createDataFrame(spark, "amazon.csv")
    amazonInfo = selectInfoFromDataFrame(amazonDataFrame, "amazon")

    googDataFrame = createDataFrame(spark, "google.csv")
    googInfo = selectInfoFromDataFrame(googDataFrame, "google")

    facebookDataFrame = createDataFrame(spark, "facebook.csv")
    facebookInfo = selectInfoFromDataFrame(facebookDataFrame, "facebook")

    # Collect all Date and closing into one dataFrame
    dataTable = amazonInfo.join(googInfo,
                                amazonInfo.amazonDate == googInfo.googleDate).select("amazonDate",
                                                                                     "closeAmazon", "closeGoogle")
    dataTable = dataTable.join(facebookInfo,
                               dataTable.amazonDate == facebookInfo.facebookDate).select(dataTable["amazonDate"].alias("date"),
                                                                                         "closeAmazon", "closeGoogle", "closeFacebook")

    # We want to format the data into the format such that first column is all date, second column is symbols and last
    # column is all about the closing price of that day
    amazFormatted = selectInfoAsNewNames(dataTable, "amazon")
    faceBookFormatted = selectInfoAsNewNames(dataTable, "facebook")
    googFormatted = selectInfoAsNewNames(dataTable, "google")
    # We union the columns together, then reorder them by dates
    formattedDataTable = amazFormatted.union(faceBookFormatted).union(googFormatted)
    formattedDataTable = formattedDataTable.orderBy(formattedDataTable.date.asc())

    # We construct the final DataFrame
    # 1: We add timestamp and price as two new columns based on date and closing Price
    finalDf = formattedDataTable.withColumn("timestamp",
                                            to_timestamp(formattedDataTable.date)).withColumn("price",
                                                                                              formattedDataTable["closingPrice"].cast("double"))
    # 2: After that we drop the original price and closingPrice
    finalDf = finalDf.drop("date", "closingPrice").sort("timestamp")
    finalDf.registerTempTable("preData")
    finalDf.show()

    # We gather the necessary data to create a time series RDD
    minDate = finalDf.selectExpr("min(timestamp)").collect()[0]["min(timestamp)"]
    maxDate = finalDf.selectExpr("max(timestamp)").alias("timestamp").collect()[0]["max(timestamp)"]
    frequency = DayFrequency(1, sc)

    dtIndex = datetimeindex.DateTimeIndex.uniform(start=minDate, end=maxDate, freq=frequency, sc=sc)
    tsRdd = timeseriesrdd.time_series_rdd_from_observations(dtIndex, finalDf, "timestamp", "symbol", "price")


    # Last step BRO, we perform the prediction
    df = tsRdd.map_series(train_transform_func)

    # Let's avoid the zone check in python here. it is way too annoying if we care about that
    finalDf.show()
    spark.stop()



if __name__ == '__main__':
    run_spark_application()