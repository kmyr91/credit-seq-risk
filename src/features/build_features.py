
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# 1. Start Spark
spark = (
    SparkSession.builder
    .appName("CreditRiskFeatures")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# 2. Load monthly histories
history = spark.read.parquet(
    "data/processed/uci/history.parquet"
)


# 3. Aggregate features
features = history.groupBy("customer_id").agg(
    F.count("*").alias("n_statements"),
    F.avg("bill_amount").alias("mean_bill"),
    F.max("bill_amount").alias("max_bill"),
    F.sum("payment_amount").alias("total_payment"),
    F.avg("repayment_status").alias("mean_repayment_status"),

    # New features
    F.max("repayment_status").alias("max_repayment_status"),

    F.sum(
        F.when(F.col("repayment_status") > 0, 1).otherwise(0)
    ).alias("months_late"),

    F.sum(
        F.greatest(F.col("bill_amount"), F.lit(0))
    ).alias("total_positive_bill"),
)


# 4. Find latest and earliest observations
latest_window = (
    Window.partitionBy("customer_id")
    .orderBy(F.col("statement_date").desc())
)

earliest_window = (
    Window.partitionBy("customer_id")
    .orderBy(F.col("statement_date").asc())
)

ranked = (
    history
    .withColumn("latest_rank", F.row_number().over(latest_window))
    .withColumn("earliest_rank", F.row_number().over(earliest_window))
)

temporal = ranked.groupBy("customer_id").agg(
    F.max(
        F.when(
            F.col("latest_rank") == 1,
            F.col("repayment_status")
        )
    ).alias("latest_repayment_status"),

    F.max(
        F.when(
            F.col("latest_rank") == 1,
            F.col("bill_amount")
        )
    ).alias("latest_bill"),

    F.max(
        F.when(
            F.col("earliest_rank") == 1,
            F.col("bill_amount")
        )
    ).alias("earliest_bill"),
)

temporal = temporal.withColumn(
    "bill_change",
    F.col("latest_bill") - F.col("earliest_bill")
).drop("earliest_bill")


# 5. Combine features
features = features.join(
    temporal,
    on="customer_id",
    how="inner"
)

# Prevent duplicate customer records
assert features.count() == 30000

assert not (
    features.groupBy("customer_id")
    .count()
    .filter(F.col("count") > 1)
    .limit(1)
    .count()
)

# Check sequence lengths
assert features.filter(
    F.col("n_statements") != 6
).count() == 0

features.orderBy("customer_id").show(5, truncate=False)


# 6. Save
output = Path(
    "data/processed/uci/customer_features.parquet"
)

features.toPandas().to_parquet(
    output,
    index=False
)

spark.stop()

print("\nSaved:", output)
