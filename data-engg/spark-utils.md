
### Creating spark table using parquet
```sql
CREATE TABLE table_test
USING parquet
OPTIONS (Path 'viewfs://aggregate/20230201/data',header "true",delimiter '\t', inferSchema "true")
;
```

### Creating spark table using CSV
```sql
CREATE TABLE table_test(C1 STRING, C2 STRING, C3 STRING, C4 STRING, C5 STRING, C6 STRING, C7 STRING) using CSV
OPTIONS (Path '/apps/hd/data/all',header "false",delimiter '\t')
;
```

https://github.com/adipolak/scaling-machine-learning-course/blob/main/notebooks