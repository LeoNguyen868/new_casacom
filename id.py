import duckdb
home_maid_gt=duckdb.query("""
copy(
select distinct ad_id from read_parquet("/home/hieu/Work/casacom/import/home/2025/**/*.parquet")
) to '/home/hieu/Work/new_casacom/result/home_maid_gt.parquet' (format parquet)
""")
work_maid_gt=duckdb.query("""
copy(
select distinct ad_id from read_parquet("/home/hieu/Work/casacom/import/work/2025/**/*.parquet")
) to '/home/hieu/Work/new_casacom/result/work_maid_gt.parquet' (format parquet)
""")