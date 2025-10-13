import glob
import pandas as pd
work=pd.read_csv('/home/hieu/Work/new_casacom/result3/work.csv', usecols=['maid', 'geohash'])
all_work_gt=glob.glob("/home/hieu/Work/casacom/import/work/2025/**/*.parquet")
work['geohash_5']=work['geohash'].apply(lambda x: x[:5])
import duckdb

# Register work if it's a Pandas DataFrame (assuming work already exists)
duckdb.register('work', work)
all_df=[]
for i in range(len(all_work_gt)):
    # Perform the entire query in DuckDB without creating intermediate DataFrames
    print(f"Processing {i} of {all_work_gt[i]}")
    
    # Match by both MAID and geohash_5
    maid_match = duckdb.query(f"""
        WITH gt AS (
            SELECT 
                ad_id,
                geohash_5
            FROM read_parquet('{all_work_gt[i]}')
        )
        
        SELECT 
            w.maid,
            w.geohash,
            w.geohash_5,
            gt.ad_id AS ad_id_gt,
            gt.geohash_5 AS gt_geohash_5
        FROM work AS w
        INNER JOIN gt
        ON w.maid = gt.ad_id
        
    """).df()
    all_df.append(maid_match)

pd.concat(all_df).to_parquet("match_work_all.parquet")