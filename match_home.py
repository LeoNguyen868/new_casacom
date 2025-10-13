import glob
import pandas as pd
home=pd.read_csv('/home/hieu/Work/new_casacom/result3/home.csv', usecols=['maid', 'geohash'])
all_home_gt=glob.glob("/home/hieu/Work/casacom/import/home/2025/**/*.parquet")
home['geohash_5']=home['geohash'].apply(lambda x: x[:5])
import duckdb

# Register home if it's a Pandas DataFrame (assuming home already exists)
duckdb.register('home', home)
all_df=[]
for i in range(len(all_home_gt)):
    # Perform the entire query in DuckDB without creating intermediate DataFrames
    print(f"Processing {i} of {all_home_gt[i]}")
    
    # Match by both MAID and geohash_5
    maid_match = duckdb.query(f"""
        WITH gt AS (
            SELECT 
                ad_id,
                geohash_5
            FROM read_parquet('{all_home_gt[i]}')
        )
        
        SELECT 
            h.maid,
            h.geohash,
            h.geohash_5,
            gt.ad_id AS ad_id_gt,
            gt.geohash_5 AS gt_geohash_5
        FROM home AS h
        INNER JOIN gt
        ON h.maid = gt.ad_id AND h.geohash_5 = gt.geohash_5
        
    """).df()
    all_df.append(maid_match)

pd.concat(all_df).to_parquet("match_home.parquet")