import glob
import pandas as pd
home=pd.read_csv('/home/hieu/Work/new_casacom/maids.csv', usecols=['maid'])
all_home_gt=glob.glob("/home/hieu/Work/casacom/import/home/2025/**/*.parquet")
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
            FROM read_parquet('{all_home_gt[i]}')
        )
        
        SELECT 
            h.maid,
            gt.ad_id AS ad_id_gt,
        FROM home AS h
        INNER JOIN gt
        ON h.maid = gt.ad_id
        
    """).df()
    all_df.append(maid_match)

pd.concat(all_df).to_parquet("match_all_maids_home.parquet")