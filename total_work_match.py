import pandas as pd
all_maid=pd.read_parquet('/home/hieu/Work/new_casacom/data/months/all_maid.parquet', columns=['maid'])
import glob
all_work_gt=glob.glob("/home/hieu/Work/casacom/import/work/2025/**/*.parquet")
import duckdb
all_df=[]
for i, file_path in enumerate(all_work_gt):
    print(i,len(all_work_gt))
    cc=duckdb.query(f"""
    SELECT DISTINCT gt.ad_id 
    FROM read_parquet('{file_path}') AS gt
    INNER JOIN all_maid
    ON gt.ad_id = all_maid.maid
""").df()
    all_df.append(cc)
all_df=pd.concat(all_df)
all_df.to_parquet('/home/hieu/Work/new_casacom/data/months/all_work_match.parquet')