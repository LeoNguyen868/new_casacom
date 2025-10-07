import duckdb
months=["03","04","05","06","07","08","09"]
for month in months:
    print(f"Processing {month}")
    duckdb.query(f"""copy (select maid, flux, sum(maid_flux) as maid_flux from read_parquet('/home/hieu/Work/new_casacom/data/processed/2025-{month}-*.parquet') group by maid, flux) to '/home/hieu/Work/new_casacom/data/months/2025-{month}.parquet' (format parquet)""")