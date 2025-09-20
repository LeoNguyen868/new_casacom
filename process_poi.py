import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import os

# Get PostgreSQL connection parameters from environment variables or use defaults
db_params = {
    "dbname": os.environ.get("POSTGRES_DB", "osm"),
    "user": os.environ.get("POSTGRES_USER", "postgres"),
    "password": os.environ.get("POSTGRES_PASSWORD", "postgres"),
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": os.environ.get("POSTGRES_PORT", "5432")
}

try:
    # Try to connect to PostgreSQL
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    print("Connected to PostgreSQL!")
except Exception as e:
    print(f"Failed to connect to PostgreSQL: {e}")
    raise  # Raise the exception to indicate failure

import plotly.graph_objects as go
from shapely.wkt import loads
import psycopg2
from pyproj import Transformer

def extract_coordinates(wkt_string):
    geom = loads(wkt_string)
    if geom.geom_type == 'Polygon':
        coords = list(geom.exterior.coords)
        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        return [transformer.transform(x, y) for x, y in coords]
    elif geom.geom_type == 'LineString':
        coords = list(geom.coords)
        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        return [transformer.transform(x, y) for x, y in coords]
    else:
        return []
import json
key_tag=json.load(open('./poi_tag_keys.json'))
value_tag=json.load(open('./poi_tag_values.json'))
building_data=json.load(open('./building_data.json'))
k_mapping={}
for k,v in key_tag.items():
    for i in v:
        if i not in k_mapping:
            k_mapping[i] = []
        k_mapping[i].append(k)
v_mapping={}
for k,v in value_tag.items():
    for i in v:
        if i not in v_mapping:
            v_mapping[i] = []
        v_mapping[i].append(k)
building_mapping={}
for k,v in building_data.items():
    for i in v:
        if i not in building_mapping:
            building_mapping[i] = []
        building_mapping[i].append(k)
def get_poi_category(lat, lon,v_mapping,k_mapping,building_mapping):
    reference_point = f"ST_Transform(ST_SetSRID(ST_MakePoint({lon}, {lat}), 4326), 3857)"
    
    polygon_query = f"""
    SELECT 
        *,
        ST_AsText(way) AS geometry
    FROM planet_osm_polygon
    WHERE ST_Contains(way, {reference_point})
    and way_area < 1000000
    order by way_area asc
    limit 1
    """

        
    cursor.execute(polygon_query)
    polygon_results = cursor.fetchall()
    if len(polygon_results)==0:
        # Return three None values to match expected unpacking
        return None, None, None
    else:
        non_null_pairs = {k: v for k, v in polygon_results[0].items() if v is not None and k not in ['osm_id','geometry','way','z_order']}
        
        suggest=[]
        for k,v in non_null_pairs.items():
            if k in k_mapping:
                if k == 'building' and v in building_mapping:
                    suggest.extend(building_mapping[v])
                else:
                    suggest.extend(k_mapping[k])
            if v in v_mapping:
                suggest.extend(v_mapping[v])
            # Special handling for building key
        count={}
        for i in suggest:
            count[i]=count.get(i,0)+1
        confidence={}
        for k,v in count.items():
            confidence[k]=v/len(suggest)
        
        # Ensure we always return 3 values
        coordinates = extract_coordinates(polygon_results[0]['geometry']) if 'geometry' in polygon_results[0] else None
        return confidence, non_null_pairs, coordinates
def get_road(lat,lon):
    reference_point = f"ST_Transform(ST_SetSRID(ST_MakePoint({lon}, {lat}), 4326), 3857)"
    line_query = f"""
    SELECT 
        *,ST_AsText(way) AS geometry,
        ST_Distance(way, {reference_point}) as distance
    FROM planet_osm_line
    WHERE highway IS NOT NULL
    AND ST_Intersects(way, ST_Buffer({reference_point}, 30))
    order by distance
    limit 1
    """

    # Execute the query
    cursor.execute(line_query)
    line_results = cursor.fetchall()
    if len(line_results)==0:
        return None,None,None
    non_null_pairs = {k: v for k, v in line_results[0].items() if v is not None and k not in ['geometry','way','z_order']}
    return {"path":(100-line_results[0]['distance'])/100},non_null_pairs,extract_coordinates(line_results[0]['geometry'])
# Define function to calculate combined scores
def calculate_combined_scores(row):
    # Extract scores from the row
    envidence_score = row['evidence_score']
    poi_score = row['poi_score']
    
    # Initialize merged dictionary with envidence scores
    merged = {}
    for category in envidence_score:
        merged[category] = envidence_score[category]
    
    # If poi_score exists, enhance the corresponding categories in merged
    if poi_score:
        for category in poi_score:
            if category in merged:
                # Weighted combination: 60% evidence score, 40% POI score
                merged[category] = merged[category] * 0.5 + (poi_score[category]) * 0.5
            else:
                merged[category] = (poi_score[category])
    
    # Find highest scoring category
    if merged:
        highest_category = max(merged.items(), key=lambda x: x[1])
        category = highest_category[0]
        confidence = highest_category[1]
    else:
        category = ''
        confidence = 0
    
    # Create result dictionary with final category and confidence
    result = {
        'category': category,
        'confidence': confidence
    }
    
    # Add all category scores (home, work, leisure, path)
    for category in ['home', 'work', 'leisure', 'path']:
        result[category] = merged.get(category, 0.0)
    
    return pd.Series(result)

import numpy as np
from envidence import *
import timezonefinder as tf
tz=tf.TimezoneFinder()

def load_maid(maid_file):
    store=EvidenceStore()
    store.load_from_pickle(maid_file)
    geo = []
    d = store.store
    maid_total_pings=0
    for i in d.keys():
        dr=store.derive(i)
        maid_total_pings+=dr['level_1_primary']['pings']
    for i in d.keys():
        
        dr=store.derive(i)
        meta,l1,l2,l4=dr['meta'],dr['level_1_primary'],dr['level_2_secondary'],dr['level_4_duration']
        l5=dr['level_5_flux']
        s=store.overall_score(dr)
        days=pd.DataFrame({'unique_days':list(d[i]['unique_days'])})
        days['month']=pd.to_datetime(days['unique_days']).dt.month
        days['week']=pd.to_datetime(days['unique_days']).dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        # Count occurrences for each day of the week
        week_counts = days.groupby(['week']).size().reindex(day_order, fill_value=0).to_dict()
        
        # Count occurrences for each month (1-12)
        month_counts = days.groupby(['month']).size().reindex(range(1, 13), fill_value=0).to_dict()
        
        # Create entry with all the data
        entry = {
            'geohash': i,
            'lat': meta['mean_coordinate'][0],
            'lon': meta['mean_coordinate'][1],
            'first_seen': meta['first_seen'],
            'last_seen': meta['last_seen'],
            'est_duration': l4['est_duration'],
            'pings': l1['pings'],
            'unique_days': l1['unique_days'],
            'entropy_hour_norm': round(l2['entropy_hour_norm'], 2),
            'evidence_score': s,
            'maid_total_pings': maid_total_pings,
            'fluxB': l5['flux_counts']['B'],
            'fluxC': l5['flux_counts']['C'],
            'fluxD': l5['flux_counts']['D'],
            'fluxE': l5['flux_counts']['E'],
            'fluxF': l5['flux_counts']['F']
        }
        
        # Add day of week columns
        for day in day_order:
            entry[f'{day.lower()}'] = week_counts[day]
            
        # Add month columns
        for month in range(1, 13):
            entry[f'month_{month}'] = month_counts[month]
            
        # Add hour columns (0-23)
        for hour in range(24):
            entry[f'hour_{hour}'] = d[i]['hourly_hist'][hour]
            
        geo.append(entry)
    return pd.DataFrame(geo)    

def process_maid_data(maid):
    # Load data for the MAID
    pdf = load_maid(maid)
    
    # Skip processing if dataframe is empty
    if pdf.empty:
        return pd.DataFrame()
    
    # Calculate quantiles for filtering stationary vs movement points
    q_base = 0.9
    q = min(q_base + np.log2(len(pdf))/100, 0.99)
    est_duration = pdf.est_duration.quantile(q)
    pings = pdf.pings.quantile(q)
    
    # Split data into stationary and movement points
    is_stationary = (pdf.est_duration >= est_duration) & (pdf.pings >= pings)
    stationary = pdf[is_stationary].reset_index(drop=True)
    movement = pdf[~is_stationary].reset_index(drop=True)
    
    # Process POI information based on whether we have both stationary and movement data
    if stationary.empty and movement.empty:
        # No data available
        return pd.DataFrame()
    
    elif stationary.empty:
        # Only movement data available
        movement[['poi_score', 'poi_info', 'poi_coordinates']] = movement.apply(
            lambda row: pd.Series(get_road(row['lat'], row['lon'])),
            axis=1
        )
        movement.dropna(inplace=True)
        all_pdf = movement
    
    elif movement.empty:
        # Only stationary data available
        # stationary[['poi_score', 'poi_info', 'poi_coordinates']] = stationary.apply(
        #     lambda row: pd.Series(get_poi_category(row['lat'], row['lon'], v_mapping, k_mapping,building_mapping)),
        #     axis=1
        # )
        # all_pdf = stationary
        pass
    
    else:
        # # Both stationary and movement data available
        # stationary[['poi_score', 'poi_info', 'poi_coordinates']] = stationary.apply(
        #     lambda row: pd.Series(get_poi_category(row['lat'], row['lon'], v_mapping, k_mapping,building_mapping)),
        #     axis=1
        # )
        movement[['poi_score', 'poi_info', 'poi_coordinates']] = movement.apply(
            lambda row: pd.Series(get_road(row['lat'], row['lon'])),
            axis=1
        )
        movement.dropna(inplace=True)
        all_pdf = pd.concat([stationary, movement], ignore_index=True)
    
    # Calculate combined scores and add to dataframe
    if all_pdf.empty:
        return pd.DataFrame()
        
    score_columns = all_pdf.apply(calculate_combined_scores, axis=1)
    final_pdf = pd.concat([all_pdf, score_columns], axis=1)
    final_pdf.reset_index(drop=True, inplace=True)
    
    return final_pdf