# evidence_store_no_poi.py
import datetime as dt
import math
import pickle
import gzip
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
import pandas as pd
import duckdb
import pygeohash as pgh
from collections import defaultdict
import json

conn=duckdb.connect()

def _to_dt(x: Union[str, dt.datetime]) -> dt.datetime:
    return dt.datetime.fromisoformat(x) if isinstance(x, str) else x

class EvidenceStore:
    def __init__(self):
        self.store: Dict[str, dict] = {}

    def _init(self):
        return {
            'pings': 0,
            'first_seen_ts': None,
            'last_seen_ts': None,
            'unique_days': set(),
            'hourly_hist': [0]*24,
            'weekday_hist': [0]*7,
            'hourly_hist_weekday': [0]*24,
            'hourly_hist_weekend': [0]*24,
            'monthly_hist': {},
            'daily_flags': {},
            'max_seen_date': None,
            'gap_bins': {'0d':0, '1-3d':0, '4-7d':0, '8-30d':0, '>30d':0},
            'poi_info': None,
            'poi_calculated': False,
            'mean_lat': None,
            'mean_lon': None,
            'coord_count': 0,
            # Simplified duration tracking
            'hourly_minutes': {}, # Format: {hour: {'min': min_minute, 'max': max_minute}}
            'est_duration': 0,
            # Flux tracking
            'flux_counts': {'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0},
        }
    
    def _update_mean_coordinates(self, c: dict, lat: float, lon: float):
        """Update running mean of coordinates using incremental formula"""
        if c['mean_lat'] is None or c['mean_lon'] is None:
            c['mean_lat'] = lat
            c['mean_lon'] = lon
            c['coord_count'] = 1
        else:
            n = c['coord_count']
            c['mean_lat'] = (c['mean_lat'] * n + lat) / (n + 1)
            c['mean_lon'] = (c['mean_lon'] * n + lon) / (n + 1)
            c['coord_count'] += 1

    @staticmethod
    def _shrink_ratio(ratio: float, n: int, p0: float, a: float = 8.0) -> float:
        return (ratio * n + a * p0) / (n + a) if (n + a) > 0 else p0

    @staticmethod
    def _mask_for(ts: dt.datetime) -> int:
        h = ts.hour
        wd = ts.weekday()
        mask = 0
        if 4 <= h <= 6:
            mask |= 0x1
        if 20 <= h <= 23:
            mask |= 0x2
        if (9 <= h <= 17) and (wd < 5):
            mask |= 0x4
        if (h >= 22) or (h <= 5):
            mask |= 0x8
        return mask

    def update(self, new_data: Dict[str, List[Union[str, dt.datetime]]], 
               coordinates: Optional[Dict[str, Tuple[float, float]]] = None,
               flux_data: Optional[Dict[str, List[str]]] = None):
        """Update evidence store with timestamp data and simplified duration calculation"""
        for gh, ts_list in (new_data or {}).items():
            if not ts_list: 
                continue
            ts_objs = [_to_dt(x) for x in ts_list]
            ts_objs.sort()
            c = self.store.get(gh)
            if c is None:
                c = self._init()
                self.store[gh] = c

            # Update mean coordinates if provided
            if coordinates and gh in coordinates:
                lat, lon = coordinates[gh]
                self._update_mean_coordinates(c, lat, lon)

            # Update flux counts if provided
            if flux_data and gh in flux_data:
                for flux_type in flux_data[gh]:
                    if flux_type in c['flux_counts']:
                        c['flux_counts'][flux_type] += 1

            # Initialize or reset hourly_minutes for this update
            hourly_minutes = {}
            
            for ts in ts_objs:
                d = ts.date()
                h = ts.hour
                minute = ts.minute
                wd = ts.weekday()
                
                # Track min/max minute for each hour
                if h not in hourly_minutes:
                    hourly_minutes[h] = {'min': minute, 'max': minute}
                else:
                    hourly_minutes[h]['min'] = min(hourly_minutes[h]['min'], minute)
                    hourly_minutes[h]['max'] = max(hourly_minutes[h]['max'], minute)
                
                # Update other tracking fields
                c['first_seen_ts'] = ts if c['first_seen_ts'] is None or ts < c['first_seen_ts'] else c['first_seen_ts']
                c['last_seen_ts']  = ts if c['last_seen_ts']  is None or ts > c['last_seen_ts']  else c['last_seen_ts']
                c['pings'] += 1
                c['unique_days'].add(d)
                c['hourly_hist'][h] += 1
                c['weekday_hist'][wd] += 1
                if wd >= 5:
                    c['hourly_hist_weekend'][h] += 1
                else:
                    c['hourly_hist_weekday'][h] += 1
                mkey = f"{ts.year:04d}-{ts.month:02d}"
                c['monthly_hist'][mkey] = c['monthly_hist'].get(mkey, 0) + 1
                ord_key = d.toordinal()
                c['daily_flags'][ord_key] = c['daily_flags'].get(ord_key, 0) | self._mask_for(ts)
                if c['max_seen_date'] is None or d > c['max_seen_date']:
                    if c['max_seen_date'] is not None:
                        delta = (d - c['max_seen_date']).days
                        if delta == 0:
                            c['gap_bins']['0d'] += 1
                        elif 1 <= delta <= 3:
                            c['gap_bins']['1-3d'] += 1
                        elif 4 <= delta <= 7:
                            c['gap_bins']['4-7d'] += 1
                        elif 8 <= delta <= 30:
                            c['gap_bins']['8-30d'] += 1
                        else:
                            c['gap_bins']['>30d'] += 1
                    c['max_seen_date'] = d
            
            # Merge hourly_minutes from this update into stored data
            if not c.get('hourly_minutes'):
                c['hourly_minutes'] = {}
                
            for hour, minmax in hourly_minutes.items():
                if hour not in c['hourly_minutes']:
                    c['hourly_minutes'][hour] = minmax
                else:
                    c['hourly_minutes'][hour]['min'] = min(c['hourly_minutes'][hour]['min'], minmax['min'])
                    c['hourly_minutes'][hour]['max'] = max(c['hourly_minutes'][hour]['max'], minmax['max'])
            
            # Always recalculate estimated duration from the complete hourly_minutes
            # This ensures consistency regardless of batch size
            total_duration = 0
            for hour, minmax in c['hourly_minutes'].items():
                duration = minmax['max'] - minmax['min']
                total_duration += duration
            
            c['est_duration'] = total_duration

    def save_to_pickle(self, filepath: str, compress: bool = False) -> None:
        """Save EvidenceStore to pickle format for fast I/O
        
        Args:
            filepath: Path to save the pickle file
            compress: Whether to use gzip compression (default: False for <1MB files)
        """
        try:
            # Prepare data for serialization
            data = {
                'store': self.store,
                'version': '1.0',  # For future compatibility
                'created_at': dt.datetime.now().isoformat()
            }
            
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            if compress:
                # Add .gz extension if not present
                if not filepath.endswith('.gz'):
                    filepath += '.gz'
                    
                with gzip.open(filepath, 'wb', compresslevel=9) as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                # Add .pkl extension if not present and no extension exists
                if not any(filepath.endswith(ext) for ext in ['.pkl', '.pickle']):
                    filepath += '.pkl'
                    
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    
            
        except Exception as e:
            raise
    
    def load_from_pickle(self, filepath: str) -> None:
        """Load EvidenceStore from pickle format
        
        Args:
            filepath: Path to the pickle file
        """
        try:
            # Auto-detect compression based on file extension
            if filepath.endswith('.gz'):
                with gzip.open(filepath, 'rb') as f:
                    data = pickle.load(f)
            elif any(filepath.endswith(ext) for ext in ['.pkl', '.pickle']):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            else:
                # Try uncompressed first, fallback to compressed
                try:
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                except (pickle.UnpicklingError, UnicodeDecodeError):
                    with gzip.open(filepath, 'rb') as f:
                        data = pickle.load(f)
            
            # Load store data
            if 'store' in data:
                self.store = data['store']
            else:
                # Legacy format support (direct store)
                self.store = data
                
            # Initialize flux_counts for older data formats
            for gh, c in self.store.items():
                if 'flux_counts' not in c:
                    c['flux_counts'] = {'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0}
                
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            raise
        except Exception as e:
            print(f"Error loading EvidenceStore from {filepath}: {e}")
            raise
    
    def save(self, filepath: str, compress: bool = False) -> None:
        """Convenience method to save the store
        
        Args:
            filepath: Path to save the file
            compress: Whether to use compression (default: False for fast I/O)
        """
        self.save_to_pickle(filepath, compress=compress)
    
    def load(self, filepath: str) -> None:
        """Convenience method to load the store
        
        Args:
            filepath: Path to load from
        """
        self.load_from_pickle(filepath)
    
    def clear_store(self) -> None:
        """Clear all evidence data"""
        self.store.clear()
        
    def recalculate_durations(self) -> None:
        """Recalculate all duration estimates in the store
        
        This can be useful when loading older data formats or ensuring
        consistency across different processing batches.
        """
        for gh, data in self.store.items():
            if 'hourly_minutes' in data:
                total_duration = 0
                for hour, minmax in data['hourly_minutes'].items():
                    duration = minmax['max'] - minmax['min']
                    total_duration += duration
                data['est_duration'] = total_duration

    def derive(self, gh: str) -> dict:
        if gh not in self.store: 
            return None
        c = self.store[gh]
        visits = c['pings']
        if visits == 0:
            return None

        first, last = c['first_seen_ts'], c['last_seen_ts']
        # Convert string timestamps to datetime objects if needed
        if isinstance(first, str):
            first = _to_dt(first)
        if isinstance(last, str):
            last = _to_dt(last)
        
        span_days = (last.date() - first.date()).days + 1 if first and last else 0
        unique_days = len(c['unique_days'])
        
        # Improved active_day_ratio with span capping and continuity boost
        capped_span = max(30, span_days)
        base_active_ratio = min(1.0, unique_days / capped_span) if capped_span > 0 else 0.0
        
        # Calculate continuity boost from gap patterns
        total_gaps = max(1, sum(c['gap_bins'].values()))
        continuity = (c['gap_bins'].get('0d', 0) + c['gap_bins'].get('1-3d', 0)) / total_gaps
        active_day_ratio = base_active_ratio * (0.5 + 0.5 * continuity)

        # visit-based ratios
        night_idxs = [22, 23, 0, 1, 2, 3, 4, 5]
        night_ratio = sum(c['hourly_hist'][i] for i in night_idxs) / visits
        weekday_day_ratio = sum(c['hourly_hist_weekday'][i] for i in range(9, 18)) / visits
        weekend_ratio = (c['weekday_hist'][5] + c['weekday_hist'][6]) / visits
        midday_weekday_ratio = sum(c['hourly_hist_weekday'][i] for i in range(11, 15)) / visits
        
        # Add evening_ratio for leisure scoring
        evening_ratio = sum(c['hourly_hist'][i] for i in range(18, 22)) / visits

        # day-level ratios from flags (presence-only)
        flags = list(c['daily_flags'].values())
        days_night = sum(1 for m in flags if (m & 0x8))
        days_weekday_work = sum(1 for m in flags if (m & 0x4))
        days_late_evening = sum(1 for m in flags if (m & 0x2))
        days_early_morning = sum(1 for m in flags if (m & 0x1))
        if unique_days > 0:
            night_days_ratio = days_night / unique_days
            weekday_work_days_ratio = days_weekday_work / unique_days
            late_evening_days_ratio = days_late_evening / unique_days
            early_morning_days_ratio = days_early_morning / unique_days
        else:
            night_days_ratio = weekday_work_days_ratio = late_evening_days_ratio = early_morning_days_ratio = 0.0

        # entropy of hourly distribution (normalized)
        H = 0.0
        for cnt in c['hourly_hist']:
            if cnt > 0:
                p = cnt / visits
                H -= p * math.log(p + 1e-12)
        entropy_hour_norm = H / math.log(24)

        # monthly stability via CV -> 1/(1+CV)
        months = list(c['monthly_hist'].values())
        if len(months) >= 2:
            mean_m = sum(months) / len(months)
            var_m = sum((x - mean_m) ** 2 for x in months) / len(months)
            std_m = math.sqrt(var_m)
            cv = (std_m / mean_m) if mean_m > 0 else 0.0
        else:
            cv = 0.0
        monthly_stability = 1.0 / (1.0 + cv)

        # recency: active days in last 30 days
        last_date = last.date()
        last30_cut = last_date.toordinal() - 30
        # Convert string keys to integers for comparison
        active_days_last_30d = sum(1 for d_ord_str in c['daily_flags'].keys() if int(d_ord_str) >= last30_cut)

        # Get POI info from cached data
        poi_info = c.get('poi_info', None)
        poi_available = c.get('poi_calculated', False) and poi_info is not None

        # Simple duration info from the simplified calculation
        duration_info = {
            'est_duration': c.get('est_duration', 0),
            'hourly_minutes': c.get('hourly_minutes', {})
        }
            
        return {
            'meta': {
                'first_seen': first.isoformat() if first else None,
                'last_seen': last.isoformat() if last else None,
                'span_days': span_days,
                'mean_coordinate': [c.get('mean_lat', None), c.get('mean_lon', None)] if c.get('mean_lat') is not None else None,
            },
            'level_1_primary': {
                'pings': visits,
                'unique_days': unique_days,
                'active_day_ratio': active_day_ratio,
                'gap_bins': dict(c['gap_bins']),
            },
            'level_2_secondary': {
                'hourly_hist': list(c['hourly_hist']),
                'weekday_hist': list(c['weekday_hist']),
                'monthly_hist': dict(c['monthly_hist']),
                'night_ratio': night_ratio,
                'weekday_day_ratio': weekday_day_ratio,
                'weekend_ratio': weekend_ratio,
                'midday_weekday_ratio': midday_weekday_ratio,
                'evening_ratio': evening_ratio,
                'early_late_overlap_day_ratio': late_evening_days_ratio,  # kept for backward compat
                'night_days_ratio': night_days_ratio,
                'weekday_work_days_ratio': weekday_work_days_ratio,
                'late_evening_days_ratio': late_evening_days_ratio,
                'early_morning_days_ratio': early_morning_days_ratio,
                'entropy_hour_norm': entropy_hour_norm,
                'monthly_stability': monthly_stability,
                'active_days_last_30d': active_days_last_30d,
            },
            'level_3_tertiary': {
                "poi_available": poi_available,
                "poi_info": poi_info
            },
            'level_4_duration': duration_info,
            'level_5_flux': {
                'flux_counts': dict(c.get('flux_counts', {'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0}))
            }
        }
    
    @staticmethod
    def score_home(ev: dict, a: float = 2.0) -> float:
        l1, l2 = ev['level_1_primary'], ev['level_2_secondary']
        visits = l1['pings']
        days = l1['unique_days']

        # Bayesian shrinkage priors
        p0_night = 8.0 / 24.0            # baseline night share
        night_ratio_shrunk = EvidenceStore._shrink_ratio(l2['night_ratio'], visits, p0_night, a=a)

        # sample-size weights
        w_visits = 1.0 - math.exp(-visits / 5.0)
        w_days = 1.0 - math.exp(-days / 3.0)    

        base = (
            0.375 * l2['night_days_ratio']
          + 0.10 * night_ratio_shrunk
          + 0.15 * l2['late_evening_days_ratio']
          + 0.10 * l2['early_morning_days_ratio']  # Added early morning evidence
          + 0.075 * (1.0 - l2['entropy_hour_norm'])
          + 0.25 * l1['active_day_ratio']  # Reduced from 0.30 to accommodate new features
          + 0.05 * l2['monthly_stability']  # Added monthly stability
        )
        s = base * w_visits * w_days

        # recency boost
        s *= min(1.0, 0.5 + 0.5 * (l2['active_days_last_30d'] / 10))
        return max(0.0, min(1.0, s))

    @staticmethod
    def score_work(ev: dict, a: float = 2.0) -> float:
        l1, l2 = ev['level_1_primary'], ev['level_2_secondary']
        visits = l1['pings']
        days = l1['unique_days']

        # Bayesian shrinkage priors
        p0_wd_day = 45.0 / 168.0         # (5 weekdays * 9 hours) / (7 days * 24 hours)
        weekday_day_ratio_shrunk = EvidenceStore._shrink_ratio(l2['weekday_day_ratio'], visits, p0_wd_day, a=a)

        # sample-size weights
        w_visits = 1.0 - math.exp(-visits / 5.0)
        w_days = 1.0 - math.exp(-days / 3.0)    

        base = (
            0.425 * l2['weekday_work_days_ratio']
          + 0.15 * weekday_day_ratio_shrunk
          + 0.10 * l2['midday_weekday_ratio']
          + 0.075 * (1.0 - l2['entropy_hour_norm'])
          + 0.20 * l1['active_day_ratio']  # Reduced from 0.25 to accommodate new features
          + 0.05 * l2['monthly_stability']  # Added monthly stability
        )
        s = base * w_visits * w_days

        # recency boost
        s *= min(1.0, 0.5 + 0.5 * (l2['active_days_last_30d'] / 10))
        return max(0.0, min(1.0, s))

    @staticmethod
    def score_leisure(ev: dict, a: float = 2.0) -> float:
        l1, l2 = ev['level_1_primary'], ev['level_2_secondary']
        visits = l1['pings']
        days = l1['unique_days']

        # Calculate home and work scores to invert them for leisure
        home_score = EvidenceStore.score_home(ev, a)
        work_score = EvidenceStore.score_work(ev, a)
        
        # Combine and invert home and work patterns
        combined_home_work = (home_score + work_score) / 2
        inverse_pattern = 1.0 - combined_home_work
        
        # Bayesian shrinkage priors
        p0_weekend = 2.0 / 7.0           # baseline weekend share
        weekend_ratio_shrunk = EvidenceStore._shrink_ratio(l2['weekend_ratio'], visits, p0_weekend, a=a)
        
        # Add evening_ratio (18-21h) with shrinkage
        p0_evening = 4.0 / 24.0
        evening_ratio_shrunk = EvidenceStore._shrink_ratio(l2['evening_ratio'], visits, p0_evening, a=a)
    
    

        # sample-size weights
        w_visits = 1.0 - math.exp(-visits / 5.0)
        w_days = 1.0 - math.exp(-days / 3.0)    
        
        base = (
            0.25 * weekend_ratio_shrunk
          + 0.20 * evening_ratio_shrunk
          + 0.15 * (1-l2['entropy_hour_norm'])
          + 0.10 * (1.0 - l2['monthly_stability'])
          + 0.30 * inverse_pattern  # Add inverse pattern from home and work
        )
        s = base * w_visits * w_days

        # Relaxed recency boost (leisure may not be recent frequently)
        s *= min(1.0, 0.5 + 0.5 * (l2['active_days_last_30d'] / 15))
        return max(0.0, min(1.0, s))
    
    def overall_score(self, ev: dict, a: float = 2.0) -> dict:
        home_score = self.score_home(ev, a)
        work_score = self.score_work(ev, a)
        leisure_score = self.score_leisure(ev, a)
        overall_score = {
            'home': home_score,
            'work': work_score,
            'leisure': leisure_score
        }
        if ev['level_3_tertiary']['poi_available']:
            poi_info = ev['level_3_tertiary']['poi_info']
            if poi_info:
                poi_label = poi_info['primary_category']
                if poi_label != 'path':
                    if poi_label in overall_score:
                        overall_score[poi_label] = 0.25 * overall_score[poi_label] + 0.75 * poi_info.get('confidence', 0.01) / 100.0
                elif poi_label == 'path':
                    overall_score[poi_label] = poi_info.get('confidence', 0.01) / 100.0

        return overall_score