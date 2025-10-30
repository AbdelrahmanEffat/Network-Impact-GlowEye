# main_API.py (updated)
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Literal, Dict, Any, List
from backend.core.analyzer import UnifiedNetworkImpactAnalyzer
import pandas as pd
import numpy as np
import logging
import io
import zipfile
import json
import time
from functools import lru_cache
import threading
# redis update
from backend.api.redis_utils import RedisDataManager
from backend.api.config import RedisConfig, get_production_config
import os
from dotenv import load_dotenv
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize Redis manager with environment variables
try:
    redis_manager = RedisDataManager(config=get_production_config())
except Exception as e:
    logger.error(f"Failed to initialize Redis manager: {str(e)}")
    redis_manager = None


app = FastAPI(
    title="Network Impact Analysis API",
    description="API for analyzing network impact from node or exchange failures",
    version="2.0.0"
)

# Request model
class AnalysisRequest(BaseModel):
    identifier: str
    identifier_type: Optional[Literal['node', 'exchange', 'auto']] = 'auto'

# Response models
class AnalysisResponse(BaseModel):
    status: str
    message: str
    total_records: int
    unique_msans: int
    analysis_type: str
    execution_time_seconds: float
    results_preview: dict
    impact_summary: dict

# Global analyzer instances and cached base results
we_analyzer = None
others_analyzer = None
we_base_results = None
others_base_results = None
# Global lock for thread-safe operations
analyzer_lock = threading.Lock()
# download update
latest_we_results = None
latest_others_results = None
latest_mobile_data = None 
latest_identifier = None
latest_identifier_type = None

@app.on_event("startup")
async def startup_event():
    """Initialize the analyzers with CSV files on startup and precompute base results"""
    global we_analyzer, others_analyzer, we_base_results, others_base_results
    
    with analyzer_lock:
        try:
            env_path = Path(__file__).resolve().parents[1] / 'secrets.env'
            load_dotenv(dotenv_path=env_path)
            #print("directory", Path(__file__).resolve())
            # Convert to Path object
            data_path = os.getenv('DATA_PATH')
            value = os.getenv('PRODUCTION')
            #print(f"Value: '{value}', Path: {data_path}")

            logger.info(f"Loading CSV files from: {data_path}")

            if value=='false':
                df_report_we = pd.read_csv(f'{data_path}\\we_igw.csv', low_memory=False)  # WE data
                df_report_others = pd.read_csv(f'{data_path}\\others.csv', low_memory=False)  # Others data
                df_res_ospf = pd.read_csv(f'{data_path}\\res_ospf.csv', low_memory=False)
                df_wan = pd.read_csv(f'{data_path}\\wan.csv', low_memory=False)
                df_agg = pd.read_csv(f'{data_path}\\agg.csv', low_memory=False)
                df_noms = pd.read_csv(f'{data_path}\\NOMS.csv', low_memory=False)
                df_mobile = pd.read_csv(f'{data_path}\\enodbs.csv', low_memory=False)
                #print(df_report_we[df_report_we['MSANCODE']=='MSAN12345'])
            else:
                # Load CSV files using the path from env
                # df_report_we = pd.read_csv(f'{data_path}/we_igw.csv', low_memory=False)  # WE data
                # df_report_others = pd.read_csv(f'{data_path}/others.csv', low_memory=False)  # Others data
                # df_res_ospf = pd.read_csv(f'{data_path}/res_ospf.csv', low_memory=False)
                # df_wan = pd.read_csv(f'{data_path}/wan.csv', low_memory=False)
                # df_agg = pd.read_csv(f'{data_path}/agg.csv', low_memory=False)
                # df_noms = pd.read_csv(f'{data_path}/NOMS.csv', low_memory=False)
                # df_mobile = pd.read_csv(f'{data_path}/enodbs.csv', low_memory=False)
                logger.info("Loading data from Redis cache with date-based keys...")

                # Check Redis connection
                if not redis_manager.health_check():
                    raise Exception("Redis connection failed")
                
                # Load data from Redis using date-based keys
                df_report_we = redis_manager.get_dataframe("we")
                df_report_others = redis_manager.get_dataframe("others")
                df_res_ospf = redis_manager.get_dataframe("res_ospf")
                df_wan = redis_manager.get_dataframe("wanData")
                df_agg = redis_manager.get_dataframe("agg")
                df_noms = redis_manager.get_dataframe('NOMS')
                df_mobile = redis_manager.get_dataframe('enodbs')
                
                # Log which keys we're using
                logger.info(f"Using Redis keys: we={redis_manager.get_latest_key('we')}, "
                        f"others={redis_manager.get_latest_key('others')}")
                
                # Validate that all data was loaded
                if any(df is None for df in [df_report_we, df_report_others, df_res_ospf, df_wan, df_agg]):
                    missing = []
                    if df_report_we is None: missing.append("we")
                    if df_report_others is None: missing.append("others")
                    if df_res_ospf is None: missing.append("res_ospf")
                    if df_wan is None: missing.append("wan")
                    if df_agg is None: missing.append("agg")
                    if df_noms is None: missing.append("NOMS")
                    if df_mobile is None: missing.append("enodbs")
                    raise Exception(f"Failed to load data from Redis for keys: {missing}")

            ## maping columns names
            df_report_others.columns = df_report_others.columns.str.upper()
            df_report_we.columns = df_report_we.columns.str.upper()

            df_report_others.rename(columns={'BITSTREAM_EXCHANGE':'Bitstream_exchange', 'EDGE_EXCHANGE':'EDGE_exchange'}, inplace=True)
            df_report_we.rename(columns={'DISTRIBUTION_EMS':'distribution_ems', 'DISTRIBUTION_EXCHANGE':'distribution_Exchange',
                                        'DISTRIBUTION_HOSTNAME':'distribution_hostname', 'DISTRIBUTION_INT':'distribution_INT',
                                        'DISTRIBUTION_IP':'distribution_IP', 'DISTRIBUTION_NAME_EX':'distribution_name_ex',
                                        'EDGE_EXCHANGE':'edge_exchange', 'EDGE_NAME_EX':'edge_name_ex',
                                        'EDGE_PORT':'edge_port'}, inplace=True)
            
            # Initialize analyzers for both data types
            we_analyzer = UnifiedNetworkImpactAnalyzer(df_report_we, df_res_ospf, df_wan, df_agg, df_noms, df_mobile)
            others_analyzer = UnifiedNetworkImpactAnalyzer(df_report_others, df_res_ospf, df_wan, df_agg, df_noms, df_mobile)
            
            # Preprocess data and generate base results
            logger.info("Preprocessing data and generating base results...")
            we_analyzer.preprocess_data()
            others_analyzer.preprocess_data()
            
            # Generate base results with a dummy identifier (we'll use actual identifier for impact analysis)
            we_base_results = we_analyzer.generate_base_results("dummy_initialization")
            others_base_results = others_analyzer.generate_base_results("dummy_initialization")
            
            #logger.info(f"Data loaded successfully. WE shape: {df_report_we.shape}, Others shape: {df_report_others.shape}")
            logger.info(f"Base results computed. WE: {we_base_results.shape}, Others: {others_base_results.shape}, Mobile: {df_mobile.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Network Impact Analysis API", 
        "version": "2.0.0",
        "endpoints": {
            "/analyze": "POST - Analyze network impact for both WE and Others",
            "/analyze/csv": "POST - Download analysis results as CSV",
            "/analyze/complete": "POST - Get detailed analysis results",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if we_analyzer is None or others_analyzer is None:
        raise HTTPException(status_code=503, detail="Service not ready - analyzers not initialized")
    
    return {
        "status": "healthy", 
        "we_analyzer_ready": we_analyzer is not None, 
        "others_analyzer_ready": others_analyzer is not None,
        "base_results_computed": we_base_results is not None and others_base_results is not None
    }


@app.post("/analyze/complete")
async def analyze_network_impact_complete(request: AnalysisRequest):
    """Single endpoint that returns both summary and detailed results - UPDATED WITH MOBILE DATA"""
    global latest_we_results, latest_others_results, latest_identifier, latest_identifier_type, latest_mobile_data
    
    if we_analyzer is None or others_analyzer is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        logger.info(f"Starting complete analysis for {request.identifier}")
        start_time = time.time()
        
        # Clear caches before new analysis
        if hasattr(we_analyzer, '_exchange_nodes_cache'):
            we_analyzer._exchange_nodes_cache.clear()
        if hasattr(others_analyzer, '_exchange_nodes_cache'):
            others_analyzer._exchange_nodes_cache.clear()
        
        # Use precomputed base results
        we_analyzer.final_df = we_base_results
        others_analyzer.final_df = others_base_results
        
        # Run impact analysis once
        if request.identifier_type == 'exchange' or (request.identifier_type == 'auto' and _is_exchange_identifier(request.identifier)):
            we_results = we_analyzer.analyze_exchange_impact(request.identifier)
            others_results = others_analyzer.analyze_exchange_impact(request.identifier)
            analysis_type = "Exchange"
        else:
            we_results = we_analyzer.analyze_node_impact(request.identifier)
            others_results = others_analyzer.analyze_node_impact(request.identifier)
            analysis_type = "Node"
        

        # Combine mobile data and remove duplicates based on CINUM
        mobile_data = we_analyzer._process_mobile_data(request.identifier)#.drop_duplicates(subset=['CINUM'])
        
        execution_time = time.time() - start_time
        
        # FIX: Ensure Path2 column exists for both datasets
        if not we_results.empty and 'Path2' not in we_results.columns:
            we_results['Path2'] = None
        if not others_results.empty and 'Path2' not in others_results.columns:
            others_results['Path2'] = None
        
        # Apply MSAN-level Route_Status calculation for WE data
        if not we_results.empty:
            we_results = we_analyzer._calculate_msan_level_route_status(we_results)

        # For Others data, use individual Route_Status calculation
        if not others_results.empty:
            others_results['Route_Status'] = others_results.apply(
                lambda row: others_analyzer._calculate_route_status_individual(
                    row['Path'], 
                    row['Path2'] if 'Path2' in row else None,
                    row.get('STATUS', 'UP'), 
                    row['Impact']
                ),
                axis=1
            )
        
        # STORE THE COMPUTED RESULTS FOR DOWNLOAD
        latest_we_results = we_results.copy()
        latest_others_results = others_results.copy()
        latest_mobile_data = mobile_data.copy()  # Store mobile data for download
        latest_identifier = request.identifier
        latest_identifier_type = request.identifier_type
        
        # Create summaries
        we_impact_summary = _create_impact_summary(we_results)
        others_impact_summary = _create_impact_summary(others_results)
        
        # Convert to serializable format
        we_results_data = convert_dataframe_to_serializable(we_results)
        others_results_data = convert_dataframe_to_serializable(others_results)
        
        # Remove duplicates for dashboard
        others_dash = others_results.drop_duplicates(subset=['MSANCODE', 'EDGE', 'BITSTREAM_HOSTNAME'])
        others_dash_data = convert_dataframe_to_serializable(others_dash)
        
        # Calculate statistics
        we_stats = _calculate_we_statistics(we_results)
        others_stats = _calculate_others_statistics(others_results)
        
        # Add mobile data count to statistics
        mobile_count = len(mobile_data)
        
        # Ensure statistics are also serializable
        we_stats_serializable = ensure_serializable(we_stats)
        others_stats_serializable = ensure_serializable(others_stats)
        
        result = {
            "status": "success",
            "message": f"Analysis completed successfully for {request.identifier}",
            "summary": {
                "total_records": we_impact_summary.get("total_records", 0) + others_impact_summary.get("total_records", 0),
                "unique_msans": we_impact_summary.get("unique_msans", 0) + others_impact_summary.get("unique_msans", 0),
                "mobile_cinum_count": mobile_count,  # Add mobile count to summary
                "analysis_type": analysis_type,
                "execution_time_seconds": round(execution_time, 3),
                "impact_summary": {
                    "we": ensure_serializable(we_impact_summary),
                    "others": ensure_serializable(others_impact_summary)
                }
            },
            "data": {
                "we_results": we_results_data,
                "others_results": others_results_data,
                "others_dash": others_dash_data
            },
            "statistics": {
                "we_stats": we_stats_serializable,
                "others_stats": others_stats_serializable
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Complete analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Complete analysis failed: {str(e)}")


@app.post("/analyze/csv")
async def analyze_and_return_csv(request: AnalysisRequest):
    """
    Analyze network impact and return results as a zip file containing WE, Others, and Mobile CSVs
    """
    global latest_we_results, latest_others_results, latest_mobile_data, latest_identifier, latest_identifier_type
    
    if we_analyzer is None or others_analyzer is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        logger.info(f"Starting CSV export for {request.identifier}")
        
        # Check if we have pre-computed results for this identifier
        use_precomputed = (
            latest_we_results is not None and 
            latest_others_results is not None and
            latest_mobile_data is not None and
            latest_identifier == request.identifier and
            latest_identifier_type == request.identifier_type
        )
        
        if use_precomputed:
            logger.info("Using pre-computed results for CSV export")
            we_results = latest_we_results
            others_results = latest_others_results
            mobile_data = latest_mobile_data
        else:
            logger.info("No pre-computed results found, running analysis...")
            # Use precomputed base results and only run impact analysis
            we_analyzer.final_df = we_base_results
            others_analyzer.final_df = others_base_results
            
            # Run the impact analysis on both data types
            if request.identifier_type == 'exchange' or (request.identifier_type == 'auto' and _is_exchange_identifier(request.identifier)):
                we_results = we_analyzer.analyze_exchange_impact(request.identifier)
                others_results = others_analyzer.analyze_exchange_impact(request.identifier)
            else:
                we_results = we_analyzer.analyze_node_impact(request.identifier)
                others_results = others_analyzer.analyze_node_impact(request.identifier)
            
            # Process mobile data
            we_mobile_data = we_analyzer._process_mobile_data(request.identifier)
            others_mobile_data = others_analyzer._process_mobile_data(request.identifier)
            mobile_data = pd.concat([we_mobile_data, others_mobile_data]).drop_duplicates(subset=['CINUM'])
            
            # Apply the same transformations as in complete analysis
            if not we_results.empty:
                we_results = we_analyzer._calculate_msan_level_route_status(we_results)
            
            if not others_results.empty:
                others_results['Route_Status'] = others_results.apply(
                    lambda row: others_analyzer._calculate_route_status_individual(
                        row['Path'], 
                        row['Path2'] if 'Path2' in row else None,
                        row.get('STATUS', 'UP'), 
                        row['Impact']
                    ),
                    axis=1
                )
        
        # Create zip file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
            # Add WE results - include ALL columns
            we_csv = we_results.to_csv(index=False)
            zip_file.writestr(f"we_impact_{request.identifier}.csv", we_csv)
            
            # Add Others results - include ALL columns
            others_csv = others_results.to_csv(index=False)
            zip_file.writestr(f"others_impact_{request.identifier}.csv", others_csv)
            
            # Add Mobile data - include ALL columns
            mobile_csv = mobile_data.to_csv(index=False)
            zip_file.writestr(f"mobile_impact_{request.identifier}.csv", mobile_csv)
            
            # Add a metadata file with info about the export
            metadata = {
                "identifier": request.identifier,
                "identifier_type": request.identifier_type,
                "export_timestamp": pd.Timestamp.now().isoformat(),
                "we_records_count": len(we_results),
                "others_records_count": len(others_results),
                "mobile_records_count": len(mobile_data),
                "used_precomputed_results": use_precomputed,
                "columns_in_we": list(we_results.columns) if not we_results.empty else [],
                "columns_in_others": list(others_results.columns) if not others_results.empty else [],
                "columns_in_mobile": list(mobile_data.columns) if not mobile_data.empty else []
            }
            zip_file.writestr("export_metadata.json", json.dumps(metadata, indent=2))
        
        zip_buffer.seek(0)
        
        # Generate filename
        safe_identifier = request.identifier.replace('/', '_').replace('\\', '_').replace('.', '_')
        filename = f"network_impact_{safe_identifier}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.zip"
        
        # Return as downloadable zip
        return StreamingResponse(
            iter([zip_buffer.getvalue()]),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
        
    except Exception as e:
        logger.error(f"CSV export failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"CSV export failed: {str(e)}")


@app.get("/debug/columns")
async def debug_columns():
    """Debug endpoint to check column consistency between computed results"""
    global latest_we_results, latest_others_results
    
    if latest_we_results is None or latest_others_results is None:
        return {"message": "No computed results available"}
    
    we_columns = list(latest_we_results.columns) if not latest_we_results.empty else []
    others_columns = list(latest_others_results.columns) if not latest_others_results.empty else []
    
    return {
        "we_columns": we_columns,
        "others_columns": others_columns,
        "we_shape": latest_we_results.shape,
        "others_shape": latest_others_results.shape,
        "we_memory_usage": latest_we_results.memory_usage(deep=True).sum() if not latest_we_results.empty else 0,
        "others_memory_usage": latest_others_results.memory_usage(deep=True).sum() if not latest_others_results.empty else 0
    }

# Add statistics calculation functions
def _calculate_we_statistics(we_results):
    """Calculate WE statistics (moved from frontend)"""
    if we_results.empty:
        return {}
    
    we_unique = we_results.drop_duplicates(subset=['MSANCODE'])
    isolated_we = we_unique[we_unique['Impact'] == 'Isolated']
    partial_we = we_unique[we_unique['Impact'] == 'Path Changed']
    st_rerouted = we_results[
    (we_results['STATUS'] == 'ST') & 
    (we_results['Route_Status'] == 'Traffic Rerouted') &
    (we_results['Impact'] == 'Path Changed')
    ]
    
    st_rerouted_unique = st_rerouted.drop_duplicates(subset=['MSANCODE'])
    st_rerouted_summary = (
    st_rerouted_unique.groupby('BNG_HOSTNAME', as_index=False)
        ['TOTAL_DATA_CUST']
        .sum()
        .rename(columns={'TOTAL_DATA_CUST': 'SUM_TOTAL_DATA_CUST_ST_REROUTED'})
    )
    # env_path = Path(__file__).resolve().parents[1] / 'secrets.env'
    # load_dotenv(dotenv_path=env_path)
 
    # data_path = os.getenv('DATA_PATH')
    # value = os.getenv('PRODUCTION')
    # if value=='false':
    #     df_report_we = pd.read_csv(f'{data_path}\\we.csv')  # WE data
    # else:
    #     # Load CSV files using the path from env
    #     df_report_we = pd.read_csv(f'{data_path}/we.csv')  # WE data
    #print("----- ST (Rerouted) Total per BNG -----")
    #print(st_rerouted_summary)
    df_report_we_up = we_results[we_results['STATUS'] == 'UP']
    we_up_sum = (
    df_report_we_up
    .groupby('BNG_HOSTNAME', as_index=False)['TOTAL_DATA_CUST']
    .sum()
    .rename(columns={'TOTAL_DATA_CUST': 'SUM_TOTAL_DATA_CUST_WE_UP'})
    )

    # Merge rerouted and WE UP sums by BNG_HOSTNAME
    merged = pd.merge(st_rerouted_summary, we_up_sum, on='BNG_HOSTNAME', how='left')
    # --- 7️⃣ Replace NaN with 0 ---
    merged[['SUM_TOTAL_DATA_CUST_ST_REROUTED', 'SUM_TOTAL_DATA_CUST_WE_UP']] = (
        merged[['SUM_TOTAL_DATA_CUST_ST_REROUTED', 'SUM_TOTAL_DATA_CUST_WE_UP']].fillna(0)
    )

    # --- 8️⃣ Add combined total column ---
    merged['SUM_TOTAL_DATA_CUST_WE_UP_AND_REROUTED'] = (
        merged['SUM_TOTAL_DATA_CUST_ST_REROUTED'] + merged['SUM_TOTAL_DATA_CUST_WE_UP']
    )


    #print("-----  (Merged) Total per BNG -----")
    #print(merged)
    merged_dict = merged.to_dict('records') if not merged.empty else [] # Convert to list of dictionaries for serialization as can't be returned as DataFrame directly
    #print(merged_dict)
    

    return {
        'merged_bng_data': merged_dict, 
        'isolated_msans': len(isolated_we),
        'partial_msans': len(partial_we),
        'isolated_sub': isolated_we['CUST'].sum() if not isolated_we.empty else 0,
        'partial_sub': partial_we['CUST'].sum() if not partial_we.empty else 0,
        'isolated_sub_voice': isolated_we['TOTAL_VOICE_CUST'].sum() if not isolated_we.empty else 0,
        'isolated_sub_data': isolated_we['TOTAL_DATA_CUST'].sum() if not isolated_we.empty else 0,
        'partial_sub_voice': partial_we['TOTAL_VOICE_CUST'].sum() if not partial_we.empty else 0,
        'partial_sub_data': partial_we['TOTAL_DATA_CUST'].sum() if not partial_we.empty else 0,
        'partial_vic': (partial_we['VIC'] == 'VIC').sum() if not partial_we.empty else 0,
        'iso_vic': (isolated_we['VIC'] == 'VIC').sum() if not isolated_we.empty else 0
      
    }


def _calculate_others_statistics(others_results):
    """Calculate Others statistics (moved from frontend)"""
    if others_results.empty:
        return {}
    
    # Step 1: Group by MSANCODE and analyze Impact values
    msan_impact_analysis = others_results.groupby('MSANCODE')['Impact'].agg([
        ('impact_values', 'unique'),  # Get all unique Impact values for this MSAN
        ('total_entries', 'count'),   # Total number of entries for this MSAN
        ('isolated_count', lambda x: (x == 'Isolated').sum()),
        ('partial_count', lambda x: (x == 'Partial').sum()),
        ('path_changed_count', lambda x: (x == 'Path Changed').sum())
    ]).reset_index()
    
    # Step 2: Categorize each MSAN based on the updated rules
    isolated_msans = []
    partial_msans = []
    path_changed_msans = []
    
    for _, row in msan_impact_analysis.iterrows():
        msan = row['MSANCODE']
        total = row['total_entries']
        isolated_count = row['isolated_count']
        partial_count = row['partial_count']
        path_changed_count = row['path_changed_count']
        impact_values = row['impact_values']
        
        # Rule 1: Isolated if ALL impacts are "Isolated"
        if isolated_count == total:
            isolated_msans.append(msan)
        
        # Rule 2: Partial if contains both "Partial" and "Isolated"
        elif 'Partial' in impact_values and 'Isolated' in impact_values:
            partial_msans.append(msan)
        
        # Rule 3: Path Changed if:
        # - Majority are "Path Changed" OR
        # - Has mix of "Path Changed" and "Partial" (without Isolated)
        elif (path_changed_count > total / 2) or ('Path Changed' in impact_values and 'Partial' in impact_values and isolated_count == 0):
            path_changed_msans.append(msan)
        
        # Rule 4: Default case - handle any remaining edge cases as Partial
        else:
            partial_msans.append(msan)
    
    # Step 3: Get the actual data for each category
    isolated_data = others_results[others_results['MSANCODE'].isin(isolated_msans)]
    partial_data = others_results[others_results['MSANCODE'].isin(partial_msans)]
    path_changed_data = others_results[others_results['MSANCODE'].isin(path_changed_msans)]
    
    # For counting unique MSANs
    isolated_unique = isolated_data.drop_duplicates(subset=['MSANCODE'])
    partial_unique = partial_data.drop_duplicates(subset=['MSANCODE'])
    path_changed_unique = path_changed_data.drop_duplicates(subset=['MSANCODE'])
    
    # Step 4: Customer counting functions - only count relevant ISP/service combinations
    def count_affected_customers(df):
        """Count customers only for the specific ISP/service combination in each row"""
        total = 0
        for _, row in df.iterrows():
            isp = row['ISP']
            service = row['SERVICE']
            
            if isp == 'VODAFONE':
                if service == 'UBBT':
                    total += row.get('VODA_UBBT_CUST', 0)
                elif service == 'HS-BT':
                    total += row.get('VODA_HS_CUST', 0)
            elif isp == 'ORANGE':
                if service == 'UBBT':
                    total += row.get('ORANGE_UBBT_CUST', 0)
                elif service == 'HS-BT':
                    total += row.get('ORANGE_HS_CUST', 0)
            elif isp == 'ETISALAT':
                if service == 'UBBT':
                    total += row.get('ETISLAT_UBBT_CUST', 0)
                elif service == 'HS-BT':
                    total += row.get('ETISLAT_HS_CUST', 0)
            elif isp == 'NOOR':
                if service == 'UBBT':
                    total += row.get('NOOR_UBBT_CUST', 0)
                elif service == 'HS-BT':
                    total += row.get('NOOR_HS_CUST', 0)
        
        return total
    
    def count_affected_customers_ftth(df):
        """Count FTTH customers only for the specific ISP/service combination in each row"""
        total_ftth = 0
        for _, row in df.iterrows():
            isp = row['ISP']
            service = row['SERVICE']
            
            if isp == 'VODAFONE':
                if service == 'UBBT-FTTH':
                    total_ftth += row.get('VODA_UBBT_FTTH_CUST', 0)
                elif service == 'HS-BT-FTTH':
                    total_ftth += row.get('VODA_HS_FTTH_CUST', 0)
            elif isp == 'ORANGE':
                if service == 'UBBT-FTTH':
                    total_ftth += row.get('ORANGE_UBBT_FTTH_CUST', 0)
                elif service == 'HS-BT-FTTH':
                    total_ftth += row.get('ORANGE_HS_FTTH_CUST', 0)
            elif isp == 'ETISALAT':
                if service == 'UBBT-FTTH':
                    total_ftth += row.get('ETISLAT_UBBT_FTTH_CUST', 0)
                elif service == 'HS-BT-FTTH':
                    total_ftth += row.get('ETISLAT_HS_FTTH_CUST', 0)
            elif isp == 'NOOR':
                if service == 'UBBT-FTTH':
                    total_ftth += row.get('NOOR_UBBT_FTTH_CUST', 0)
                elif service == 'HS-BT-FTTH':
                    total_ftth += row.get('NOOR_HS_FTTH_CUST', 0)
        
        return total_ftth
   
    isolated_results = others_results[others_results['Impact'].str.strip() == 'Isolated']
    partial_results = others_results[others_results['Impact'].str.strip() == 'Partial']
    path_results = others_results[others_results['Impact'].str.strip() == 'Path Changed']
    # Step 5: Return the statistics with updated customer counting
    return {
        # Counts of unique MSANs by category
        'isolated_msans': len(isolated_unique),
        'partial_msans': len(partial_unique),
        'path_changed_msans': len(path_changed_unique),
        
        # VIC counts
        'o_partial_vic': (partial_unique['VIC'] == 'VIC').sum() if not partial_unique.empty else 0,
        'o_iso_vic': (isolated_unique['VIC'] == 'VIC').sum() if not isolated_unique.empty else 0,
        'o_path_changed_vic': (path_changed_unique['VIC'] == 'VIC').sum() if not path_changed_unique.empty else 0,
        
        # Customer totals (only counting relevant ISP/service combinations)
        'isolated_sub': count_affected_customers(isolated_results)+count_affected_customers_ftth(isolated_results),
        'partial_sub': count_affected_customers(partial_results)+count_affected_customers_ftth(partial_results),
        'path_changed_sub': count_affected_customers(path_results)+count_affected_customers_ftth(path_results),
        
        # Vodafone statistics (using the new counting method)
        'isolated_voda': count_affected_customers(isolated_results[isolated_results['ISP'] == 'VODAFONE']),
        'isolated_voda_ubb': count_affected_customers(isolated_results[(isolated_results['ISP'] == 'VODAFONE') & (isolated_results['SERVICE'] == 'UBBT')]),
        'isolated_voda_hs': count_affected_customers(isolated_results[(isolated_results['ISP'] == 'VODAFONE') & (isolated_results['SERVICE'] == 'HS-BT')]),
        'isolated_voda_ubb_ftth': count_affected_customers_ftth(isolated_results[(isolated_results['ISP'] == 'VODAFONE') & (isolated_results['SERVICE'] == 'UBBT-FTTH')]),
        'isolated_voda_hs_ftth': count_affected_customers_ftth(isolated_results[(isolated_results['ISP'] == 'VODAFONE') & (isolated_results['SERVICE'] == 'HS-BT-FTTH')]),
        
        # Orange statistics
        	'isolated_orange': count_affected_customers(isolated_results[isolated_results['ISP'] == 'ORANGE']),
        'isolated_orange_ubb': count_affected_customers(isolated_results[(isolated_results['ISP'] == 'ORANGE') & (isolated_results['SERVICE'] == 'UBBT')]),
        'isolated_orange_hs': count_affected_customers(isolated_results[(isolated_results['ISP'] == 'ORANGE') & (isolated_results['SERVICE'] == 'HS-BT')]),
        'isolated_orange_ubb_ftth': count_affected_customers_ftth(isolated_results[(isolated_results['ISP'] == 'ORANGE') & (isolated_results['SERVICE'] == 'UBBT-FTTH')]),
        'isolated_orange_hs_ftth': count_affected_customers_ftth(isolated_results[(isolated_results['ISP'] == 'ORANGE') & (isolated_results['SERVICE'] == 'HS-BT-FTTH')]),
        
        # Etisalat statistics
        'isolated_etisalat': count_affected_customers(isolated_results[isolated_results['ISP'] == 'ETISALAT']),
        'isolated_etisalat_ubb': count_affected_customers(isolated_results[(isolated_results['ISP'] == 'ETISALAT') & (isolated_results['SERVICE'] == 'UBBT')]),
        'isolated_etisalat_hs': count_affected_customers(isolated_results[(isolated_results['ISP'] == 'ETISALAT') & (isolated_results['SERVICE'] == 'HS-BT')]),
        'isolated_etisalat_ubb_ftth': count_affected_customers_ftth(isolated_results[(isolated_results['ISP'] == 'ETISALAT') & (isolated_results['SERVICE'] == 'UBBT-FTTH')]),
        'isolated_etisalat_hs_ftth': count_affected_customers_ftth(isolated_results[(isolated_results['ISP'] == 'ETISALAT') & (isolated_results['SERVICE'] == 'HS-BT-FTTH')]),
        
        # Noor statistics
        'isolated_noor': count_affected_customers(isolated_results[isolated_results['ISP'] == 'NOOR']),
        'isolated_noor_ubb': count_affected_customers(isolated_results[(isolated_results['ISP'] == 'NOOR') & (isolated_results['SERVICE'] == 'UBBT')]),
        'isolated_noor_hs': count_affected_customers(isolated_results[(isolated_results['ISP'] == 'NOOR') & (isolated_results['SERVICE'] == 'HS-BT')]),
        'isolated_noor_ubb_ftth': count_affected_customers_ftth(isolated_results[(isolated_results['ISP'] == 'NOOR') & (isolated_results['SERVICE'] == 'UBBT-FTTH')]),
        'isolated_noor_hs_ftth': count_affected_customers_ftth(isolated_results[(isolated_results['ISP'] == 'NOOR') & (isolated_results['SERVICE'] == 'HS-BT-FTTH')]),
        
        # Wrong VLAN statistics (these are already ISP-specific, so we can sum directly)
        'wrong_noor': isolated_unique['NOOR_WRONG_VLAN'].sum() if not isolated_unique.empty else 0,
        'wrong_voda': isolated_unique['VODA_WORNG_VLAN'].sum() if not isolated_unique.empty else 0,
        'wrong_orange': isolated_unique['ORANGE_WORNG_VLAN'].sum() if not isolated_unique.empty else 0,
        'wrong_ets': isolated_unique['ETISLAT_WRONG_VLAN'].sum() if not isolated_unique.empty else 0
    }




def _create_impact_summary(results_df):
    """Create impact summary statistics"""
    if results_df.empty:
        return {"total_records": 0, "unique_msans": 0, "impact_breakdown": {}}
    
    summary = {
        "total_records": len(results_df),
        "unique_msans": len(results_df['MSANCODE'].unique()) if 'MSANCODE' in results_df.columns else 0,
    }
    
    # Impact breakdown
    if 'Impact' in results_df.columns:
        impact_counts = results_df['Impact'].value_counts().to_dict()
        summary["impact_breakdown"] = impact_counts
    
    # Status breakdown
    if 'STATUS' in results_df.columns:
        status_counts = results_df['STATUS'].value_counts().to_dict()
        summary["status_breakdown"] = status_counts
    
    # Circuit type breakdown
    if 'cir_type' in results_df.columns:
        cir_type_counts = results_df['cir_type'].value_counts().to_dict()
        summary["circuit_type_breakdown"] = cir_type_counts
    
    return summary



def convert_dataframe_to_serializable(df):
    """
    Convert DataFrame to JSON-serializable format by handling all numpy types and NaN values
    """
    if df.empty:
        return []
    
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Convert each column to handle numpy types and NaN values
    for col in df_copy.columns:
        # Handle numeric columns with numpy types
        if pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].apply(
                lambda x: None if pd.isna(x) else int(x) if isinstance(x, (np.integer, np.int64, np.int32)) else float(x) if isinstance(x, (np.floating, np.float64, np.float32)) else x
            )
        else:
            # For object/string columns, just replace NaN with None
            df_copy[col] = df_copy[col].where(pd.notna(df_copy[col]), None)
    
    # Convert to dictionary and ensure all values are serializable
    records = df_copy.to_dict('records')
    return ensure_serializable(records)

def ensure_serializable(obj):
    """
    Recursively ensure all objects are JSON serializable
    """
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return [ensure_serializable(item) for item in obj.tolist()]
    elif isinstance(obj, dict):
        return {key: ensure_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [ensure_serializable(item) for item in obj]
    elif hasattr(obj, 'item'):  # Other numpy scalars
        return obj.item()
    elif pd.isna(obj):  # Handle remaining NaN/NaT
        return None
    else:
        # Try to convert to string as last resort
        try:
            return str(obj)
        except:
            return None

def _is_exchange_identifier(identifier):
    """Check if identifier is likely an exchange (contains dots or is shorter)"""
    return '.' in identifier or len(identifier.split('-')) < 4

if __name__ == "__main__":
    import uvicorn
    
    # Run the API
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )