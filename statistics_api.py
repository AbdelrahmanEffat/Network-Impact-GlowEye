from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import pandas as pd
import logging
import time
import threading
from backend.core.analyzer import UnifiedNetworkImpactAnalyzer
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

# Request/Response Models
class StatsRequest(BaseModel):
    nodes: List[str] = Field(..., min_items=1, description="List of node identifiers to analyze")
    
    @validator('nodes')
    def validate_nodes(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one node is required")
        # Validate node format (basic check)
        for node in v:
            if not isinstance(node, str) or len(node.strip()) == 0:
                raise ValueError(f"Invalid node identifier: {node}")
        return v

class StatsResponse(BaseModel):
    status: str
    message: str
    nodes_analyzed: List[str]
    analysis_type: str
    we_stats: Dict[str, Any]
    others_stats: Dict[str, Any]

# Statistics Analyzer Wrapper
class StatisticsAnalyzer:
    """Wrapper for unified network impact analysis focused on statistics only"""
    
    def __init__(self, we_analyzer, others_analyzer):
        self.we_analyzer = we_analyzer
        self.others_analyzer = others_analyzer
        self.lock = threading.Lock()
    
    def analyze_nodes(self, nodes: List[str]):
        """
        Analyze impact for given nodes and return statistics only
        
        Args:
            nodes: List of node identifiers (single or multiple)
            
        Returns:
            tuple: (we_stats, others_stats, analysis_type)
        """
        with self.lock:
            try:
                # Determine analysis type
                analysis_type = "single_node" if len(nodes) == 1 else "multi_node"
                
                logger.info(f"Starting {analysis_type} analysis for nodes: {nodes}")
                
                # Run analysis for both data types
                if analysis_type == "single_node":
                    we_results = self.we_analyzer.analyze_node_impact(nodes[0])
                    others_results = self.others_analyzer.analyze_node_impact(nodes[0])
                else:
                    # Multiple nodes: treat as collective failure (like exchange scenario)
                    we_results = self._analyze_multi_node_impact(self.we_analyzer, nodes)
                    others_results = self._analyze_multi_node_impact(self.others_analyzer, nodes)
                
                # Calculate statistics
                we_stats = self._calculate_we_statistics(we_results)
                others_stats = self._calculate_others_statistics(others_results)
                
                logger.info(f"Analysis completed for {len(nodes)} node(s)")
                
                return we_stats, others_stats, analysis_type
                
            except Exception as e:
                logger.error(f"Analysis failed for nodes {nodes}: {str(e)}")
                raise
    
    def _analyze_multi_node_impact(self, analyzer, nodes: List[str]):
        """
        Analyze impact treating all nodes as a collective failure unit
        Similar to exchange failure but with explicit node list
        """
        if analyzer.final_df is None:
            raise ValueError("Analyzer not initialized. Call preprocess_data() and generate_base_results() first.")
        
        results = []
        
        # Case 1: Check for direct impact on target nodes
        target_impact = self._analyze_direct_target_impact(analyzer, nodes)
        if not target_impact.empty:
            results.append(target_impact)
        
        # Case 2: Physical path impact (nodes in paths)
        path_impact = self._analyze_multi_node_path_impact(analyzer, nodes)
        if not path_impact.empty:
            results.append(path_impact)
        
        # Combine results
        combined = analyzer._combine_results(results)
        
        # Apply impact calculation based on data type
        if not combined.empty:
            if analyzer.data_type == 'network':
                combined = analyzer._calculate_msan_level_impact(combined)
            else:
                combined = analyzer._calculate_impact_for_others(combined)
        
        return combined
    
    def _analyze_direct_target_impact(self, analyzer, nodes: List[str]):
        """Analyze direct impact on target nodes (all nodes in list)"""
        target_hostname_col = analyzer.column_map['target_hostname']
        bng_hostname_col = analyzer.column_map.get('bng_hostname')
        
        # Find records where target node is in the nodes list
        if analyzer.data_type == 'network' and bng_hostname_col:
            # Network data: check both BNG and distribution hostname
            direct_impact = analyzer.final_df[
                (analyzer.final_df[bng_hostname_col].isin(nodes)) | 
                (analyzer.final_df[target_hostname_col].isin(nodes))
            ]
        else:
            # Others data: only check target hostname
            direct_impact = analyzer.final_df[analyzer.final_df[target_hostname_col].isin(nodes)]
        
        if direct_impact.empty:
            return pd.DataFrame()
        
        if analyzer.data_type == 'network':
            # Get all records for affected MSANs
            all_affected = analyzer.final_df[
                analyzer.final_df.MSANCODE.isin(direct_impact.MSANCODE.unique())
            ].copy()
            
            # Determine impact based on circuit type
            msan_impacts = {}
            for msan in all_affected.MSANCODE.unique():
                msan_df = all_affected[all_affected.MSANCODE == msan]
                
                if 'cir_type' in msan_df.columns and 'Single' in msan_df['cir_type'].values:
                    msan_impacts[msan] = 'Isolated'
                else:
                    msan_impacts[msan] = 'Path Changed'
            
            all_affected['Impact'] = all_affected['MSANCODE'].map(msan_impacts)
        else:
            # Others data: get all records but only mark specific ones
            affected_msans = direct_impact.MSANCODE.unique()
            all_affected = analyzer.final_df[
                analyzer.final_df.MSANCODE.isin(affected_msans)
            ].copy()
            
            # Create mask for directly affected records
            direct_impact_mask = all_affected[target_hostname_col].isin(nodes)
            
            # Initialize Impact column
            all_affected['Path2'] = None
            all_affected['Impact'] = 'Partial'
            
            # Calculate paths only for directly affected records
            graph = analyzer.model.draw_graph(excluded_nodes=nodes)
            affected_indices = all_affected[direct_impact_mask].index
            
            for idx in affected_indices:
                row = all_affected.loc[idx]
                path2 = analyzer.model.calculate_path(graph, row['EDGE'], row[target_hostname_col])
                all_affected.at[idx, 'Path2'] = path2
                all_affected.at[idx, 'Impact'] = 'Path Changed' if isinstance(path2, list) else 'Isolated'
        
        return all_affected
    
    def _analyze_multi_node_path_impact(self, analyzer, nodes: List[str]):
        """Analyze physical path impact when all nodes are excluded together"""
        # Find records with any of the nodes in their paths
        affected_records = analyzer._find_records_with_nodes_in_path(nodes)
        
        if affected_records.empty:
            return pd.DataFrame()
        
        # Calculate alternative paths excluding all nodes
        graph = analyzer.model.draw_graph(excluded_nodes=nodes)
        target_hostname_col = analyzer.column_map['target_hostname']
        
        if analyzer.data_type == 'network':
            # Get all records for affected MSANs
            affected_msans = affected_records.MSANCODE.unique()
            all_affected = analyzer.final_df[
                analyzer.final_df.MSANCODE.isin(affected_msans)
            ].copy()
            
            all_affected['Path2'] = all_affected.apply(
                lambda row: analyzer.model.calculate_path(graph, row['EDGE'], row[target_hostname_col]),
                axis=1
            )
            
            all_affected['Impact'] = all_affected['Path2'].apply(
                lambda x: 'Path Changed' if isinstance(x, list) else 'Isolated'
            )
            
            all_affected = analyzer._calculate_msan_level_impact(all_affected)
        else:
            # Others data: get all records but only mark specific ones
            affected_msans = affected_records.MSANCODE.unique()
            all_affected = analyzer.final_df[
                analyzer.final_df.MSANCODE.isin(affected_msans)
            ].copy()
            
            all_affected['Path2'] = None
            all_affected['Impact'] = 'Partial'
            
            # Calculate paths only for records with nodes in their paths
            for idx, row in all_affected.iterrows():
                path = row['Path']
                if isinstance(path, list) and any(node in path[1:-1] for node in nodes):
                    path2 = analyzer.model.calculate_path(graph, row['EDGE'], row[target_hostname_col])
                    all_affected.at[idx, 'Path2'] = path2
                    all_affected.at[idx, 'Impact'] = 'Path Changed' if isinstance(path2, list) else 'Isolated'
        
        return all_affected
    
    def _calculate_we_statistics(self, we_results):
        """Calculate WE statistics from results"""
        if we_results.empty:
            return {
                'isolated_msans': 0,
                'path_changed_msans': 0,
                'isolated_sub': 0,
                'path_changed_sub': 0,
                'isolated_sub_voice': 0,
                'isolated_sub_data': 0,
                'path_changed_sub_voice': 0,
                'path_changed_sub_data': 0,
                'path_changed_vic': 0,
                'iso_vic': 0
            }
        
        we_unique = we_results.drop_duplicates(subset=['MSANCODE'])
        isolated_we = we_unique[we_unique['Impact'] == 'Isolated']
        partial_we = we_unique[we_unique['Impact'] == 'Path Changed'] # path_changed
        
        return {
            'isolated_msans': len(isolated_we),
            'path_changed_msans': len(partial_we),
            'isolated_sub': int(isolated_we['CUST'].sum()) if not isolated_we.empty else 0,
            'path_changed_sub': int(partial_we['CUST'].sum()) if not partial_we.empty else 0,
            'isolated_sub_voice': int(isolated_we['TOTAL_VOICE_CUST'].sum()) if not isolated_we.empty else 0,
            'isolated_sub_data': int(isolated_we['TOTAL_DATA_CUST'].sum()) if not isolated_we.empty else 0,
            'path_changed_sub_voice': int(partial_we['TOTAL_VOICE_CUST'].sum()) if not partial_we.empty else 0,
            'path_changed_sub_data': int(partial_we['TOTAL_DATA_CUST'].sum()) if not partial_we.empty else 0,
            'path_changed_vic': int((partial_we['VIC'] == 'VIC').sum()) if not partial_we.empty else 0,
            'iso_vic': int((isolated_we['VIC'] == 'VIC').sum()) if not isolated_we.empty else 0
        }
    
    def _calculate_others_statistics(self, others_results):
        """Calculate Others statistics from results"""
        if others_results.empty:
            return {
                'isolated_msans': 0,
                'partial_msans': 0,
                'path_changed_msans': 0,
                'o_partial_vic': 0,
                'o_iso_vic': 0,
                'o_path_changed_vic': 0,
                'isolated_sub': 0,
                'partial_sub': 0,
                'path_changed_sub': 0,
                'isolated_voda': 0,
                'isolated_voda_ubb': 0,
                'isolated_voda_hs': 0,
                'isolated_voda_ubb_ftth': 0,
                'isolated_voda_hs_ftth': 0,
                'isolated_orange': 0,
                'isolated_orange_ubb': 0,
                'isolated_orange_hs': 0,
                'isolated_orange_ubb_ftth': 0,
                'isolated_orange_hs_ftth': 0,
                'isolated_etisalat': 0,
                'isolated_etisalat_ubb': 0,
                'isolated_etisalat_hs': 0,
                'isolated_etisalat_ubb_ftth': 0,
                'isolated_etisalat_hs_ftth': 0,
                'isolated_noor': 0,
                'isolated_noor_ubb': 0,
                'isolated_noor_hs': 0,
                'isolated_noor_ubb_ftth': 0,
                'isolated_noor_hs_ftth': 0,
                'wrong_noor': 0,
                'wrong_voda': 0,
                'wrong_orange': 0,
                'wrong_ets': 0
            }
        
        # Step 1: Group by MSANCODE and analyze Impact values
        msan_impact_analysis = others_results.groupby('MSANCODE')['Impact'].agg([
            ('impact_values', 'unique'),
            ('total_entries', 'count'),
            ('isolated_count', lambda x: (x == 'Isolated').sum()),
            ('partial_count', lambda x: (x == 'Partial').sum()),
            ('path_changed_count', lambda x: (x == 'Path Changed').sum())
        ]).reset_index()
        
        # Step 2: Categorize each MSAN
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
            
            if isolated_count == total:
                isolated_msans.append(msan)
            elif 'Partial' in impact_values and 'Isolated' in impact_values:
                partial_msans.append(msan)
            elif (path_changed_count > total / 2) or ('Path Changed' in impact_values and 'Partial' in impact_values and isolated_count == 0):
                path_changed_msans.append(msan)
            else:
                partial_msans.append(msan)
        
        # Step 3: Get data for each category
        isolated_data = others_results[others_results['MSANCODE'].isin(isolated_msans)]
        partial_data = others_results[others_results['MSANCODE'].isin(partial_msans)]
        path_changed_data = others_results[others_results['MSANCODE'].isin(path_changed_msans)]
        
        isolated_unique = isolated_data.drop_duplicates(subset=['MSANCODE'])
        partial_unique = partial_data.drop_duplicates(subset=['MSANCODE'])
        path_changed_unique = path_changed_data.drop_duplicates(subset=['MSANCODE'])
        
        # Step 4: Customer counting functions
        def count_affected_customers(df):
            """Count customers for specific ISP/service combinations"""
            total = 0
            for _, row in df.iterrows():
                isp = row.get('ISP', '')
                service = row.get('SERVICE', '')
                
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
            return int(total) if total > 0 else 0
        
        def count_affected_customers_ftth(df):
            """Count FTTH customers"""
            total_ftth = 0
            for _, row in df.iterrows():
                isp = row.get('ISP', '')
                service = row.get('SERVICE', '')
                
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
            return int(total_ftth) if total_ftth > 0 else 0
        
        # Step 5: Build statistics
        return {
            'isolated_msans': len(isolated_unique),
            'partial_msans': len(partial_unique),
            'path_changed_msans': len(path_changed_unique),
            'o_partial_vic': int((partial_unique['VIC'] == 'VIC').sum()) if not partial_unique.empty else 0,
            'o_iso_vic': int((isolated_unique['VIC'] == 'VIC').sum()) if not isolated_unique.empty else 0,
            'o_path_changed_vic': int((path_changed_unique['VIC'] == 'VIC').sum()) if not path_changed_unique.empty else 0,
            'isolated_sub': count_affected_customers(isolated_data),
            'partial_sub': count_affected_customers(partial_data),
            'path_changed_sub': count_affected_customers(path_changed_data),
            'isolated_voda': count_affected_customers(isolated_data[isolated_data['ISP'] == 'VODAFONE']),
            'isolated_voda_ubb': count_affected_customers(isolated_data[(isolated_data['ISP'] == 'VODAFONE') & (isolated_data['SERVICE'] == 'UBBT')]),
            'isolated_voda_hs': count_affected_customers(isolated_data[(isolated_data['ISP'] == 'VODAFONE') & (isolated_data['SERVICE'] == 'HS-BT')]),
            'isolated_voda_ubb_ftth': count_affected_customers_ftth(isolated_data[(isolated_data['ISP'] == 'VODAFONE') & (isolated_data['SERVICE'] == 'UBBT-FTTH')]),
            'isolated_voda_hs_ftth': count_affected_customers_ftth(isolated_data[(isolated_data['ISP'] == 'VODAFONE') & (isolated_data['SERVICE'] == 'HS-BT-FTTH')]),
            'isolated_orange': count_affected_customers(isolated_data[isolated_data['ISP'] == 'ORANGE']),
            'isolated_orange_ubb': count_affected_customers(isolated_data[(isolated_data['ISP'] == 'ORANGE') & (isolated_data['SERVICE'] == 'UBBT')]),
            'isolated_orange_hs': count_affected_customers(isolated_data[(isolated_data['ISP'] == 'ORANGE') & (isolated_data['SERVICE'] == 'HS-BT')]),
            'isolated_orange_ubb_ftth': count_affected_customers_ftth(isolated_data[(isolated_data['ISP'] == 'ORANGE') & (isolated_data['SERVICE'] == 'UBBT-FTTH')]),
            'isolated_orange_hs_ftth': count_affected_customers_ftth(isolated_data[(isolated_data['ISP'] == 'ORANGE') & (isolated_data['SERVICE'] == 'HS-BT-FTTH')]),
            'isolated_etisalat': count_affected_customers(isolated_data[isolated_data['ISP'] == 'ETISALAT']),
            'isolated_etisalat_ubb': count_affected_customers(isolated_data[(isolated_data['ISP'] == 'ETISALAT') & (isolated_data['SERVICE'] == 'UBBT')]),
            'isolated_etisalat_hs': count_affected_customers(isolated_data[(isolated_data['ISP'] == 'ETISALAT') & (isolated_data['SERVICE'] == 'HS-BT')]),
            'isolated_etisalat_ubb_ftth': count_affected_customers_ftth(isolated_data[(isolated_data['ISP'] == 'ETISALAT') & (isolated_data['SERVICE'] == 'UBBT-FTTH')]),
            'isolated_etisalat_hs_ftth': count_affected_customers_ftth(isolated_data[(isolated_data['ISP'] == 'ETISALAT') & (isolated_data['SERVICE'] == 'HS-BT-FTTH')]),
            'isolated_noor': count_affected_customers(isolated_data[isolated_data['ISP'] == 'NOOR']),
            'isolated_noor_ubb': count_affected_customers(isolated_data[(isolated_data['ISP'] == 'NOOR') & (isolated_data['SERVICE'] == 'UBBT')]),
            'isolated_noor_hs': count_affected_customers(isolated_data[(isolated_data['ISP'] == 'NOOR') & (isolated_data['SERVICE'] == 'HS-BT')]),
            'isolated_noor_ubb_ftth': count_affected_customers_ftth(isolated_data[(isolated_data['ISP'] == 'NOOR') & (isolated_data['SERVICE'] == 'UBBT-FTTH')]),
            'isolated_noor_hs_ftth': count_affected_customers_ftth(isolated_data[(isolated_data['ISP'] == 'NOOR') & (isolated_data['SERVICE'] == 'HS-BT-FTTH')]),
            'wrong_noor': int(isolated_data['NOOR_WRONG_VLAN'].sum()) if not isolated_data.empty else 0,
            'wrong_voda': int(isolated_data['VODA_WORNG_VLAN'].sum()) if not isolated_data.empty else 0,
            'wrong_orange': int(isolated_data['ORANGE_WORNG_VLAN'].sum()) if not isolated_data.empty else 0,
            'wrong_ets': int(isolated_data['ETISLAT_WRONG_VLAN'].sum()) if not isolated_data.empty else 0
        }

# FastAPI Application
app = FastAPI(
    title="Network Impact Statistics API",
    description="API for analyzing network impact statistics from node failures",
    version="1.0.0"
)

# Global instances
stats_analyzer = None
analyzer_lock = threading.Lock()

@app.on_event("startup")
async def startup_event():
    """Initialize analyzers and precompute base results on startup"""
    global stats_analyzer
    
    with analyzer_lock:
        try:
            env_path = Path(__file__).resolve().parents[0] / 'secrets.env'
            load_dotenv(dotenv_path=env_path)

            # Debug: Check if file exists
            if env_path.exists():
                print(f"Found secrets.env at: {env_path}")
            else:
                print(f"secrets.env not found at: {env_path}")

            # Now you can access the environment variables
            value = os.getenv('PRODUCTION')
            print(f"Value: '{value}'")
            
            logger.info("Loading CSV files...")
            
            if value=='false':

                data_path = r"C:\Users\secre\OneDrive\Desktop\network-impact-analysis\backend\data"
                
                # Load your CSV files
                df_report_we = pd.read_csv(f'{data_path}\\we.csv')  # WE data
                df_report_others = pd.read_csv(f'{data_path}\\others.csv')  # Others data
                df_res_ospf = pd.read_csv(f'{data_path}\\res_ospf.csv')
                df_wan = pd.read_csv(f'{data_path}\\wan.csv')
                df_agg = pd.read_csv(f'{data_path}\\agg.csv')
            else:    

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
                    raise Exception(f"Failed to load data from Redis for keys: {missing}")
            
            # Normalize column names
            df_report_others.columns = df_report_others.columns.str.upper()
            df_report_we.columns = df_report_we.columns.str.upper()
            
            df_report_others.rename(columns={
                'BITSTREAM_EXCHANGE': 'Bitstream_exchange', 
                'EDGE_EXCHANGE': 'EDGE_exchange'
            }, inplace=True)
            
            df_report_we.rename(columns={
                'DISTRIBUTION_EMS': 'distribution_ems',
                'DISTRIBUTION_EXCHANGE': 'distribution_Exchange',
                'DISTRIBUTION_HOSTNAME': 'distribution_hostname',
                'DISTRIBUTION_INT': 'distribution_INT',
                'DISTRIBUTION_IP': 'distribution_IP',
                'DISTRIBUTION_NAME_EX': 'distribution_name_ex',
                'EDGE_EXCHANGE': 'edge_exchange',
                'EDGE_NAME_EX': 'edge_name_ex',
                'EDGE_PORT': 'edge_port'
            }, inplace=True)
            
            # Initialize analyzers
            we_analyzer = UnifiedNetworkImpactAnalyzer(df_report_we, df_res_ospf, df_wan, df_agg)
            others_analyzer = UnifiedNetworkImpactAnalyzer(df_report_others, df_res_ospf, df_wan, df_agg)
            
            # Preprocess data
            logger.info("Preprocessing data...")
            we_analyzer.preprocess_data()
            others_analyzer.preprocess_data()
            
            # Generate base results
            logger.info("Generating base results...")
            we_analyzer.generate_base_results("initialization")
            others_analyzer.generate_base_results("initialization")
            
            # Initialize statistics analyzer
            stats_analyzer = StatisticsAnalyzer(we_analyzer, others_analyzer)
            
            logger.info("Statistics API startup completed successfully")
            
        except Exception as e:
            logger.error(f"Startup failed: {str(e)}")
            raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Network Impact Statistics API",
        "version": "1.0.0",
        "endpoints": {
            "/stats": "POST - Analyze and return statistics for node(s)",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if stats_analyzer is None:
        raise HTTPException(status_code=503, detail="Service not ready - analyzer not initialized")
    
    return {
        "status": "healthy",
        "analyzer_ready": stats_analyzer is not None
    }

@app.post("/stats", response_model=StatsResponse)
async def analyze_stats(request: StatsRequest):
    """
    Analyze network impact and return statistics only
    
    Args:
        request: StatsRequest containing list of nodes
        
    Returns:
        StatsResponse with we_stats and others_stats
    """
    if stats_analyzer is None:
        raise HTTPException(status_code=503, detail="Service not ready - analyzer not initialized")
    
    try:
        logger.info(f"Analyzing nodes: {request.nodes}")
        
        start_time = time.time()
        
        # Run analysis
        we_stats, others_stats, analysis_type = stats_analyzer.analyze_nodes(request.nodes)
        
        execution_time = time.time() - start_time
        
        logger.info(f"Analysis completed in {execution_time:.3f} seconds")
        
        return StatsResponse(
            status="success",
            message=f"Analysis completed successfully for {len(request.nodes)} node(s)",
            nodes_analyzed=request.nodes,
            analysis_type=analysis_type,
            we_stats=we_stats,
            others_stats=others_stats
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "statistics_api:app",
        host="0.0.0.0",
        port=8002,
        reload=True
    )