from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import uvicorn
from statistics_analyzer import StatisticsNetworkAnalyzer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Network Impact Statistics API",
    description="API for analyzing network node impact and returning statistics only",
    version="1.0.0"
)

# Request models
class NodeAnalysisRequest(BaseModel):
    nodes: List[str]
    include_detailed_breakdown: Optional[bool] = False

class SingleNodeRequest(BaseModel):
    node: str
    include_detailed_breakdown: Optional[bool] = False

# Global analyzer instance
analyzer = None

@app.on_event("startup")
async def startup_event():
    """Initialize the analyzer with data on startup"""
    global analyzer
    
    try:
        logger.info("Loading CSV files for statistics API...")
        
        # Update this path to your data directory
        data_path = r"C:\Users\secre\OneDrive\Desktop\network-impact-analysis\backend\data"
        
        # Load CSV files
        df_report_we = pd.read_csv(f'{data_path}\\we.csv')
        df_report_others = pd.read_csv(f'{data_path}\\others.csv')
        df_res_ospf = pd.read_csv(f'{data_path}\\res_ospf.csv')
        df_wan = pd.read_csv(f'{data_path}\\wan.csv')
        df_agg = pd.read_csv(f'{data_path}\\agg.csv')

        # Standardize column names
        df_report_others.columns = df_report_others.columns.str.upper()
        df_report_we.columns = df_report_we.columns.str.upper()

        # Rename columns to match expected format
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
        
        # Initialize analyzer
        analyzer = StatisticsNetworkAnalyzer(
            df_report_we, df_report_others, df_res_ospf, df_wan, df_agg
        )
        
        logger.info("Statistics analyzer initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize statistics analyzer: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Network Impact Statistics API",
        "version": "1.0.0", 
        "endpoints": {
            "/analyze/nodes": "POST - Analyze multiple nodes",
            "/analyze/node": "POST - Analyze single node",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "status": "healthy",
        "analyzer_ready": analyzer is not None
    }

@app.post("/analyze/nodes")
async def analyze_multiple_nodes(request: NodeAnalysisRequest):
    """
    Analyze impact for multiple nodes and return comprehensive statistics
    
    - **nodes**: List of node identifiers to analyze
    - **include_detailed_breakdown**: Whether to include detailed ISP/service breakdown
    """
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    if not request.nodes:
        raise HTTPException(status_code=400, detail="No nodes provided")
    
    try:
        logger.info(f"Analyzing {len(request.nodes)} nodes: {request.nodes}")
        
        results = analyzer.analyze_nodes(request.nodes)
        
        # Simplify response if detailed breakdown not requested
        if not request.include_detailed_breakdown:
            for node_result in results['node_results'].values():
                if 'error' not in node_result:
                    node_result.pop('we_statistics', None)
                    node_result.pop('others_statistics', None)
        
        return {
            "status": "success",
            "request": {
                "nodes_analyzed": request.nodes,
                "total_nodes": len(request.nodes)
            },
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/node") 
async def analyze_single_node(request: SingleNodeRequest):
    """
    Analyze impact for a single node and return comprehensive statistics
    
    - **node**: Single node identifier to analyze
    - **include_detailed_breakdown**: Whether to include detailed ISP/service breakdown
    """
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        logger.info(f"Analyzing single node: {request.node}")
        
        # Use the multi-node analyzer for consistency
        results = analyzer.analyze_nodes([request.node])
        node_result = results['node_results'].get(request.node, {})
        
        # Simplify response if detailed breakdown not requested
        if not request.include_detailed_breakdown and 'error' not in node_result:
            node_result.pop('we_statistics', None)
            node_result.pop('others_statistics', None)
        
        return {
            "status": "success",
            "request": {
                "node": request.node
            },
            "results": {
                "node_analysis": node_result,
                "analysis_metadata": {
                    "total_analysis_time_seconds": results['total_analysis_time_seconds']
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Single node analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/nodes/counts")
async def get_data_counts():
    """Get basic counts of data loaded"""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "we_records": len(analyzer.df_report_we),
        "others_records": len(analyzer.df_report_others),
        "unique_we_msans": analyzer.df_report_we['MSANCODE'].nunique(),
        "unique_others_msans": analyzer.df_report_others['MSANCODE'].nunique(),
        "total_nodes_in_graph": len(analyzer.base_graph.nodes())
    }

if __name__ == "__main__":
    uvicorn.run(
        "statistics_api:app",
        host="0.0.0.0", 
        port=8002,  # Different port to avoid conflict with existing API
        reload=True
    )