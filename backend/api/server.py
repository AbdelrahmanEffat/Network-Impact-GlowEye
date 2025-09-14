# main_API.py (updated)
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Literal, Dict, Any, List
from backend.core.analyzer import UnifiedNetworkImpactAnalyzer
import pandas as pd
import logging
import io
import zipfile
import json
import time
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

@app.on_event("startup")
async def startup_event():
    """Initialize the analyzers with CSV files on startup and precompute base results"""
    global we_analyzer, others_analyzer, we_base_results, others_base_results
    
    try:
        logger.info("Loading CSV files...")
        
        data_path = r"C:\Users\secre\OneDrive\Desktop\network-impact-web\endpoint\data"
        # Load your CSV files
        df_report_we = pd.read_csv(f'{data_path}\\Report(11).csv')  # WE data
        df_report_others = pd.read_csv(f'{data_path}\\Report(12).csv')  # Others data
        df_res_ospf = pd.read_csv(f'{data_path}\\res_ospf.csv')
        df_wan = pd.read_csv(f'{data_path}\\wan.csv')
        df_agg = pd.read_csv(f'{data_path}\\agg.csv')
        
        # Initialize analyzers for both data types
        we_analyzer = UnifiedNetworkImpactAnalyzer(df_report_we, df_res_ospf, df_wan, df_agg)
        others_analyzer = UnifiedNetworkImpactAnalyzer(df_report_others, df_res_ospf, df_wan, df_agg)
        
        # Preprocess data and generate base results
        logger.info("Preprocessing data and generating base results...")
        we_analyzer.preprocess_data()
        others_analyzer.preprocess_data()
        
        # Generate base results with a dummy identifier (we'll use actual identifier for impact analysis)
        we_base_results = we_analyzer.generate_base_results("dummy_initialization")
        others_base_results = others_analyzer.generate_base_results("dummy_initialization")
        
        logger.info(f"Data loaded successfully. WE shape: {df_report_we.shape}, Others shape: {df_report_others.shape}")
        logger.info(f"Base results computed. WE: {we_base_results.shape}, Others: {others_base_results.shape}")
        
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
            "/analyze/detailed": "POST - Get detailed analysis results",
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

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_network_impact(request: AnalysisRequest):
    """
    Analyze network impact from node or exchange failure for both WE and Others data
    """
    if we_analyzer is None or others_analyzer is None:
        raise HTTPException(
            status_code=503, 
            detail="Service not ready - analyzers not initialized"
        )
    
    try:
        logger.info(f"Starting impact analysis for {request.identifier} (type: {request.identifier_type})")
        
        start_time = time.time()
        
        # Use precomputed base results and only run impact analysis
        we_analyzer.final_df = we_base_results
        others_analyzer.final_df = others_base_results
        
        # Run the impact analysis on both data types
        if request.identifier_type == 'exchange' or (request.identifier_type == 'auto' and _is_exchange_identifier(request.identifier)):
            we_results = we_analyzer.analyze_exchange_impact(request.identifier)
            we_results = we_results.drop_duplicates(subset=['MSANCODE', 'STATUS'])
            others_results = others_analyzer.analyze_exchange_impact(request.identifier)
            analysis_type = "Exchange"
        else:
            we_results = we_analyzer.analyze_node_impact(request.identifier)
            others_results = others_analyzer.analyze_node_impact(request.identifier)
            analysis_type = "Node"
        
        execution_time = time.time() - start_time
        
        # Create impact summaries
        we_impact_summary = _create_impact_summary(we_results)
        others_impact_summary = _create_impact_summary(others_results)
        
        # Get previews
        we_preview = _get_results_preview(we_results)
        others_preview = _get_results_preview(others_results)
        
        # Combine results
        combined_impact_summary = {
            "we": we_impact_summary,
            "others": others_impact_summary,
            "total_records": we_impact_summary.get("total_records", 0) + others_impact_summary.get("total_records", 0),
            "total_unique_msans": we_impact_summary.get("unique_msans", 0) + others_impact_summary.get("unique_msans", 0)
        }
        
        return AnalysisResponse(
            status="success",
            message=f"Analysis completed successfully for {request.identifier}",
            total_records=combined_impact_summary["total_records"],
            unique_msans=combined_impact_summary["total_unique_msans"],
            analysis_type=analysis_type,
            execution_time_seconds=round(execution_time, 3),
            results_preview={
                "we": we_preview,
                "others": others_preview
            },
            impact_summary=combined_impact_summary
        )
        
    except Exception as e:
        logger.error(f"Analysis failed for {request.identifier}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/analyze/csv")
async def analyze_and_return_csv(request: AnalysisRequest):
    """
    Analyze network impact and return results as a zip file containing both WE and Others CSVs
    """
    if we_analyzer is None or others_analyzer is None:
        raise HTTPException(
            status_code=503, 
            detail="Service not ready - analyzers not initialized"
        )
    
    try:
        logger.info(f"Starting CSV export for {request.identifier}")
        
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
        
        # Create zip file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
            # Add WE results
            we_csv = we_results.to_csv(index=False)
            zip_file.writestr(f"we_impact_{request.identifier}.csv", we_csv)
            
            # Add Others results
            others_csv = others_results.to_csv(index=False)
            zip_file.writestr(f"others_impact_{request.identifier}.csv", others_csv)
        
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
        raise HTTPException(
            status_code=500,
            detail=f"CSV export failed: {str(e)}"
        )

@app.post("/analyze/detailed")
async def analyze_network_impact_detailed(request: AnalysisRequest):
    """Get detailed analysis results including full data"""
    if we_analyzer is None or others_analyzer is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        logger.info(f"Starting detailed analysis for {request.identifier}")
        
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

        # Convert NaN values to None (which becomes null in JSON)
        # Use a more robust method to handle all NaN cases
        def replace_nan_with_none(obj):
            if isinstance(obj, dict):
                return {k: replace_nan_with_none(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_nan_with_none(v) for v in obj]
            elif pd.isna(obj):  # This handles both NaN and NaT values
                return None
            else:
                return obj
        
        # Convert to dict first, then clean NaN values
        we_results_dict = we_results.to_dict(orient='records')
        others_results_dict = others_results.to_dict(orient='records')
        
        # Clean NaN values from both results
        we_results_clean = replace_nan_with_none(we_results_dict)
        others_results_clean = replace_nan_with_none(others_results_dict)
        
        return {
            "we_results": we_results_clean,
            "others_results": others_results_clean
        }
    except Exception as e:
        logger.error(f"Detailed analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detailed analysis failed: {str(e)}")

def _get_results_preview(results_df, num_records=5):
    """Get preview of results with key columns"""
    if results_df.empty:
        return []
    
    preview_columns = [
        'MSANCODE', 'EDGE', 'distribution_hostname', 'BNG_HOSTNAME', 
        'STATUS', 'CUST', 'cir_type', 'Impact'
    ]
    
    # Adjust columns based on available data
    available_preview_cols = [col for col in preview_columns if col in results_df.columns]
    
    # Add BITSTREAM_HOSTNAME if available
    if 'BITSTREAM_HOSTNAME' in results_df.columns and 'distribution_hostname' not in available_preview_cols:
        available_preview_cols.append('BITSTREAM_HOSTNAME')
    
    return results_df[available_preview_cols].head(num_records).to_dict('records')

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