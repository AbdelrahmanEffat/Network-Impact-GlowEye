from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests
import pandas as pd
import json
from datetime import datetime
import io
import zipfile
import os
from pathlib import Path

app = FastAPI(
    title="Network Impact Analysis Web Interface",
    description="Web interface for visualizing network impact analysis results",
    version="2.0.0"
)

# Get the current directory of this file
current_dir = Path(__file__).parent

# Mount static files with correct path
static_dir = current_dir / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Setup templates with correct path
templates_dir = current_dir / "templates"
templates = Jinja2Templates(directory=templates_dir)

# Configuration - update with your API URL
API_BASE_URL = "http://localhost:8000"

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with analysis form"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_network_impact(
    request: Request,
    identifier: str = Form(...),
    identifier_type: str = Form("auto")
        ):
    """Analyze network impact and display results"""
    try:
        # Call the analysis API
        api_url = f"{API_BASE_URL}/analyze"
        response = requests.post(
            api_url,
            json={"identifier": identifier, "identifier_type": identifier_type}
        )
        
        if response.status_code != 200:
            error_msg = f"API request failed: {response.status_code} - {response.text}"
            return templates.TemplateResponse(
                "error.html", 
                {"request": request, "error": error_msg}
            )
        
        # Parse the API response
        result = response.json()
        
        # Get detailed data for both WE and Others
        we_data = await get_detailed_data(identifier, identifier_type, "we")
        others_data = await get_detailed_data(identifier, identifier_type, "others")
        
        # In the analyze_network_impact function, after getting we_data and others_data
        # Add code to calculate the statistics

        # Calculate WE statistics
        we_stats = {}
        if we_data is not None and not we_data.empty:
            we_unique = we_data.drop_duplicates(subset=['MSANCODE'])
            isolated_we = we_unique[we_unique['Impact'] == 'Isolated']
            partial_we = we_unique[we_unique['Impact'] == 'Partially Impacted']
            
            we_stats = {
                'isolated_msans': len(isolated_we),
                'partial_msans': len(partial_we),
                'isolated_sub': isolated_we['CUST'].sum() if not isolated_we.empty else 0,
                'partial_sub': partial_we['CUST'].sum() if not partial_we.empty else 0,
                'isolated_sub_voice': isolated_we['TOTAL_VOICE_CUST'].sum() if not isolated_we.empty else 0,
                'isolated_sub_data': isolated_we['TOTAL_DATA_CUST'].sum() if not isolated_we.empty else 0,
                'partial_sub_voice': partial_we['TOTAL_VOICE_CUST'].sum() if not partial_we.empty else 0,
                'partial_sub_data': partial_we['TOTAL_DATA_CUST'].sum() if not partial_we.empty else 0,

                 'partial_vic':(partial_we['VIC'] == 'VIC').sum() if not partial_we.empty else 0,
                 'iso_vic':(isolated_we['VIC']=='VIC').sum() if not isolated_we.empty else 0
            }

        # Calculate Others statistics
        others_stats = {}
        if others_data is not None and not others_data.empty:
            others_unique = others_data.drop_duplicates(subset=['MSANCODE'])
            isolated_others = others_unique[others_unique['Impact'] == 'Isolated']
            partial_others = others_unique[others_unique['Impact'] == 'Partially Impacted']
            print(isolated_others['VODA_WORNG_VLAN'].sum())
            others_stats = {
                'isolated_msans': len(isolated_others),
                'partial_msans': len(partial_others),
                'o_partial_vic':(partial_others['VIC'] == 'VIC').sum() if not partial_others.empty else 0,
                'o_iso_vic':(isolated_others['VIC']=='VIC').sum() if not isolated_others.empty else 0,
                'isolated_sub': isolated_others['TOTAL_OTHER_CUST'].sum() if not isolated_others.empty else 0,
                'partial_sub': partial_others['TOTAL_OTHER_CUST'].sum() if not partial_others.empty else 0,
                'isolated_voda': isolated_others['VODA_CUST'].sum() if not isolated_others.empty else 0,
                'isolated_voda_ubb': isolated_others['VODA_UBBT_CUST'].sum() if not isolated_others.empty else 0,
                'isolated_voda_hs': isolated_others['VODA_HS_CUST'].sum() if not isolated_others.empty else 0,
                'isolated_voda_ubb_ftth': isolated_others['VODA_UBBT_FTTH_CUST'].sum() if not isolated_others.empty else 0,
                'isolated_voda_hs_ftth': isolated_others['VODA_HS_FTTH_CUST'].sum() if not isolated_others.empty else 0,
                'isolated_orange': isolated_others['ORANGE_CUST'].sum() if not isolated_others.empty else 0,
                'isolated_orange_ubb': isolated_others['ORANGE_UBBT_CUST'].sum() if not isolated_others.empty else 0,
                'isolated_orange_hs': isolated_others['ORANGE_HS_CUST'].sum() if not isolated_others.empty else 0,
                'isolated_orange_ubb_ftth': isolated_others['ORANGE_UBBT_FTTH_CUST'].sum() if not isolated_others.empty else 0,
                'isolated_orange_hs_ftth': isolated_others['ORANGE_HS_FTTH_CUST'].sum() if not isolated_others.empty else 0,
                'isolated_etisalat': isolated_others['ETISLAT_CUST'].sum() if not isolated_others.empty else 0,
                'isolated_etisalat_ubb': isolated_others['ETISLAT_UBBT_CUST'].sum() if not isolated_others.empty else 0,
                'isolated_etisalat_hs': isolated_others['ETISLAT_HS_CUST'].sum() if not isolated_others.empty else 0,
                'isolated_etisalat_ubb_ftth': isolated_others['ETISLAT_UBBT_FTTH_CUST'].sum() if not isolated_others.empty else 0,
                'isolated_etisalat_hs_ftth': isolated_others['ETISLAT_HS_FTTH_CUST'].sum() if not isolated_others.empty else 0,
                'isolated_noor': isolated_others['NOOR_CUST'].sum() if not isolated_others.empty else 0,
                'isolated_noor_ubb': isolated_others['NOOR_UBBT_CUST'].sum() if not isolated_others.empty else 0,
                'isolated_noor_hs': isolated_others['NOOR_HS_CUST'].sum() if not isolated_others.empty else 0,
                'isolated_noor_ubb_ftth': isolated_others['NOOR_UBBT_FTTH_CUST'].sum() if not isolated_others.empty else 0,
                'isolated_noor_hs_ftth': isolated_others['NOOR_HS_FTTH_CUST'].sum() if not isolated_others.empty else 0,
                'wrong_noor': isolated_others['NOOR_WRONG_VLAN'].sum() if not isolated_others.empty else 0,
                'wrong_voda': isolated_others['VODA_WORNG_VLAN'].sum() if not isolated_others.empty else 0 ,
                'wrong_orange': isolated_others['ORANGE_WORNG_VLAN'].sum() if not isolated_others.empty else 0,
                'wrong_ets': isolated_others['ETISLAT_WRONG_VLAN'].sum() if not isolated_others.empty else 0
           

           
            }
        
        # Remove Duplicated records for others path, based on ISP
        others_dash = others_data.drop_duplicates(subset=['MSANCODE', 'EDGE', 'BITSTREAM_HOSTNAME'])

        # Add these to template_data
        template_data = {
            "request": request,
            "identifier": identifier,
            "identifier_type": identifier_type,
            "result": result,
            "we_data": we_data,
            "others_data": others_data,
            "others_dash": others_dash,
            "we_stats": we_stats,
            "others_stats": others_stats,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        }
        
        return templates.TemplateResponse("results.html", template_data)
        
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        return templates.TemplateResponse(
            "error.html", 
            {"request": request, "error": error_msg}
        )

@app.post("/api/analyze")

async def api_analyze_network_impact(identifier: str, identifier_type: str = "auto"):
    """API endpoint to get analysis results"""
    try:
        # Call the analysis API
        api_url = f"{API_BASE_URL}/analyze"
        response = requests.post(
            api_url,
            json={"identifier": identifier, "identifier_type": identifier_type}
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        return response.json()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/download")
async def download_results(identifier: str, identifier_type: str = "auto"):
    """Download analysis results as CSV"""
    try:
        # Call the CSV API
        api_url = f"{API_BASE_URL}/analyze/csv"
        response = requests.post(
            api_url,
            json={"identifier": identifier, "identifier_type": identifier_type}
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        # Generate filename
        safe_identifier = identifier.replace('/', '_').replace('\\', '_').replace('.', '_')
        filename = f"network_impact_{safe_identifier}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        
        # Return the file directly from the API
        return Response(
            content=response.content,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

async def get_detailed_data(identifier, identifier_type, data_type):
    """Get detailed data from the API"""
    try:
        api_url = f"{API_BASE_URL}/analyze/detailed"
        response = requests.post(
            api_url,
            json={"identifier": identifier, "identifier_type": identifier_type}
        )
        
        if response.status_code != 200:
            return pd.DataFrame()
            
        result = response.json()
        if data_type == "we":
            return pd.DataFrame(result['we_results'])
        else:
            return pd.DataFrame(result['others_results'])
            
    except Exception as e:
        print(f"Error getting detailed data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)