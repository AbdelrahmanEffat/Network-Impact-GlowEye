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
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Network Impact Analysis Web Interface",
    description="Web interface for visualizing network impact analysis results",
    version="2.0.0"
)


env_path = Path(__file__).resolve().parents[0] / 'secrets.env'
load_dotenv(dotenv_path=env_path)

# Get the current directory of this file
current_dir = Path(__file__).parent

# Mount static files with correct path
static_dir = current_dir / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Setup templates with correct path
templates_dir = current_dir / "templates"
templates = Jinja2Templates(directory=templates_dir)

# Configuration
print("directory", Path(__file__).resolve())
print(os.getenv('BACKEND_API'))
print(os.getenv('PORT'))
API_BASE_URL = os.getenv('BACKEND_API')  # "http://localhost:8000"

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
    """Analyze network impact with single API call"""
    try:
        # Single API call to get complete results
        api_url = f"{API_BASE_URL}/analyze/complete"
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
        
        # Parse the complete response
        result = response.json()
        
        # Convert data back to DataFrames for template
        we_data = pd.DataFrame(result['data']['we_results']) if result['data']['we_results'] else pd.DataFrame()
        others_data = pd.DataFrame(result['data']['others_results']) if result['data']['others_results'] else pd.DataFrame()
        others_dash = pd.DataFrame(result['data']['others_dash']) if result['data']['others_dash'] else pd.DataFrame()
        
        template_data = {
            "request": request,
            "identifier": identifier,
            "identifier_type": identifier_type,
            "result": result['summary'],
            "we_data": we_data,
            "others_data": others_data,
            "others_dash": others_dash,
            "we_stats": result['statistics']['we_stats'],
            "others_stats": result['statistics']['others_stats'],
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return templates.TemplateResponse("results.html", template_data)
        
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        return templates.TemplateResponse(
            "error.html", 
            {"request": request, "error": error_msg}
        )


# ========== NEW PAGINATION PROXY ENDPOINTS ==========

@app.get("/api/we-data")
async def proxy_we_data(
    page: int = 1,
    per_page: int = 50,
    msancode: str = "",
    impact: str = "",
    edge: str = "",
    distribution: str = "",
    bng: str = ""
):
    """Proxy WE data pagination to backend API"""
    try:
        params = {
            'page': page,
            'per_page': per_page,
            'msancode': msancode,
            'impact': impact,
            'edge': edge,
            'distribution': distribution,
            'bng': bng
        }
        
        # Remove empty params
        params = {k: v for k, v in params.items() if v}
        
        api_url = f"{API_BASE_URL}/api/we-data"
        response = requests.get(api_url, params=params)
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        return response.json()
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Backend connection failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proxy failed: {str(e)}")


@app.get("/api/others-data")
async def proxy_others_data(
    page: int = 1,
    per_page: int = 50,
    msancode: str = "",
    impact: str = "",
    edge: str = "",
    bitstream: str = "",
    isp: str = "",
    service: str = ""
):
    """Proxy Others data pagination to backend API"""
    try:
        params = {
            'page': page,
            'per_page': per_page,
            'msancode': msancode,
            'impact': impact,
            'edge': edge,
            'bitstream': bitstream,
            'isp': isp,
            'service': service
        }
        
        # Remove empty params
        params = {k: v for k, v in params.items() if v}
        
        api_url = f"{API_BASE_URL}/api/others-data"
        response = requests.get(api_url, params=params)
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        return response.json()
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Backend connection failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proxy failed: {str(e)}")


@app.get("/api/filter-options/we")
async def proxy_we_filter_options():
    """Proxy WE filter options to backend API"""
    try:
        api_url = f"{API_BASE_URL}/api/filter-options/we"
        response = requests.get(api_url)
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        return response.json()
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Backend connection failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proxy failed: {str(e)}")


@app.get("/api/filter-options/others")
async def proxy_others_filter_options():
    """Proxy Others filter options to backend API"""
    try:
        api_url = f"{API_BASE_URL}/api/filter-options/others"
        response = requests.get(api_url)
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        return response.json()
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Backend connection failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proxy failed: {str(e)}")

# ========== END OF NEW PAGINATION ENDPOINTS ==========


@app.post("/download")
async def download_results(request: Request):
    """Download analysis results as CSV - UPDATED FOR POST"""
    try:
        # Get data from request body instead of query parameters
        data = await request.json()
        identifier = data.get('identifier')
        identifier_type = data.get('identifier_type', 'auto')
        
        if not identifier:
            raise HTTPException(status_code=400, detail="Identifier is required")
        
        # Call the backend CSV API
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

@app.get("/download")
async def download_results_get(identifier: str, identifier_type: str = "auto"):
    """GET version for backward compatibility"""
    # Redirect to POST or handle similarly
    return await download_results(Request(scope={"type": "http"}, receive=None))


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
