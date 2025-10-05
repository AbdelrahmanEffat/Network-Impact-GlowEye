import pandas as pd
import numpy as np
import networkx as nx
import time
import warnings
import os
from typing import List, Dict, Any
from collections import defaultdict
warnings.filterwarnings("ignore")

class StatisticsNetworkAnalyzer:
    """
    Proper node impact analyzer with combined node failure
    """
    
    def __init__(self, df_report_we, df_report_others, df_res_ospf, df_wan, df_agg):
        """Initialize with network data"""
        self.df_report_we = df_report_we.copy()
        self.df_report_others = df_report_others.copy()
        self.df_res_ospf = df_res_ospf.copy()
        self.df_wan = df_wan.copy()
        self.df_agg = df_agg.copy()
        self.export_dir = "impact_analysis_results"
        
        # Create export directory
        os.makedirs(self.export_dir, exist_ok=True)
        
        # Preprocess data
        self._preprocess_data()
        
        # Create network graph and calculate initial paths
        self.base_graph = self._create_base_graph()
        self._calculate_initial_paths()
        
        print(f"Statistics analyzer initialized. WE records: {len(self.df_report_we)}, Others records: {len(self.df_report_others)}")
    
    def _preprocess_data(self):
        """Clean and preprocess the data with better deduplication"""
        # WE data preprocessing
        self.df_report_we.drop(columns=['ID', 'ROWVERSION'], inplace=True, errors='ignore')
        
        # Filter WE data
        if 'distribution_INT' in self.df_report_we.columns and 'STATUS' in self.df_report_we.columns:
            df_filtered = self.df_report_we[
                (self.df_report_we.distribution_INT.isnull()) & (self.df_report_we.STATUS != 'ST')
            ]
            self.df_report_we = self.df_report_we[
                ~self.df_report_we.MSANCODE.isin(df_filtered.MSANCODE.unique())
            ]
        
        # Process port columns and deduplicate WE data
        if 'edge_port' in self.df_report_we.columns:
            self.df_report_we['edge_port'] = self.df_report_we['edge_port'].apply(
                lambda x: x.split('.')[0] if isinstance(x, str) else x
            )
        
        # Remove duplicates from WE data - keep first occurrence per MSAN
        self.df_report_we = self.df_report_we.drop_duplicates(subset=['MSANCODE'], keep='first')
        
        # Others data preprocessing
        self.df_report_others.drop(columns=['ID', 'ROWVERSION'], inplace=True, errors='ignore')
        if 'EDGE_PORT' in self.df_report_others.columns:
            self.df_report_others['EDGE_PORT'] = self.df_report_others['EDGE_PORT'].apply(
                lambda x: x.split('.')[0] if isinstance(x, str) else x
            )
        
        # Remove duplicates from Others data - keep first occurrence per MSAN
        self.df_report_others = self.df_report_others.drop_duplicates(subset=['MSANCODE'], keep='first')
        
        print(f"After deduplication - WE records: {len(self.df_report_we)}, Others records: {len(self.df_report_others)}")
    
    def _create_base_graph(self):
        """Create the base network graph"""
        always_excluded = ['INSOMNA-R02J-C-EG', 'INSOMNA-R01J-C-EG']
        
        df_filtered = self.df_wan[~self.df_wan['NODENAME'].isin(always_excluded)]
        df_filtered = df_filtered[~df_filtered['NEIGHBOR_HOSTNAME'].isin(always_excluded)]
        df_filtered = df_filtered.drop_duplicates()

        G = nx.Graph()
        for idx, row in df_filtered.iterrows():
            G.add_edge(row[0], row[1])
        return G
    
    def _calculate_initial_paths(self):
        """Calculate initial paths for all records"""
        print("Calculating initial paths...")
        
        # WE data paths
        if 'EDGE' in self.df_report_we.columns and 'distribution_hostname' in self.df_report_we.columns:
            self.df_report_we['Path1'] = self.df_report_we.apply(
                lambda row: self._calculate_path(self.base_graph, row['EDGE'], row['distribution_hostname']),
                axis=1
            )
        
        # Others data paths
        if 'EDGE' in self.df_report_others.columns and 'BITSTREAM_HOSTNAME' in self.df_report_others.columns:
            self.df_report_others['Path1'] = self.df_report_others.apply(
                lambda row: self._calculate_path(self.base_graph, row['EDGE'], row['BITSTREAM_HOSTNAME']),
                axis=1
            )
    
    def _calculate_path(self, graph, source, target):
        """Calculate path between two nodes"""
        if source not in graph or target not in graph:
            return None
        
        try:
            return nx.shortest_path(graph, source=source, target=target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def _ensure_serializable(self, obj):
        """Ensure all objects are JSON serializable"""
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
            return [self._ensure_serializable(item) for item in obj.tolist()]
        elif isinstance(obj, dict):
            return {key: self._ensure_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._ensure_serializable(item) for item in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        elif pd.isna(obj):
            return None
        else:
            try:
                return str(obj)
            except:
                return None
    
    def _save_impact_dataframes(self, we_impact: pd.DataFrame, others_impact: pd.DataFrame, failed_nodes: List[str]) -> Dict[str, str]:
        """Save impact DataFrames to CSV files for validation"""
        timestamp = int(time.time())
        node_str = "_".join(failed_nodes).replace("-", "_")
        
        file_paths = {}
        
        # Save WE impact data
        we_affected = we_impact[we_impact['Impact'].isin(['Isolated', 'Partially Impacted'])]
        if not we_affected.empty:
            we_filename = f"we_impact_{node_str}_{timestamp}.csv"
            we_filepath = os.path.join(self.export_dir, we_filename)
            we_affected.to_csv(we_filepath, index=False)
            file_paths['we_impact_csv'] = we_filepath
        
        # Save Others impact data
        others_affected = others_impact[others_impact['Impact'].isin(['Isolated', 'Partially Impacted'])]
        if not others_affected.empty:
            others_filename = f"others_impact_{node_str}_{timestamp}.csv"
            others_filepath = os.path.join(self.export_dir, others_filename)
            others_affected.to_csv(others_filepath, index=False)
            file_paths['others_impact_csv'] = others_filepath
        
        # Save combined summary
        combined_summary = self._create_combined_summary(we_affected, others_affected, failed_nodes)
        if combined_summary is not None:
            combined_filename = f"combined_impact_{node_str}_{timestamp}.csv"
            combined_filepath = os.path.join(self.export_dir, combined_filename)
            combined_summary.to_csv(combined_filepath, index=False)
            file_paths['combined_impact_csv'] = combined_filepath
        
        return file_paths
    
    def _create_combined_summary(self, we_affected: pd.DataFrame, others_affected: pd.DataFrame, failed_nodes: List[str]) -> pd.DataFrame:
        """Create a combined summary DataFrame for validation"""
        summary_data = []
        
        # WE impact summary
        if not we_affected.empty:
            we_isolated = we_affected[we_affected['Impact'] == 'Isolated']
            we_partial = we_affected[we_affected['Impact'] == 'Partially Impacted']
            
            we_summary = {
                'Data_Source': 'WE',
                'Failed_Nodes': ', '.join(failed_nodes),
                'Total_Affected_MSANs': we_affected['MSANCODE'].nunique(),
                'Isolated_MSANs': we_isolated['MSANCODE'].nunique(),
                'Partial_MSANs': we_partial['MSANCODE'].nunique(),
                'Total_Affected_Subscribers': we_affected['CUST'].sum() if 'CUST' in we_affected.columns else 0,
                'Isolated_Subscribers': we_isolated['CUST'].sum() if 'CUST' in we_isolated.columns else 0,
                'Partial_Subscribers': we_partial['CUST'].sum() if 'CUST' in we_partial.columns else 0,
                'Voice_Subscribers': we_affected['TOTAL_VOICE_CUST'].sum() if 'TOTAL_VOICE_CUST' in we_affected.columns else 0,
                'Data_Subscribers': we_affected['TOTAL_DATA_CUST'].sum() if 'TOTAL_DATA_CUST' in we_affected.columns else 0,
                'VIC_Cabinets': we_affected[we_affected['VIC'] == 'VIC']['MSANCODE'].nunique() if 'VIC' in we_affected.columns else 0
            }
            summary_data.append(we_summary)
        
        # Others impact summary
        if not others_affected.empty:
            others_isolated = others_affected[others_affected['Impact'] == 'Isolated']
            others_partial = others_affected[others_affected['Impact'] == 'Partially Impacted']
            
            others_summary = {
                'Data_Source': 'Others',
                'Failed_Nodes': ', '.join(failed_nodes),
                'Total_Affected_MSANs': others_affected['MSANCODE'].nunique(),
                'Isolated_MSANs': others_isolated['MSANCODE'].nunique(),
                'Partial_MSANs': others_partial['MSANCODE'].nunique(),
                'Total_Affected_Subscribers': others_affected['TOTAL_OTHER_CUST'].sum() if 'TOTAL_OTHER_CUST' in others_affected.columns else 0,
                'Isolated_Subscribers': others_isolated['TOTAL_OTHER_CUST'].sum() if 'TOTAL_OTHER_CUST' in others_isolated.columns else 0,
                'Partial_Subscribers': others_partial['TOTAL_OTHER_CUST'].sum() if 'TOTAL_OTHER_CUST' in others_partial.columns else 0,
                'VIC_Cabinets': others_affected[others_affected['VIC'] == 'VIC']['MSANCODE'].nunique() if 'VIC' in others_affected.columns else 0
            }
            summary_data.append(others_summary)
        
        if summary_data:
            return pd.DataFrame(summary_data)
        return None
    
    def analyze_nodes(self, nodes: List[str], export_csv: bool = True) -> Dict[str, Any]:
        """
        Analyze impact for multiple nodes failing together as a unit
        """
        start_time = time.time()
        
        try:
            print(f"Analyzing combined failure of nodes: {nodes}")
            
            # Create graph without ALL failed nodes together
            failed_graph = self.base_graph.copy()
            for node in nodes:
                if node in failed_graph:
                    failed_graph.remove_node(node)
            
            print(f"Graph after removing {len(nodes)} nodes: {len(failed_graph.nodes())} nodes remaining")
            
            # Analyze WE data impact with all nodes removed
            we_impact = self._analyze_we_impact(nodes, failed_graph)
            
            # Analyze Others data impact with all nodes removed  
            others_impact = self._analyze_others_impact(nodes, failed_graph)
            
            # Save DataFrames for validation
            csv_files = {}
            if export_csv:
                csv_files = self._save_impact_dataframes(we_impact, others_impact, nodes)
            
            # Calculate comprehensive statistics for the combined failure
            combined_stats = self._calculate_combined_stats(we_impact, others_impact, nodes)
            combined_stats['analysis_time_seconds'] = round(time.time() - start_time, 3)
            
            # Add CSV file paths to results
            if csv_files:
                combined_stats['exported_files'] = csv_files
            
            result = {
                'status': 'success',
                'failed_nodes': nodes,
                'results': self._ensure_serializable(combined_stats)
            }
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'analysis_time_seconds': round(time.time() - start_time, 3)
            }
    
    def _analyze_we_impact(self, failed_nodes: List[str], failed_graph: nx.Graph) -> pd.DataFrame:
        """Analyze WE data impact with all nodes removed together"""
        we_impact = self.df_report_we.copy()
        
        # Calculate new paths with ALL failed nodes removed
        we_impact['Path2'] = we_impact.apply(
            lambda row: self._calculate_path(failed_graph, row['EDGE'], row['distribution_hostname']),
            axis=1
        )
        
        # Determine impact based on path comparison
        we_impact['Impact'] = we_impact.apply(
            lambda row: self._determine_impact(row['Path1'], row['Path2'], failed_nodes),
            axis=1
        )
        
        return we_impact
    
    def _analyze_others_impact(self, failed_nodes: List[str], failed_graph: nx.Graph) -> pd.DataFrame:
        """Analyze Others data impact with all nodes removed together"""
        others_impact = self.df_report_others.copy()
        
        # Calculate new paths with ALL failed nodes removed
        others_impact['Path2'] = others_impact.apply(
            lambda row: self._calculate_path(failed_graph, row['EDGE'], row['BITSTREAM_HOSTNAME']),
            axis=1
        )
        
        # Determine impact based on path comparison
        others_impact['Impact'] = others_impact.apply(
            lambda row: self._determine_impact(row['Path1'], row['Path2'], failed_nodes),
            axis=1
        )
        
        return others_impact
    
    def _determine_impact(self, path1, path2, failed_nodes):
        """Determine impact based on path comparison with multiple failed nodes"""
        # If no initial path, consider as no service
        if path1 is None:
            return "No Service"
        
        # Check if ANY failed node is in initial path
        any_node_in_path = False
        if isinstance(path1, list):
            for failed_node in failed_nodes:
                if failed_node in path1:
                    any_node_in_path = True
                    break
        
        # If no failed nodes in path, no impact
        if not any_node_in_path:
            return "No Impact"
        
        # If failed nodes were in path and new path exists, partially impacted
        if path2 is not None:
            return "Partially Impacted"
        else:
            return "Isolated"
    
    def _calculate_combined_stats(self, we_impact: pd.DataFrame, others_impact: pd.DataFrame, failed_nodes: List[str]) -> Dict[str, Any]:
        """Calculate comprehensive statistics for combined node failure"""
        we_stats = self._calculate_we_statistics(we_impact)
        others_stats = self._calculate_others_statistics(others_impact)
        
        total_affected_subscribers = we_stats['affected_subscribers'] + others_stats['affected_subscribers']
        total_isolated_subscribers = we_stats['isolated_subscribers'] + others_stats['isolated_subscribers']
        total_partial_subscribers = we_stats['partial_subscribers'] + others_stats['partial_subscribers']
        
        # Calculate ratios based on total unique records, not sum of all records
        total_we_subscribers = self.df_report_we['CUST'].sum() if 'CUST' in self.df_report_we.columns else len(self.df_report_we)
        total_others_subscribers = self.df_report_others['TOTAL_OTHER_CUST'].sum() if 'TOTAL_OTHER_CUST' in self.df_report_others.columns else len(self.df_report_others)
        
        combined_stats = {
            'failed_nodes': failed_nodes,
            'timestamp': time.time(),
            'we_statistics': we_stats,
            'others_statistics': others_stats,
            'summary': {
                'total_affected_msans': we_stats['affected_msans'] + others_stats['affected_msans'],
                'total_affected_subscribers': total_affected_subscribers,
                'total_isolated_subscribers': total_isolated_subscribers,
                'total_partial_subscribers': total_partial_subscribers,
                'we_impact_ratio': round(we_stats['affected_subscribers'] / max(1, total_we_subscribers), 4),
                'others_impact_ratio': round(others_stats['affected_subscribers'] / max(1, total_others_subscribers), 4),
                'overall_severity': self._calculate_severity(total_affected_subscribers),
                'failure_scenario': f"Combined failure of {len(failed_nodes)} nodes",
                'data_quality_notes': {
                    'we_unique_msans_analyzed': len(self.df_report_we),
                    'others_unique_msans_analyzed': len(self.df_report_others)
                }
            }
        }
        
        return self._ensure_serializable(combined_stats)
    
    def _calculate_we_statistics(self, we_impact: pd.DataFrame) -> Dict[str, Any]:
        """Calculate WE-specific statistics - FIXED to only count affected records"""
        if we_impact.empty:
            return self._get_empty_we_stats()
        
        # Only consider records that are actually affected
        affected_we = we_impact[we_impact['Impact'].isin(['Isolated', 'Partially Impacted'])]
        
        if affected_we.empty:
            return self._get_empty_we_stats()
        
        # Impact breakdown
        isolated = affected_we[affected_we['Impact'] == 'Isolated']
        partial = affected_we[affected_we['Impact'] == 'Partially Impacted']
        
        # Unique MSANs affected
        isolated_msans = isolated['MSANCODE'].nunique() if 'MSANCODE' in isolated.columns else 0
        partial_msans = partial['MSANCODE'].nunique() if 'MSANCODE' in partial.columns else 0
        affected_msans = isolated_msans + partial_msans
        
        # Customer counts - ONLY from affected records
        isolated_cust = isolated['CUST'].sum() if 'CUST' in isolated.columns else 0
        partial_cust = partial['CUST'].sum() if 'CUST' in partial.columns else 0
        total_affected_cust = isolated_cust + partial_cust
        
        # Service type breakdown - ONLY from affected records
        voice_cust = affected_we['TOTAL_VOICE_CUST'].sum() if 'TOTAL_VOICE_CUST' in affected_we.columns else 0
        data_cust = affected_we['TOTAL_DATA_CUST'].sum() if 'TOTAL_DATA_CUST' in affected_we.columns else 0
        
        # VIC cabinet impact - ONLY from affected records
        vic_affected = affected_we[affected_we['VIC'] == 'VIC']['MSANCODE'].nunique() if 'VIC' in affected_we.columns and 'MSANCODE' in affected_we.columns else 0
        
        stats = {
            'affected_msans': int(affected_msans),
            'affected_subscribers': int(total_affected_cust),
            'isolated_subscribers': int(isolated_cust),
            'partial_subscribers': int(partial_cust),
            'voice_subscribers_affected': int(voice_cust),
            'data_subscribers_affected': int(data_cust),
            'vic_cabinets_affected': int(vic_affected),
            'impact_breakdown': {
                'isolated_msans': int(isolated_msans),
                'partial_msans': int(partial_msans),
                'no_impact_msans': int(we_impact[we_impact['Impact'] == 'No Impact']['MSANCODE'].nunique() if 'MSANCODE' in we_impact.columns else 0)
            },
            'records_analyzed': int(len(we_impact)),
            'affected_records': int(len(affected_we))
        }
        
        return self._ensure_serializable(stats)
    
    def _calculate_others_statistics(self, others_impact: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Others-specific statistics - FIXED to only count affected records"""
        if others_impact.empty:
            return self._get_empty_others_stats()
        
        # Only consider records that are actually affected
        affected_others = others_impact[others_impact['Impact'].isin(['Isolated', 'Partially Impacted'])]
        
        if affected_others.empty:
            return self._get_empty_others_stats()
        
        # Impact breakdown
        isolated = affected_others[affected_others['Impact'] == 'Isolated']
        partial = affected_others[affected_others['Impact'] == 'Partially Impacted']
        
        # Unique MSANs affected
        isolated_msans = isolated['MSANCODE'].nunique() if 'MSANCODE' in isolated.columns else 0
        partial_msans = partial['MSANCODE'].nunique() if 'MSANCODE' in partial.columns else 0
        affected_msans = isolated_msans + partial_msans
        
        # ISP breakdown - ONLY from affected records
        isp_stats = {}
        for isp in ['VODA', 'ORANGE', 'ETISLAT', 'NOOR']:
            isp_cust_col = f'{isp}_CUST'
            if isp_cust_col in affected_others.columns:
                isp_stats[f'{isp.lower()}_subscribers'] = int(affected_others[isp_cust_col].sum())
        
        # Service type breakdown - ONLY from affected records
        service_stats = {}
        for service in ['UBB', 'HS']:
            for isp in ['VODA', 'ORANGE', 'ETISLAT', 'NOOR']:
                service_col = f'{isp}_{service}_CUST'
                if service_col in affected_others.columns:
                    service_stats[f'{isp.lower()}_{service.lower()}_subscribers'] = int(affected_others[service_col].sum())
        
        # Total customers - ONLY from affected records
        isolated_cust = isolated['TOTAL_OTHER_CUST'].sum() if 'TOTAL_OTHER_CUST' in isolated.columns else 0
        partial_cust = partial['TOTAL_OTHER_CUST'].sum() if 'TOTAL_OTHER_CUST' in partial.columns else 0
        total_affected_cust = isolated_cust + partial_cust
        
        # VIC cabinet impact - ONLY from affected records
        vic_affected = affected_others[affected_others['VIC'] == 'VIC']['MSANCODE'].nunique() if 'VIC' in affected_others.columns and 'MSANCODE' in affected_others.columns else 0
        
        stats = {
            'affected_msans': int(affected_msans),
            'affected_subscribers': int(total_affected_cust),
            'isolated_subscribers': int(isolated_cust),
            'partial_subscribers': int(partial_cust),
            'vic_cabinets_affected': int(vic_affected),
            'isp_breakdown': isp_stats,
            'service_breakdown': service_stats,
            'impact_breakdown': {
                'isolated_msans': int(isolated_msans),
                'partial_msans': int(partial_msans),
                'no_impact_msans': int(others_impact[others_impact['Impact'] == 'No Impact']['MSANCODE'].nunique() if 'MSANCODE' in others_impact.columns else 0)
            },
            'records_analyzed': int(len(others_impact)),
            'affected_records': int(len(affected_others))
        }
        
        return self._ensure_serializable(stats)
    
    def _calculate_severity(self, total_affected_subscribers: int) -> str:
        """Calculate overall severity level"""
        if total_affected_subscribers == 0:
            return "None"
        elif total_affected_subscribers < 100:
            return "Low"
        elif total_affected_subscribers < 1000:
            return "Medium"
        elif total_affected_subscribers < 10000:
            return "High"
        else:
            return "Critical"
    
    def _get_empty_we_stats(self) -> Dict[str, Any]:
        """Return empty WE statistics"""
        return self._ensure_serializable({
            'affected_msans': 0,
            'affected_subscribers': 0,
            'isolated_subscribers': 0,
            'partial_subscribers': 0,
            'voice_subscribers_affected': 0,
            'data_subscribers_affected': 0,
            'vic_cabinets_affected': 0,
            'impact_breakdown': {'isolated_msans': 0, 'partial_msans': 0, 'no_impact_msans': 0},
            'records_analyzed': 0,
            'affected_records': 0
        })
    
    def _get_empty_others_stats(self) -> Dict[str, Any]:
        """Return empty Others statistics"""
        return self._ensure_serializable({
            'affected_msans': 0,
            'affected_subscribers': 0,
            'isolated_subscribers': 0,
            'partial_subscribers': 0,
            'vic_cabinets_affected': 0,
            'isp_breakdown': {},
            'service_breakdown': {},
            'impact_breakdown': {'isolated_msans': 0, 'partial_msans': 0, 'no_impact_msans': 0},
            'records_analyzed': 0,
            'affected_records': 0
        })