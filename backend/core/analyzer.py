import pandas as pd
import numpy as np
import networkx as nx
import time
import json
import warnings
warnings.filterwarnings("ignore")

class UnifiedNetworkImpactAnalyzer:
    """
    Unified module for analyzing network impact from node or exchange failures.
    Handles both WE (network topology) and Others (bitstream topology) scenarios.
    """
    
    # Define column mappings as class constants
    NETWORK_COLUMN_MAP = {
        'target_hostname': 'distribution_hostname',
        'target_exchange': 'distribution_Exchange',
        'edge_exchange': 'edge_exchange',
        'port': 'edge_port',
        'vlan': 'VLAN',
        'bng_hostname': 'BNG_HOSTNAME'
    }
    
    BITSTREAM_COLUMN_MAP = {
        'target_hostname': 'BITSTREAM_HOSTNAME',
        'target_exchange': 'Bitstream_exchange',
        'edge_exchange': 'EDGE_exchange',
        'port': 'EDGE_PORT',
        'vlan': 'EDGE_VLAN',
        'bng_hostname': None
    }
    
    def __init__(self, df_report, df_res_ospf, df_wan, df_agg):
        """Initialize with network data"""
        self.df_report = df_report.copy()
        self.df_res_ospf = df_res_ospf.copy()
        self.df_wan = df_wan.copy()
        self.df_agg = df_agg.copy()
        self.model = None
        self.final_df = None
        self.data_type = self._detect_data_type()
        self.column_map = self._get_column_mappings()
        
        print(f"Detected data type: {self.data_type}")
        
    def _detect_data_type(self):
        """Auto-detect data type based on available columns"""
        if 'distribution_hostname' in self.df_report.columns and 'BNG_HOSTNAME' in self.df_report.columns:
            return 'network'
        elif 'BITSTREAM_HOSTNAME' in self.df_report.columns:
            return 'bitstream'
        else:
            raise ValueError("Unable to detect data type. Missing required columns.")
    
    def _get_column_mappings(self):
        """Get column mappings based on data type"""
        return self.NETWORK_COLUMN_MAP if self.data_type == 'network' else self.BITSTREAM_COLUMN_MAP
    
    def preprocess_data(self):
        """Clean and preprocess the report data based on data type"""
        # Remove ID and ROWVERSION columns if they exist
        self.df_report.drop(columns=['ID', 'ROWVERSION'], inplace=True, errors='ignore')
        
        if self.data_type == 'network':
            # Filter out records with null BNG_HOSTNAME and non-ST status (WE specific)
            df_filtered = self.df_report[
                (self.df_report.distribution_INT.isnull()) & (self.df_report.STATUS != 'ST')
            ]
            
            # Remove MSANCODEs from filtered records
            self.df_report = self.df_report[
                ~self.df_report.MSANCODE.isin(df_filtered.MSANCODE.unique())
            ]
        
        # Process port column based on data type
        port_col = self.column_map['port']
        if port_col in self.df_report.columns:
            self.df_report[port_col] = self.df_report[port_col].apply(lambda x: x.split('.')[0])
        
        # Process OSPF data (common for both types)
        self.df_res_ospf['LOCAL_INTERFACE'] = self.df_res_ospf['LOCAL_INTERFACE'].apply(
            lambda x: x.split(':')[0] if isinstance(x, str) and ':' in x else x
        )
        self.df_res_ospf['NEIGHBOR_INTERFACE'] = self.df_res_ospf['NEIGHBOR_INTERFACE'].apply(
            lambda x: x.split(':')[0] if isinstance(x, str) and ':' in x else x
        )
        
        print(f"Data preprocessed. Final shape: {self.df_report.shape}")
    

    def _find_records_with_nodes_in_path(self, affected_nodes):
        """Find MSANs that have any of the affected nodes in their paths - returns MSANs, not individual records"""
        # Check Path column for affected nodes
        path_mask = self.final_df['Path'].apply(
            lambda path: (isinstance(path, list) and len(path) >= 3 and 
                        any(node in path[1:-1] for node in affected_nodes))
        )
        
        # Check Path2 column if it exists
        path2_mask = pd.Series([False] * len(self.final_df))
        if 'Path2' in self.final_df.columns:
            path2_mask = self.final_df['Path2'].apply(
                lambda path: (isinstance(path, list) and len(path) >= 3 and 
                            any(node in path[1:-1] for node in affected_nodes))
            )
        
        # Find all affected MSANs (not individual records)
        if self.data_type == 'network':
            # For network type, get MSANs with UP status that have affected nodes in paths
            affected_up = self.final_df[
                (path_mask | path2_mask) & (self.final_df['STATUS'] == 'UP')
            ]
            
            if affected_up.empty:
                return pd.DataFrame()
            
            affected_msans = affected_up.MSANCODE.unique()
        else:
            # For Others type, get MSANs that have affected nodes in paths
            affected_records = self.final_df[path_mask | path2_mask]
            if affected_records.empty:
                return pd.DataFrame()
            
            affected_msans = affected_records.MSANCODE.unique()
        
        # Return a small DataFrame with just the affected MSANs for consistency
        return pd.DataFrame({'MSANCODE': affected_msans})


    # Helper method to determine MSAN-level impact:, it should be replaced by abvove methode
    def _calculate_msan_level_impact(self, results_df):
        """
        Calculate impact at MSAN level instead of record level.
        If any record for an MSAN is 'Partially Impacted', the entire MSAN is 'Partially Impacted'
        """
        if results_df.empty or 'MSANCODE' not in results_df.columns:
            return results_df
        
        # Ensure Impact column exists
        if 'Impact' not in results_df.columns:
            print("Warning: Impact column not found, setting default impact")
            results_df['Impact'] = 'Isolated'  # Default to most conservative impact
        
        # Group by MSANCODE and determine the overall impact
        msan_impact = {}
        for msan in results_df['MSANCODE'].unique():
            msan_records = results_df[results_df['MSANCODE'] == msan]
            
            # If ANY record is Partially Impacted, the entire MSAN is Partially Impacted
            if 'Partially Impacted' in msan_records['Impact'].values:
                msan_impact[msan] = 'Partially Impacted'
            else:
                # Check if all records are Isolated
                if 'Isolated' in msan_records['Impact'].values:
                    msan_impact[msan] = 'Isolated'
                else:
                    # Default to the first impact type found
                    msan_impact[msan] = msan_records['Impact'].iloc[0] if not msan_records.empty else 'Unknown'
        
        # Apply the MSAN-level impact to all records
        results_df = results_df.copy()
        results_df['Impact'] = results_df['MSANCODE'].map(msan_impact)
        
        return results_df
    

    def generate_base_results(self, dwn_identifier):
        """Generate base results with path calculations"""
        start_time = time.time()
        
        # Create the unified CIR model
        self.model = UnifiedCIRModel(
            self.df_report, self.df_res_ospf, self.df_wan, self.df_agg, 
            dwn_identifier, self.data_type, self.column_map
        )
        self.final_df = self.model.generate_results()
        
        elapsed_time = time.time() - start_time
        print(f"Base results generated in {elapsed_time:.3f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"Final DataFrame shape: {self.final_df.shape}")
        
        return self.final_df
    
    ## ISP-Case, ips lvl not whole MSAN
    def _calculate_impact_for_others(self, results_df):
        """
        Calculate impact for Others data - ensure we have all MSAN records
        and only mark affected ones
        """
        if results_df.empty:
            return results_df
        
        # For Others data, we want to keep all records but only mark affected ones
        # The impact should already be set by the individual analysis methods
        
        # Ensure we have 'No Impact' for records without specific impact
        if 'Impact' not in results_df.columns:
            results_df['Impact'] = 'No Impact'
        else:
            results_df['Impact'] = results_df['Impact'].fillna('No Impact')
        
        return results_df

    def analyze_exchange_impact(self, dwn_exchange):
        """Analyze impact when an exchange fails - with correct logic for Others data"""
        if self.final_df is None:
            raise ValueError("Must call generate_base_results() first")
        
        results = []
        
        # Case 1: Edge Exchange directly impacted
        edge_col = self.column_map['edge_exchange']
        edge_impact = self._analyze_direct_impact(edge_col, dwn_exchange, 'Isolated')
        if not edge_impact.empty:
            results.append(edge_impact)
            
        # Case 2: Target Exchange directly impacted (AGG/Bitstream)
        target_impact = self._analyze_target_exchange_impact(dwn_exchange)
        if not target_impact.empty:
            results.append(target_impact)
            
        # Case 3: Physical path impact
        physical_impact = self._analyze_exchange_physical_path_impact(dwn_exchange)
        if not physical_impact.empty:
            results.append(physical_impact)
            
        # Combine results
        combined_results = self._combine_results(results)
        
        if not combined_results.empty:
            if self.data_type == 'network':
                # For WE data, use MSAN-level impact
                combined_results = self._calculate_msan_level_impact(combined_results)
            else:
                # For Others data, ensure proper impact calculation
                combined_results = self._calculate_impact_for_others(combined_results)
        
        return combined_results

    def analyze_node_impact(self, dwn_node):
        """Analyze impact when a node fails - with correct logic for Others data"""
        if self.final_df is None:
            raise ValueError("Must call generate_base_results() first")
        
        results = []
        
        # Case 1: Edge directly impacted
        edge_impact = self._analyze_direct_impact('EDGE', dwn_node, 'Isolated')
        if not edge_impact.empty:
            results.append(edge_impact)
            
        # Case 2: Target node directly impacted (AGG/BNG/Bitstream)
        target_impact = self._analyze_target_node_impact(dwn_node)
        if not target_impact.empty:
            results.append(target_impact)
            
        # Case 3: Physical path impact
        physical_impact = self._analyze_node_physical_path_impact(dwn_node)
        if not physical_impact.empty:
            results.append(physical_impact)
            
        # Combine results
        combined_results = self._combine_results(results)
        
        if not combined_results.empty:
            if self.data_type == 'network':
                # For WE data, use MSAN-level impact
                combined_results = self._calculate_msan_level_impact(combined_results)
            else:
                # For Others data, ensure proper impact calculation
                combined_results = self._calculate_impact_for_others(combined_results)
        
        return combined_results
    
    def _analyze_direct_impact(self, column, value, impact_type):
        """Generic method for analyzing direct impact on a column - with Impact column guarantee"""
        impact_df = self.final_df[self.final_df[column] == value].copy()
        if not impact_df.empty:
            impact_df['Impact'] = impact_type
        return impact_df
    
    def _analyze_target_exchange_impact(self, dwn_exchange):
        """Analyze direct impact on target exchange (AGG/Bitstream) - fixed to keep all MSAN records"""
        target_exchange_col = self.column_map['target_exchange']
        target_hostname_col = self.column_map['target_hostname']
        
        # Find records directly impacted by the exchange failure
        direct_impact = self.final_df[self.final_df[target_exchange_col] == dwn_exchange]
        
        if direct_impact.empty:
            return pd.DataFrame()
        
        if self.data_type == 'network':
            # For network type, get all records for affected MSANs
            all_affected = self.final_df[
                self.final_df.MSANCODE.isin(direct_impact.MSANCODE.unique())
            ].copy()
            
            # Get ALL nodes in the affected exchange
            affected_nodes = self._get_exchange_nodes(dwn_exchange)
            
            # Calculate alternative paths
            graph = self.model.draw_graph(excluded_nodes=affected_nodes)
            
            all_affected['Path2'] = all_affected.apply(
                lambda row: self.model.calculate_path(graph, row['EDGE'], row[target_hostname_col]),
                axis=1
            )
            
            all_affected['Impact'] = all_affected['Path2'].apply(
                lambda x: 'Partially Impacted' if isinstance(x, list) else 'Isolated'
            )
        else:
            # For Others data, get ALL records for the affected MSANs but only mark specific ones
            affected_msans = direct_impact.MSANCODE.unique()
            all_affected = self.final_df[
                self.final_df.MSANCODE.isin(affected_msans)
            ].copy()
            
            # Get ALL nodes in the affected exchange
            affected_nodes = self._get_exchange_nodes(dwn_exchange)
            
            # Calculate alternative paths ONLY for the directly affected records
            graph = self.model.draw_graph(excluded_nodes=affected_nodes)
            
            # Create a mask for directly affected records
            direct_impact_mask = all_affected[target_exchange_col] == dwn_exchange
            
            # Initialize Path2 and Impact columns
            all_affected['Path2'] = None
            all_affected['Impact'] = 'No Impact'  # Default to no impact
            
            # Calculate paths and set impact only for directly affected records
            affected_indices = all_affected[direct_impact_mask].index
            for idx in affected_indices:
                row = all_affected.loc[idx]
                path2 = self.model.calculate_path(graph, row['EDGE'], row[target_hostname_col])
                all_affected.at[idx, 'Path2'] = path2
                all_affected.at[idx, 'Impact'] = 'Partially Impacted' if isinstance(path2, list) else 'Isolated'
        
        return all_affected

    def _analyze_target_node_impact(self, dwn_node):
        """Analyze direct impact on target nodes (AGG/BNG/Bitstream) - fixed to keep all MSAN records"""
        target_hostname_col = self.column_map['target_hostname']
        bng_hostname_col = self.column_map['bng_hostname']
        
        if self.data_type == 'network' and bng_hostname_col:
            # Network data: BNG or distribution hostname affected
            direct_impact = self.final_df[
                (self.final_df[bng_hostname_col] == dwn_node) | 
                (self.final_df[target_hostname_col] == dwn_node)
            ]
        else:
            # Others data: Only bitstream hostname affected
            direct_impact = self.final_df[self.final_df[target_hostname_col] == dwn_node]
        
        if direct_impact.empty:
            return pd.DataFrame()
        
        if self.data_type == 'network':
            # For network type, get all records for affected MSANs
            all_affected = self.final_df[
                self.final_df.MSANCODE.isin(direct_impact.MSANCODE.unique())
            ].copy()
            
            # Determine impact based on circuit type
            msan_impacts = {}
            for msan in all_affected.MSANCODE.unique():
                msan_df = all_affected[all_affected.MSANCODE == msan]
                
                if 'cir_type' in msan_df.columns and 'Single' in msan_df['cir_type'].values:
                    msan_impacts[msan] = 'Isolated'
                else:
                    msan_impacts[msan] = 'Partially Impacted'
            
            all_affected['Impact'] = all_affected['MSANCODE'].map(msan_impacts)
        else:
            # For Others data, get ALL records for the affected MSANs but only mark specific ones
            affected_msans = direct_impact.MSANCODE.unique()
            all_affected = self.final_df[
                self.final_df.MSANCODE.isin(affected_msans)
            ].copy()
            
            # Calculate alternative paths ONLY for the directly affected records
            graph = self.model.draw_graph(excluded_nodes=[dwn_node])
            
            # Create a mask for directly affected records
            if self.data_type == 'network' and bng_hostname_col:
                direct_impact_mask = (all_affected[bng_hostname_col] == dwn_node) | (all_affected[target_hostname_col] == dwn_node)
            else:
                direct_impact_mask = all_affected[target_hostname_col] == dwn_node
            
            # Initialize Path2 and Impact columns
            all_affected['Path2'] = None
            all_affected['Impact'] = 'No Impact'  # Default to no impact
            
            # Calculate paths and set impact only for directly affected records
            affected_indices = all_affected[direct_impact_mask].index
            for idx in affected_indices:
                row = all_affected.loc[idx]
                path2 = self.model.calculate_path(graph, row['EDGE'], row[target_hostname_col])
                all_affected.at[idx, 'Path2'] = path2
                all_affected.at[idx, 'Impact'] = 'Partially Impacted' if isinstance(path2, list) else 'Isolated'
        
        return all_affected

    def _analyze_exchange_physical_path_impact(self, dwn_exchange):
        """Analyze physical path impact for exchange failure - fixed to keep all MSAN records"""
        # Get nodes in the affected exchange
        affected_nodes = self._get_exchange_nodes(dwn_exchange)
        
        if not affected_nodes:
            return pd.DataFrame()
        
        # Find records with affected nodes in their paths
        affected_records = self._find_records_with_nodes_in_path(affected_nodes)
        
        if affected_records.empty:
            return pd.DataFrame()
        
        # Calculate alternative paths
        graph = self.model.draw_graph(excluded_nodes=affected_nodes)
        target_hostname_col = self.column_map['target_hostname']
        
        if self.data_type == 'network':
            # For network data, get all records for affected MSANs
            affected_msans = affected_records.MSANCODE.unique()
            all_affected = self.final_df[
                self.final_df.MSANCODE.isin(affected_msans)
            ].copy()
            
            all_affected['Path2'] = all_affected.apply(
                lambda row: self.model.calculate_path(graph, row['EDGE'], row[target_hostname_col]),
                axis=1
            )
            
            # Set impact based on path existence
            all_affected['Impact'] = all_affected['Path2'].apply(
                lambda x: 'Partially Impacted' if isinstance(x, list) else 'Isolated'
            )
            
            # Apply MSAN-level impact
            all_affected = self._calculate_msan_level_impact(all_affected)
        else:
            # For Others data, get ALL records for affected MSANs but only mark specific ones
            affected_msans = affected_records.MSANCODE.unique()
            all_affected = self.final_df[
                self.final_df.MSANCODE.isin(affected_msans)
            ].copy()
            
            # Initialize Path2 and Impact columns
            all_affected['Path2'] = None
            all_affected['Impact'] = 'No Impact'  # Default to no impact
            
            # Calculate paths and set impact only for records that have affected nodes in their paths
            for idx, row in all_affected.iterrows():
                path = row['Path']
                if isinstance(path, list) and any(node in path[1:-1] for node in affected_nodes):
                    path2 = self.model.calculate_path(graph, row['EDGE'], row[target_hostname_col])
                    all_affected.at[idx, 'Path2'] = path2
                    all_affected.at[idx, 'Impact'] = 'Partially Impacted' if isinstance(path2, list) else 'Isolated'
        
        return all_affected

    def _analyze_node_physical_path_impact(self, dwn_node):
        """Analyze physical path impact for node failure - fixed to keep all MSAN records"""
        # Find records with the node in their paths
        affected_records = self._find_records_with_nodes_in_path([dwn_node])
        
        if affected_records.empty:
            return pd.DataFrame()
        
        # Calculate alternative paths excluding the specific node
        graph = self.model.draw_graph(excluded_nodes=[dwn_node])
        target_hostname_col = self.column_map['target_hostname']
        
        if self.data_type == 'network':
            # For network data, get all records for affected MSANs
            affected_msans = affected_records.MSANCODE.unique()
            all_affected = self.final_df[
                self.final_df.MSANCODE.isin(affected_msans)
            ].copy()
            
            all_affected['Path2'] = all_affected.apply(
                lambda row: self.model.calculate_path(graph, row['EDGE'], row[target_hostname_col]),
                axis=1
            )
            
            # Set impact based on path existence
            all_affected['Impact'] = all_affected['Path2'].apply(
                lambda x: 'Partially Impacted' if isinstance(x, list) else 'Isolated'
            )
            
            # Apply MSAN-level impact
            all_affected = self._calculate_msan_level_impact(all_affected)
        else:
            # For Others data, get ALL records for affected MSANs but only mark specific ones
            affected_msans = affected_records.MSANCODE.unique()
            all_affected = self.final_df[
                self.final_df.MSANCODE.isin(affected_msans)
            ].copy()
            
            # Initialize Path2 and Impact columns
            all_affected['Path2'] = None
            all_affected['Impact'] = 'No Impact'  # Default to no impact
            
            # Calculate paths and set impact only for records that have the node in their paths
            for idx, row in all_affected.iterrows():
                path = row['Path']
                if isinstance(path, list) and dwn_node in path[1:-1]:
                    path2 = self.model.calculate_path(graph, row['EDGE'], row[target_hostname_col])
                    all_affected.at[idx, 'Path2'] = path2
                    all_affected.at[idx, 'Impact'] = 'Partially Impacted' if isinstance(path2, list) else 'Isolated'
        
        return all_affected
        

    
    def _get_exchange_nodes(self, dwn_exchange):
        """Get all nodes belonging to a specific exchange - enhanced version"""
        # Extract exchange code from exchange name
        exchange_code = dwn_exchange.split('.')[-1] if '.' in dwn_exchange else dwn_exchange
        
        edge_exchange_col = self.column_map['edge_exchange']
        target_exchange_col = self.column_map['target_exchange']
        target_hostname_col = self.column_map['target_hostname']
        
        # Find all nodes in the exchange from various sources
        edge_nodes = self.final_df[
            self.final_df[edge_exchange_col] == dwn_exchange
        ]['EDGE'].unique().tolist()
        
        target_nodes = self.final_df[
            self.final_df[target_exchange_col] == dwn_exchange
        ][target_hostname_col].unique().tolist()
        
        # Also check WAN data for nodes in this exchange
        wan_nodes_in_exchange = []
        if hasattr(self, 'df_wan') and self.df_wan is not None:
            # Look for nodes that have the exchange code in their name
            wan_nodes_in_exchange = self.df_wan[
                self.df_wan['NODENAME'].str.contains(exchange_code, na=False)
            ]['NODENAME'].unique().tolist()
            
            wan_neighbors_in_exchange = self.df_wan[
                self.df_wan['NEIGHBOR_HOSTNAME'].str.contains(exchange_code, na=False)
            ]['NEIGHBOR_HOSTNAME'].unique().tolist()
            
            wan_nodes_in_exchange.extend(wan_neighbors_in_exchange)
        
        # Combine and deduplicate all nodes
        all_nodes = list(set(edge_nodes + target_nodes + wan_nodes_in_exchange))
        
        # Filter nodes that actually belong to the exchange based on naming convention
        exchange_nodes = [
            node for node in all_nodes 
            if node and (
                # Check if node follows the naming convention and contains exchange code
                (len(node.split('-')) > 2 and node.split('-')[2] == exchange_code) or
                # Or if the node name contains the exchange code
                exchange_code in node
            )
        ]
        
        print(f"Found {len(exchange_nodes)} nodes in exchange {dwn_exchange}")#: {exchange_nodes}
        return exchange_nodes
    
    def _find_msans_with_nodes_in_path(self, affected_nodes):
        """Find MSANs that have any of the affected nodes in their paths"""
        # Check Path column
        path_mask = self.final_df['Path'].apply(
            lambda path: (isinstance(path, list) and len(path) >= 3 and 
                         any(node in path[1:-1] for node in affected_nodes))
        )
        
        # Check Path2 column if it exists
        path2_mask = pd.Series([False] * len(self.final_df))
        if 'Path2' in self.final_df.columns:
            path2_mask = self.final_df['Path2'].apply(
                lambda path: (isinstance(path, list) and len(path) >= 3 and 
                             any(node in path[1:-1] for node in affected_nodes))
            )
        
        # Different logic based on data type
        if self.data_type == 'network':
            # For network type, get MSANs with UP status that have affected nodes in paths
            affected_up = self.final_df[
                (path_mask | path2_mask) & (self.final_df['STATUS'] == 'UP')
            ]
            
            if affected_up.empty:
                return pd.DataFrame()
            
            # Get all records for affected MSANs (both UP and ST)
            all_affected = self.final_df[
                self.final_df.MSANCODE.isin(affected_up.MSANCODE.unique())
            ].copy()
        else:
            # For bitstream type, just get records with affected nodes in paths
            all_affected = self.final_df[path_mask | path2_mask].copy()
        
        return all_affected
    
    def _combine_results(self, result_list):
        """Combine and deduplicate results from different impact analyses"""
        if not result_list:
            return pd.DataFrame()
        
        # Remove duplicates between result sets
        combined = result_list[0].copy()
        
        for result_df in result_list[1:]:
            # Only add records not already in combined
            new_records = result_df[
                ~result_df.MSANCODE.isin(combined.MSANCODE.unique())
            ]
            if not new_records.empty:
                combined = pd.concat([combined, new_records], ignore_index=True)
        
        # Final deduplication
        combined_temp = combined.copy()
        
        # Convert lists to tuples for deduplication
        for col in ['Path', 'Path2']:
            if col in combined_temp.columns:
                combined_temp[col] = combined_temp[col].apply(
                    lambda x: tuple(x) if isinstance(x, list) else x
                )
        
        return combined_temp.drop_duplicates()
    
    def run_complete_analysis(self, identifier, identifier_type='auto'):
        """
        Run complete analysis for a given identifier
        
        Args:
            identifier (str): Node or exchange identifier
            identifier_type (str): 'node', 'exchange', or 'auto' to detect
        
        Returns:
            pd.DataFrame: Analysis results
        """
        # Preprocess data
        self.preprocess_data()
        
        # Generate base results
        self.generate_base_results(identifier)
        
        # Auto-detect type if needed
        if identifier_type == 'auto':
            identifier_type = self._detect_identifier_type(identifier)
        
        # Run appropriate analysis
        if identifier_type == 'exchange':
            results = self.analyze_exchange_impact(identifier)
            analysis_type = "Exchange"
        else:
            results = self.analyze_node_impact(identifier)
            analysis_type = "Node"
        
        print(f"{analysis_type} impact analysis completed. Results shape: {results.shape}")
        
        return results
    
    def _detect_identifier_type(self, identifier):
        """Auto-detect if identifier is a node or exchange"""
        # Simple heuristic: exchanges typically contain dots or are shorter
        if '.' in identifier or len(identifier.split('-')) < 4:
            return 'exchange'
        else:
            return 'node'
    
    def export_results(self, results, filename):
        """Export results to CSV"""
        results.to_csv(filename, index=False)
        print(f"Results exported to {filename}")

    # method to calculate Route_Status based on the MSAN-level conditions
    def _calculate_route_status_for_msan(self, msan_records):
        """
        Calculate Route_Status for an entire MSAN based on multiple conditions across records
        """
        # Condition 1: Check if any UP record with Partially Impacted and valid backup path
        condition1_records = msan_records[
            (msan_records['STATUS'] == 'UP') & 
            (msan_records['Impact'] == 'Partially Impacted') &
            (msan_records['Path2'].apply(lambda x: (isinstance(x, (list, tuple)) and len(x) > 0)))
        ]
        
        if not condition1_records.empty:
            return 'Primary Path Active'
        
        # Condition 2: Check ST records with Partially Impacted and valid primary path
        condition2_records = msan_records[
            (msan_records['STATUS'] == 'ST') & 
            (msan_records['Impact'] == 'Partially Impacted') &
            (msan_records['Path'].apply(lambda x: (isinstance(x, (list, tuple)) and len(x) > 0)))
        ]
        
        if not condition2_records.empty:
            return 'Traffic Rerouted'
        
        # Condition 3: Check if all records are Isolated
        if (msan_records['Impact'] == 'Isolated').all():
            return 'Inactive Paths'
        
        # Default fallback
        return 'UNKN'

    def _calculate_msan_level_route_status(self, results_df):
        """
        Apply MSAN-level Route_Status calculation to all records
        """
        if results_df.empty or 'MSANCODE' not in results_df.columns:
            return results_df
        
        # Group by MSANCODE and calculate Route_Status for each MSAN
        msan_route_status = {}
        for msan in results_df['MSANCODE'].unique():
            msan_records = results_df[results_df['MSANCODE'] == msan]
            route_status = self._calculate_route_status_for_msan(msan_records)
            msan_route_status[msan] = route_status
        
        # Apply the MSAN-level Route_Status to all records
        results_df = results_df.copy()
        results_df['Route_Status'] = results_df['MSANCODE'].map(msan_route_status)
        
        return results_df

    def _calculate_route_status_individual(self, path1, path2, status, impact):
        """
        Individual record Route_Status calculation (for non-MSAN level data)
        """
        def is_valid_path(path):
            if not (isinstance(path, (list, tuple)) and len(path) > 0):
                return False
            # Also check it's not an error message disguised as a tuple
            if isinstance(path, (list, tuple)) and len(path) == 1:
                if isinstance(path[0], str) and any(error in path[0] for error in ['NetworkXNoPath', 'Error', 'NodeNotFound']):
                    return False
            return True
        
        # Condition 1: UP + Partially Impacted + valid backup path
        if status == 'UP' and impact == 'Partially Impacted' and is_valid_path(path2):
            return 'Primary Path Active'
        
        # Condition 2: ST + Partially Impacted + valid primary path  
        elif status == 'ST' and impact == 'Partially Impacted' and is_valid_path(path1):
            return 'Traffic Rerouted'
        
        # Condition 3: Isolated impact
        elif impact == 'Isolated':
            return 'Inactive Paths'
        
        # Default
        return 'UNKN'

    '''
    def _calculate_route_status(self, path1, path2):
        """
        Calculate Route_Status by comparing final destinations of two paths
        Returns: 'Primary Path Active' or 'Traffic Rerouted'
        """
        def extract_final_destination(path):
            if isinstance(path, list) and len(path) > 0:
                return path[-1]  # Last element of the list
            elif isinstance(path, str):
                # Handle error messages - treat entire message as destination
                if "NetworkXNoPath" in path or "Error" in path or "NodeNotFound" in path:
                    return path
                # Try to parse string representation of list
                try:
                    import ast
                    parsed = ast.literal_eval(path)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        return parsed[-1]
                except:
                    pass
                return path  # Return original string if parsing fails
            else:
                return str(path) if path is not None else "Unknown"
        
        final1 = extract_final_destination(path1)
        final2 = extract_final_destination(path2)
        
        # Compare the final destinations
        if final1 == final2:
            return 'Primary Path Active'
        else:
            return 'Traffic Rerouted'
        
    '''





class UnifiedCIRModel:
    """Unified CIR Model that handles both network and bitstream scenarios"""
    
    def __init__(self, df_report, df_res_ospf, df_wan, df_agg, dwn_node, data_type, column_map):
        self.df = df_report.copy()
        self.resOspf = df_res_ospf.copy()
        self.data = df_wan.copy()
        self.agg = df_agg.copy()
        self.dwn_node = dwn_node
        self.data_type = data_type
        self.column_map = column_map
        
        # Precompute base graph once
        self._base_graph = self._create_base_graph()
        
    def _create_base_graph(self):
        """Create the base graph without any exclusions (called once during init)"""
        # Always exclude these nodes
        always_excluded = ['INSOMNA-R02J-C-EG', 'INSOMNA-R01J-C-EG']
        
        df_filtered = self.data[~self.data['NODENAME'].isin(always_excluded)]
        df_filtered = df_filtered[~df_filtered['NEIGHBOR_HOSTNAME'].isin(always_excluded)]
        df_filtered = df_filtered.drop_duplicates()

        G = nx.Graph()
        for idx, row in df_filtered.iterrows():
            G.add_edge(row[0], row[1])
        return G
    
    def draw_graph(self, excluded_nodes=None):
        """Create network graph with optional node exclusions"""
        if excluded_nodes is None or len(excluded_nodes) == 0:
            return self._base_graph.copy()
        
        # Create a copy of the base graph and remove excluded nodes
        graph_copy = self._base_graph.copy()
        graph_copy.remove_nodes_from(excluded_nodes)
        return graph_copy
    
    def calculate_path(self, graph, source, target):
        """Calculate path between two nodes - optimized version"""
        # Quick check if nodes exist in graph
        if source not in graph or target not in graph:
            return f"NodeNotFound: {source} or {target} not in graph"
        
        try:
            return nx.shortest_path(graph, source=source, target=target)
        except nx.NetworkXNoPath:
            return f"NetworkXNoPath: No path between {source} and {target}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_results(self):
        """Main method to generate the final results dataframe"""
        if self.data_type == 'network':
            return self._generate_network_results()
        else:
            return self._generate_bitstream_results()
    
    def _generate_network_results(self):
        """Generate results for network model (WE data)"""
        # Calculate initial paths
        specific_columns = ['EDGE', 'distribution_hostname']
        
        self.df['Path'] = self.df[specific_columns].apply(
            lambda row: self.calculate_path(self._base_graph, row['EDGE'], row['distribution_hostname']), axis=1
        )

        # Split data by status
        df_st = self.df[self.df['STATUS'] == 'ST'].copy()
        df_up = self.df[self.df['STATUS'] == 'UP'].copy()
        df_st.reset_index(inplace=True, drop=True)
        df_up.reset_index(inplace=True, drop=True)

        # Prepare merge dataframe
        df01 = df_up[['MSANCODE', 'distribution_hostname', 'edge_port', 'VLAN']].copy()
        df01.rename(columns={'distribution_hostname': 'distribution_hostname_UP'}, inplace=True)

        # Merge data
        res_df = pd.merge(df_st, df01, how='left', on=['MSANCODE', 'edge_port', 'VLAN'])
        dfx = res_df.copy()

        # Process paths and calculate masks
        dfx = self._process_paths(dfx, 'Path')
        dfx = self._calculate_network_masks(dfx)

        # Combine results
        final_df = pd.concat([df_up, dfx], ignore_index=True)
        return final_df
    
    def _generate_bitstream_results(self):
        """Generate results for bitstream model (Others data)"""
        # Calculate initial paths
        target_hostname_col = self.column_map['target_hostname']
        specific_columns = ['EDGE', target_hostname_col]
        
        self.df['Path'] = self.df[specific_columns].apply(
            lambda row: self.calculate_path(self._base_graph, row['EDGE'], row[target_hostname_col]), axis=1
        )
        
        res_df = self.df.copy()
        res_df = self._process_paths(res_df, 'Path')
        
        return res_df
    
    def _process_paths(self, dfx, column):
        """Process paths and remove duplicates"""
        path_backup = dfx[column].copy()
        dfx[column] = dfx[column].apply(lambda x: json.dumps(x) if isinstance(x, list) else str(x))
        dfx = dfx.drop_duplicates()
        dfx[column] = dfx.index.map(path_backup)
        return dfx

    def _calculate_network_masks(self, dfx):
        """Calculate mask values for paths (network model only)"""
        specific_columns = ['Path', 'distribution_hostname_UP']
        dfx['mask'] = dfx[specific_columns].apply(
            lambda row: (
                np.nan if not isinstance(row['Path'], list)
                else 'Single' if row['distribution_hostname_UP'] in row['Path']
                else 'Dual'
            ), axis=1
        )

        # Calculate optimized paths
        dfx = self._calculate_optimized_paths(dfx)
        
        # Calculate final mask
        specific_columns = ['Path2', 'distribution_hostname_UP']
        dfx['mask2'] = dfx[specific_columns].apply(
            lambda row: (
                np.nan if not isinstance(row['Path2'], list)
                else 'Single' if row['distribution_hostname_UP'] in row['Path2']
                else 'Dual'
            ), axis=1
        )

        # Clean up columns
        dfx['cir_type'] = np.where((dfx['mask'] == 'Single') & (dfx['mask2'].isna()), 
                                'Single', 
                                dfx['mask2'])
        dfx.drop(columns=['mask', 'mask2'], axis=1, inplace=True)
        
        return dfx

    def _calculate_optimized_paths(self, dfx):
        """Calculate optimized paths using smarter caching"""
        start_time = time.perf_counter()
        
        # Group by distribution_hostname_UP to minimize graph operations
        hostname_groups = dfx.groupby('distribution_hostname_UP')
        
        path_results = []
        
        for hostname, group_df in hostname_groups:
            # Create graph excluding this specific hostname
            graph = self.draw_graph(excluded_nodes=[hostname])
            
            # Calculate paths for all rows in this group using the same graph
            group_paths = []
            for _, row in group_df.iterrows():
                path = self.calculate_path(graph, row['EDGE'], row['distribution_hostname'])
                group_paths.append(path)
            
            # Add paths to results
            group_df = group_df.copy()
            group_df['Path2'] = group_paths
            path_results.append(group_df)
        
        # Combine results
        dfx_optimized = pd.concat(path_results, ignore_index=True)
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Optimized path calculation time: {execution_time:.3f} seconds")
        
        return dfx_optimized
    