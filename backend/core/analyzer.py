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
    
    def __init__(self, df_report, df_res_ospf, df_wan, df_agg, df_noms, df_mobile, df_sbc):
        """Initialize with network data"""
        self.df_report = df_report.copy()
        self.df_res_ospf = df_res_ospf.copy()
        self.df_wan = df_wan.copy()
        self.df_agg = df_agg.copy()
        self.df_noms = df_noms.copy()
        self.df_mobile = df_mobile.copy()
        self.df_sbc = df_sbc.copy()
        
        self.model = None
        self.final_df = None
        self.data_type = self._detect_data_type()
        self.column_map = self._get_column_mappings()
        # Add cache for exchange nodes ot solve performance issue
        self._exchange_nodes_cache = {}
        
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
        If any record for an MSAN is 'Path Changed', the entire MSAN is 'Path Changed'
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
            
            # If ANY record is Path Changed, the entire MSAN is Path Changed
            if 'Path Changed' in msan_records['Impact'].values:
                msan_impact[msan] = 'Path Changed'
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
    

    def _process_mobile_data(self, dwn_identifier, identifier_type='auto'):
        """Process mobile data and return mobile sites related to the down identifier"""
        if identifier_type == 'auto':
            identifier_type = self._detect_identifier_type(dwn_identifier)
        
        if identifier_type == 'exchange':
            # For exchange failure, get all nodes in that exchange first
            exchange_nodes = self._get_exchange_nodes(dwn_identifier)
            print(f"Found {len(exchange_nodes)} nodes in exchange {dwn_identifier}")
            
            if len(exchange_nodes) == 0:
                return pd.DataFrame()
            
            # Find mobile sites where hostname matches any node in the exchange
            mobile_data = self.df_mobile[self.df_mobile['hostname'].isin(exchange_nodes)]
            print(f"Found {len(mobile_data)} mobile sites in exchange {dwn_identifier}")
            
            return mobile_data
        else:
            # For node failure, filter by hostname
            mobile_data = self.df_mobile[self.df_mobile['hostname'] == dwn_identifier]
            print(f"Found {len(mobile_data)} mobile sites for node {dwn_identifier}")
            return mobile_data


    def _process_sbc_data(self, dwn_identifier, identifier_type='auto'):
        """Process sbc data and return affected customers count related to the down identifier"""
        if identifier_type == 'auto':
            identifier_type = self._detect_identifier_type(dwn_identifier)
        
        if identifier_type == 'exchange':
            # For exchange failure, get all nodes in that exchange first
            
            sbc = self.df_sbc[self.df_sbc.exchange == dwn_identifier]['sbc'].unique()
            if len(sbc) == 0:
                return 0
            
            sbc = sbc[0]
            #print(self.df_report.columns)
            cust_count = int(self.df_report[self.df_report.SBC == sbc].drop_duplicates(subset='MSANCODE')['CUST'].sum())
            
            return cust_count
        else:
            # For node failure, filter by hostname
            sbc = self.df_sbc[self.df_sbc.exchange == dwn_identifier]['sbc'].unique()
            
            if len(sbc) == 0 or len(self.df_sbc[self.df_sbc.sbc.isin(sbc)]['nodename'].unique()) > 1:
                return 0
            
            
            sbc = sbc[0]
            cust_count = int(self.df_report[self.df_report.SBC == sbc].drop_duplicates(subset='MSANCODE')['CUST'].sum())

            return cust_count


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
        """Calculate impact for Others data - FIXED VERSION"""
        if results_df.empty:
            return results_df
        
        # For Others data, we need to ensure impact is properly calculated
        # Don't default everything to 'Partial'
        
        # Only set default for records without impact
        if 'Impact' not in results_df.columns:
            results_df['Impact'] = 'Partial'
        else:
            # Only fill NaN values, don't override existing impacts
            results_df['Impact'] = results_df['Impact'].fillna('Partial')
        
        return results_df

    def analyze_exchange_impact(self, dwn_exchange):
        """Analyze impact when an exchange fails - FIXED VERSION"""
        if self.final_df is None:
            raise ValueError("Must call generate_base_results() first")
        
        # Get exchange nodes ONCE and cache them
        affected_nodes = self._get_exchange_nodes(dwn_exchange)
        
        # Create graph ONCE for this entire analysis
        graph = self.model.draw_graph(excluded_nodes=affected_nodes)
        
        # STEP 1: Find MSANs where UP records are affected
        affected_msans = self._find_msans_with_up_records_affected(dwn_exchange, affected_nodes)
        
        if affected_msans.empty:
            return pd.DataFrame()  # No affected MSANs found
        
        results = []
        
        # Case 1: Edge Exchange directly impacted (ONLY for affected MSANs)
        edge_col = self.column_map['edge_exchange']
        edge_impact = self._analyze_direct_impact_modified(edge_col, dwn_exchange, 'Isolated', affected_msans)
        if not edge_impact.empty:
            results.append(edge_impact)
            
        # Case 2: Target Exchange directly impacted - ONLY for affected MSANs
        target_impact = self._analyze_target_exchange_impact_optimized_modified(
            dwn_exchange, affected_nodes, graph, affected_msans
        )
        if not target_impact.empty:
            results.append(target_impact)
            
        # Case 3: Physical path impact - ONLY for affected MSANs  
        physical_impact = self._analyze_exchange_physical_path_impact_optimized_modified(
            dwn_exchange, affected_nodes, graph, affected_msans
        )
        if not physical_impact.empty:
            results.append(physical_impact)
            
        # Combine results
        combined_results = self._combine_results(results)
        
        if not combined_results.empty:
            if self.data_type == 'network':
                combined_results = self._calculate_msan_level_impact(combined_results)
            else:
                combined_results = self._calculate_impact_for_others(combined_results)
        
        return combined_results
    

    def _analyze_direct_impact_modified(self, column, value, impact_type, affected_msans):
        """Modified direct impact analysis - only for affected MSANs"""
        impact_df = self.final_df[
            (self.final_df[column] == value) & 
            (self.final_df.MSANCODE.isin(affected_msans['MSANCODE']))
        ].copy()
        if not impact_df.empty:
            impact_df['Impact'] = impact_type
        return impact_df


    def _find_msans_with_up_records_affected(self, dwn_exchange, affected_nodes):
        """Find MSANs where UP records contain the affected exchange nodes"""
        
        # For network data, we only care about UP records as triggers
        if self.data_type == 'network':
            base_records = self.final_df[self.final_df['STATUS'] == 'UP']
        else:
            # For Others data, all records can trigger
            base_records = self.final_df
        
        if base_records.empty:
            return pd.DataFrame()
        
        # Check multiple conditions for records being affected
        edge_exchange_col = self.column_map['edge_exchange']
        target_exchange_col = self.column_map['target_exchange']
        
        # Condition 1: Record's edge exchange matches
        edge_affected = base_records[base_records[edge_exchange_col] == dwn_exchange]
        
        # Condition 2: Record's target exchange matches  
        target_affected = base_records[base_records[target_exchange_col] == dwn_exchange]
        
        # Condition 3: Record's path contains affected nodes
        path_affected_mask = base_records['Path'].apply(
            lambda path: (isinstance(path, list) and len(path) >= 3 and 
                        any(node in path[1:-1] for node in affected_nodes))
        )
        path_affected = base_records[path_affected_mask]
        
        # Combine all affected MSANs
        all_affected_msans = set()
        for df in [edge_affected, target_affected, path_affected]:
            if not df.empty:
                all_affected_msans.update(df['MSANCODE'].unique())
        
        # Return DataFrame with affected MSANs
        return pd.DataFrame({'MSANCODE': list(all_affected_msans)})


    def analyze_node_impact(self, dwn_node):
        """Analyze impact when a node fails - FIXED VERSION"""
        if self.final_df is None:
            raise ValueError("Must call generate_base_results() first")
        
        # STEP 1: Find MSANs where UP records are affected by the node
        affected_msans = self._find_msans_with_up_records_affected_by_node(dwn_node)
        
        if affected_msans.empty:
            return pd.DataFrame()  # No affected MSANs found
        
        results = []
        
        # Case 1: Edge directly impacted (ONLY for affected MSANs)
        edge_impact = self._analyze_direct_impact_modified('EDGE', dwn_node, 'Isolated', affected_msans)
        if not edge_impact.empty:
            results.append(edge_impact)
            
        # Case 2: Target node directly impacted - ONLY for affected MSANs
        target_impact = self._analyze_target_node_impact_modified(dwn_node, affected_msans)
        if not target_impact.empty:
            results.append(target_impact)
            
        # Case 3: Physical path impact - ONLY for affected MSANs  
        physical_impact = self._analyze_node_physical_path_impact_modified(dwn_node, affected_msans)
        if not physical_impact.empty:
            results.append(physical_impact)
            
        # Combine results
        combined_results = self._combine_results(results)
        
        if not combined_results.empty:
            if self.data_type == 'network':
                combined_results = self._calculate_msan_level_impact(combined_results)
            else:
                combined_results = self._calculate_impact_for_others(combined_results)
        
        return combined_results
    

    def _find_msans_with_up_records_affected_by_node(self, dwn_node):
        """Find MSANs where UP records are affected by the node failure"""
        
        # For network data, we only care about UP records as triggers
        if self.data_type == 'network':
            base_records = self.final_df[self.final_df['STATUS'] == 'UP']
        else:
            # For Others data, all records can trigger
            base_records = self.final_df
        
        if base_records.empty:
            return pd.DataFrame()
        
        target_hostname_col = self.column_map['target_hostname']
        bng_hostname_col = self.column_map['bng_hostname']
        
        # Condition 1: Record's EDGE matches the node
        edge_affected = base_records[base_records['EDGE'] == dwn_node]
        
        # Condition 2: Record's target hostname matches
        target_affected = base_records[base_records[target_hostname_col] == dwn_node]
        
        # Condition 3: For network data, check BNG hostname
        bng_affected = pd.DataFrame()
        if self.data_type == 'network' and bng_hostname_col:
            bng_affected = base_records[base_records[bng_hostname_col] == dwn_node]
        
        # Condition 4: Record's path contains the node
        path_affected_mask = base_records['Path'].apply(
            lambda path: (isinstance(path, list) and len(path) >= 3 and 
                        dwn_node in path[1:-1])
        )
        path_affected = base_records[path_affected_mask]
        
        # Combine all affected MSANs
        all_affected_msans = set()
        for df in [edge_affected, target_affected, bng_affected, path_affected]:
            if not df.empty:
                all_affected_msans.update(df['MSANCODE'].unique())
        
        # Return DataFrame with affected MSANs
        return pd.DataFrame({'MSANCODE': list(all_affected_msans)})


    def _analyze_direct_impact(self, column, value, impact_type):
        """Generic method for analyzing direct impact on a column - with Impact column guarantee"""
        impact_df = self.final_df[self.final_df[column] == value].copy()
        if not impact_df.empty:
            impact_df['Impact'] = impact_type
        return impact_df
    

    def _analyze_target_exchange_impact_optimized_modified(self, dwn_exchange, affected_nodes, graph, affected_msans):
        """Modified version that only processes MSANs with affected UP records"""
        target_exchange_col = self.column_map['target_exchange']
        target_hostname_col = self.column_map['target_hostname']
        
        # Only consider records from affected MSANs
        direct_impact = self.final_df[
            (self.final_df[target_exchange_col] == dwn_exchange) & 
            (self.final_df.MSANCODE.isin(affected_msans['MSANCODE']))
        ]
        
        if direct_impact.empty:
            return pd.DataFrame()
        
        if self.data_type == 'network':
            # For network type, get all records for affected MSANs
            all_affected = self.final_df[
                self.final_df.MSANCODE.isin(direct_impact.MSANCODE.unique())
            ].copy()
            
            # Use the precomputed graph
            all_affected['Path2'] = all_affected.apply(
                lambda row: self.model.calculate_path(graph, row['EDGE'], row[target_hostname_col]),
                axis=1
            )
            
            all_affected['Impact'] = all_affected['Path2'].apply(
                lambda x: 'Path Changed' if isinstance(x, list) else 'Isolated'
            )
        else:
            # For Others data, get ALL records for the affected MSANs
            affected_msans_list = direct_impact.MSANCODE.unique()
            all_affected = self.final_df[
                self.final_df.MSANCODE.isin(affected_msans_list)
            ].copy()
            
            # Initialize Path2 and Impact columns for ALL records
            all_affected['Path2'] = None
            all_affected['Impact'] = 'Partial'  # Default to Partial
            
            # Calculate paths and set impact for ALL records in affected MSANs
            # (not just directly impacted ones)
            for idx, row in all_affected.iterrows():
                # Check if this specific record needs path recalculation
                target_exchange = row[target_exchange_col]
                
                # Calculate Path2 if target exchange matches OR if path contains affected nodes
                should_calculate_path = False
                
                # Condition 1: Target exchange matches
                if target_exchange == dwn_exchange:
                    should_calculate_path = True
                
                # Condition 2: Path contains affected nodes
                path = row['Path']
                if isinstance(path, list) and len(path) >= 3:
                    intermediate_nodes = path[1:-1]
                    if any(node in intermediate_nodes for node in affected_nodes):
                        should_calculate_path = True
                
                if should_calculate_path:
                    path2 = self.model.calculate_path(graph, row['EDGE'], row[target_hostname_col])
                    all_affected.at[idx, 'Path2'] = path2
                    
                    # Determine impact based on path2
                    if isinstance(path2, list) and len(path2) > 0:
                        all_affected.at[idx, 'Impact'] = 'Path Changed'
                    else:
                        all_affected.at[idx, 'Impact'] = 'Isolated'
        
        return all_affected

    def _analyze_target_node_impact_modified(self, dwn_node, affected_msans):
        """Analyze direct impact on target nodes - ONLY for affected MSANs"""
        target_hostname_col = self.column_map['target_hostname']
        bng_hostname_col = self.column_map['bng_hostname']
        
        # Only consider records from affected MSANs
        if self.data_type == 'network' and bng_hostname_col:
            # Network data: BNG or distribution hostname affected
            direct_impact = self.final_df[
                ((self.final_df[bng_hostname_col] == dwn_node) | 
                (self.final_df[target_hostname_col] == dwn_node)) &
                (self.final_df.MSANCODE.isin(affected_msans['MSANCODE']))
            ]
        else:
            # Others data: Only bitstream hostname affected
            direct_impact = self.final_df[
                (self.final_df[target_hostname_col] == dwn_node) &
                (self.final_df.MSANCODE.isin(affected_msans['MSANCODE']))
            ]
        
        if direct_impact.empty:
            return pd.DataFrame()
        
        # Calculate alternative paths excluding the specific node
        graph = self.model.draw_graph(excluded_nodes=[dwn_node])
        
        if self.data_type == 'network':
            # For network type, get all records for affected MSANs
            affected_msans_list = direct_impact.MSANCODE.unique()
            all_affected = self.final_df[
                self.final_df.MSANCODE.isin(affected_msans_list)
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
            
            # Calculate Path2 for all records
            all_affected['Path2'] = all_affected.apply(
                lambda row: self.model.calculate_path(graph, row['EDGE'], row[target_hostname_col]),
                axis=1
            )
        else:
            # For Others data, get ALL records for the affected MSANs but only mark specific ones
            affected_msans_list = direct_impact.MSANCODE.unique()
            all_affected = self.final_df[
                self.final_df.MSANCODE.isin(affected_msans_list)
            ].copy()
            
            # Create a mask for directly affected records
            if self.data_type == 'network' and bng_hostname_col:
                direct_impact_mask = (all_affected[bng_hostname_col] == dwn_node) | (all_affected[target_hostname_col] == dwn_node)
            else:
                direct_impact_mask = all_affected[target_hostname_col] == dwn_node
            
            # Initialize Path2 and Impact columns
            all_affected['Path2'] = None
            all_affected['Impact'] = 'Partial'  # Default to Partial
            
            # Calculate paths and set impact only for directly affected records
            affected_indices = all_affected[direct_impact_mask].index
            for idx in affected_indices:
                row = all_affected.loc[idx]
                path2 = self.model.calculate_path(graph, row['EDGE'], row[target_hostname_col])
                all_affected.at[idx, 'Path2'] = path2
                all_affected.at[idx, 'Impact'] = 'Path Changed' if isinstance(path2, list) else 'Isolated'
        
        return all_affected

    def _analyze_exchange_physical_path_impact_optimized_modified(self, dwn_exchange, affected_nodes, graph, affected_msans):
        """Analyze physical path impact for exchange failure - ONLY for MSANs with affected UP records"""
        
        # Get all records from affected MSANs
        affected_msan_records = self.final_df[
            self.final_df.MSANCODE.isin(affected_msans['MSANCODE'])
        ].copy()
        
        if affected_msan_records.empty:
            return pd.DataFrame()
        
        # Find records with affected nodes in their paths (within the affected MSANs)
        path_mask = affected_msan_records['Path'].apply(
            lambda path: (isinstance(path, list) and len(path) >= 3 and 
                        any(node in path[1:-1] for node in affected_nodes))
        )
        
        # Check Path2 column if it exists
        path2_mask = pd.Series([False] * len(affected_msan_records))
        if 'Path2' in affected_msan_records.columns:
            path2_mask = affected_msan_records['Path2'].apply(
                lambda path: (isinstance(path, list) and len(path) >= 3 and 
                            any(node in path[1:-1] for node in affected_nodes))
            )
        
        # Find records that have affected nodes in their paths
        path_affected_records = affected_msan_records[path_mask | path2_mask]
        
        if path_affected_records.empty:
            return pd.DataFrame()
        
        # Use the precomputed graph
        target_hostname_col = self.column_map['target_hostname']
        
        if self.data_type == 'network':
            # For network data, get all records for affected MSANs
            affected_msans_from_path = path_affected_records.MSANCODE.unique()
            all_affected = self.final_df[
                self.final_df.MSANCODE.isin(affected_msans_from_path)
            ].copy()
            
            # Use precomputed graph for path calculation
            all_affected['Path2'] = all_affected.apply(
                lambda row: self.model.calculate_path(graph, row['EDGE'], row[target_hostname_col]),
                axis=1
            )
            
            # Set impact based on path existence
            all_affected['Impact'] = all_affected['Path2'].apply(
                lambda x: 'Path Changed' if isinstance(x, list) else 'Isolated'
            )
            
            # Apply MSAN-level impact
            all_affected = self._calculate_msan_level_impact(all_affected)
        else:
            # ========== FIXED LOGIC FOR OTHERS DATA ==========
            # For Others data, we need to process ALL records from affected MSANs
            # and properly determine impact based on path analysis
            
            all_affected = affected_msan_records.copy()
            
            # Initialize columns
            all_affected['Path2'] = None
            all_affected['Impact'] = 'Partial'  # Default
            
            # Process each record to determine impact
            for idx, row in all_affected.iterrows():
                path = row['Path']
                
                # Check if original path contains any affected nodes
                has_affected_nodes = False
                if isinstance(path, list) and len(path) >= 3:
                    # Check intermediate nodes (excluding first and last)
                    intermediate_nodes = path[1:-1]
                    has_affected_nodes = any(node in intermediate_nodes for node in affected_nodes)
                
                if has_affected_nodes:
                    # Calculate alternative path excluding affected exchange nodes
                    path2 = self.model.calculate_path(graph, row['EDGE'], row[target_hostname_col])
                    all_affected.at[idx, 'Path2'] = path2
                    
                    # Determine impact: if no alternative path exists, it's Isolated
                    if isinstance(path2, list) and len(path2) > 0:
                        all_affected.at[idx, 'Impact'] = 'Path Changed'
                    else:
                        all_affected.at[idx, 'Impact'] = 'Isolated'
                else:
                    # No affected nodes in path, keep as Partial
                    all_affected.at[idx, 'Impact'] = 'Partial'
                    all_affected.at[idx, 'Path2'] = []  # Empty list for consistency
            # ========== END FIX ==========
        
        return all_affected
    

    def _analyze_node_physical_path_impact_modified(self, dwn_node, affected_msans):
        """Analyze physical path impact for node failure - ONLY for affected MSANs"""
        
        # Get all records from affected MSANs
        affected_msan_records = self.final_df[
            self.final_df.MSANCODE.isin(affected_msans['MSANCODE'])
        ].copy()
        
        if affected_msan_records.empty:
            return pd.DataFrame()
        
        # Find records with the node in their paths (within the affected MSANs)
        path_affected_mask = affected_msan_records['Path'].apply(
            lambda path: (isinstance(path, list) and len(path) >= 3 and 
                        dwn_node in path[1:-1])
        )
        
        # Check Path2 column if it exists
        path2_affected_mask = pd.Series([False] * len(affected_msan_records))
        if 'Path2' in affected_msan_records.columns:
            path2_affected_mask = affected_msan_records['Path2'].apply(
                lambda path: (isinstance(path, list) and len(path) >= 3 and 
                            dwn_node in path[1:-1])
            )
        
        # Find records that have the node in their paths
        path_affected_records = affected_msan_records[path_affected_mask | path2_affected_mask]
        
        if path_affected_records.empty:
            return pd.DataFrame()
        
        # Calculate alternative paths excluding the specific node
        graph = self.model.draw_graph(excluded_nodes=[dwn_node])
        target_hostname_col = self.column_map['target_hostname']
        
        if self.data_type == 'network':
            # For network data, get all records for affected MSANs
            affected_msans_from_path = path_affected_records.MSANCODE.unique()
            all_affected = self.final_df[
                self.final_df.MSANCODE.isin(affected_msans_from_path)
            ].copy()
            
            all_affected['Path2'] = all_affected.apply(
                lambda row: self.model.calculate_path(graph, row['EDGE'], row[target_hostname_col]),
                axis=1
            )
            
            # Set impact based on path existence
            all_affected['Impact'] = all_affected['Path2'].apply(
                lambda x: 'Path Changed' if isinstance(x, list) else 'Isolated'
            )
            
            # Apply MSAN-level impact
            all_affected = self._calculate_msan_level_impact(all_affected)
        else:
            # For Others data, get ALL records for affected MSANs but only mark specific ones
            affected_msans_from_path = path_affected_records.MSANCODE.unique()
            all_affected = self.final_df[
                self.final_df.MSANCODE.isin(affected_msans_from_path)
            ].copy()
            
            # Initialize Path2 and Impact columns
            all_affected['Path2'] = None
            all_affected['Impact'] = 'Partial'  # Default to Partial
            
            # Calculate paths and set impact only for records that have the node in their paths
            for idx, row in all_affected.iterrows():
                path = row['Path']
                if isinstance(path, list) and dwn_node in path[1:-1]:
                    path2 = self.model.calculate_path(graph, row['EDGE'], row[target_hostname_col])
                    all_affected.at[idx, 'Path2'] = path2
                    all_affected.at[idx, 'Impact'] = 'Path Changed' if isinstance(path2, list) else 'Isolated'
        
        return all_affected
        

    # Returniung all nodes in the exchange - FIXED VERSION, Case 'MEETGHAMR1...DK' '07-1-10-36'
    def _get_exchange_nodes(self, dwn_exchange):
        """Get all nodes belonging to a specific exchange - FIXED VERSION - WITH CACHING"""

        # Check cache first
        if dwn_exchange in self._exchange_nodes_cache:
            return self._exchange_nodes_cache[dwn_exchange]

        all_nodes = self.df_noms[self.df_noms.nodesite == dwn_exchange]['nodename'].unique()
        
        print(f"Found {len(all_nodes)} nodes in exchange {dwn_exchange}")

        # Cache the result
        self._exchange_nodes_cache[dwn_exchange] = all_nodes

        return all_nodes

    def _is_node_in_exchange(self, node, exchange_code, full_exchange_name):
        """Helper method to precisely determine if a node belongs to an exchange"""
        if not isinstance(node, str):
            return False
        
        # Method 1: Check if node name contains the full exchange name (most reliable)
        if full_exchange_name.replace('.', '').upper() in node.upper():
            return True
        
        # Method 2: Check naming convention with exchange code
        node_parts = node.split('-')
        if len(node_parts) >= 3:
            # Check if the third part matches the exchange code
            if node_parts[2] == exchange_code:
                return True
        
        # Method 3: Check for exchange code with proper context (avoid partial matches)
        # Only match if exchange code appears as a standalone part or with proper separators
        import re
        pattern = r'(^|[-_\.])' + re.escape(exchange_code) + r'($|[-_\.])'
        if re.search(pattern, node, re.IGNORECASE):
            return True
        
        return False
    
    def debug_exchange_nodes(self, dwn_exchange):
        """Debug method to trace why nodes are being included in exchange"""
        exchange_code = dwn_exchange.split('.')[-1] if '.' in dwn_exchange else dwn_exchange
        
        print(f"\n=== DEBUGGING EXCHANGE: {dwn_exchange} (code: {exchange_code}) ===")
        
        edge_exchange_col = self.column_map['edge_exchange']
        target_exchange_col = self.column_map['target_exchange']
        
        # Check edge nodes
        edge_nodes = self.final_df[
            self.final_df[edge_exchange_col] == dwn_exchange
        ]['EDGE'].unique()
        print(f"Direct edge nodes: {list(edge_nodes)}")
        
        # Check target nodes  
        target_hostname_col = self.column_map['target_hostname']
        target_nodes = self.final_df[
            self.final_df[target_exchange_col] == dwn_exchange
        ][target_hostname_col].unique()
        print(f"Direct target nodes: {list(target_nodes)}")
        
        # Check WAN data
        if hasattr(self, 'df_wan') and self.df_wan is not None:
            wan_nodes = []
            for node in self.df_wan['NODENAME'].dropna().unique():
                if exchange_code in node:
                    wan_nodes.append(node)
            print(f"WAN nodes containing '{exchange_code}': {wan_nodes}")
        
        print("=== END DEBUG ===\n")

    # Add this temporary test method to debug your specific case
    def debug_specific_msan(self, msan_code, dwn_exchange):
        """Debug why a specific MSAN is being included"""
        msan_data = self.final_df[self.final_df['MSANCODE'] == msan_code]
        if msan_data.empty:
            print(f"MSAN {msan_code} not found in data")
            return
        
        print(f"\n=== DEBUGGING MSAN {msan_code} for exchange {dwn_exchange} ===")
        
        for idx, row in msan_data.iterrows():
            path = row['Path']
            edge_exchange = row.get(self.column_map['edge_exchange'])
            target_exchange = row.get(self.column_map['target_exchange'])
            
            print(f"Record {idx}:")
            print(f"  EDGE: {row['EDGE']}, EDGE Exchange: {edge_exchange}")
            print(f"  Target: {row.get(self.column_map['target_hostname'])}, Target Exchange: {target_exchange}")
            print(f"  Path: {path}")
            
            # Check if any Tanta nodes in path
            if isinstance(path, list):
                tanta_nodes_in_path = [node for node in path if 'TANTA' in node or 'GH' in node]
                print(f"  Tanta-related nodes in path: {tanta_nodes_in_path}")
        
        print("=== END MSAN DEBUG ===\n")
    
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
        # Condition 1: Check if any UP record with Path Changed and valid backup path
        condition1_records = msan_records[
            (msan_records['STATUS'] == 'UP') & 
            (msan_records['Impact'] == 'Path Changed') &
            (msan_records['Path2'].apply(lambda x: (isinstance(x, (list, tuple)) and len(x) > 0)))
        ]
        
        if not condition1_records.empty:
            return 'Primary Path Active'
        
        # Condition 2: Check ST records with Path Changed and valid primary path
        condition2_records = msan_records[
            (msan_records['STATUS'] == 'ST') & 
            (msan_records['Impact'] == 'Path Changed') &
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
        
        # Condition 1: UP + Path Changed + valid backup path
        if status == 'UP' and impact == 'Path Changed' and is_valid_path(path2):
            return 'Primary Path Active'
        
        # Condition 2: ST + Path Changed + valid primary path  
        elif status == 'ST' and impact == 'Path Changed' and is_valid_path(path1):
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
        dfx.replace({np.nan: None}, inplace=True)
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
    