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
                (self.df_report.BNG_HOSTNAME.isnull()) & (self.df_report.STATUS != 'ST')
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
    
    def analyze_exchange_impact(self, dwn_exchange):
        """Analyze impact when an exchange fails"""
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
            
        return self._combine_results(results)
    
    def analyze_node_impact(self, dwn_node):
        """Analyze impact when a node fails"""
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
            
        return self._combine_results(results)
    
    def _analyze_direct_impact(self, column, value, impact_type):
        """Generic method for analyzing direct impact on a column"""
        impact_df = self.final_df[self.final_df[column] == value].copy()
        if not impact_df.empty:
            impact_df['Impact'] = impact_type
        return impact_df
    
    def _analyze_target_exchange_impact(self, dwn_exchange):
        """Analyze direct impact on target exchange (AGG/Bitstream)"""
        target_exchange_col = self.column_map['target_exchange']
        target_hostname_col = self.column_map['target_hostname']
        
        direct_impact = self.final_df[self.final_df[target_exchange_col] == dwn_exchange]
        
        if direct_impact.empty:
            return pd.DataFrame()
        
        # Get all records for affected MSANs
        all_affected = self.final_df[
            self.final_df.MSANCODE.isin(direct_impact.MSANCODE.unique())
        ].copy()
        
        # Get affected target nodes
        affected_nodes = direct_impact[target_hostname_col].unique().tolist()
        
        # Calculate alternative paths only for network type (Others don't need this)
        if self.data_type == 'network':
            # Create graph excluding affected nodes
            graph = self.model.draw_graph(excluded_nodes=affected_nodes)
            
            # Calculate alternative paths
            all_affected['Path2'] = all_affected.apply(
                lambda row: self.model.calculate_path(graph, row['EDGE'], row[target_hostname_col]),
                axis=1
            )
            
            all_affected['Impact'] = all_affected['Path2'].apply(
                lambda x: 'Partially Impacted' if isinstance(x, list) else 'Isolated'
            )
        else:
            # For bitstream type, just mark as isolated
            all_affected['Impact'] = 'Isolated'
        
        return all_affected
    
    def _analyze_exchange_physical_path_impact(self, dwn_exchange):
        """Analyze physical path impact for exchange failure"""
        # Get nodes in the affected exchange
        affected_nodes = self._get_exchange_nodes(dwn_exchange)
        
        if not affected_nodes:
            return pd.DataFrame()
        
        # Find MSANs with affected nodes in their paths
        affected_msans = self._find_msans_with_nodes_in_path(affected_nodes)
        
        if affected_msans.empty:
            return pd.DataFrame()
        
        # Calculate alternative paths
        graph = self.model.draw_graph(excluded_nodes=affected_nodes)
        target_hostname_col = self.column_map['target_hostname']
        
        affected_msans['Path2'] = affected_msans.apply(
            lambda row: self.model.calculate_path(graph, row['EDGE'], row[target_hostname_col]),
            axis=1
        )
        
        affected_msans['Impact'] = affected_msans['Path2'].apply(
            lambda x: 'Partially Impacted' if isinstance(x, list) else 'Isolated'
        )
        
        return affected_msans
    
    def _analyze_target_node_impact(self, dwn_node):
        """Analyze direct impact on target nodes (AGG/BNG/Bitstream)"""
        target_hostname_col = self.column_map['target_hostname']
        bng_hostname_col = self.column_map['bng_hostname']
        
        # Build condition based on data type
        if self.data_type == 'network' and bng_hostname_col:
            direct_impact = self.final_df[
                (self.final_df[bng_hostname_col] == dwn_node) | 
                (self.final_df[target_hostname_col] == dwn_node)
            ]
        else:
            direct_impact = self.final_df[self.final_df[target_hostname_col] == dwn_node]
        
        if direct_impact.empty:
            return pd.DataFrame()
        
        # Get all records for affected MSANs
        all_affected = self.final_df[
            self.final_df.MSANCODE.isin(direct_impact.MSANCODE.unique())
        ].copy()
        
        # Determine impact based on data type and circuit type
        if self.data_type == 'network':
            # For network type, use circuit type information
            msan_impacts = []
            for msan in all_affected.MSANCODE.unique():
                msan_df = all_affected[all_affected.MSANCODE == msan].copy()
                
                if 'cir_type' in msan_df.columns and 'Single' in msan_df['cir_type'].values:
                    msan_df['Impact'] = 'Isolated'
                else:
                    msan_df['Impact'] = 'Partially Impacted'
                    
                msan_impacts.append(msan_df)
            
            return pd.concat(msan_impacts, ignore_index=True)
        else:
            # For bitstream type, mark as isolated
            all_affected['Impact'] = 'Isolated'
            return all_affected
    
    def _analyze_node_physical_path_impact(self, dwn_node):
        """Analyze physical path impact for node failure"""
        # Find MSANs with the node in their paths
        affected_msans = self._find_msans_with_nodes_in_path([dwn_node])
        
        if affected_msans.empty:
            return pd.DataFrame()
        
        # Calculate alternative paths
        graph = self.model.draw_graph(excluded_nodes=[dwn_node])
        target_hostname_col = self.column_map['target_hostname']
        
        affected_msans['Path2'] = affected_msans.apply(
            lambda row: self.model.calculate_path(graph, row['EDGE'], row[target_hostname_col]),
            axis=1
        )
        
        affected_msans['Impact'] = affected_msans['Path2'].apply(
            lambda x: 'Partially Impacted' if isinstance(x, list) else 'Isolated'
        )
        
        return affected_msans
    
    def _get_exchange_nodes(self, dwn_exchange):
        """Get all nodes belonging to a specific exchange"""
        # Extract exchange code from exchange name
        exchange_code = dwn_exchange.split('.')[-1] if '.' in dwn_exchange else dwn_exchange
        
        edge_exchange_col = self.column_map['edge_exchange']
        target_exchange_col = self.column_map['target_exchange']
        target_hostname_col = self.column_map['target_hostname']
        
        # Find all nodes in the exchange from direct impacts
        edge_nodes = self.final_df[
            self.final_df[edge_exchange_col] == dwn_exchange
        ]['EDGE'].unique().tolist()
        
        target_nodes = self.final_df[
            self.final_df[target_exchange_col] == dwn_exchange
        ][target_hostname_col].unique().tolist()
        
        # Combine and filter nodes belonging to the exchange
        all_nodes = list(set(edge_nodes + target_nodes))
        exchange_nodes = [
            node for node in all_nodes 
            if len(node.split('-')) > 2 and node.split('-')[2] == exchange_code
        ]
        
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
        
        self.g = self.draw_graph()
        
    def draw_graph(self, excluded_nodes=None):
        """Create network graph from stored data, optionally excluding nodes"""
        if excluded_nodes is None:
            excluded_nodes = []
        elif not isinstance(excluded_nodes, list):
            excluded_nodes = [excluded_nodes]
            
        # Always exclude these nodes
        always_excluded = ['INSOMNA-R02J-C-EG', 'INSOMNA-R01J-C-EG']
        all_excluded = always_excluded + excluded_nodes
        
        df_filtered = self.data[~self.data['NODENAME'].isin(all_excluded)]
        df_filtered = df_filtered[~df_filtered['NEIGHBOR_HOSTNAME'].isin(all_excluded)]
        df_filtered = df_filtered.drop_duplicates()

        G = nx.Graph()
        for idx, row in df_filtered.iterrows():
            G.add_edge(row[0], row[1])
        return G

    def calculate_path(self, graph, source, target):
        """Calculate path between two nodes"""
        try:
            return nx.shortest_path(graph, source=source, target=target)
        except nx.NetworkXNoPath:
            return f"NetworkXNoPath: No path between {source} and {target}."
        except nx.NodeNotFound as e:
            return e
        except Exception as f:
            return f"Error: {f}"
    
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
            lambda row: self.calculate_path(self.g, row['EDGE'], row['distribution_hostname']), axis=1
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
            lambda row: self.calculate_path(self.g, row['EDGE'], row[target_hostname_col]), axis=1
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
        """Calculate optimized paths using cached graphs"""
        start_time = time.perf_counter()
        
        # Precompute all unique graphs
        unique_hostnames = dfx['distribution_hostname_UP'].unique()
        graph_cache = {}
        
        for hostname in unique_hostnames:
            graph_cache[hostname] = self.draw_graph(excluded_nodes=[hostname])

        # Apply path finding using cached graphs
        def optimized_path_function(row):
            graph = graph_cache[row['distribution_hostname_UP']]
            return self.calculate_path(graph, row['EDGE'], row['distribution_hostname'])

        dfx['Path2'] = dfx.apply(optimized_path_function, axis=1)
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Optimized path calculation time: {execution_time:.3f} seconds")
        
        return dfx