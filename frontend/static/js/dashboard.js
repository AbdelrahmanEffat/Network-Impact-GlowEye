// Improved Dashboard functionality for network path analysis
class NetworkDashboard {
    constructor() {
        this.weData = null;
        this.currentSimulation = null;
        this.observers = [];
        this.eventListeners = [];
    }

    initialize(data) {
        this.weData = data;
        
        // Check dependencies
        if (typeof d3 === 'undefined') {
            console.error('D3.js library is required but not loaded');
            return false;
        }

        document.addEventListener('DOMContentLoaded', () => {
            this.initializeDashboard();
        });
        
        return true;
    }

    initializeDashboard() {
        const dashboardTab = document.getElementById('dashboard-tab');
        if (!dashboardTab) return;

        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                    if (dashboardTab.classList.contains('active')) {
                        this.renderDashboard();
                    } else {
                        this.cleanup();
                    }
                }
            });
        });

        observer.observe(dashboardTab, { attributes: true });
        this.observers.push(observer);
    }

    renderDashboard() {
        console.log("Rendering dashboard...");

        try {
            // Validate data
            if (!this.validateData()) {
                this.showErrorMessage("No valid WE data available for dashboard visualization.");
                return;
            }
            
            // Clear previous content and cleanup
            this.cleanup();
            
            const dashboardContainer = document.getElementById('dashboard-container');
            if (!dashboardContainer) {
                this.showErrorMessage("Dashboard container not found.");
                return;
            }
            
            dashboardContainer.innerHTML = '';
            
            // Create dashboard components
            this.createDashboardControls();
            
            // Initialize with first MSANCODE
            const firstMsan = this.weData[0].MSANCODE;
            this.updateVisualization(firstMsan);
            
        } catch (error) {
            console.error("Error rendering dashboard:", error);
            this.showErrorMessage(`Failed to render dashboard: ${error.message}`);
        }
    }

    validateData() {
        return this.weData && 
               Array.isArray(this.weData) && 
               this.weData.length > 0 &&
               this.weData.some(item => item.MSANCODE);
    }

    createDashboardControls() {
        const dashboardContainer = document.getElementById('dashboard-container');
        if (!dashboardContainer) return;

        // Create control container
        const controlsDiv = document.createElement('div');
        controlsDiv.className = 'dashboard-controls';

        // MSANCODE selector
        const msanSelectorDiv = this.createMsanSelector();
        
        // Visualization container
        const vizContainer = document.createElement('div');
        vizContainer.id = 'dashboard-viz';
        vizContainer.className = 'dashboard-viz-container';

        controlsDiv.appendChild(msanSelectorDiv);
        dashboardContainer.appendChild(controlsDiv);
        dashboardContainer.appendChild(vizContainer);
    }

    createMsanSelector() {
        const msanSelectorDiv = document.createElement('div');
        msanSelectorDiv.className = 'dashboard-control';

        const msanLabel = document.createElement('label');
        msanLabel.textContent = 'Select MSANCODE:';
        msanLabel.htmlFor = 'msan-selector';

        const msanSelector = document.createElement('select');
        msanSelector.id = 'msan-selector';

        // Get unique MSANCODEs with validation
        const msanCodes = [...new Set(
            this.weData
                .filter(item => item.MSANCODE)
                .map(item => item.MSANCODE)
        )].sort();

        if (msanCodes.length === 0) {
            const option = document.createElement('option');
            option.textContent = 'No MSANCODEs available';
            option.disabled = true;
            msanSelector.appendChild(option);
        } else {
            msanCodes.forEach(code => {
                const option = document.createElement('option');
                option.value = code;
                option.textContent = code;
                msanSelector.appendChild(option);
            });
        }

        // Add event listener with cleanup tracking
        const changeHandler = () => this.updateVisualization(msanSelector.value);
        msanSelector.addEventListener('change', changeHandler);
        this.eventListeners.push({ element: msanSelector, event: 'change', handler: changeHandler });

        msanSelectorDiv.appendChild(msanLabel);
        msanSelectorDiv.appendChild(msanSelector);
        
        return msanSelectorDiv;
    }

    updateVisualization(msanCode) {
        const vizContainer = document.getElementById('dashboard-viz');
        if (!vizContainer) return;

        // Stop previous simulation
        if (this.currentSimulation) {
            this.currentSimulation.stop();
        }

        // Filter and validate data
        const filteredData = this.weData.filter(item => 
            item.MSANCODE === msanCode && (item.Path || item.Path2)
        );

        if (filteredData.length === 0) {
            vizContainer.innerHTML = '<div class="no-data-message"><p>No path data available for selected MSANCODE.</p></div>';
            return;
        }

        vizContainer.innerHTML = '';

        try {
            this.createNetworkVisualization(vizContainer, filteredData, msanCode);
            this.createPathDetails(vizContainer, filteredData, msanCode);
        } catch (error) {
            console.error('Error creating visualization:', error);
            this.showErrorMessage(`Visualization error: ${error.message}`);
        }
    }

    createNetworkVisualization(container, data, msanCode) {
        // Create header
        const header = document.createElement('h3');
        header.textContent = `Network Path Visualization for ${msanCode}`;
        container.appendChild(header);

        // Create SVG container
        const svgContainer = document.createElement('div');
        svgContainer.id = 'dashboard-network-viz';
        svgContainer.className = 'network-viz-svg';
        svgContainer.style.minHeight = '400px';
        container.appendChild(svgContainer);

        // Process data
        const { nodes, links } = this.processNetworkData(data);
        
        if (nodes.length === 0) {
            svgContainer.innerHTML = '<div class="no-data-message"><p>No valid path data for visualization.</p></div>';
            return;
        }

        this.renderD3Visualization(svgContainer, nodes, links);
    }

    processNetworkData(data) {
        const allNodes = new Set();
        const links = [];
        const linkMap = new Map(); // Prevent duplicate links

        data.forEach(record => {
            const path1 = this.parsePath(record.Path);
            const path2 = this.parsePath(record.Path2);
            
            // Process primary path
            this.processPath(path1, 'primary', allNodes, links, linkMap);
            
            // Process backup path
            if (path2.length > 0) {
                this.processPath(path2, 'backup', allNodes, links, linkMap);
            }
        });

        return {
            nodes: Array.from(allNodes).map(id => ({ id })),
            links: links
        };
    }

    processPath(path, type, allNodes, links, linkMap) {
        path.forEach(node => allNodes.add(node));
        
        for (let i = 0; i < path.length - 1; i++) {
            const linkKey = `${path[i]}-${path[i+1]}-${type}`;
            if (!linkMap.has(linkKey)) {
                links.push({
                    source: path[i],
                    target: path[i+1],
                    type: type
                });
                linkMap.set(linkKey, true);
            }
        }
    }

    renderD3Visualization(container, nodes, links) {
        const width = Math.max(container.clientWidth || 800, 600);
        const height = 400;

        const svg = d3.select(`#${container.id}`)
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .attr('viewBox', `0 0 ${width} ${height}`)
            .style('max-width', '100%')
            .style('height', 'auto');

        // Create simulation
        this.currentSimulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(80))
            .force('charge', d3.forceManyBody().strength(-200))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(25));

        // Create links
        const link = svg.append('g')
            .attr('class', 'links')
            .selectAll('line')
            .data(links)
            .enter().append('line')
            .attr('stroke-width', 2)
            .attr('stroke', d => d.type === 'primary' ? '#4CAF50' : '#FF9800')
            .attr('stroke-dasharray', d => d.type === 'backup' ? '5,5' : null);

        // Create nodes
        const node = svg.append('g')
            .attr('class', 'nodes')
            .selectAll('circle')
            .data(nodes)
            .enter().append('circle')
            .attr('r', 12)
            .attr('fill', '#2196F3')
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .style('cursor', 'grab')
            .call(d3.drag()
                .on('start', (event, d) => this.dragStarted(event, d))
                .on('drag', (event, d) => this.dragged(event, d))
                .on('end', (event, d) => this.dragEnded(event, d)));

        // Add labels
        const label = svg.append('g')
            .attr('class', 'labels')
            .selectAll('text')
            .data(nodes)
            .enter().append('text')
            .text(d => d.id)
            .attr('font-size', '11px')
            .attr('font-family', 'Arial, sans-serif')
            .attr('dx', 15)
            .attr('dy', 4)
            .style('pointer-events', 'none');

        // Update on tick
        this.currentSimulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            node
                .attr('cx', d => Math.max(12, Math.min(width - 12, d.x)))
                .attr('cy', d => Math.max(12, Math.min(height - 12, d.y)));

            label
                .attr('x', d => Math.max(12, Math.min(width - 12, d.x)))
                .attr('y', d => Math.max(12, Math.min(height - 12, d.y)));
        });

        this.addLegend(svg);
    }

    addLegend(svg) {
        const legend = svg.append('g')
            .attr('class', 'legend')
            .attr('transform', 'translate(20, 20)');

        // Primary path legend
        legend.append('line')
            .attr('x1', 0).attr('y1', 0)
            .attr('x2', 20).attr('y2', 0)
            .attr('stroke', '#4CAF50')
            .attr('stroke-width', 3);

        legend.append('text')
            .text('Primary Path')
            .attr('x', 25).attr('y', 5)
            .attr('font-size', '12px')
            .attr('font-family', 'Arial, sans-serif');

        // Backup path legend
        legend.append('line')
            .attr('x1', 0).attr('y1', 20)
            .attr('x2', 20).attr('y2', 20)
            .attr('stroke', '#FF9800')
            .attr('stroke-width', 3)
            .attr('stroke-dasharray', '5,5');

        legend.append('text')
            .text('Backup Path')
            .attr('x', 25).attr('y', 25)
            .attr('font-size', '12px')
            .attr('font-family', 'Arial, sans-serif');
    }

    dragStarted(event, d) {
        if (!event.active) this.currentSimulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    dragEnded(event, d) {
        if (!event.active) this.currentSimulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    createPathDetails(container, data, msanCode) {
        const detailsContainer = document.createElement('div');
        detailsContainer.className = 'path-details';

        const header = document.createElement('h3');
        header.textContent = `Path Details for ${msanCode}`;
        detailsContainer.appendChild(header);

        data.forEach((record, index) => {
            const recordDiv = this.createPathRecord(record, index);
            detailsContainer.appendChild(recordDiv);
        });

        container.appendChild(detailsContainer);
    }

    createPathRecord(record, index) {
        const recordDiv = document.createElement('div');
        recordDiv.className = 'path-record';
        
        const recordHeader = document.createElement('h4');
        recordHeader.textContent = `Record ${index + 1}`;
        recordDiv.appendChild(recordHeader);
        
        // Impact information
        const impactPara = document.createElement('p');
        const impactText = document.createElement('strong');
        impactText.textContent = 'Impact: ';
        const impactValue = document.createTextNode(record.Impact || 'Unknown');
        impactPara.appendChild(impactText);
        impactPara.appendChild(impactValue);
        impactPara.className = `impact-${(record.Impact || 'unknown').toLowerCase().replace(/\s+/g, '-')}`;
        recordDiv.appendChild(impactPara);
        
        // Primary path
        if (record.Path) {
            this.addPathSection(recordDiv, 'Primary Path', record.Path);
        }
        
        // Backup path
        if (record.Path2) {
            this.addPathSection(recordDiv, 'Backup Path', record.Path2);
        }
        
        return recordDiv;
    }

    addPathSection(container, title, pathData) {
        const pathHeader = document.createElement('h5');
        pathHeader.textContent = title + ':';
        container.appendChild(pathHeader);
        
        const pathList = document.createElement('ul');
        const path = this.parsePath(pathData);
        
        if (path.length === 0) {
            const li = document.createElement('li');
            li.textContent = 'No path data';
            li.style.fontStyle = 'italic';
            pathList.appendChild(li);
        } else {
            path.forEach(node => {
                const li = document.createElement('li');
                li.textContent = node;
                pathList.appendChild(li);
            });
        }
        
        container.appendChild(pathList);
    }

    parsePath(path) {
        if (!path) return [];
        
        if (Array.isArray(path)) {
            return path.filter(item => item != null && item !== '');
        }

        if (typeof path === 'string') {
            try {
                const parsed = JSON.parse(path);
                if (Array.isArray(parsed)) {
                    return parsed.filter(item => item != null && item !== '');
                }
            } catch (e) {
                // If JSON parsing fails, try other formats
                if (path.startsWith('[') && path.endsWith(']')) {
                    return path.slice(1, -1)
                        .split(',')
                        .map(item => item.trim().replace(/['"]/g, ''))
                        .filter(item => item !== '');
                }
                
                // Return as single item if not empty
                return path.trim() ? [path.trim()] : [];
            }
        }

        return [];
    }

    showErrorMessage(message) {
        const dashboardContainer = document.getElementById('dashboard-container');
        if (dashboardContainer) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'dashboard-error';
            
            const errorPara = document.createElement('p');
            errorPara.textContent = message;
            
            errorDiv.appendChild(errorPara);
            dashboardContainer.innerHTML = '';
            dashboardContainer.appendChild(errorDiv);
        }
    }

    cleanup() {
        // Stop simulation
        if (this.currentSimulation) {
            this.currentSimulation.stop();
            this.currentSimulation = null;
        }

        // Remove event listeners
        this.eventListeners.forEach(({ element, event, handler }) => {
            if (element && element.removeEventListener) {
                element.removeEventListener(event, handler);
            }
        });
        this.eventListeners = [];
    }

    destroy() {
        this.cleanup();
        
        // Disconnect observers
        this.observers.forEach(observer => observer.disconnect());
        this.observers = [];
        
        this.weData = null;
    }
}

// Usage example:
// const dashboard = new NetworkDashboard();
// dashboard.initialize(weData);