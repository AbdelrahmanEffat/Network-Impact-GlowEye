
class NetworkDashboard {
    constructor() {
        console.log('NetworkDashboard constructor called');
        this.weData = null;
        this.currentSimulation = null;
        this.observers = [];
        this.eventListeners = [];
        this.failedNodeId = null;
        this.isInitialized = false;
    }

    initialize(data, failedNodeId = null) {
        console.log('NetworkDashboard.initialize called with failedNodeId:', failedNodeId);
        
        this.weData = data;
        this.failedNodeId = failedNodeId;
        
        if (typeof d3 === 'undefined') {
            console.error('D3.js library is required but not loaded');
            return false;
        }

        this.setupTabHandling();
        
        if (document.getElementById('dashboard-tab')?.classList.contains('active')) {
            setTimeout(() => this.renderDashboard(), 100);
        }
    
        return true;
    }

    setupTabHandling() {
        console.log('Setting up tab handling...');
        
        const tabButtons = document.querySelectorAll('.tab-button');
        tabButtons.forEach(button => {
            const clickHandler = () => {
                const tabId = button.getAttribute('data-tab');
                if (tabId === 'dashboard-tab') {
                    console.log('Dashboard tab clicked, rendering...');
                    setTimeout(() => this.renderDashboard(), 100);
                }
            };
            
            button.addEventListener('click', clickHandler);
            this.eventListeners.push({ 
                element: button, 
                event: 'click', 
                handler: clickHandler 
            });
        });
        
        console.log('Tab handling setup complete');
    }

    renderDashboard() {
        console.log('=== RENDER DASHBOARD CALLED ===');
        console.log('Dashboard state:', {
            weDataLength: this.weData ? this.weData.length : 'null',
            failedNodeId: this.failedNodeId,
            isInitialized: this.isInitialized
        });

        try {
            if (!this.validateData()) {
                this.showErrorMessage("No valid WE data available for dashboard visualization.");
                return;
            }
            
            this.cleanup();
            
            const dashboardContainer = document.getElementById('dashboard-container');
            if (!dashboardContainer) {
                this.showErrorMessage("Dashboard container not found.");
                return;
            }
            
            dashboardContainer.innerHTML = '';
            
            this.createDashboardControls();
            
            const firstMsan = this.weData[0].MSANCODE;
            console.log('Updating visualization with first MSAN:', firstMsan);
            this.updateVisualization(firstMsan);
            
        } catch (error) {
            console.error("Error rendering dashboard:", error);
            this.showErrorMessage(`Failed to render dashboard: ${error.message}`);
        }
    }

    validateData() {
        const isValid = this.weData && 
               Array.isArray(this.weData) && 
               this.weData.length > 0 &&
               this.weData.some(item => item.MSANCODE);
        
        console.log('Data validation result:', isValid);
        return isValid;
    }

    createDashboardControls() {
        const dashboardContainer = document.getElementById('dashboard-container');
        if (!dashboardContainer) return;

        const controlsDiv = document.createElement('div');
        controlsDiv.className = 'dashboard-controls';

        const msanSelectorDiv = this.createMsanSelector();
        
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

        const msanCodes = [...new Set(
            this.weData
                .filter(item => item.MSANCODE)
                .map(item => item.MSANCODE)
        )].sort();

        msanCodes.forEach(code => {
            const option = document.createElement('option');
            option.value = code;
            option.textContent = code;
            msanSelector.appendChild(option);
        });

        const changeHandler = () => this.updateVisualization(msanSelector.value);
        msanSelector.addEventListener('change', changeHandler);
        this.eventListeners.push({ element: msanSelector, event: 'change', handler: changeHandler });

        msanSelectorDiv.appendChild(msanLabel);
        msanSelectorDiv.appendChild(msanSelector);
        
        return msanSelectorDiv;
    }

    updateVisualization(msanCode) {
        console.log('=== UPDATE VISUALIZATION ===');
        console.log('MSAN Code:', msanCode);
        console.log('Failed Node ID:', this.failedNodeId);
        
        const vizContainer = document.getElementById('dashboard-viz');
        if (!vizContainer) return;

        if (this.currentSimulation) {
            this.currentSimulation.stop();
        }

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
        console.log('=== CREATE NETWORK VISUALIZATION ===');
        
        const header = document.createElement('h3');
        header.textContent = `Network Path Visualization for ${msanCode}`;
        container.appendChild(header);

        // Create a wrapper for visualization and info panel
        const vizWrapper = document.createElement('div');
        vizWrapper.style.display = 'flex';
        vizWrapper.style.gap = '20px';
        vizWrapper.style.alignItems = 'flex-start';

        const svgContainer = document.createElement('div');
        svgContainer.id = 'dashboard-network-viz';
        svgContainer.className = 'network-viz-svg';
        svgContainer.style.minHeight = '400px';
        svgContainer.style.flex = '1';
        
        // Create info panel on the right
        const infoPanel = this.createInfoPanel(data,msanCode);
        
        vizWrapper.appendChild(svgContainer);
        vizWrapper.appendChild(infoPanel);
        container.appendChild(vizWrapper);

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
        const linkMap = new Map();

        data.forEach(record => {
            const path1 = this.parsePath(record.Path);
            const path2 = this.parsePath(record.Path2);
            
            this.processPath(path1, 'primary', allNodes, links, linkMap);
            
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
        console.log('=== RENDER D3 VISUALIZATION ===');
        console.log('Nodes to render:', nodes.map(n => n.id));
        console.log('Failed node to highlight:', this.failedNodeId);
        
        const width = Math.max(container.clientWidth || 800, 600);
        const height = 400;

        const svg = d3.select(`#${container.id}`)
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .attr('viewBox', `0 0 ${width} ${height}`)
            .style('max-width', '100%')
            .style('height', 'auto');

        this.currentSimulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(80))
            .force('charge', d3.forceManyBody().strength(-200))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(25));

        const link = svg.append('g')
            .attr('class', 'links')
            .selectAll('line')
            .data(links)
            .enter().append('line')
            .attr('stroke-width', 2)
            .attr('stroke', d => d.type === 'primary' ? '#4CAF50' : '#FF9800')
            .attr('stroke-dasharray', d => d.type === 'backup' ? '5,5' : null);

        const node = svg.append('g')
            .attr('class', 'nodes')
            .selectAll('circle')
            .data(nodes)
            .enter().append('circle')
            .attr('r', 12)
            .attr('fill', d => {
                const isFailed = this.isFailedNode(d.id);
                const isEdge = this.isEdgeNode(d.id);
                if (isFailed) return '#ff0000';
                if (isEdge) return '#4CAF50';
                return '#2196F3';
            })
            .attr('stroke', d => this.isFailedNode(d.id) ? '#ff0000' : '#fff')
            .attr('stroke-width', d => this.isFailedNode(d.id) ? 3 : 2)
            .attr('class', d => this.isFailedNode(d.id) ? 'failed-node' : '')
            .style('cursor', 'grab')
            .call(d3.drag()
                .on('start', (event, d) => this.dragStarted(event, d))
                .on('drag', (event, d) => this.dragged(event, d))
                .on('end', (event, d) => this.dragEnded(event, d)));

        const label = svg.append('g')
            .attr('class', 'labels')
            .selectAll('text')
            .data(nodes)
            .enter().append('text')
            .text(d => d.id)
            .attr('font-size', '11px')
            .attr('font-family', 'Arial, sans-serif')
            .attr('font-weight', d => this.isFailedNode(d.id) ? 'bold' : 'normal')
            .attr('fill', d => this.isFailedNode(d.id) ? '#ff0000' : '#333')
            .attr('dx', 15)
            .attr('dy', 4)
            .attr('class', d => this.isFailedNode(d.id) ? 'failed-node' : '')
            .style('pointer-events', 'none');

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

    isFailedNode(nodeId) {
        if (!this.failedNodeId || !nodeId) {
            return false;
        }
        
        const normalize = (str) => {
            return String(str).toLowerCase().trim().replace(/\s+/g, '');
        };
        
        const normalizedNode = normalize(nodeId);
        const normalizedFailed = normalize(this.failedNodeId);
        
        const isFailed = normalizedNode === normalizedFailed || 
                        normalizedNode.includes(normalizedFailed) ||
                        normalizedFailed.includes(normalizedNode);
        
        if (isFailed) {
            console.log(`FAILED NODE MATCH: ${nodeId} vs ${this.failedNodeId}`);
        }
        
        return isFailed;
    }

    isEdgeNode(nodeId) {
        if (!nodeId || !this.weData) {
            return false;
        }
        
        const normalize = (str) => {
            return String(str).toLowerCase().trim().replace(/\s+/g, '');
        };
        
        const normalizedNode = normalize(nodeId);
        
        for (const record of this.weData) {
            const primaryPath = this.parsePath(record.Path);
            if (primaryPath.length > 0) {
                const firstNode = normalize(primaryPath[0]);
                if (firstNode === normalizedNode) {
                    return true;
                }
            }
        }
        
        return false;
    }

    addLegend(svg) {
        const legend = svg.append('g')
            .attr('class', 'legend')
            .attr('transform', 'translate(20, 20)');

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

        legend.append('circle')
            .attr('cx', 10).attr('cy', 45)
            .attr('r', 6)
            .attr('fill', '#4CAF50')
            .attr('stroke', '#fff')
            .attr('stroke-width', 1);

        legend.append('text')
            .text('Edge Node')
            .attr('x', 25).attr('y', 50)
            .attr('font-size', '12px')
            .attr('font-family', 'Arial, sans-serif');

        legend.append('circle')
            .attr('cx', 10).attr('cy', 65)
            .attr('r', 6)
            .attr('fill', '#ff0000')
            .attr('stroke', '#fff')
            .attr('stroke-width', 1);

        legend.append('text')
            .text('Failed Node')
            .attr('x', 25).attr('y', 70)
            .attr('font-size', '12px')
            .attr('font-family', 'Arial, sans-serif');

        legend.append('circle')
            .attr('cx', 10).attr('cy', 85)
            .attr('r', 6)
            .attr('fill', '#2196F3')
            .attr('stroke', '#fff')
            .attr('stroke-width', 1);

        legend.append('text')
            .text('Normal Node')
            .attr('x', 25).attr('y', 90)
            .attr('font-size', '12px')
            .attr('font-family', 'Arial, sans-serif');
    }
createInfoPanel(data, msanCode) {
    const panel = document.createElement('div');
    panel.className = 'info-panel';
    panel.style.minWidth = '140px'; // smaller width
    panel.style.padding = '3px 4px'; // less padding
    panel.style.backgroundColor = '#fafafa';
    panel.style.borderRadius = '5px';
    panel.style.border = '1px solid #ccc';
    panel.style.fontSize = '10.5px'; // smaller font
    panel.style.lineHeight = '1.2';

    // Title
    const title = document.createElement('h4');
    title.textContent = 'Network Information';
    title.style.margin = '0 0 4px 0'; // tighter margin
    title.style.fontSize = '11px'; // smaller title
    title.style.color = '#333';
    title.style.borderBottom = '1px solid #bbb';
    title.style.paddingBottom = '2px';
    panel.appendChild(title);

    // MSAN Display
    const msanDiv = document.createElement('div');
    msanDiv.style.marginBottom = '6px';
    msanDiv.style.padding = '3px 6px';
    msanDiv.style.backgroundColor = '#e8f0fe';
    msanDiv.style.borderRadius = '3px';
    msanDiv.style.borderLeft = '3px solid #2196F3';
    msanDiv.style.fontSize = '10.5px';
    msanDiv.innerHTML = `<strong style="color: #1976d2;">MSAN:</strong> ${msanCode}`;
    panel.appendChild(msanDiv);

    // Status groups
    const upItems = new Set();
    const stItems = new Set();

    const filtered = data.filter(r => r.MSANCODE === msanCode);
    filtered.forEach(record => {
        const status = (record.STATUS || '').toLowerCase();
        const entries = [];
        if (record.BNG_HOSTNAME) entries.push(`BNG HOSTNAME: ${record.BNG_HOSTNAME}`);
        if (record.IGW_HOSTNAME) entries.push(`IGW HOSTNAME: ${record.IGW_HOSTNAME}`);
        if (record.IGW_INTERFACES) entries.push(`IGW INTERFACE : ${record.IGW_INTERFACES}`);
        entries.forEach(item => {
            if (status === 'up') upItems.add(item);
            else if (status === 'st') stItems.add(item);
        });
    });

    // Grid layout (smaller and tighter)
    const grid = document.createElement('div');
    grid.style.display = 'grid';
    grid.style.gridTemplateColumns = '1fr 1fr';
    grid.style.gap = '4px'; // smaller gap

    const makeColumn = (titleText, color, items) => {
        const col = document.createElement('div');
        const header = document.createElement('div');
        header.textContent = `${titleText} (${items.size})`;
        header.style.color = color;
        header.style.fontWeight = '600';
        header.style.fontSize = '10.5px';
        header.style.marginBottom = '2px';
        col.appendChild(header);

        if (items.size === 0) {
            const empty = document.createElement('div');
            empty.textContent = '—';
            empty.style.color = '#aaa';
            empty.style.fontSize = '10px';
            col.appendChild(empty);
        } else {
            items.forEach(v => {
                const div = document.createElement('div');
                div.textContent = v;
                div.style.padding = '2px 2px';
                div.style.marginBottom = '2px';
                div.style.backgroundColor = '#fff';
                div.style.border = '1px solid #ddd';
                div.style.borderRadius = '2px';
                div.style.fontSize = '10px';
                col.appendChild(div);
            });
        }
        return col;
    };

    grid.appendChild(makeColumn('UP', '#2E7D32', upItems));
    grid.appendChild(makeColumn('ST', '#C62828', stItems));

    panel.appendChild(grid);
    return panel;
}


    addInfoSection(container, title, values, color) {
        const section = document.createElement('div');
        section.style.marginBottom = '15px';

        const sectionTitle = document.createElement('div');
        sectionTitle.style.fontWeight = 'bold';
        sectionTitle.style.marginBottom = '5px';
        sectionTitle.style.color = color;
        sectionTitle.style.fontSize = '13px';
        sectionTitle.innerHTML = `<span style="display: inline-block; width: 8px; height: 8px; background-color: ${color}; border-radius: 50%; margin-right: 6px;"></span>${title}:`;
        section.appendChild(sectionTitle);

        if (values.length === 0) {
            const noData = document.createElement('div');
            noData.textContent = 'N/A';
            noData.style.fontStyle = 'italic';
            noData.style.color = '#999';
            noData.style.fontSize = '12px';
            noData.style.paddingLeft = '14px';
            section.appendChild(noData);
        } else {
            values.forEach(value => {
                const valueDiv = document.createElement('div');
                valueDiv.textContent = value;
                valueDiv.style.fontSize = '12px';
                valueDiv.style.padding = '4px 8px';
                valueDiv.style.marginLeft = '14px';
                valueDiv.style.backgroundColor = '#fff';
                valueDiv.style.borderRadius = '4px';
                valueDiv.style.marginBottom = '4px';
                valueDiv.style.border = '1px solid #e0e0e0';
                valueDiv.style.wordBreak = 'break-word';
                section.appendChild(valueDiv);
            });
        }

        container.appendChild(section);
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
        
        const recordNames = ['Main logical Path', 'Backup Logical Path'];
        const recordHeader = document.createElement('h4');
        recordHeader.textContent = recordNames[index] || `Record ${index + 1}`;
        recordDiv.appendChild(recordHeader);
        
        const impactPara = document.createElement('p');
        const impactText = document.createElement('strong');
        impactText.textContent = 'Impact: ';
        const impactValue = document.createTextNode(record.Impact || 'Unknown');
        impactPara.appendChild(impactText);
        impactPara.appendChild(impactValue);
        impactPara.className = `impact-${(record.Impact || 'unknown').toLowerCase().replace(/\s+/g, '-')}`;
        recordDiv.appendChild(impactPara);
        
        if (record.Path) {
            this.addPathSection(recordDiv, 'Primary Physical Path', record.Path);
        }
        
        if (record.Path2) {
            this.addPathSection(recordDiv, 'Backup Physical Path', record.Path2);
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
                
                if (this.isFailedNode(node)) {
                    li.style.color = '#ff0000';
                    li.style.fontWeight = 'bold';
                    li.style.backgroundColor = '#ffebee';
                    li.style.padding = '2px 4px';
                    li.style.borderRadius = '3px';
                }
                
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
                if (path.startsWith('[') && path.endsWith(']')) {
                    return path.slice(1, -1)
                        .split(',')
                        .map(item => item.trim().replace(/['"]/g, ''))
                        .filter(item => item !== '');
                }
                
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
        if (this.currentSimulation) {
            this.currentSimulation.stop();
            this.currentSimulation = null;
        }

        this.eventListeners.forEach(({ element, event, handler }) => {
            if (element && element.removeEventListener) {
                element.removeEventListener(event, handler);
            }
        });
        this.eventListeners = [];
    }

    destroy() {
        this.cleanup();
        
        this.observers.forEach(observer => observer.disconnect());
        this.observers = [];
        
        this.weData = null;
        this.failedNodeId = null;
    }
    
}
