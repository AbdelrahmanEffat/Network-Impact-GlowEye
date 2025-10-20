// Fixed version of the network visualization code

document.addEventListener('DOMContentLoaded', function() {
    // Set up tab functionality
    setupTabs();
    
    // Initialize table filtering
    initializeFiltering();
    
    // Initialize pagination for each table
    initializePagination('we-table', 50);
    initializePagination('others-table', 50);

    // Network Topology
    initializeNetworkVisualization();

    // Add debug
    setTimeout(debugFiltering, 1000);
    
    // Test filtering after a short delay
    setTimeout(testFiltering, 2000);

    const chartsTabButton = document.querySelector('[data-tab="charts-tab"]');
        if (chartsTabButton) {
            chartsTabButton.addEventListener('click', function() {
                setTimeout(colorIsolatedText, 100);
            });
        }
        
        // Also run if charts tab is already active
        if (document.getElementById('charts-tab')?.classList.contains('active')) {
            setTimeout(colorIsolatedText, 100);
        }

    // Tab switching functionality (update existing code)
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetTab = this.getAttribute('data-tab');
            
            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to clicked button and corresponding content
            this.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
            
            // Special handling for dashboard tab
            if (targetTab === 'dashboard-tab') {
                // Trigger dashboard rendering if data is available
                if (window.networkDashboard && weData) {
                    // Small delay to ensure DOM is ready
                    setTimeout(() => {
                        window.networkDashboard.renderDashboard();
                    }, 100);
                }
            }
             if (targetTab === 'dashboard-tab-other') {
                if (window.othersDashboard && othersData && othersData.length > 0) {
        // Ensure DOM is ready
        setTimeout(() => {
            // Manually trigger render (only if needed)
            window.othersDashboard.renderDashboard();
        }, 100);
    }
}
        });
    });
});

// Function to color "Isolated" text red in charts tab
function colorIsolatedText() {
    const chartsTab = document.getElementById('charts-tab');
    if (!chartsTab) return;
    
    // Find all elements that might contain "Isolated" text
    const elements = chartsTab.querySelectorAll('.stat-label, .stat-value, .subsection-title, h3, h4, h5');
    
    elements.forEach(element => {
        const html = element.innerHTML;
        // Replace "Isolated" with red version
        const newHtml = html.replace(/Isolated/g, '<span style="color: #ff0000; font-weight: bold;">Isolated</span>');
        if (newHtml !== html) {
            element.innerHTML = newHtml;
        }
    });
}

function setupTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to clicked button and corresponding content
            this.classList.add('active');
            const tabId = this.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });
}

function initializeFiltering() {
    console.log('Initializing filtering...');
    
    // Initialize filtering for WE table
    initializeTableFiltering('we-tab', 'we-table');
    
    // Initialize filtering for Others table
    initializeTableFiltering('others-tab', 'others-table');
}

function initializeTableFiltering(tabId, tableId) {
    const tab = document.getElementById(tabId);
    if (!tab) {
        console.log(`Tab ${tabId} not found`);
        return;
    }
    
    const table = document.getElementById(tableId);
    if (!table) {
        console.log(`Table ${tableId} not found`);
        return;
    }
    
    const filterSelects = tab.querySelectorAll('.filter-select');
    console.log(`Found ${filterSelects.length} filter selects for ${tabId}`);
    
    if (filterSelects.length === 0) {
        console.log(`No filter selects found in ${tabId}`);
        return;
    }
    
    // Function to apply all active filters for this specific table
    function applyAllFilters() {
        const tbody = table.querySelector('tbody');
        const rows = tbody ? tbody.querySelectorAll('tr') : [];
        const activeFilters = [];
        
        // Collect all active filters from this tab only
        filterSelects.forEach(select => {
            const columnIndex = parseInt(select.getAttribute('data-column'));
            const filterValue = select.value.trim();
            if (filterValue !== "") {
                activeFilters.push({ columnIndex, filterValue });
            }
        });
        
        let visibleCount = 0;
        
        rows.forEach(row => {
            let matchesAll = true;
            
            // Check if row matches all active filters
            activeFilters.forEach(filter => {
                const cell = row.cells[filter.columnIndex];
                if (cell) {
                    const cellValue = cell.textContent.trim();
                    if (cellValue !== filter.filterValue) {
                        matchesAll = false;
                    }
                } else {
                    matchesAll = false;
                }
            });
            
            // Show/hide row based on filter match
            if (activeFilters.length === 0 || matchesAll) {
                row.style.display = '';
                visibleCount++;
            } else {
                row.style.display = 'none';
            }
        });
        
        console.log(`Table ${tableId}: ${visibleCount} rows visible with ${activeFilters.length} active filters`);
        return visibleCount;
    }
    
    // Add event listener to all filter dropdowns in this tab
    filterSelects.forEach(select => {
        select.addEventListener('change', function() {
            console.log(`Filter changed in ${tabId}: column ${this.getAttribute('data-column')}, value: "${this.value}"`);
            const visibleCount = applyAllFilters();
            
            // Update pagination after filtering
            updatePaginationAfterFilter(tableId);
        });
    });
    
    // Apply initial filters if any are set
    applyAllFilters();
    
    console.log(`Filtering initialized for ${tableId}`);
}

function updatePaginationAfterFilter(tableId) {
    const table = document.getElementById(tableId);
    if (!table) {
        console.log(`Table ${tableId} not found for pagination update`);
        return;
    }
    
    // Get all rows from tbody, but only count visible ones
    const tbody = table.querySelector('tbody');
    if (!tbody) return;
    
    const allRows = tbody.querySelectorAll('tr');
    const visibleRows = Array.from(allRows).filter(row => 
        row.style.display !== 'none'
    );
    
    console.log(`Pagination update for ${tableId}: ${visibleRows.length} visible rows out of ${allRows.length} total`);
    
    // Find the pagination controls
    const section = table.closest('.section') || table.parentElement;
    const paginationControls = section.querySelector('.pagination-controls');
    
    if (paginationControls) {
        // Update pagination to work with filtered results
        updatePaginationForFilteredRows(tableId, visibleRows);
    }
}

function updatePaginationForFilteredRows(tableId, visibleRows) {
    const pageSize = 50;
    const pageCount = Math.ceil(visibleRows.length / pageSize);
    let currentPage = 1;
    
    const table = document.getElementById(tableId);
    const section = table.closest('.section') || table.parentElement;
    const paginationControls = section.querySelector('.pagination-controls');
    
    if (!paginationControls) return;
    
    const prevButton = paginationControls.querySelector('.prev-page');
    const nextButton = paginationControls.querySelector('.next-page');
    const pageInfo = paginationControls.querySelector('.page-info');
    
    function showFilteredPage(page) {
        currentPage = page;
        const start = (page - 1) * pageSize;
        const end = start + pageSize;
        
        // Hide all rows first
        const tbody = table.querySelector('tbody');
        if (tbody) {
            const allRows = tbody.querySelectorAll('tr');
            allRows.forEach(row => {
                if (row.style.display !== 'none') {
                    row.style.display = 'none';
                    row.setAttribute('data-hidden-by-pagination', 'true');
                }
            });
        }
        
        // Show only the rows for current page
        visibleRows.forEach((row, index) => {
            if (index >= start && index < end) {
                row.style.display = '';
                row.removeAttribute('data-hidden-by-pagination');
            }
        });
        
        pageInfo.textContent = `Page ${page} of ${pageCount} (${visibleRows.length} filtered results)`;
        
        prevButton.disabled = currentPage === 1;
        nextButton.disabled = currentPage === pageCount || pageCount === 0;
    }
    
    // Remove old event listeners and add new ones
    const newPrevButton = prevButton.cloneNode(true);
    const newNextButton = nextButton.cloneNode(true);
    prevButton.parentNode.replaceChild(newPrevButton, prevButton);
    nextButton.parentNode.replaceChild(newNextButton, nextButton);
    
    newPrevButton.addEventListener('click', () => {
        if (currentPage > 1) showFilteredPage(currentPage - 1);
    });
    
    newNextButton.addEventListener('click', () => {
        if (currentPage < pageCount) showFilteredPage(currentPage + 1);
    });
    
    showFilteredPage(1);
}

function debugFiltering() {
    console.log('=== DEBUG FILTERING ===');
    
    // Check if filter elements exist with more detail
    const weTab = document.getElementById('we-tab');
    const othersTab = document.getElementById('others-tab');
    
    if (weTab) {
        const weFilters = weTab.querySelectorAll('.filter-select');
        console.log(`WE tab filters found: ${weFilters.length}`);
        
        weFilters.forEach((filter, index) => {
            console.log(`WE Filter ${index}: column=${filter.getAttribute('data-column')}, options=${filter.children.length}`);
        });
    }
    
    if (othersTab) {
        const othersFilters = othersTab.querySelectorAll('.filter-select');
        console.log(`Others tab filters found: ${othersFilters.length}`);
        
        othersFilters.forEach((filter, index) => {
            console.log(`Others Filter ${index}: column=${filter.getAttribute('data-column')}, options=${filter.children.length}`);
        });
    }
    
    // Check table rows
    const weTable = document.getElementById('we-table');
    const othersTable = document.getElementById('others-table');
    
    if (weTable) {
        const weRows = weTable.querySelectorAll('tbody tr');
        console.log(`WE table rows: ${weRows.length}`);
        
        // Check first few rows to verify data structure
        if (weRows.length > 0) {
            const firstRow = weRows[0];
            const cells = firstRow.querySelectorAll('td');
            console.log(`First WE row has ${cells.length} cells`);
            cells.forEach((cell, index) => {
                console.log(`WE Cell ${index}: ${cell.textContent.trim().substring(0, 50)}...`);
            });
        }
    }
    
    if (othersTable) {
        const othersRows = othersTable.querySelectorAll('tbody tr');
        console.log(`Others table rows: ${othersRows.length}`);
    }
    
    // Test filter functionality
    console.log('Testing filter change events...');
    const testFilter = document.querySelector('.filter-select');
    if (testFilter) {
        console.log('Dispatching test change event...');
        testFilter.dispatchEvent(new Event('change'));
    }
    
    console.log('=== END DEBUG ===');
}

function testFiltering() {
    console.log('=== TESTING FILTERING ===');
    
    // Test WE table filtering
    const weFilter = document.querySelector('#we-tab .filter-select');
    if (weFilter) {
        console.log('Testing WE filter...');
        
        // Get the first non-empty option value
        for (let i = 1; i < weFilter.options.length; i++) {
            if (weFilter.options[i].value) {
                const testValue = weFilter.options[i].value;
                console.log(`Setting WE filter to: ${testValue}`);
                weFilter.value = testValue;
                weFilter.dispatchEvent(new Event('change'));
                break;
            }
        }
    }
    
    setTimeout(() => {
        // Test Others table filtering
        const othersFilter = document.querySelector('#others-tab .filter-select');
        if (othersFilter) {
            console.log('Testing Others filter...');
            
            // Get the first non-empty option value
            for (let i = 1; i < othersFilter.options.length; i++) {
                if (othersFilter.options[i].value) {
                    const testValue = othersFilter.options[i].value;
                    console.log(`Setting Others filter to: ${testValue}`);
                    othersFilter.value = testValue;
                    othersFilter.dispatchEvent(new Event('change'));
                    break;
                }
            }
        }
    }, 1000);
}

function initializePagination(tableId, pageSize = 50) {
    const table = document.getElementById(tableId);
    if (!table) return;
    
    const rows = table.querySelectorAll('tbody tr');
    const pageCount = Math.ceil(rows.length / pageSize);
    let currentPage = 1;
    
    // Create pagination controls if they don't exist
    let paginationControls = table.closest('.section').querySelector('.pagination-controls');
    if (!paginationControls) {
        paginationControls = document.createElement('div');
        paginationControls.className = 'pagination-controls';
        paginationControls.innerHTML = `
            <button class="prev-page">Previous</button>
            <span class="page-info">Page 1 of ${pageCount}</span>
            <button class="next-page">Next</button>
        `;
        table.parentNode.insertBefore(paginationControls, table.nextSibling);
    }
    
    const prevButton = paginationControls.querySelector('.prev-page');
    const nextButton = paginationControls.querySelector('.next-page');
    const pageInfo = paginationControls.querySelector('.page-info');
    
    function showPage(page) {
        currentPage = page;
        const start = (page - 1) * pageSize;
        const end = start + pageSize;
        
        rows.forEach((row, index) => {
            row.style.display = (index >= start && index < end) ? '' : 'none';
        });
        
        pageInfo.textContent = `Page ${page} of ${pageCount}`;
        
        // Enable/disable buttons
        prevButton.disabled = currentPage === 1;
        nextButton.disabled = currentPage === pageCount;
    }
    
    prevButton.addEventListener('click', () => {
        if (currentPage > 1) showPage(currentPage - 1);
    });
    
    nextButton.addEventListener('click', () => {
        if (currentPage < pageCount) showPage(currentPage + 1);
    });
    
    showPage(1);
}

function initializeNetworkVisualization() {
    const topologyTab = document.getElementById('topology-tab');
    if (!topologyTab) return;
    
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                if (topologyTab.classList.contains('active')) {
                    renderNetworkTopology();
                }
            }
        });
    });
    
    observer.observe(topologyTab, { attributes: true });
}

function renderNetworkTopology() {
    console.log("Rendering network topology...");
    
    try {
        // Clear previous visualization
        d3.select('#network-topology').selectAll('*').remove();
        
        // Check if required variables exist
        const hasWeData = typeof weData !== 'undefined' && weData && weData.length > 0;
        const hasOthersData = typeof othersData !== 'undefined' && othersData && othersData.length > 0;
        const hasIdentifier = typeof identifier !== 'undefined' && identifier;
        const hasIdentifierType = typeof identifierType !== 'undefined' && identifierType;
        
        console.log("Data availability:", { hasWeData, hasOthersData, hasIdentifier, hasIdentifierType });
        
        if (!hasWeData && !hasOthersData) {
            showErrorMessage("No network data available. Please ensure data is loaded properly.");
            return;
        }
        
        // Extract node and link data from the results
        const graphData = extractGraphData();
        console.log("Graph data:", graphData);
        
        if (graphData.nodes.length === 0) {
            showErrorMessage("No network nodes found in the data.");
            return;
        }
        
        // Hide loading message, show visualization container
        hideMessages();
        
        // Create SVG container
        const container = document.getElementById('network-topology');
        const width = container.clientWidth || 800;
        const height = container.clientHeight || 600;
        
        const svg = d3.select('#network-topology')
            .append('svg')
            .attr('width', width)
            .attr('height', height);
        
        // Add zoom group
        const g = svg.append('g');
        
        // Create tooltip
        const tooltip = d3.select('body')
            .append('div')
            .attr('class', 'tooltip')
            .style('position', 'absolute')
            .style('padding', '10px')
            .style('background-color', 'rgba(0, 0, 0, 0.8)')
            .style('color', 'white')
            .style('border-radius', '4px')
            .style('pointer-events', 'none')
            .style('font-size', '12px')
            .style('opacity', 0);
        
        // Create force simulation
        const simulation = d3.forceSimulation(graphData.nodes)
            .force('link', d3.forceLink(graphData.links).id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(15));
        
        // Draw links
        const link = g.append('g')
            .attr('class', 'links')
            .selectAll('line')
            .data(graphData.links)
            .enter()
            .append('line')
            .attr('class', d => `link ${d.type}`)
            .attr('stroke', d => {
                switch(d.type) {
                    case 'affected': return '#ff0000';
                    case 'alternative': return '#ff9900';
                    default: return '#999';
                }
            })
            .attr('stroke-width', d => d.type === 'normal' ? 1 : 2)
            .attr('stroke-opacity', 0.6);
        
        // Draw nodes
        const node = g.append('g')
            .attr('class', 'nodes')
            .selectAll('circle')
            .data(graphData.nodes)
            .enter()
            .append('circle')
            .attr('class', d => `node ${d.type}`)
            .attr('r', 8)
            .attr('fill', d => d.color)
            .attr('stroke', d => d.failed ? '#ff0000' : '#fff')
            .attr('stroke-width', d => d.failed ? 3 : 2)
            .style('cursor', 'pointer')
            .call(d3.drag()
                .on('start', dragStarted)
                .on('drag', dragged)
                .on('end', dragEnded))
            .on('mouseover', function(event, d) {
                tooltip.transition()
                    .duration(200)
                    .style('opacity', .9);
                tooltip.html(`
                    <div><strong>${d.id}</strong></div>
                    <div>Type: ${d.type}</div>
                    <div>Status: ${d.failed ? 'Failed' : 'Operational'}</div>
                    ${d.impact ? `<div>Impact: ${d.impact}</div>` : ''}
                `)
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 28) + 'px');
            })
            .on('mouseout', function() {
                tooltip.transition()
                    .duration(500)
                    .style('opacity', 0);
            });
        
        // Add node labels
        const label = g.append('g')
            .attr('class', 'labels')
            .selectAll('text')
            .data(graphData.nodes)
            .enter()
            .append('text')
            .attr('class', 'node-label')
            .text(d => d.id.length > 10 ? d.id.substring(0, 10) + '...' : d.id)
            .style('font-size', '10px')
            .style('text-anchor', 'middle')
            .style('dy', -12)
            .style('pointer-events', 'none');
        
        // Update simulation on tick
        simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);
            
            label
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        });
        
        // Drag functions
        function dragStarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        
        function dragEnded(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
        
        // Add zoom functionality
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', function(event) {
                g.attr('transform', event.transform);
            });
        
        svg.call(zoom);
        
        // Add control button functionality
        const resetBtn = document.getElementById('reset-view');
        const zoomInBtn = document.getElementById('zoom-in');
        const zoomOutBtn = document.getElementById('zoom-out');
        
        if (resetBtn) {
            resetBtn.addEventListener('click', function() {
                svg.transition()
                    .duration(750)
                    .call(zoom.transform, d3.zoomIdentity);
            });
        }
        
        if (zoomInBtn) {
            zoomInBtn.addEventListener('click', function() {
                svg.transition()
                    .duration(300)
                    .call(zoom.scaleBy, 1.5);
            });
        }
        
        if (zoomOutBtn) {
            zoomOutBtn.addEventListener('click', function() {
                svg.transition()
                    .duration(300)
                    .call(zoom.scaleBy, 0.75);
            });
        }

    } catch (error) {
        console.error("Error rendering network topology:", error);
        showErrorMessage(`Failed to load network visualization: ${error.message}`);
    }
}

function extractGraphData() {
    console.log("Extracting graph data...");
    
    const nodes = [];
    const links = [];
    const nodeMap = new Map();
    
    // Function to add a node if it doesn't exist
    function addNode(id, type, failed = false) {
        if (!id || typeof id !== 'string' || nodeMap.has(id)) return;
        
        let color;
        switch(type) {
            case 'edge': color = '#4CAF50'; break;
            case 'distribution': color = '#2196F3'; break;
            case 'bng': color = '#FF9800'; break;
            case 'bitstream': color = '#9C27B0'; break;
            case 'failed': color = '#ff0000'; break;
            default: color = '#795548';
        }
        
        nodes.push({
            id: id,
            type: type,
            color: color,
            failed: failed
        });
        
        nodeMap.set(id, true);
    }
    
    // Function to add a link if it doesn't exist
    function addLink(source, target, type = 'normal') {
        if (!source || !target || typeof source !== 'string' || typeof target !== 'string') return;
        if (source === target) return; // No self-links
        
        // Check if link already exists
        const linkExists = links.some(link => 
            (link.source === source && link.target === target) ||
            (link.source === target && link.target === source)
        );
        
        if (!linkExists) {
            links.push({
                source: source,
                target: target,
                type: type
            });
        }
    }
    
    try {
        // Add some sample data if no real data is available
        if (typeof weData === 'undefined' && typeof othersData === 'undefined') {
            console.log("No data available, creating sample network");
            addNode('Sample-EDGE-01', 'edge');
            addNode('Sample-DIST-01', 'distribution');
            addNode('Sample-BNG-01', 'bng');
            addNode('Failed-Node', 'failed', true);
            
            addLink('Sample-EDGE-01', 'Sample-DIST-01');
            addLink('Sample-DIST-01', 'Sample-BNG-01');
            addLink('Failed-Node', 'Sample-EDGE-01', 'affected');
            
            return { nodes, links };
        }
        
        // Process WE data if available
        if (typeof weData !== 'undefined' && weData && weData.length > 0) {
            console.log("Processing WE data:", weData.length, "records");
            
            weData.forEach(record => {
                if (!record) return;
                
                // Add nodes
                if (record.EDGE) addNode(record.EDGE, 'edge');
                if (record.distribution_hostname) addNode(record.distribution_hostname, 'distribution');
                if (record.BNG_HOSTNAME) addNode(record.BNG_HOSTNAME, 'bng');
                
                // Add links
                if (record.EDGE && record.distribution_hostname) {
                    addLink(record.EDGE, record.distribution_hostname, 
                           record.Impact === 'Isolated' ? 'affected' : 'normal');
                }
                
                if (record.distribution_hostname && record.BNG_HOSTNAME) {
                    addLink(record.distribution_hostname, record.BNG_HOSTNAME, 'normal');
                }
            });
        }
        
        // Process Others data if available
        if (typeof othersData !== 'undefined' && othersData && othersData.length > 0) {
            console.log("Processing Others data:", othersData.length, "records");
            
            othersData.forEach(record => {
                if (!record) return;
                
                // Add nodes
                if (record.EDGE) addNode(record.EDGE, 'edge');
                if (record.BITSTREAM_HOSTNAME) addNode(record.BITSTREAM_HOSTNAME, 'bitstream');
                
                // Add links
                if (record.EDGE && record.BITSTREAM_HOSTNAME) {
                    addLink(record.EDGE, record.BITSTREAM_HOSTNAME,
                           record.Impact === 'Isolated' ? 'affected' : 'normal');
                }
            });
        }
        
        // Add the failed node/exchange if identifier is available
        if (typeof identifier !== 'undefined' && identifier && typeof identifierType !== 'undefined') {
            if (identifierType === 'node') {
                addNode(identifier, 'failed', true);
            }
        }
        
        console.log("Extracted", nodes.length, "nodes and", links.length, "links");
        return { nodes, links };
        
    } catch (error) {
        console.error("Error in extractGraphData:", error);
        // Return sample data on error
        addNode('Error-Node', 'failed', true);
        addNode('Sample-Node', 'edge');
        addLink('Error-Node', 'Sample-Node', 'affected');
        return { nodes, links };
    }
}

function showErrorMessage(message) {
    const loadingMsg = document.getElementById('loading-message');
    const errorMsg = document.getElementById('error-message');
    
    if (loadingMsg) loadingMsg.style.display = 'none';
    if (errorMsg) {
        errorMsg.style.display = 'block';
        errorMsg.textContent = message;
    }
}

function hideMessages() {
    const loadingMsg = document.getElementById('loading-message');
    const errorMsg = document.getElementById('error-message');
    
    if (loadingMsg) loadingMsg.style.display = 'none';
    if (errorMsg) errorMsg.style.display = 'none';
}