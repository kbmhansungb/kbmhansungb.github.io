---
layout: post
title: KnowledgeGraph
---

<script src="https://d3js.org/d3.v7.min.js"></script>

<script>
document.addEventListener("DOMContentLoaded", function() {
  var nodes = [
    { id: 'A', x: 50, y: 50, size: 8 },
    { id: 'B', x: 150, y: 50, size: 8 },
    { id: 'C', x: 100, y: 100, size: 8 },
    { id: 'D', x: 50, y: 150, size: 8 }
  ];

  var links = [
    { source: 'A', target: 'B' },
    { source: 'B', target: 'C' },
    { source: 'C', target: 'A' },
    { source: 'D', target: 'A' }
  ];

  // Creating a map for node id's to nodes for quick lookup
  var nodeById = nodes.reduce(function(map, node) {
    map[node.id] = node;
    return map;
  }, {});

  // Replacing string references in links with actual node objects
  links.forEach(function(link) {
    link.source = nodeById[link.source];
    link.target = nodeById[link.target];
  });

  var svg = d3.select('#graph');

  // Defining arrowheads (if needed)
  svg.append('defs').append('marker')
    .attr('id', 'arrowhead')
    .attr('viewBox', '-0 -5 10 10')
    .attr('refX', 13)
    .attr('refY', 0)
    .attr('orient', 'auto')
    .attr('markerWidth', 5)
    .attr('markerHeight', 5)
    .attr('xoverflow', 'visible')
    .append('svg:path')
    .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
    .attr('fill', '#808080')
    .style('stroke','none');

  // Creating links
  var link = svg.selectAll('line')
    .data(links)
    .enter().append('line')
    .attr('stroke', '#999')
    .attr('stroke-width', '2')
    .attr('marker-end', 'url(#arrowhead)'); // If you want arrowheads

  // Creating nodes
  var node = svg.selectAll('circle')
    .data(nodes)
    .enter().append('circle')
    .attr('r', function(d) { return d.size; })
    // fill color
    .attr('fill', '#808080')

  // Creating labels
  var text = svg.selectAll('text')
    .data(nodes)
    .enter().append('text')
    .attr('dy', -15)
    .attr('text-anchor', 'middle')
    .text(function(d) { return d.id; });

  // Applying force simulation if needed
  // var simulation = d3.forceSimulation(nodes)
  //   .force('link', d3.forceLink(links).id(function(d) { return d.id; }))
  //   .force('charge', d3.forceManyBody())
  //   .force('center', d3.forceCenter(svg.attr('width') / 2, svg.attr('height') / 2));

  // simulation.on('tick', ticked);

  // function ticked() {
  //   link
  //     .attr('x1', function(d) { return d.source.x; })
  //     .attr('y1', function(d) { return d.source.y; })
  //     .attr('x2', function(d) { return d.target.x; })
  //     .attr('y2', function(d) { return d.target.y; });

  //   node
  //     .attr('cx', function(d) { return d.x; })
  //     .attr('cy', function(d) { return d.y; });

  //   text
  //     .attr('x', function(d) { return d.x; })
  //     .attr('y', function(d) { return d.y; });
  // }

  // Drag and drop functionality
  function dragstarted(event, d) {
    d3.select(this).raise().classed('active', true);
  }

  function dragged(event, d) {
    d.x = event.x;
    d.y = event.y;
    d3.select(this).attr('cx', d.x).attr('cy', d.y);
    updateLinkPositions();
    updateTextPositions();
  }

  function dragended(event, d) {
    d3.select(this).classed('active', false);
  }

  function updateLinkPositions() {
    link
      .attr('x1', function(l) { return l.source.x; })
      .attr('y1', function(l) { return l.source.y; })
      .attr('x2', function(l) { return l.target.x; })
      .attr('y2', function(l) { return l.target.y; });
  }

  function updateTextPositions() {
    text
      .attr('x', function(d) { return d.x; })
      .attr('y', function(d) { return d.y - 15; });
  }

  // Initialize node positions
  node.attr('cx', function(d) { return d.x; })
      .attr('cy', function(d) { return d.y; });

  // Initialize text positions
  text.attr('x', function(d) { return d.x; })
      .attr('y', function(d) { return d.y - 15; });

  // Initialize link positions
  updateLinkPositions();

  // Enable drag functionality
  node.call(d3.drag()
    .on('start', dragstarted)
    .on('drag', dragged)
    .on('end', dragended));
});
</script>

<svg id="graph" width="100%" height=400></svg>