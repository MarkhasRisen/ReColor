#!/usr/bin/env node
/*
Exports a thesis-grade Daltonization Algorithm Flow Diagram:
LMS -> Simulation -> Error -> Correction

Generates an SVG (vector) and a PNG (raster). Supports letter sizing or exact WxH.

Usage examples:
  node scripts/export_flow_diagram.js \
    --outputSvg out/flow_diagram.svg \
    --outputPng out/flow_diagram.png \
    --title "Daltonization Algorithm Flow" \
    --outer 64 --boxW 520 --boxH 200 --gapX 96 --titleHeight 140

  node scripts/export_flow_diagram.js --letter --orientation landscape --dpi 300
  node scripts/export_flow_diagram.js --size 3300x2550
*/

const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

function arg(k, def) { const i = process.argv.indexOf(k); if (i>-1 && i+1<process.argv.length) return process.argv[i+1]; return def; }

const outputSvg = arg('--outputSvg', 'out/flow_diagram.svg');
const outputPng = arg('--outputPng', 'out/flow_diagram.png');
const title = arg('--title', 'Daltonization Algorithm Flow');
const outer = parseInt(arg('--outer', '64'), 10);
const boxW = parseInt(arg('--boxW', '500'), 10);
const boxH = parseInt(arg('--boxH', '220'), 10);
const gapX = parseInt(arg('--gapX', '88'), 10);
const titleHeight = parseInt(arg('--titleHeight', '140'), 10);

const sizeArg = arg('--size', '');
const letterFlag = process.argv.includes('--letter');
const orientation = arg('--orientation', 'landscape');
const dpi = parseInt(arg('--dpi', '300'), 10);

const palette = {
  bg: '#ffffff',
  text: '#111111',
  boxFill: '#f3f6ff',
  boxStroke: '#276ef1',
  arrow: '#111111',
  annotation: '#333333'
};

function svgEscape(s){
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function makeSVG() {
  const boxes = [
    {
      key: 'LIN',
      label: 'Linearization',
      lines: [
        'Input: sRGB (Gamma-Corrected)',
        'Output: RGB_Linear',
        'RGB_Linear = linearize(sRGB)'
      ]
    },
    {
      key: 'LMS',
      label: 'LMS Transform',
      lines: [
        'LMS_Orig = M_RGB→LMS · RGB_Linear'
      ]
    },
    {
      key: 'SIM',
      label: 'Simulation',
      lines: [
        'LMS_Sim = M_Sim · LMS_Orig',
        '(Viénot/Brettel-style)'
      ]
    },
    {
      key: 'ERR',
      label: 'Error',
      lines: [
        'E = LMS_Orig − LMS_Sim'
      ]
    },
    {
      key: 'COR',
      label: 'Correction',
      lines: [
        'LMS_Corr = LMS_Orig + k · (M_Corr · E)'
      ]
    },
  ];

  const cols = boxes.length;
  const contentW = cols*boxW + (cols-1)*gapX;
  const contentH = boxH;
  const W = outer*2 + contentW;
  const H = outer + titleHeight + contentH + outer;

  // Positions
  const yTop = outer + titleHeight; // top-left y of boxes row
  const positions = boxes.map((b, i) => ({
    x: outer + i*(boxW + gapX),
    y: yTop
  }));

  // Title font size scales with titleHeight
  const titleSize = Math.max(36, Math.floor(titleHeight * 0.45));
  const titleY = outer + Math.floor(titleHeight * 0.65);

  // Box label / note sizes
  const boxTitleSize = 28;
  const boxNoteSize = 16;
  const boxLineHeight = 22;

  // Arrow marker id
  const arrowId = 'arrowHead';

  const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="${arrowId}" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="10" markerHeight="10" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="${palette.arrow}" />
    </marker>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#000000" flood-opacity="0.15" />
    </filter>
  </defs>
  <rect x="0" y="0" width="${W}" height="${H}" fill="${palette.bg}" />

  <!-- Title -->
  <text x="${Math.floor(W/2)}" y="${titleY}" text-anchor="middle" font-family="sans-serif" font-size="${titleSize}" font-weight="700" fill="${palette.text}">${svgEscape(title)}</text>

  <!-- Boxes -->
  ${positions.map((p,i)=>{
    const b = boxes[i];
    const startY = p.y + 96;
    const note = Array.isArray(b.lines) ? b.lines : (b.note ? [b.note] : []);
    const tspans = note.map((ln, idx)=>`<tspan x="${p.x + Math.floor(boxW/2)}" dy="${idx===0?0:boxLineHeight}">${svgEscape(ln)}</tspan>`).join('');
    return `
    <g filter="url(#shadow)">
      <rect x="${p.x}" y="${p.y}" rx="16" ry="16" width="${boxW}" height="${boxH}" fill="${palette.boxFill}" stroke="${palette.boxStroke}" stroke-width="3" />
    </g>
    <text x="${p.x + Math.floor(boxW/2)}" y="${p.y + 56}" text-anchor="middle" font-family="sans-serif" font-size="${boxTitleSize}" font-weight="700" fill="${palette.text}">${svgEscape(b.label)}</text>
    <text x="${p.x + Math.floor(boxW/2)}" y="${startY}" text-anchor="middle" font-family="sans-serif" font-size="${boxNoteSize}" fill="${palette.annotation}">${tspans}</text>
    `;
  }).join('\n')}

  <!-- Arrows -->
  ${positions.slice(0,-1).map((p,i)=>{
    const fromX = p.x + boxW;
    const fromY = p.y + Math.floor(boxH/2);
    const toX = positions[i+1].x;
    const toY = positions[i+1].y + Math.floor(boxH/2);
    const midX = (fromX + toX)/2;
    return `
      <path d="M ${fromX} ${fromY} C ${midX} ${fromY}, ${midX} ${toY}, ${toX} ${toY}" fill="none" stroke="${palette.arrow}" stroke-width="4" marker-end="url(#${arrowId})" />
    `;
  }).join('\n')}
</svg>`;

  return { svg, W, H };
}

async function main(){
  const { svg, W, H } = makeSVG();

  const outDirSvg = path.dirname(outputSvg);
  fs.mkdirSync(outDirSvg, { recursive: true });
  fs.writeFileSync(outputSvg, svg, 'utf8');

  // Rasterize via sharp
  let image = sharp(Buffer.from(svg));
  let targetW = W, targetH = H;
  if (sizeArg) {
    const m = sizeArg.match(/^(\d+)x(\d+)$/); if (m) { targetW=parseInt(m[1],10); targetH=parseInt(m[2],10); }
  } else if (letterFlag) {
    const inches = orientation === 'portrait' ? { w: 8.5, h: 11 } : { w: 11, h: 8.5 };
    targetW = Math.round(inches.w * dpi); targetH = Math.round(inches.h * dpi);
  }
  image = image.resize(targetW, targetH);

  const outDirPng = path.dirname(outputPng);
  fs.mkdirSync(outDirPng, { recursive: true });
  await image.png({ compressionLevel: 9 }).toFile(outputPng);

  console.log(`Exported: ${outputSvg} (${W}x${H}) and ${outputPng} (${targetW}x${targetH})`);
}

main().catch(err => { console.error(err); process.exit(1); });
