#!/usr/bin/env node
/*
Combines two 1x3 row images (simulated and corrected) into a single 2x3 matrix
with outer margins and a title header. Defaults target files created by other scripts.

Usage:
  node scripts/combine_rows.js \
    --sim out/row_simulated.png \
    --corr out/row_corrected.png \
    --output out/matrix_combined.png \
    --title "CVD Simulation (Top) and Daltonization (Bottom)" \
    [--outer 48] [--between 64] [--titleHeight 100] [--bg #ffffff]

Optional sizing (like other exporters):
  --size WxH or --letter --orientation portrait|landscape --dpi 300
*/

const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

function arg(k, def) {
  const i = process.argv.indexOf(k);
  if (i > -1 && i + 1 < process.argv.length) return process.argv[i + 1];
  return def;
}

const simPath = arg('--sim', 'out/row_simulated.png');
const corrPath = arg('--corr', 'out/row_corrected.png');
const outputPath = arg('--output', 'out/matrix_combined.png');
const title = arg('--title', 'CVD Simulation (Top) and Daltonization (Bottom)');
const outer = parseInt(arg('--outer', '48'), 10);
const between = parseInt(arg('--between', '64'), 10);
const titleHeight = parseInt(arg('--titleHeight', '100'), 10);
const bg = arg('--bg', '#ffffff');

const sizeArg = arg('--size', '');
const letterFlag = process.argv.includes('--letter');
const orientation = arg('--orientation', 'portrait');
const dpi = parseInt(arg('--dpi', '300'), 10);

async function main() {
  if (!fs.existsSync(simPath)) { console.error(`Simulated row not found: ${simPath}`); process.exit(1); }
  if (!fs.existsSync(corrPath)) { console.error(`Corrected row not found: ${corrPath}`); process.exit(1); }

  const simImg = sharp(simPath).removeAlpha().toColorspace('srgb');
  const corrImg = sharp(corrPath).removeAlpha().toColorspace('srgb');
  const simMeta = await simImg.metadata();
  const corrMeta = await corrImg.metadata();

  // Ensure same width; if heights differ, we will center vertically (unlikely if source rows match)
  const W = Math.max(simMeta.width || 0, corrMeta.width || 0);
  const H1 = simMeta.height || 0;
  const H2 = corrMeta.height || 0;

  const padLeft = outer, padRight = outer;
  const padTop = outer + titleHeight;
  const padBottom = outer;

  const outW = padLeft + W + padRight;
  const outH = padTop + H1 + between + H2 + padBottom;

  // Base white canvas
  let canvas = sharp({ create: { width: outW, height: outH, channels: 3, background: bg } });

  // Prepare row buffers (extend/crop to desired width if necessary)
  const simBuf = await sharp(simPath).resize({ width: W }).jpeg({ quality: 95 }).toBuffer();
  const corrBuf = await sharp(corrPath).resize({ width: W }).jpeg({ quality: 95 }).toBuffer();

  const composites = [];

  // Title header
  const titleSVG = Buffer.from(`<?xml version="1.0"?><svg width="${outW}" height="${titleHeight}" xmlns="http://www.w3.org/2000/svg">
    <text x="${Math.floor(outW/2)}" y="${Math.max(32, Math.floor(titleHeight*0.6))}" text-anchor="middle" font-family="sans-serif" font-size="${Math.max(28, Math.floor(titleHeight*0.36))}" fill="#111" font-weight="700">${title}</text>
  </svg>`);
  composites.push({ input: titleSVG, left: 0, top: outer });

  // Place rows
  composites.push({ input: simBuf, left: padLeft, top: padTop });
  composites.push({ input: corrBuf, left: padLeft, top: padTop + H1 + between });

  canvas = canvas.composite(composites);

  const outDir = path.dirname(outputPath);
  fs.mkdirSync(outDir, { recursive: true });
  const base = await canvas.png({ compressionLevel: 9 }).toBuffer();

  let target = base; let targetW = outW; let targetH = outH;
  if (sizeArg) {
    const m = sizeArg.match(/^(\d+)x(\d+)$/); if (m) { targetW = parseInt(m[1],10); targetH = parseInt(m[2],10); }
  } else if (letterFlag) {
    const inches = orientation === 'portrait' ? { w: 8.5, h: 11 } : { w: 11, h: 8.5 };
    targetW = Math.round(inches.w * dpi); targetH = Math.round(inches.h * dpi);
  }
  if (targetW !== outW || targetH !== outH) {
    target = await sharp(base).resize(targetW, targetH).png({ compressionLevel: 9 }).toBuffer();
  }
  await sharp(target).toFile(outputPath);
  console.log(`Exported: ${outputPath} (${targetW}x${targetH})`);
}

main().catch(err => { console.error(err); process.exit(1); });
