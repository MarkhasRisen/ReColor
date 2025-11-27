#!/usr/bin/env node
/*
Generates a 1x3 horizontal strip PNG showing corrected outputs (Protan/Deutan/Tritan)
at a specified strength (default 1.0). Uses Sharp for image IO and CPU math
consistent with export_corrected.js.

Usage:
  node scripts/export_corrected_strip.js --input src/assets/colorful-flower-1.jpg --output out/corrected_strip.png --strength 1.0
*/

const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

function arg(k, def) {
  const i = process.argv.indexOf(k);
  if (i > -1 && i + 1 < process.argv.length) return process.argv[i + 1];
  return def;
}

const inputPath = arg('--input', 'src/assets/colorful-flower-1.jpg');
const outputPath = arg('--output', 'out/corrected_strip.png');
const sizeArg = arg('--size', ''); // e.g., 3300x2550
const letterFlag = process.argv.includes('--letter');
const orientation = arg('--orientation', 'landscape');
const dpi = parseInt(arg('--dpi', '300'), 10);
const strength = parseFloat(arg('--strength', '1.0'));
const gap = 16; // px between columns

// Matrices (row-major for CPU multiply)
const HPE_RGB_TO_LMS = [
  0.31399022, 0.63951294, 0.04649755,
  0.15537241, 0.75789446, 0.08670142,
  0.01775239, 0.10944209, 0.87256922,
];
const HPE_LMS_TO_RGB = [
   5.47221206, -4.6419601 ,  0.16963708,
  -1.1252419 ,  2.29317094, -0.1678952 ,
   0.02980165, -0.19318073,  1.16364789,
];

const SIM = {
  protan: [
    0.0, 1.05118294, -0.05116099,
    0.0,  1.0,        0.0,
    0.0,  0.0,        1.0,
  ],
  deutan: [
    1.0, 0.0,        0.0,
    0.9513092, 0.0,  0.04866992,
    0.0, 0.0,        1.0,
  ],
  tritan: [
    1.0, 0.0,        0.0,
    0.0,  1.0,       0.0,
    -0.86744736, 1.86727089, 0.0,
  ],
};

const CORR = {
  protan: [
    0.0, 0.0, 0.0,
    0.7, 1.0, 0.0,
    0.7, 0.0, 1.0,
  ],
  deutan: [
    1.0, 0.7, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.7, 1.0,
  ],
  tritan: [
    1.0, 0.0, 0.7,
    0.0, 1.0, 0.7,
    0.0, 0.0, 0.0,
  ],
};

function srgbToLinear(u) {
  const c = u / 255;
  if (c <= 0.04045) return c / 12.92;
  return Math.pow((c + 0.055) / 1.055, 2.4);
}
function linearToSrgb(u) {
  const c = Math.max(0, Math.min(1, u));
  if (c <= 0.0031308) return Math.round(255 * 12.92 * c);
  return Math.round(255 * (1.055 * Math.pow(c, 1/2.4) - 0.055));
}
function mul3x3(M, v) {
  return [
    M[0]*v[0] + M[1]*v[1] + M[2]*v[2],
    M[3]*v[0] + M[4]*v[1] + M[5]*v[2],
    M[6]*v[0] + M[7]*v[1] + M[8]*v[2],
  ];
}

function correct(buffer, width, height, type) {
  const out = Buffer.alloc(buffer.length);
  const Msim = SIM[type];
  const Mcorr = CORR[type];

  for (let i = 0; i < width*height; i++) {
    const idx = i*3;
    const R = buffer[idx];
    const G = buffer[idx+1];
    const B = buffer[idx+2];

    const r = srgbToLinear(R);
    const g = srgbToLinear(G);
    const b = srgbToLinear(B);

    const lms = mul3x3(HPE_RGB_TO_LMS, [r,g,b]);
    const lmsSim = mul3x3(Msim, lms);
    const err = [ lms[0]-lmsSim[0], lms[1]-lmsSim[1], lms[2]-lmsSim[2] ];
    const comp = mul3x3(Mcorr, err);
    const lmsCorr = [ lms[0] + strength*comp[0], lms[1] + strength*comp[1], lms[2] + strength*comp[2] ];
    const rgbCorr = mul3x3(HPE_LMS_TO_RGB, lmsCorr);

    out[idx]   = linearToSrgb(rgbCorr[0]);
    out[idx+1] = linearToSrgb(rgbCorr[1]);
    out[idx+2] = linearToSrgb(rgbCorr[2]);
  }
  return { data: out, info: { width, height, channels: 3 } };
}

async function main() {
  if (!fs.existsSync(inputPath)) {
    console.error(`Input image not found: ${inputPath}`);
    process.exit(1);
  }
  const { data, info } = await sharp(inputPath).removeAlpha().toColorspace('srgb').raw().toBuffer({ resolveWithObject: true });
  const width = info.width;
  const height = info.height;

  const prot = correct(data, width, height, 'protan');
  const deut = correct(data, width, height, 'deutan');
  const trit = correct(data, width, height, 'tritan');

  const panelW = width;
  const panelH = height;
  const outW = 3*panelW + 2*gap;
  const outH = panelH;

  function fromRaw(p) { return sharp(p.data, { raw: { width: panelW, height: panelH, channels: 3 } }); }
  let canvas = sharp({ create: { width: outW, height: outH, channels: 3, background: { r:255, g:255, b:255 }}});

  const panels = [prot, deut, trit];
  const labels = [
    'Protanopia (Corrected, Strength 1.0)',
    'Deuteranopia (Corrected, Strength 1.0)',
    'Tritanopia (Corrected, Strength 1.0)'
  ];

  const composites = [];
  for (let i = 0; i < panels.length; i++) {
    const left = i*(panelW + gap);
    const top = 0;

    const panelBuf = await fromRaw(panels[i]).jpeg({ quality: 95 }).toBuffer();
    composites.push({ input: panelBuf, left, top });

    const label = labels[i];
    const bandH = 56;
    const fontSize = 28;
    const strokeW = 3;
    const textY = panelH - Math.floor(bandH/2) + Math.floor(fontSize/2) - 2;
    const svg = Buffer.from(`<?xml version="1.0"?><svg width="${panelW}" height="${panelH}" xmlns="http://www.w3.org/2000/svg">
      <rect x=\"0\" y=\"${panelH-bandH}\" width=\"${panelW}\" height=\"${bandH}\" fill=\"rgba(0,0,0,0.35)\"/>
      <text x=\"${Math.floor(panelW/2)}\" y=\"${textY}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"${fontSize}\" font-weight=\"700\" fill=\"#FFFFFF\" stroke=\"#000000\" stroke-width=\"${strokeW}\" paint-order=\"stroke fill\">${label}</text>
    </svg>`);
    composites.push({ input: svg, left, top });
  }

  canvas = canvas.composite(composites);

  const outDir = path.dirname(outputPath);
  fs.mkdirSync(outDir, { recursive: true });
  const base = await canvas.png({ compressionLevel: 9 }).toBuffer();

  let target = base;
  let targetW = outW;
  let targetH = outH;
  if (sizeArg) {
    const m = sizeArg.match(/^(\d+)x(\d+)$/);
    if (m) { targetW = parseInt(m[1],10); targetH = parseInt(m[2],10); }
  } else if (letterFlag) {
    const inches = orientation === 'portrait' ? { w: 8.5, h: 11 } : { w: 11, h: 8.5 };
    targetW = Math.round(inches.w * dpi);
    targetH = Math.round(inches.h * dpi);
  }
  if (targetW !== outW || targetH !== outH) {
    target = await sharp(base).resize(targetW, targetH).png({ compressionLevel: 9 }).toBuffer();
  }
  await sharp(target).toFile(outputPath);
  console.log(`Exported: ${outputPath} (${targetW}x${targetH}) strength=${strength}`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
