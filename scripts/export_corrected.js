#!/usr/bin/env node
/*
Generate three corrected images at a given strength for Protanopia, Deuteranopia, Tritanopia.
Outputs: out/protan_corrected.png, out/deutan_corrected.png, out/tritan_corrected.png

Usage:
  node scripts/export_corrected.js --input src/assets/colorful-flower-1.jpg --outDir out --strength 1.0
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
const outDir = arg('--outDir', 'out');
const strength = parseFloat(arg('--strength', '1.0'));

// Matrices (row-major for CPU multiply), consistent with export_matrix.js
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

// CVD simulation matrices (single-matrix approximation)
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

// Correction matrices (heuristic redistribution)
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

function correct(buffer, width, height, type /* protan|deutan|tritan */) {
  const out = Buffer.alloc(buffer.length);
  const Msim = SIM[type];
  const Mcorr = CORR[type];

  for (let i = 0; i < width*height; i++) {
    const idx = i*3;
    const R = buffer[idx];
    const G = buffer[idx+1];
    const B = buffer[idx+2];

    // sRGB -> linear
    const r = srgbToLinear(R);
    const g = srgbToLinear(G);
    const b = srgbToLinear(B);

    // RGB -> LMS
    const lms = mul3x3(HPE_RGB_TO_LMS, [r,g,b]);

    // Simulate CVD
    const lmsSim = mul3x3(Msim, lms);

    // Error and correction with strength
    const err = [ lms[0]-lmsSim[0], lms[1]-lmsSim[1], lms[2]-lmsSim[2] ];
    const comp = mul3x3(Mcorr, err);
    const lmsCorr = [ lms[0] + strength*comp[0], lms[1] + strength*comp[1], lms[2] + strength*comp[2] ];

    // Back to RGB
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

  const types = ['protan','deutan','tritan'];
  fs.mkdirSync(outDir, { recursive: true });
  for (const t of types) {
    const corrected = correct(data, width, height, t);
    const outPath = path.join(outDir, `${t}_corrected.png`);
    await sharp(corrected.data, { raw: { width, height, channels: 3 } })
      .png({ compressionLevel: 9 })
      .toFile(outPath);
    console.log(`Exported: ${outPath} (${width}x${height}) strength=${strength}`);
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
