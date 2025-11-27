#!/usr/bin/env node
/*
Generates a research-style grid inspired by the provided layout.
For each CVD type (Protan/Deutan/Tritan), creates two rows:
- Row 1: Simulation at multiple severities (s in [0..1]) with Original as first column
- Row 2: Correction at multiple strengths (k in [0..1]) with Original as first column

Defaults:
  simLevels = 0.25,0.5,0.75,1.0
  corrLevels = 0.25,0.5,0.75,1.0

Usage examples:
  node scripts/export_research_grid.js --input src/assets/colorful-flower-1.jpg --output out/research_grid.png
  node scripts/export_research_grid.js --letter --orientation portrait --dpi 300
  node scripts/export_research_grid.js --sim 0.2,0.4,0.6,1.0 --corr 0.25,0.5,0.75,1.0
*/

const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

function arg(k, def) { const i = process.argv.indexOf(k); if (i>-1 && i+1<process.argv.length) return process.argv[i+1]; return def; }
function parseList(s, defArr) { if (!s) return defArr; return s.split(',').map(x => parseFloat(x)).filter(x => !isNaN(x)); }

const inputPath = arg('--input', 'src/assets/colorful-flower-1.jpg');
const outputPath = arg('--output', 'out/research_grid.png');
const simLevels = parseList(arg('--sim', ''), [0.25,0.5,0.75,1.0]);
const corrLevels = parseList(arg('--corr', ''), [0.25,0.5,0.75,1.0]);
const gap = 16; // between columns
const rowGap = 40; // between rows inside type group
const groupGap = 56; // between different CVD type groups
const headerTop = 64; // document title top area
const typeHeaderH = 40; // per-type header height

const sizeArg = arg('--size', '');
const letterFlag = process.argv.includes('--letter');
const orientation = arg('--orientation', 'portrait');
const dpi = parseInt(arg('--dpi', '300'), 10);

// Matrices (row-major for CPU)
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
  protan: [ 0.0, 1.05118294, -0.05116099,  0.0,1.0,0.0,  0.0,0.0,1.0 ],
  deutan: [ 1.0,0.0,0.0,  0.9513092,0.0,0.04866992,  0.0,0.0,1.0 ],
  tritan: [ 1.0,0.0,0.0,  0.0,1.0,0.0,  -0.86744736,1.86727089,0.0 ],
};
const CORR = {
  protan: [ 0.0,0.0,0.0,  0.7,1.0,0.0,  0.7,0.0,1.0 ],
  deutan: [ 1.0,0.7,0.0,  0.0,0.0,0.0,  0.0,0.7,1.0 ],
  tritan: [ 1.0,0.0,0.7,  0.0,1.0,0.7,  0.0,0.0,0.0 ],
};

function srgbToLinear(u){ const c=u/255; return c<=0.04045? c/12.92: Math.pow((c+0.055)/1.055,2.4); }
function linearToSrgb(u){ const c=Math.max(0,Math.min(1,u)); return c<=0.0031308? Math.round(255*12.92*c): Math.round(255*(1.055*Math.pow(c,1/2.4)-0.055)); }
function mul3x3(M,v){ return [ M[0]*v[0]+M[1]*v[1]+M[2]*v[2], M[3]*v[0]+M[4]*v[1]+M[5]*v[2], M[6]*v[0]+M[7]*v[1]+M[8]*v[2] ]; }

function simulate(buffer,w,h,type,severity){
  const out=Buffer.alloc(buffer.length); const Ms=SIM[type];
  for(let i=0;i<w*h;i++){
    const idx=i*3; const R=buffer[idx], G=buffer[idx+1], B=buffer[idx+2];
    const r=srgbToLinear(R), g=srgbToLinear(G), b=srgbToLinear(B);
    const lms=mul3x3(HPE_RGB_TO_LMS,[r,g,b]);
    const lmsFull=mul3x3(Ms,lms);
    const lmsSev=[ lms[0]*(1-severity)+lmsFull[0]*severity, lms[1]*(1-severity)+lmsFull[1]*severity, lms[2]*(1-severity)+lmsFull[2]*severity ];
    const rgb=mul3x3(HPE_LMS_TO_RGB,lmsSev);
    out[idx]=linearToSrgb(rgb[0]); out[idx+1]=linearToSrgb(rgb[1]); out[idx+2]=linearToSrgb(rgb[2]);
  }
  return { data: out, info: { width:w, height:h, channels:3 } };
}

function correct(buffer,w,h,type,strength){
  const out=Buffer.alloc(buffer.length); const Ms=SIM[type]; const Mc=CORR[type];
  for(let i=0;i<w*h;i++){
    const idx=i*3; const R=buffer[idx], G=buffer[idx+1], B=buffer[idx+2];
    const r=srgbToLinear(R), g=srgbToLinear(G), b=srgbToLinear(B);
    const lms=mul3x3(HPE_RGB_TO_LMS,[r,g,b]);
    const lmsSim=mul3x3(Ms,lms);
    const err=[ lms[0]-lmsSim[0], lms[1]-lmsSim[1], lms[2]-lmsSim[2] ];
    const comp=mul3x3(Mc,err);
    const lmsCorr=[ lms[0]+strength*comp[0], lms[1]+strength*comp[1], lms[2]+strength*comp[2] ];
    const rgb=mul3x3(HPE_LMS_TO_RGB,lmsCorr);
    out[idx]=linearToSrgb(rgb[0]); out[idx+1]=linearToSrgb(rgb[1]); out[idx+2]=linearToSrgb(rgb[2]);
  }
  return { data: out, info: { width:w, height:h, channels:3 } };
}

async function main(){
  if(!fs.existsSync(inputPath)){ console.error(`Input image not found: ${inputPath}`); process.exit(1); }
  const { data, info } = await sharp(inputPath).removeAlpha().toColorspace('srgb').raw().toBuffer({ resolveWithObject: true });
  const W=info.width, H=info.height;

  const types=['protan','deutan','tritan'];
  const cols = 1 + Math.max(simLevels.length, corrLevels.length);
  const panelW=W, panelH=H;
  const groupH = typeHeaderH + panelH*2 + rowGap; // header + two rows + inner gap
  const outW = cols*panelW + (cols-1)*gap;
  const outH = headerTop + (types.length*groupH) + (types.length-1)*groupGap;

  let canvas = sharp({ create: { width: outW, height: outH, channels: 3, background: { r:255,g:255,b:255 } } });
  const composites=[];

  // Document title
  const docTitle = Buffer.from(`<?xml version="1.0"?><svg width="${outW}" height="${headerTop}" xmlns="http://www.w3.org/2000/svg">
    <text x="${Math.floor(outW/2)}" y="48" text-anchor="middle" font-family="sans-serif" font-size="32" fill="#111" font-weight="700">CVD Simulation and Daltonization Grid</text>
  </svg>`);
  composites.push({ input: docTitle, left:0, top:0 });

  for(let tIndex=0; tIndex<types.length; tIndex++){
    const type=types[tIndex];
    const y0 = headerTop + tIndex*(groupH + groupGap);

    // Type header with sublabels
    const typeName = type.charAt(0).toUpperCase()+type.slice(1);
    const hdr = Buffer.from(`<?xml version="1.0"?><svg width="${outW}" height="${typeHeaderH}" xmlns="http://www.w3.org/2000/svg">
      <text x="8" y="28" font-family="sans-serif" font-size="26" fill="#111" font-weight="700">${typeName}</text>
      <text x="8" y="${typeHeaderH-4}" font-family="sans-serif" font-size="16" fill="#333">Row 1: Simulation (s), Row 2: Correction (strength)</text>
    </svg>`);
    composites.push({ input: hdr, left:0, top: y0 });

    // Row 1: Original + simulated severities
    const row1Top = y0 + typeHeaderH;
    // Original
    const origBuf = await sharp(data, { raw: { width: W, height: H, channels: 3 } }).jpeg({ quality: 95 }).toBuffer();
    composites.push({ input: origBuf, left: 0, top: row1Top });
    // Label
    let svg = Buffer.from(`<?xml version="1.0"?><svg width="${panelW}" height="${panelH}" xmlns="http://www.w3.org/2000/svg">
      <rect x="0" y="${panelH-56}" width="${panelW}" height="56" fill="rgba(0,0,0,0.35)"/>
      <text x="${Math.floor(panelW/2)}" y="${panelH - Math.floor(56/2) + Math.floor(28/2) - 2}" text-anchor="middle" font-family="sans-serif" font-size="28" font-weight="700" fill="#FFFFFF" stroke="#000" stroke-width="3" paint-order="stroke fill">Original</text>
    </svg>`);
    composites.push({ input: svg, left: 0, top: row1Top });

    for(let i=0;i<simLevels.length;i++){
      const sev=simLevels[i];
      const sim = simulate(data, W, H, type, sev);
      const buf = await sharp(sim.data, { raw: { width: W, height: H, channels: 3 } }).jpeg({ quality: 95 }).toBuffer();
      const left = (i+1)*(panelW + gap);
      composites.push({ input: buf, left, top: row1Top });
      const txt = `${typeName} (Simulated s=${sev})`;
      const lab = Buffer.from(`<?xml version="1.0"?><svg width="${panelW}" height="${panelH}" xmlns="http://www.w3.org/2000/svg">
        <rect x="0" y="${panelH-56}" width="${panelW}" height="56" fill="rgba(0,0,0,0.35)"/>
        <text x="${Math.floor(panelW/2)}" y="${panelH - Math.floor(56/2) + Math.floor(28/2) - 2}" text-anchor="middle" font-family="sans-serif" font-size="28" font-weight="700" fill="#FFFFFF" stroke="#000" stroke-width="3" paint-order="stroke fill">${txt}</text>
      </svg>`);
      composites.push({ input: lab, left, top: row1Top });
    }

    // Row 2: Original + corrected with strengths
    const row2Top = row1Top + panelH + rowGap;
    composites.push({ input: origBuf, left: 0, top: row2Top });
    svg = Buffer.from(`<?xml version="1.0"?><svg width="${panelW}" height="${panelH}" xmlns="http://www.w3.org/2000/svg">
      <rect x="0" y="${panelH-56}" width="${panelW}" height="56" fill="rgba(0,0,0,0.35)"/>
      <text x="${Math.floor(panelW/2)}" y="${panelH - Math.floor(56/2) + Math.floor(28/2) - 2}" text-anchor="middle" font-family="sans-serif" font-size="28" font-weight="700" fill="#FFFFFF" stroke="#000" stroke-width="3" paint-order="stroke fill">Original</text>
    </svg>`);
    composites.push({ input: svg, left: 0, top: row2Top });

    for(let i=0;i<corrLevels.length;i++){
      const k=corrLevels[i];
      const cor = correct(data, W, H, type, k);
      const buf = await sharp(cor.data, { raw: { width: W, height: H, channels: 3 } }).jpeg({ quality: 95 }).toBuffer();
      const left = (i+1)*(panelW + gap);
      composites.push({ input: buf, left, top: row2Top });
      const txt = `${typeName} (Corrected k=${k})`;
      const lab = Buffer.from(`<?xml version="1.0"?><svg width="${panelW}" height="${panelH}" xmlns="http://www.w3.org/2000/svg">
        <rect x="0" y="${panelH-56}" width="${panelW}" height="56" fill="rgba(0,0,0,0.35)"/>
        <text x="${Math.floor(panelW/2)}" y="${panelH - Math.floor(56/2) + Math.floor(28/2) - 2}" text-anchor="middle" font-family="sans-serif" font-size="28" font-weight="700" fill="#FFFFFF" stroke="#000" stroke-width="3" paint-order="stroke fill">${txt}</text>
      </svg>`);
      composites.push({ input: lab, left, top: row2Top });
    }
  }

  canvas = canvas.composite(composites);

  // Output and optional resize
  const outDir = path.dirname(outputPath);
  fs.mkdirSync(outDir, { recursive: true });
  const base = await canvas.png({ compressionLevel: 9 }).toBuffer();

  let target = base; let targetW = outW; let targetH = outH;
  if (sizeArg) {
    const m = sizeArg.match(/^(\d+)x(\d+)$/); if (m) { targetW=parseInt(m[1],10); targetH=parseInt(m[2],10); }
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
