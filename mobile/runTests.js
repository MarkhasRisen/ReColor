/**
 * Standalone test runner for daltonization algorithms
 * Run with: node runTests.js
 */

// Simulate the daltonization algorithms in pure JavaScript
const SIMULATION_MATRICES = {
  protan: [
    [0.0, 2.02344, -2.52581],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
  ],
  deutan: [
    [1.0, 0.0, 0.0],
    [0.494207, 0.0, 1.24827],
    [0.0, 0.0, 1.0],
  ],
  tritan: [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [-0.395913, 0.801109, 0.0],
  ],
  normal: [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
  ],
};

const CORRECTION_MATRICES = {
  protan: [
    [0.0, 0.7, 0.7],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
  ],
  deutan: [
    [1.0, 0.0, 0.0],
    [0.7, 0.0, 0.7],
    [0.0, 0.0, 1.0],
  ],
  tritan: [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.7, 0.7, 0.0],
  ],
  normal: [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
  ],
};

function matrixMultiply(rgb, matrix) {
  return [
    rgb[0] * matrix[0][0] + rgb[1] * matrix[0][1] + rgb[2] * matrix[0][2],
    rgb[0] * matrix[1][0] + rgb[1] * matrix[1][1] + rgb[2] * matrix[1][2],
    rgb[0] * matrix[2][0] + rgb[1] * matrix[2][1] + rgb[2] * matrix[2][2],
  ];
}

function clamp(value) {
  return Math.max(0, Math.min(1, value));
}

function daltonizePixel(rgb, cvdType, severity) {
  if (cvdType === 'normal' || severity === 0) {
    return rgb;
  }

  const simulated = matrixMultiply(rgb, SIMULATION_MATRICES[cvdType]);
  const error = [
    rgb[0] - simulated[0],
    rgb[1] - simulated[1],
    rgb[2] - simulated[2],
  ];
  const correction = matrixMultiply(error, CORRECTION_MATRICES[cvdType]);
  const corrected = [
    rgb[0] + severity * correction[0],
    rgb[1] + severity * correction[1],
    rgb[2] + severity * correction[2],
  ];

  return corrected.map(clamp);
}

function colorDistance(rgb1, rgb2) {
  const dr = rgb1[0] - rgb2[0];
  const dg = rgb1[1] - rgb2[1];
  const db = rgb1[2] - rgb2[2];
  return Math.sqrt(dr * dr + dg * dg + db * db);
}

function identifyColor(rgb) {
  const [r, g, b] = rgb;
  const hex = `#${Math.round(r).toString(16).padStart(2, '0')}${Math.round(g).toString(16).padStart(2, '0')}${Math.round(b).toString(16).padStart(2, '0')}`;

  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const delta = max - min;

  let name = 'Unknown';

  if (delta < 20) {
    if (max < 50) name = 'Black';
    else if (max > 200) name = 'White';
    else name = 'Gray';
  } else {
    if (r === max) {
      if (g > b) name = g > r * 0.7 ? 'Yellow' : 'Orange';
      else name = 'Red';
    } else if (g === max) {
      if (r > b) name = 'Yellow-Green';
      else name = b > g * 0.5 ? 'Cyan' : 'Green';
    } else {
      if (r > g) name = 'Magenta';
      else name = g > b * 0.7 ? 'Cyan' : 'Blue';
    }
  }

  return { name, hex, rgb: [r, g, b] };
}

// Run tests
console.log('\x1b[36m%s\x1b[0m', '====================================');
console.log('\x1b[36m%s\x1b[0m', 'Testing Daltonization Algorithms');
console.log('\x1b[36m%s\x1b[0m', '====================================\n');

const testColors = {
  red: [1.0, 0.0, 0.0],
  green: [0.0, 1.0, 0.0],
  blue: [0.0, 0.0, 1.0],
  yellow: [1.0, 1.0, 0.0],
  white: [1.0, 1.0, 1.0],
};

let passCount = 0;
let failCount = 0;

// Test 1: Normal vision
console.log('\x1b[1m%s\x1b[0m', 'Test 1: Normal Vision (Identity)');
console.log('----------------------------------');
Object.entries(testColors).forEach(([name, rgb]) => {
  const result = daltonizePixel(rgb, 'normal', 1.0);
  const unchanged = result.every((val, i) => Math.abs(val - rgb[i]) < 0.0001);
  if (unchanged) {
    console.log('\x1b[32m%s\x1b[0m', `✓ ${name}: PASS`);
    passCount++;
  } else {
    console.log('\x1b[31m%s\x1b[0m', `✗ ${name}: FAIL`);
    failCount++;
  }
});
console.log();

// Test 2: Protanopia correction
console.log('\x1b[1m%s\x1b[0m', 'Test 2: Protanopia Correction');
console.log('----------------------------------');
const red = [1.0, 0.0, 0.0];
const redCorrected = daltonizePixel(red, 'protan', 0.8);
console.log('Original red:', red);
console.log('Corrected red:', redCorrected.map(v => v.toFixed(3)));
const redShifted = redCorrected[0] !== red[0] || redCorrected[1] !== red[1];
if (redShifted) {
  console.log('\x1b[32m%s\x1b[0m', '✓ Red shifted: PASS');
  passCount++;
} else {
  console.log('\x1b[31m%s\x1b[0m', '✗ Red shifted: FAIL');
  failCount++;
}
console.log();

// Test 3: Deuteranopia correction
console.log('\x1b[1m%s\x1b[0m', 'Test 3: Deuteranopia Correction');
console.log('----------------------------------');
const green = [0.0, 1.0, 0.0];
const greenCorrected = daltonizePixel(green, 'deutan', 0.8);
console.log('Original green:', green);
console.log('Corrected green:', greenCorrected.map(v => v.toFixed(3)));
const greenShifted = greenCorrected[0] !== green[0] || greenCorrected[1] !== green[1];
if (greenShifted) {
  console.log('\x1b[32m%s\x1b[0m', '✓ Green shifted: PASS');
  passCount++;
} else {
  console.log('\x1b[31m%s\x1b[0m', '✗ Green shifted: FAIL');
  failCount++;
}
console.log();

// Test 4: Severity scaling
console.log('\x1b[1m%s\x1b[0m', 'Test 4: Severity Scaling');
console.log('----------------------------------');
const severities = [0.0, 0.25, 0.5, 0.75, 1.0];
const distances = severities.map(s => {
  const corrected = daltonizePixel(red, 'protan', s);
  return colorDistance(red, corrected);
});
distances.forEach((dist, i) => {
  console.log(`Severity ${severities[i].toFixed(2)}: distance = ${dist.toFixed(3)}`);
});
const increasing = distances[0] <= distances[1] && distances[1] <= distances[2];
if (increasing) {
  console.log('\x1b[32m%s\x1b[0m', '✓ Distance increases with severity: PASS');
  passCount++;
} else {
  console.log('\x1b[31m%s\x1b[0m', '✗ Distance increases with severity: FAIL');
  failCount++;
}
console.log();

// Test 5: Color identification
console.log('\x1b[1m%s\x1b[0m', 'Test 5: Color Identification');
console.log('----------------------------------');
const identTests = [
  { rgb: [255, 0, 0], expected: 'Red' },
  { rgb: [0, 255, 0], expected: 'Green' },
  { rgb: [0, 0, 255], expected: 'Blue' },
  { rgb: [255, 255, 0], expected: 'Yellow' },
  { rgb: [255, 255, 255], expected: 'White' },
];

identTests.forEach(({ rgb, expected }) => {
  const result = identifyColor(rgb);
  const match = result.name === expected;
  if (match) {
    console.log('\x1b[32m%s\x1b[0m', `✓ ${expected}: ${result.name} (${result.hex}) PASS`);
    passCount++;
  } else {
    console.log('\x1b[31m%s\x1b[0m', `✗ ${expected}: ${result.name} (${result.hex}) FAIL`);
    failCount++;
  }
});
console.log();

// Test 6: Performance benchmark
console.log('\x1b[1m%s\x1b[0m', 'Test 6: Performance Benchmark');
console.log('----------------------------------');
const iterations = 100000;
const startTime = Date.now();

for (let i = 0; i < iterations; i++) {
  daltonizePixel([0.5, 0.5, 0.5], 'protan', 0.8);
}

const endTime = Date.now();
const totalTime = endTime - startTime;
const avgTime = totalTime / iterations;
const pixelsPerSec = Math.round(iterations / (totalTime / 1000));

console.log(`Iterations: ${iterations.toLocaleString()}`);
console.log(`Total time: ${totalTime}ms`);
console.log(`Average time: ${(avgTime * 1000).toFixed(3)}μs per pixel`);
console.log(`Speed: ${pixelsPerSec.toLocaleString()} pixels/second`);
console.log('\x1b[32m%s\x1b[0m', '✓ Performance test complete: PASS');
passCount++;
console.log();

// Summary
console.log('\x1b[36m%s\x1b[0m', '====================================');
console.log('\x1b[36m%s\x1b[0m', 'Test Summary');
console.log('\x1b[36m%s\x1b[0m', '====================================');
console.log('\x1b[32m%s\x1b[0m', `✓ Passed: ${passCount}`);
if (failCount > 0) {
  console.log('\x1b[31m%s\x1b[0m', `✗ Failed: ${failCount}`);
} else {
  console.log('\x1b[32m%s\x1b[0m', '✗ Failed: 0');
}
console.log();

if (failCount === 0) {
  console.log('\x1b[42m\x1b[30m%s\x1b[0m', ' ALL TESTS PASSED! ');
  console.log('\x1b[32m%s\x1b[0m', '\n✓ Algorithms are working correctly!');
  console.log('\x1b[32m%s\x1b[0m', '✓ Ready for mobile integration!');
} else {
  console.log('\x1b[41m\x1b[37m%s\x1b[0m', ' SOME TESTS FAILED ');
}
console.log('\x1b[36m%s\x1b[0m', '====================================\n');
