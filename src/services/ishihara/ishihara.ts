/**
 * On-Device Ishihara Test Evaluation
 * No backend required - all plates bundled in assets
 * Uses DaltonLens-generated plates with clinical scoring
 */

export interface IshiharaPlate {
  plateNumber: number;
  plateType: 'control' | 'transformation' | 'vanishing' | 'hidden_digit' | 'classification';
  normalAnswer: string | null;
  protanAnswer: string | null;
  deutanAnswer: string | null;
  description: string;
  imageSource: any;
}

export interface TestResult {
  totalPlates: number;
  correctNormal: number;
  correctProtan: number;
  correctDeutan: number;
  incorrect: number;
  controlFailed: number;
  cvdType: 'normal' | 'protan' | 'deutan' | 'unknown';
  severity: number; // 0.0-1.0
  confidence: number; // 0.0-1.0
  interpretation: string;
  recommendations: string[];
}

// All 38 Ishihara plates with correct answers
export const COMPREHENSIVE_PLATES: IshiharaPlate[] = [
  // Control plates (1-2)
  { plateNumber: 1, plateType: 'control', normalAnswer: '12', protanAnswer: null, deutanAnswer: null, description: 'Control - everyone sees 12', imageSource: require('../../../assets/ishihara/plate_01.png') },
  { plateNumber: 2, plateType: 'control', normalAnswer: '8', protanAnswer: null, deutanAnswer: null, description: 'Control - everyone sees 8', imageSource: require('../../../assets/ishihara/plate_02.png') },
  
  // Transformation plates (3-9)
  { plateNumber: 3, plateType: 'transformation', normalAnswer: '6', protanAnswer: '5', deutanAnswer: '5', description: 'Normal: 6, CVD: 5', imageSource: require('../../../assets/ishihara/plate_03.png') },
  { plateNumber: 4, plateType: 'transformation', normalAnswer: '29', protanAnswer: '70', deutanAnswer: '70', description: 'Normal: 29, CVD: 70', imageSource: require('../../../assets/ishihara/plate_04.png') },
  { plateNumber: 5, plateType: 'transformation', normalAnswer: '57', protanAnswer: '35', deutanAnswer: '35', description: 'Normal: 57, CVD: 35', imageSource: require('../../../assets/ishihara/plate_05.png') },
  { plateNumber: 6, plateType: 'transformation', normalAnswer: '5', protanAnswer: '2', deutanAnswer: '2', description: 'Normal: 5, CVD: 2', imageSource: require('../../../assets/ishihara/plate_06.png') },
  { plateNumber: 7, plateType: 'transformation', normalAnswer: '3', protanAnswer: '5', deutanAnswer: '5', description: 'Normal: 3, CVD: 5', imageSource: require('../../../assets/ishihara/plate_07.png') },
  { plateNumber: 8, plateType: 'transformation', normalAnswer: '15', protanAnswer: '17', deutanAnswer: '17', description: 'Normal: 15, CVD: 17', imageSource: require('../../../assets/ishihara/plate_08.png') },
  { plateNumber: 9, plateType: 'transformation', normalAnswer: '74', protanAnswer: '21', deutanAnswer: '21', description: 'Normal: 74, CVD: 21', imageSource: require('../../../assets/ishihara/plate_09.png') },
  
  // Vanishing plates (10-15)
  { plateNumber: 10, plateType: 'vanishing', normalAnswer: '2', protanAnswer: null, deutanAnswer: null, description: 'Normal: 2, CVD: nothing', imageSource: require('../../../assets/ishihara/plate_10.png') },
  { plateNumber: 11, plateType: 'vanishing', normalAnswer: '6', protanAnswer: null, deutanAnswer: null, description: 'Normal: 6, CVD: nothing', imageSource: require('../../../assets/ishihara/plate_11.png') },
  { plateNumber: 12, plateType: 'vanishing', normalAnswer: '97', protanAnswer: null, deutanAnswer: null, description: 'Normal: 97, CVD: nothing', imageSource: require('../../../assets/ishihara/plate_12.png') },
  { plateNumber: 13, plateType: 'vanishing', normalAnswer: '45', protanAnswer: null, deutanAnswer: null, description: 'Normal: 45, CVD: nothing', imageSource: require('../../../assets/ishihara/plate_13.png') },
  { plateNumber: 14, plateType: 'vanishing', normalAnswer: '5', protanAnswer: null, deutanAnswer: null, description: 'Normal: 5, CVD: nothing', imageSource: require('../../../assets/ishihara/plate_14.png') },
  { plateNumber: 15, plateType: 'vanishing', normalAnswer: '7', protanAnswer: null, deutanAnswer: null, description: 'Normal: 7, CVD: nothing', imageSource: require('../../../assets/ishihara/plate_15.png') },
  
  // Hidden digit plates (16-17)
  { plateNumber: 16, plateType: 'hidden_digit', normalAnswer: null, protanAnswer: '45', deutanAnswer: '45', description: 'Normal: nothing, CVD: 45', imageSource: require('../../../assets/ishihara/plate_16.png') },
  { plateNumber: 17, plateType: 'hidden_digit', normalAnswer: null, protanAnswer: '5', deutanAnswer: '5', description: 'Normal: nothing, CVD: 5', imageSource: require('../../../assets/ishihara/plate_17.png') },
  
  // Classification plates (18-21)
  { plateNumber: 18, plateType: 'classification', normalAnswer: '26', protanAnswer: '6', deutanAnswer: '2', description: 'Protan/Deutan classification', imageSource: require('../../../assets/ishihara/plate_18.png') },
  { plateNumber: 19, plateType: 'classification', normalAnswer: '42', protanAnswer: '2', deutanAnswer: '4', description: 'Protan/Deutan classification', imageSource: require('../../../assets/ishihara/plate_19.png') },
  { plateNumber: 20, plateType: 'classification', normalAnswer: '35', protanAnswer: '5', deutanAnswer: '3', description: 'Protan/Deutan classification', imageSource: require('../../../assets/ishihara/plate_20.png') },
  { plateNumber: 21, plateType: 'classification', normalAnswer: '96', protanAnswer: '6', deutanAnswer: '9', description: 'Protan/Deutan classification', imageSource: require('../../../assets/ishihara/plate_21.png') },
  
  // Additional vanishing plates (22-25)
  { plateNumber: 22, plateType: 'vanishing', normalAnswer: '5', protanAnswer: null, deutanAnswer: null, description: 'Normal: 5, CVD: nothing', imageSource: require('../../../assets/ishihara/plate_22.png') },
  { plateNumber: 23, plateType: 'vanishing', normalAnswer: '7', protanAnswer: null, deutanAnswer: null, description: 'Normal: 7, CVD: nothing', imageSource: require('../../../assets/ishihara/plate_23.png') },
  { plateNumber: 24, plateType: 'vanishing', normalAnswer: '16', protanAnswer: null, deutanAnswer: null, description: 'Normal: 16, CVD: nothing', imageSource: require('../../../assets/ishihara/plate_24.png') },
  { plateNumber: 25, plateType: 'vanishing', normalAnswer: '73', protanAnswer: null, deutanAnswer: null, description: 'Normal: 73, CVD: nothing', imageSource: require('../../../assets/ishihara/plate_25.png') },
  
  // Tracing plates (26-38) - simplified for mobile (accept "line" as answer)
  { plateNumber: 26, plateType: 'vanishing', normalAnswer: 'line', protanAnswer: null, deutanAnswer: null, description: 'Normal: trace line, CVD: nothing', imageSource: require('../../../assets/ishihara/plate_26.png') },
  { plateNumber: 27, plateType: 'vanishing', normalAnswer: 'line', protanAnswer: null, deutanAnswer: null, description: 'Normal: trace line, CVD: nothing', imageSource: require('../../../assets/ishihara/plate_27.png') },
  { plateNumber: 28, plateType: 'hidden_digit', normalAnswer: null, protanAnswer: 'line', deutanAnswer: 'line', description: 'Normal: nothing, CVD: trace line', imageSource: require('../../../assets/ishihara/plate_28.png') },
  { plateNumber: 29, plateType: 'hidden_digit', normalAnswer: null, protanAnswer: 'line', deutanAnswer: 'line', description: 'Normal: nothing, CVD: trace line', imageSource: require('../../../assets/ishihara/plate_29.png') },
  { plateNumber: 30, plateType: 'classification', normalAnswer: 'purple', protanAnswer: 'red', deutanAnswer: 'blue', description: 'Line color classification', imageSource: require('../../../assets/ishihara/plate_30.png') },
  { plateNumber: 31, plateType: 'classification', normalAnswer: 'purple', protanAnswer: 'red', deutanAnswer: 'blue', description: 'Line color classification', imageSource: require('../../../assets/ishihara/plate_31.png') },
  { plateNumber: 32, plateType: 'vanishing', normalAnswer: 'line', protanAnswer: null, deutanAnswer: null, description: 'Normal: trace line, CVD: nothing', imageSource: require('../../../assets/ishihara/plate_32.png') },
  { plateNumber: 33, plateType: 'vanishing', normalAnswer: 'line', protanAnswer: null, deutanAnswer: null, description: 'Normal: trace line, CVD: nothing', imageSource: require('../../../assets/ishihara/plate_33.png') },
  { plateNumber: 34, plateType: 'hidden_digit', normalAnswer: null, protanAnswer: 'line', deutanAnswer: 'line', description: 'Normal: nothing, CVD: trace line', imageSource: require('../../../assets/ishihara/plate_34.png') },
  { plateNumber: 35, plateType: 'hidden_digit', normalAnswer: null, protanAnswer: 'line', deutanAnswer: 'line', description: 'Normal: nothing, CVD: trace line', imageSource: require('../../../assets/ishihara/plate_35.png') },
  { plateNumber: 36, plateType: 'classification', normalAnswer: 'purple', protanAnswer: 'red', deutanAnswer: 'blue', description: 'Line color classification', imageSource: require('../../../assets/ishihara/plate_36.png') },
  { plateNumber: 37, plateType: 'classification', normalAnswer: 'purple', protanAnswer: 'red', deutanAnswer: 'blue', description: 'Line color classification', imageSource: require('../../../assets/ishihara/plate_37.png') },
  { plateNumber: 38, plateType: 'control', normalAnswer: 'line', protanAnswer: 'line', deutanAnswer: 'line', description: 'Control tracing - everyone traces', imageSource: require('../../../assets/ishihara/plate_38.png') },
];

// Quick test uses first 14 plates
export const QUICK_PLATES = COMPREHENSIVE_PLATES.slice(0, 14);

/**
 * Evaluate user responses and determine CVD type
 */
export function evaluateIshiharaTest(
  responses: Map<number, string>,
  useComprehensive: boolean = false
): TestResult {
  const plates = useComprehensive ? COMPREHENSIVE_PLATES : QUICK_PLATES;
  
  let correctNormal = 0;
  let correctProtan = 0;
  let correctDeutan = 0;
  let incorrect = 0;
  let controlFailed = 0;
  let classificationProtan = 0;
  let classificationDeutan = 0;
  
  plates.forEach(plate => {
    const response = responses.get(plate.plateNumber)?.trim().toLowerCase();
    
    if (!response || response === '') {
      incorrect++;
      return;
    }
    
    const normalMatch = response === plate.normalAnswer?.toLowerCase();
    const protanMatch = response === plate.protanAnswer?.toLowerCase();
    const deutanMatch = response === plate.deutanAnswer?.toLowerCase();
    
    if (normalMatch) correctNormal++;
    if (protanMatch) correctProtan++;
    if (deutanMatch) correctDeutan++;
    if (!normalMatch && !protanMatch && !deutanMatch) incorrect++;
    
    // Track control plate failures
    if (plate.plateType === 'control' && !normalMatch) {
      controlFailed++;
    }
    
    // Classification plates help distinguish protan from deutan
    if (plate.plateType === 'classification') {
      if (protanMatch) classificationProtan++;
      if (deutanMatch) classificationDeutan++;
    }
  });
  
  // Diagnose CVD type
  const diagnosis = diagnoseCVD(
    correctNormal,
    correctProtan,
    correctDeutan,
    classificationProtan,
    classificationDeutan,
    controlFailed,
    plates.length
  );
  
  return {
    totalPlates: plates.length,
    correctNormal,
    correctProtan,
    correctDeutan,
    incorrect,
    controlFailed,
    ...diagnosis
  };
}

function diagnoseCVD(
  correctNormal: number,
  correctProtan: number,
  correctDeutan: number,
  classificationProtan: number,
  classificationDeutan: number,
  controlFailed: number,
  totalPlates: number
): {
  cvdType: 'normal' | 'protan' | 'deutan' | 'unknown';
  severity: number;
  confidence: number;
  interpretation: string;
  recommendations: string[];
} {
  // Invalid test if control plates failed
  if (controlFailed > 0) {
    return {
      cvdType: 'unknown',
      severity: 0,
      confidence: 0,
      interpretation: 'Test invalid - control plates failed. Please retake the test in good lighting.',
      recommendations: ['Ensure good lighting', 'Avoid screen glare', 'Take test again']
    };
  }
  
  const normalScore = correctNormal / totalPlates;
  const NORMAL_THRESHOLD = 0.86; // ~12/14 or ~33/38
  
  // Normal vision
  if (normalScore >= NORMAL_THRESHOLD) {
    return {
      cvdType: 'normal',
      severity: 0,
      confidence: 0.95,
      interpretation: 'Normal color vision detected. No signs of color vision deficiency.',
      recommendations: [
        'Your color vision appears normal',
        'Continue with regular eye checkups',
        'No special accommodations needed'
      ]
    };
  }
  
  // Determine CVD type
  let cvdType: 'protan' | 'deutan' = 'protan';
  if (classificationProtan > classificationDeutan) {
    cvdType = 'protan';
  } else if (classificationDeutan > classificationProtan) {
    cvdType = 'deutan';
  } else {
    // Use higher score
    cvdType = correctProtan >= correctDeutan ? 'protan' : 'deutan';
  }
  
  // Calculate severity
  let severity: number;
  let severityLabel: string;
  if (normalScore > 0.57) {
    severity = 0.3;
    severityLabel = 'Mild';
  } else if (normalScore > 0.29) {
    severity = 0.6;
    severityLabel = 'Moderate';
  } else {
    severity = 0.9;
    severityLabel = 'Strong';
  }
  
  // Calculate confidence
  const totalClassification = classificationProtan + classificationDeutan;
  const confidence = totalClassification > 0
    ? Math.max(classificationProtan, classificationDeutan) / totalClassification
    : 0.5;
  
  const cvdTypeLabel = cvdType === 'protan' ? 'Protanopia/Protanomaly (Red deficiency)' : 'Deuteranopia/Deuteranomaly (Green deficiency)';
  
  return {
    cvdType,
    severity,
    confidence,
    interpretation: `${severityLabel} ${cvdTypeLabel} detected. ${Math.round(normalScore * 100)}% of plates answered correctly for normal vision.`,
    recommendations: [
      'Consider using color enhancement features in ReColor app',
      'Consult an eye care professional for detailed examination',
      cvdType === 'protan' 
        ? 'Use tools that highlight red colors'
        : 'Use tools that highlight green colors',
      'Inform educators/employers if accommodation is needed',
      severity > 0.7 ? 'Strong CVD may require special accommodations' : 'Mild CVD - many tasks remain accessible'
    ]
  };
}
