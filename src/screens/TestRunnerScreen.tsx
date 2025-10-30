/**
 * Test Runner Screen
 * Displays daltonization algorithm test results
 */

import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  SafeAreaView,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import {
  daltonizePixel,
  daltonizeImage,
  simulateCVD,
  identifyColor,
  colorDistance,
} from '../services/daltonization';

interface TestResult {
  name: string;
  status: 'pass' | 'fail' | 'warning';
  message: string;
}

const TestRunnerScreen = () => {
  const [results, setResults] = useState<TestResult[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [summary, setSummary] = useState({ passed: 0, failed: 0, warnings: 0 });

  const runTests = async () => {
    setIsRunning(true);
    const testResults: TestResult[] = [];

    // Test 1: Normal vision
    try {
      const red = [1.0, 0.0, 0.0];
      const result = daltonizePixel(red, 'normal', 1.0);
      const unchanged = result.every((val, i) => Math.abs(val - red[i]) < 0.0001);
      testResults.push({
        name: 'Normal Vision (Identity)',
        status: unchanged ? 'pass' : 'fail',
        message: unchanged ? 'Colors unchanged for normal vision' : 'Colors modified unexpectedly',
      });
    } catch (error) {
      testResults.push({
        name: 'Normal Vision (Identity)',
        status: 'fail',
        message: `Error: ${error}`,
      });
    }

    // Test 2: Protanopia correction
    try {
      const red = [1.0, 0.0, 0.0];
      const corrected = daltonizePixel(red, 'protan', 0.8);
      const changed = corrected[0] !== red[0] || corrected[1] !== red[1] || corrected[2] !== red[2];
      testResults.push({
        name: 'Protanopia Correction',
        status: changed ? 'pass' : 'fail',
        message: changed
          ? `Red shifted from [${red.join(', ')}] to [${corrected.map(v => v.toFixed(2)).join(', ')}]`
          : 'No correction applied',
      });
    } catch (error) {
      testResults.push({
        name: 'Protanopia Correction',
        status: 'fail',
        message: `Error: ${error}`,
      });
    }

    // Test 3: Deuteranopia correction
    try {
      const green = [0.0, 1.0, 0.0];
      const corrected = daltonizePixel(green, 'deutan', 0.8);
      const changed = corrected[0] !== green[0] || corrected[1] !== green[1] || corrected[2] !== green[2];
      testResults.push({
        name: 'Deuteranopia Correction',
        status: changed ? 'pass' : 'fail',
        message: changed
          ? `Green shifted from [${green.join(', ')}] to [${corrected.map(v => v.toFixed(2)).join(', ')}]`
          : 'No correction applied',
      });
    } catch (error) {
      testResults.push({
        name: 'Deuteranopia Correction',
        status: 'fail',
        message: `Error: ${error}`,
      });
    }

    // Test 4: Tritanopia correction
    try {
      const blue = [0.0, 0.0, 1.0];
      const corrected = daltonizePixel(blue, 'tritan', 0.8);
      const changed = corrected[0] !== blue[0] || corrected[1] !== blue[1] || corrected[2] !== blue[2];
      testResults.push({
        name: 'Tritanopia Correction',
        status: changed ? 'pass' : 'fail',
        message: changed
          ? `Blue shifted from [${blue.join(', ')}] to [${corrected.map(v => v.toFixed(2)).join(', ')}]`
          : 'No correction applied',
      });
    } catch (error) {
      testResults.push({
        name: 'Tritanopia Correction',
        status: 'fail',
        message: `Error: ${error}`,
      });
    }

    // Test 5: Severity scaling
    try {
      const red = [1.0, 0.0, 0.0];
      const severities = [0.0, 0.5, 1.0];
      const distances = severities.map(s => {
        const corrected = daltonizePixel(red, 'protan', s);
        return colorDistance(red, corrected);
      });
      const increasing = distances[0] <= distances[1] && distances[1] <= distances[2];
      testResults.push({
        name: 'Severity Scaling',
        status: increasing ? 'pass' : 'fail',
        message: increasing
          ? `Distances: ${distances.map(d => d.toFixed(2)).join(' → ')}`
          : 'Distance does not increase with severity',
      });
    } catch (error) {
      testResults.push({
        name: 'Severity Scaling',
        status: 'fail',
        message: `Error: ${error}`,
      });
    }

    // Test 6: CVD Simulation
    try {
      const testImage = new Uint8ClampedArray([
        255, 0, 0, 255,
        0, 255, 0, 255,
      ]);
      const simulated = simulateCVD(testImage, 'protan');
      const changed = !simulated.every((val, i) => val === testImage[i]);
      const alphaPreserved = simulated[3] === 255 && simulated[7] === 255;
      testResults.push({
        name: 'CVD Simulation',
        status: changed && alphaPreserved ? 'pass' : 'fail',
        message: changed && alphaPreserved
          ? 'Simulation applied, alpha preserved'
          : 'Simulation or alpha preservation failed',
      });
    } catch (error) {
      testResults.push({
        name: 'CVD Simulation',
        status: 'fail',
        message: `Error: ${error}`,
      });
    }

    // Test 7: Color identification
    try {
      const tests = [
        { rgb: [255, 0, 0], expected: 'Red' },
        { rgb: [0, 255, 0], expected: 'Green' },
        { rgb: [0, 0, 255], expected: 'Blue' },
      ];
      const allPassed = tests.every(({ rgb, expected }) => {
        const result = identifyColor(rgb);
        return result.name === expected;
      });
      testResults.push({
        name: 'Color Identification',
        status: allPassed ? 'pass' : 'fail',
        message: allPassed ? 'All primary colors identified correctly' : 'Some colors misidentified',
      });
    } catch (error) {
      testResults.push({
        name: 'Color Identification',
        status: 'fail',
        message: `Error: ${error}`,
      });
    }

    // Test 8: Performance benchmark
    try {
      const benchmarkImage = new Uint8ClampedArray(1280 * 720 * 4);
      for (let i = 0; i < benchmarkImage.length; i += 4) {
        benchmarkImage[i] = 128;
        benchmarkImage[i + 1] = 128;
        benchmarkImage[i + 2] = 128;
        benchmarkImage[i + 3] = 255;
      }

      const iterations = 5;
      const startTime = Date.now();
      for (let i = 0; i < iterations; i++) {
        daltonizeImage(benchmarkImage, 'protan', 0.8);
      }
      const endTime = Date.now();
      const avgTime = (endTime - startTime) / iterations;
      const fps = 1000 / avgTime;

      testResults.push({
        name: 'Performance (720p)',
        status: fps >= 30 ? 'pass' : fps >= 15 ? 'warning' : 'fail',
        message: `${fps.toFixed(1)} FPS (${avgTime.toFixed(0)}ms per frame)`,
      });
    } catch (error) {
      testResults.push({
        name: 'Performance (720p)',
        status: 'fail',
        message: `Error: ${error}`,
      });
    }

    // Calculate summary
    const passed = testResults.filter(r => r.status === 'pass').length;
    const failed = testResults.filter(r => r.status === 'fail').length;
    const warnings = testResults.filter(r => r.status === 'warning').length;

    setResults(testResults);
    setSummary({ passed, failed, warnings });
    setIsRunning(false);
  };

  useEffect(() => {
    runTests();
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pass':
        return '#4CAF50';
      case 'warning':
        return '#FF9800';
      case 'fail':
        return '#F44336';
      default:
        return '#999';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pass':
        return '✓';
      case 'warning':
        return '⚠';
      case 'fail':
        return '✗';
      default:
        return '○';
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Algorithm Tests</Text>
        <TouchableOpacity style={styles.runButton} onPress={runTests} disabled={isRunning}>
          <Text style={styles.runButtonText}>
            {isRunning ? 'Running...' : 'Run Tests'}
          </Text>
        </TouchableOpacity>
      </View>

      {isRunning ? (
        <View style={styles.loading}>
          <ActivityIndicator size="large" color="#4CAF50" />
          <Text style={styles.loadingText}>Running tests...</Text>
        </View>
      ) : (
        <>
          <View style={styles.summary}>
            <View style={styles.summaryItem}>
              <Text style={[styles.summaryCount, { color: '#4CAF50' }]}>
                {summary.passed}
              </Text>
              <Text style={styles.summaryLabel}>Passed</Text>
            </View>
            <View style={styles.summaryItem}>
              <Text style={[styles.summaryCount, { color: '#FF9800' }]}>
                {summary.warnings}
              </Text>
              <Text style={styles.summaryLabel}>Warnings</Text>
            </View>
            <View style={styles.summaryItem}>
              <Text style={[styles.summaryCount, { color: '#F44336' }]}>
                {summary.failed}
              </Text>
              <Text style={styles.summaryLabel}>Failed</Text>
            </View>
          </View>

          <ScrollView style={styles.results}>
            {results.map((result, index) => (
              <View key={index} style={styles.resultCard}>
                <View style={styles.resultHeader}>
                  <Text
                    style={[
                      styles.resultIcon,
                      { color: getStatusColor(result.status) },
                    ]}
                  >
                    {getStatusIcon(result.status)}
                  </Text>
                  <Text style={styles.resultName}>{result.name}</Text>
                </View>
                <Text style={styles.resultMessage}>{result.message}</Text>
              </View>
            ))}
          </ScrollView>
        </>
      )}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#FFF',
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
  runButton: {
    backgroundColor: '#4CAF50',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 8,
  },
  runButtonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: '600',
  },
  loading: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 15,
    fontSize: 16,
    color: '#666',
  },
  summary: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    padding: 20,
    backgroundColor: '#FFF',
    marginBottom: 10,
  },
  summaryItem: {
    alignItems: 'center',
  },
  summaryCount: {
    fontSize: 32,
    fontWeight: 'bold',
  },
  summaryLabel: {
    fontSize: 14,
    color: '#666',
    marginTop: 5,
  },
  results: {
    flex: 1,
  },
  resultCard: {
    backgroundColor: '#FFF',
    marginHorizontal: 15,
    marginBottom: 10,
    padding: 15,
    borderRadius: 8,
    borderLeftWidth: 4,
    borderLeftColor: '#E0E0E0',
  },
  resultHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  resultIcon: {
    fontSize: 20,
    fontWeight: 'bold',
    marginRight: 10,
  },
  resultName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    flex: 1,
  },
  resultMessage: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
  },
});

export default TestRunnerScreen;
