import { Component } from 'react';
import * as FileSystem from 'expo-file-system';
import { Ionicons } from '@expo/vector-icons';
import { Text, TouchableOpacity, View } from 'react-native';

const LOG_FILE = `${FileSystem.documentDirectory}recolor_debug.log`;
const MAX_LOG_LINES = 200;

export const AppLog = {
  _buffer: [],
  _flushTimer: null,

  async _flush() {
    if (this._buffer.length === 0) return;
    const batch = this._buffer.splice(0);
    try {
      const existing = await FileSystem.readAsStringAsync(LOG_FILE).catch(() => '');
      const lines = (existing + batch.join('\n') + '\n').split('\n');
      const trimmed = lines.slice(-MAX_LOG_LINES).join('\n');
      await FileSystem.writeAsStringAsync(LOG_FILE, trimmed);
    } catch (_) {}
  },

  log(tag, message) {
    const ts = new Date().toISOString().slice(11, 23);
    const line = `[${ts}] [${tag}] ${message}`;
    this._buffer.push(line);
    console.log(line);
    if (this._flushTimer) clearTimeout(this._flushTimer);
    this._flushTimer = setTimeout(() => this._flush(), 300);
  },

  async readAll() {
    try {
      return await FileSystem.readAsStringAsync(LOG_FILE);
    } catch (_) {
      return '(no logs yet)';
    }
  },

  async clear() {
    try {
      await FileSystem.deleteAsync(LOG_FILE, { idempotent: true });
    } catch (_) {}
  },
};

// Global error handlers — must run once at app startup (called from App.js)
export function installGlobalErrorHandlers() {
  const _defaultHandler = ErrorUtils.getGlobalHandler();
  ErrorUtils.setGlobalHandler((error, isFatal) => {
    AppLog.log('FATAL', `${isFatal ? '[FATAL] ' : ''}${error?.message || error}\n${error?.stack || ''}`);
    AppLog._flush();
    if (_defaultHandler) _defaultHandler(error, isFatal);
  });

  if (typeof global?.HermesInternal !== 'undefined') {
    try {
      const tracking = require('promise/setimmediate/rejection-tracking');
      tracking.enable({
        allRejections: true,
        onUnhandled: (id, rejection) => {
          if (rejection instanceof Error) {
            AppLog.log('PROMISE', `Unhandled rejection: ${rejection.message}\n${rejection.stack || ''}`);
          } else if (rejection) {
            AppLog.log('PROMISE', `Unhandled rejection: ${JSON.stringify(rejection)}`);
          }
          AppLog._flush();
        },
      });
    } catch (_) {}
  }
}

const PRIMARY = '#6C63FF';

export class ScreenErrorBoundary extends Component {
  state = { hasError: false, error: null };

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    AppLog.log('CRASH', `${error?.message || error}\n${info?.componentStack || ''}`);
  }

  render() {
    if (this.state.hasError) {
      return (
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#000', padding: 30 }}>
          <Ionicons name="warning" size={48} color="#FF6B6B" />
          <Text style={{ color: '#FFF', fontSize: 18, fontWeight: 'bold', marginTop: 15, textAlign: 'center' }}>
            This screen crashed
          </Text>
          <Text style={{ color: '#AAA', fontSize: 13, marginTop: 10, textAlign: 'center' }}>
            {this.state.error?.message || 'Unknown error'}
          </Text>
          <TouchableOpacity
            style={{ marginTop: 25, backgroundColor: PRIMARY, paddingHorizontal: 24, paddingVertical: 12, borderRadius: 8 }}
            onPress={() => {
              this.setState({ hasError: false, error: null });
              if (this.props.navigation?.goBack) this.props.navigation.goBack();
            }}
          >
            <Text style={{ color: '#FFF', fontWeight: 'bold' }}>
              {this.props.navigation ? 'Go Back' : 'Retry'}
            </Text>
          </TouchableOpacity>
        </View>
      );
    }
    return this.props.children;
  }
}
