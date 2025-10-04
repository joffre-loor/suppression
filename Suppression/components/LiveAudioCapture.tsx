import React, { useEffect, useState, useRef } from 'react';
import { View, Text, Pressable, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import LiveAudioStream from 'react-native-live-audio-stream';
import { Buffer } from 'buffer';

interface LiveAudioCaptureProps {
  onDataReceived?: (data: string) => void;
  onError?: (error: string) => void;
  onStreamingChange?: (streaming: boolean) => void;
  serverUrl?: string;
  onAmplitude?: (amplitude: number) => void;
}

const LiveAudioCapture: React.FC<LiveAudioCaptureProps> = ({ 
  onDataReceived, 
  onError,
  onStreamingChange,
  serverUrl = 'ws://172.20.10.2:8000/ws',
  onAmplitude
}) => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [permissionGranted, setPermissionGranted] = useState(false);
  const [chunkCount, setChunkCount] = useState(0);
  const [serverConnected, setServerConnected] = useState(false);
  const ws = useRef<WebSocket | null>(null);
  
  useEffect(() => {
    // iOS permissions are handled by app.json configuration
    setPermissionGranted(true);

    // Cleanup on unmount
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []);

  const connectToServer = () => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      console.log('Already connected to WebSocket server.');
      return;
    }

    try {
      ws.current = new WebSocket(serverUrl);

      ws.current.onopen = () => {
        setServerConnected(true);
        console.log('✅ Connected to WebSocket server');
      };

      ws.current.onclose = () => {
        setServerConnected(false);
        console.log('❌ Disconnected from WebSocket server');
      };

      ws.current.onerror = (error) => {
        onError?.(`WebSocket error: ${error}`);
        console.error('WebSocket error:', error);
      };

      ws.current.onmessage = (event) => {
        // Handle messages from the server (e.g., classification results)
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'classification' && data.result) {
            const result = data.result;
            if (result.top_class && result.top_confidence) {
              const classification = `${result.top_class} (${result.top_confidence.toFixed(3)})`;
              onDataReceived?.(classification);
            }
          }
        } catch (e) {
          console.error('Error parsing WebSocket message:', e);
        }
      };
    } catch (error) {
      onError?.(`Connection failed: ${error}`);
      console.error('Failed to connect:', error);
    }
  };

  const startStreaming = () => {
    if (!permissionGranted) {
      onError?.("Microphone permission not granted");
      return;
    }

    // Connect to the server first
    connectToServer();

    // Configure and start the audio stream
    const options = {
      sampleRate: 16000,
      channels: 1,
      bitsPerSample: 16,
      bufferSize: 4096,
      wavFile: '',
    };

    LiveAudioStream.init(options);
    LiveAudioStream.start();

    LiveAudioStream.on('data', (data: string) => {
      setChunkCount(prev => prev + 1);
      
      // Send audio data to the WebSocket server
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        ws.current.send(data);
      }

      // Compute and emit amplitude (RMS of 16-bit PCM)
      try {
        if (onAmplitude) {
          const buf = Buffer.from(data, 'base64');
          let sumSquares = 0;
          let samples = 0;
          for (let i = 0; i + 1 < buf.length; i += 2) {
            const sample = (buf[i] | (buf[i + 1] << 8));
            const signed = sample > 0x7FFF ? sample - 0x10000 : sample;
            const norm = signed / 32768;
            sumSquares += norm * norm;
            samples++;
          }
          if (samples > 0) {
            const rms = Math.sqrt(sumSquares / samples);
            // Smooth and clamp amplitude to [0, 1]
            const amplitude = Math.max(0, Math.min(1, rms));
            onAmplitude(amplitude);
          }
        }
      } catch (_) {
        // ignore amplitude calc errors
      }
    });

    setIsStreaming(true);
    onStreamingChange?.(true);
  };

  const stopStreaming = () => {
    LiveAudioStream.stop();
    setIsStreaming(false);
    onStreamingChange?.(false);
    
    // Close the WebSocket connection
    if (ws.current) {
      ws.current.close();
      ws.current = null;
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.statusRow}>
        <View style={styles.statusItem}>
          <Ionicons 
            name={permissionGranted ? "checkmark-circle" : "close-circle"} 
            size={16} 
            color={permissionGranted ? "#34C759" : "#FF3B30"} 
          />
          <Text style={styles.statusText}>Mic</Text>
        </View>
        <View style={styles.statusItem}>
          <Ionicons 
            name={serverConnected ? "checkmark-circle" : "close-circle"} 
            size={16} 
            color={serverConnected ? "#34C759" : "#FF3B30"} 
          />
          <Text style={styles.statusText}>Server</Text>
        </View>
        <View style={styles.statusItem}>
          <Ionicons 
            name={isStreaming ? "radio" : "radio-outline"} 
            size={16} 
            color={isStreaming ? "#007AFF" : "#8E8E93"} 
          />
          <Text style={styles.statusText}>Stream</Text>
        </View>
      </View>

      <Pressable 
        onPress={isStreaming ? stopStreaming : startStreaming} 
        disabled={!permissionGranted} 
        style={[
          styles.recordButton,
          isStreaming && styles.recordingButton,
          !permissionGranted && styles.disabledButton
        ]}
      >
        <Ionicons 
          name={isStreaming ? "stop" : "radio-outline"} 
          size={24} 
          color={isStreaming ? "#FFFFFF" : "#007AFF"} 
        />
        <Text style={[
          styles.buttonText,
          isStreaming && styles.recordingText
        ]}>
          {isStreaming ? "Stop Stream" : "Start Stream"}
        </Text>
      </Pressable>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
  },
  statusRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    width: '100%',
    marginBottom: 20,
  },
  statusItem: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: 4,
  },
  statusText: {
    fontSize: 12,
    fontWeight: '500',
    color: '#8E8E93',
  },
  recordButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
    paddingHorizontal: 32,
    borderRadius: 25,
    backgroundColor: '#F2F2F7',
    borderWidth: 2,
    borderColor: '#007AFF',
    minWidth: 200,
    gap: 8,
  },
  recordingButton: {
    backgroundColor: '#FF3B30',
    borderColor: '#FF3B30',
  },
  disabledButton: {
    opacity: 0.5,
    borderColor: '#C7C7CC',
  },
  buttonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#007AFF',
  },
  recordingText: {
    color: '#FFFFFF',
  },
});

export default LiveAudioCapture;