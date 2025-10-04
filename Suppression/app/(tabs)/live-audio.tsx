import React, { useState, useEffect, useRef, useMemo } from 'react';
import { View, Text, ScrollView, StyleSheet, Pressable, Animated, Dimensions } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import LiveAudioCapture from '@/components/LiveAudioCapture';

const { width } = Dimensions.get('window');

interface DetectedSound {
  id: string;
  name: string;
  confidence: number;
  timestamp: Date;
  isSuppressed: boolean;
}

export default function LiveAudioScreen() {
  const [detectedSounds, setDetectedSounds] = useState<DetectedSound[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [soundLevel, setSoundLevel] = useState(0);
  const [lastDetectionAt, setLastDetectionAt] = useState<number | null>(null);
  const [lastConfidence, setLastConfidence] = useState(0);
  const [amplitude, setAmplitude] = useState(0);

  // Animation for sound wave (persist across renders)
  const waveAnimation = useRef(new Animated.Value(0)).current;
  const pulseAnimation = useRef(new Animated.Value(1)).current;

  // Pre-generate bar seeds for visual diversity
  const BAR_COUNT = 24;
  const barSeeds = useRef(Array.from({ length: BAR_COUNT }, () => Math.random() * 0.8 + 0.2)).current;

  // Derive currently active sound types (last 10s)
  const activeTypes = useMemo(() => {
    const now = Date.now();
    const windowMs = 10000;
    const map = new Map<string, { confidence: number; isSuppressed: boolean; displayName: string }>();
    for (const s of detectedSounds) {
      if (now - s.timestamp.getTime() <= windowMs) {
        const key = s.name.trim().toLowerCase();
        const prev = map.get(key);
        if (!prev || s.confidence > prev.confidence) {
          map.set(key, { confidence: s.confidence, isSuppressed: s.isSuppressed, displayName: s.name });
        }
      }
    }
    return Array.from(map.entries())
      .sort((a, b) => b[1].confidence - a[1].confidence)
      .map(([_, meta]) => ({ name: meta.displayName, confidence: meta.confidence, isSuppressed: meta.isSuppressed }));
  }, [detectedSounds]);

  // Determine visualizer state
  const visualState = useMemo<'idle' | 'listening' | 'active'>(() => {
    if (!isStreaming) return 'idle';
    const now = Date.now();
    if (lastDetectionAt && now - lastDetectionAt < 1500 && lastConfidence > 0.1) {
      return 'active';
    }
    return 'listening';
  }, [isStreaming, lastDetectionAt, lastConfidence]);

  // Sound wave animation
  useEffect(() => {
    if (isStreaming) {
      const waveLoop = Animated.loop(
        Animated.sequence([
          Animated.timing(waveAnimation, {
            toValue: 1,
            duration: 900,
            useNativeDriver: true,
          }),
          Animated.timing(waveAnimation, {
            toValue: 0,
            duration: 900,
            useNativeDriver: true,
          }),
        ])
      );

      const pulseLoop = Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnimation, {
            toValue: 1.08,
            duration: 800,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnimation, {
            toValue: 1,
            duration: 800,
            useNativeDriver: true,
          }),
        ])
      );

      waveLoop.start();
      pulseLoop.start();

      return () => {
        waveLoop.stop();
        pulseLoop.stop();
      };
    }
  }, [isStreaming]);

  const handleDataReceived = (data: string) => {
    // Parse the classification data
    try {
      const parts = data.split(' (');
      if (parts.length === 2) {
        const name = parts[0];
        const confidence = parseFloat(parts[1].replace(')', ''));
        
        const newSound: DetectedSound = {
          id: Date.now().toString(),
          name,
          confidence,
          timestamp: new Date(),
          isSuppressed: false
        };
        
        setDetectedSounds(prev => [newSound, ...prev.slice(0, 19)]); // Keep last 20
        const level = Math.min(confidence * 100, 100);
        setSoundLevel(level); // Update sound level
        setLastDetectionAt(Date.now());
        setLastConfidence(confidence);
      }
    } catch (e) {
      console.error('Error parsing sound data:', e);
    }
  };

  const handleError = (errorMessage: string) => {
    setError(errorMessage);
  };

  const toggleSuppression = (soundId: string) => {
    setDetectedSounds(prev => 
      prev.map(sound => 
        sound.id === soundId 
          ? { ...sound, isSuppressed: !sound.isSuppressed }
          : sound
      )
    );
  };

  const handleStreamingChange = (streaming: boolean) => {
    setIsStreaming(streaming);
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.headerContainer}>
        <Text style={styles.header}>Noise Suppression</Text>
        <Text style={styles.subtitle}>Real-time audio classification & suppression</Text>
      </View>

      {/* Active Types & Sound Wave Visualizer */}
      {isStreaming && (
        <View style={styles.waveContainer}>
          {/* Active types chips */}
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={styles.chipsRow}
          >
            {activeTypes.length === 0 ? (
              <View style={styles.chipMuted}>
                <Ionicons name="pulse" size={14} color="#8E8E93" />
                <Text style={styles.chipMutedText}>Listening…</Text>
              </View>
            ) : (
              activeTypes.map((t) => (
                <View
                  key={t.name}
                  style={[styles.chip, t.isSuppressed && styles.chipSuppressed]}
                >
                  <Text style={[styles.chipText, t.isSuppressed && styles.chipTextSuppressed]}>
                    {t.name}
                  </Text>
                  <View style={styles.chipBadge}>
                    <Text style={styles.chipBadgeText}>{Math.round(t.confidence * 100)}%</Text>
                  </View>
                </View>
              ))
            )}
          </ScrollView>

          {/* Siri-like visualizer: dots when idle/listening; bars when active */}
          {visualState !== 'active' ? (
            <View style={styles.dotsRow}>
              {barSeeds.slice(0, 5).map((seed, idx) => {
                const scale = waveAnimation.interpolate({
                  inputRange: [0, 0.5, 1],
                  outputRange: [1 - seed * 0.2, 1 + seed * 0.2, 1 - seed * 0.2],
                });
                const opacity = waveAnimation.interpolate({
                  inputRange: [0, 0.5, 1],
                  outputRange: [0.6, 1, 0.6],
                });
                return (
                  <Animated.View
                    key={idx}
                    style={[styles.dot, { transform: [{ scale }], opacity }]}
                  />
                );
              })}
            </View>
          ) : (
            <Animated.View style={[styles.wave, { transform: [{ scale: pulseAnimation }] }]}>
              {Array.from({ length: 18 }).map((_, idx) => {
                const seed = barSeeds[idx] ?? 0.5;
                const base = 10 + (idx % 3) * 2;
                const dynamic = amplitude * 72 * (0.5 + seed * 0.9);
                const height = Math.max(8, Math.min(88, base + dynamic));
                const scaleY = waveAnimation.interpolate({
                  inputRange: [0, 0.5, 1],
                  outputRange: [1, 1 + seed * 0.8, 1],
                });
                return (
                  <Animated.View
                    key={idx}
                    style={[
                      styles.waveBar,
                      {
                        height,
                        transform: [{ scaleY }],
                        opacity: 0.6 + seed * 0.4,
                      },
                    ]}
                  />
                );
              })}
            </Animated.View>
          )}
          <Text style={styles.listeningText}>
            {visualState === 'active' ? '🎙️ Listening…' : '🟢 Ready'}
          </Text>
        </View>
      )}

      {/* Detected Sounds List */}
      <ScrollView style={styles.soundsList} showsVerticalScrollIndicator={false}>
        {detectedSounds.length === 0 ? (
          <View style={styles.emptyState}>
            <Ionicons name="volume-high" size={48} color="#E5E5EA" />
            <Text style={styles.emptyText}>No sounds detected yet</Text>
            <Text style={styles.emptySubtext}>Start streaming to see detected sounds</Text>
          </View>
        ) : (
          detectedSounds.map((sound) => (
            <Pressable
              key={sound.id}
              style={[
                styles.soundCard,
                sound.isSuppressed && styles.suppressedCard
              ]}
              onPress={() => toggleSuppression(sound.id)}
            >
              <View style={styles.soundContent}>
                <View style={styles.soundInfo}>
                  <Text style={[
                    styles.soundName,
                    sound.isSuppressed && styles.suppressedText
                  ]}>
                    {sound.name}
                  </Text>
                  <View style={styles.confidenceContainer}>
                    <View style={[styles.confidenceBar, { width: `${sound.confidence * 100}%` }]} />
                  </View>
                  <Text style={styles.soundConfidence}>
                    {Math.round(sound.confidence * 100)}% confidence
                  </Text>
                </View>
                <View style={styles.soundActions}>
                  <View style={[
                    styles.suppressButton,
                    sound.isSuppressed && styles.suppressedButton
                  ]}>
                    <Ionicons 
                      name={sound.isSuppressed ? "volume-mute" : "volume-high"} 
                      size={24} 
                      color={sound.isSuppressed ? "#FF3B30" : "#34C759"} 
                    />
                  </View>
                </View>
              </View>
            </Pressable>
          ))
        )}
      </ScrollView>

      {/* Voice Recorder at Bottom */}
      <View style={styles.recorderContainer}>
        <LiveAudioCapture 
          onDataReceived={handleDataReceived}
          onError={handleError}
          onStreamingChange={handleStreamingChange}
          onAmplitude={setAmplitude}
          serverUrl="ws://172.20.10.2:8000/ws"
        />
      </View>

      {error && (
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>⚠️ {error}</Text>
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { 
    flex: 1, 
    backgroundColor: '#F8F9FA' 
  },
  headerContainer: {
    paddingHorizontal: 24,
    paddingTop: 20,
    paddingBottom: 16,
    backgroundColor: '#FFFFFF',
    borderBottomWidth: 1,
    borderBottomColor: '#E5E5EA',
  },
  header: { 
    fontSize: 32, 
    fontWeight: '700', 
    color: '#1D1D1F',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 16,
    color: '#8E8E93',
    fontWeight: '400',
  },
  waveContainer: {
    backgroundColor: '#FFFFFF',
    paddingVertical: 24,
    paddingHorizontal: 24,
    alignItems: 'center',
    borderBottomWidth: 1,
    borderBottomColor: '#E5E5EA',
  },
  wave: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    height: 64,
    marginBottom: 12,
  },
  waveBar: {
    width: 5,
    backgroundColor: '#0A84FF',
    marginHorizontal: 3,
    borderRadius: 3,
    shadowColor: '#0A84FF',
    shadowOpacity: 0.2,
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: 3,
  },
  dotsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    height: 32,
    marginBottom: 12,
    gap: 10,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#0A84FF',
  },
  listeningText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#0A84FF',
  },
  chipsRow: {
    paddingBottom: 12,
  },
  chip: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#EEF5FF',
    borderWidth: 1,
    borderColor: '#D6E6FF',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 999,
    marginRight: 8,
    gap: 8,
  },
  chipSuppressed: {
    backgroundColor: '#FFF5F5',
    borderColor: '#FECACA',
  },
  chipText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#0A84FF',
  },
  chipTextSuppressed: {
    color: '#FF3B30',
  },
  chipBadge: {
    backgroundColor: '#0A84FF',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 999,
  },
  chipBadgeText: {
    fontSize: 12,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  chipMuted: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#F2F2F7',
    borderWidth: 1,
    borderColor: '#E5E5EA',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 999,
    gap: 6,
  },
  chipMutedText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#8E8E93',
  },
  soundsList: {
    flex: 1,
    paddingHorizontal: 16,
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#8E8E93',
    marginTop: 16,
  },
  emptySubtext: {
    fontSize: 14,
    color: '#C7C7CC',
    marginTop: 4,
  },
  soundCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    marginVertical: 6,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  suppressedCard: {
    backgroundColor: '#FFF5F5',
    borderWidth: 1,
    borderColor: '#FECACA',
  },
  soundContent: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
  },
  soundInfo: {
    flex: 1,
  },
  soundName: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1D1D1F',
    marginBottom: 8,
  },
  suppressedText: {
    textDecorationLine: 'line-through',
    color: '#8E8E93',
  },
  confidenceContainer: {
    height: 4,
    backgroundColor: '#E5E5EA',
    borderRadius: 2,
    marginBottom: 8,
    overflow: 'hidden',
  },
  confidenceBar: {
    height: '100%',
    backgroundColor: '#34C759',
    borderRadius: 2,
  },
  soundConfidence: {
    fontSize: 14,
    color: '#8E8E93',
    fontWeight: '500',
  },
  soundActions: {
    marginLeft: 16,
  },
  suppressButton: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#F2F2F7',
    alignItems: 'center',
    justifyContent: 'center',
  },
  suppressedButton: {
    backgroundColor: '#FFEBEE',
  },
  recorderContainer: {
    backgroundColor: '#FFFFFF',
    paddingVertical: 20,
    paddingHorizontal: 24,
    borderTopWidth: 1,
    borderTopColor: '#E5E5EA',
  },
  errorContainer: {
    backgroundColor: '#FFF5F5',
    padding: 16,
    margin: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#FECACA',
  },
  errorText: {
    color: '#FF3B30',
    fontSize: 14,
    fontWeight: '500',
    textAlign: 'center',
  },
});