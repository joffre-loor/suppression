// LiveAudioScreen.tsx
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { View, Text, StyleSheet, Pressable, FlatList, Switch, TextInput } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';

const DEFAULT_API = 'http://172.20.10.5:8000'; // <-- set to your Mac's API address
const POLL_MS = 1000;

interface SoundItem {
  id: string;
  name: string;        // friendly name shown in UI
  isSuppressed: boolean;
  confidence: number;  // for future/visualization
}

// Map friendly names -> better AudioSep prompts
const PROMPT_MAP: Record<string, string> = {
  speech: 'speech',
  music: 'music',
  dog: 'dog barking',
  bird: 'bird chirping',
  fan: 'fan noise',
  chatter: 'people talking',
};

const PRESETS = ['speech', 'music', 'dog', 'bird', 'fan', 'chatter'];

export default function LiveAudioScreen() {
  const [apiBase, setApiBase] = useState<string>(DEFAULT_API);
  const [sounds, setSounds] = useState<SoundItem[]>(
    PRESETS.map((name, i) => ({
      id: `${i + 1}`,
      name,
      isSuppressed: false,
      confidence: 0.9,
    }))
  );
  const [error, setError] = useState<string | null>(null);
  const [online, setOnline] = useState<boolean>(false);

  // Classes to push, passed through PROMPT_MAP for nicer prompts
  const classes = useMemo(
    () =>
      Array.from(
        new Set(
          sounds
            .filter((s) => s.isSuppressed)
            .map((s) => PROMPT_MAP[s.name] ?? s.name)
        )
      ),
    [sounds]
  );

  // Debounced push helper (always mode: "drop")
  const pushTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pushState = useCallback(
    (arr = sounds) => {
      if (pushTimer.current) clearTimeout(pushTimer.current);
      pushTimer.current = setTimeout(async () => {
        try {
          const payload = {
            mode: 'drop',
            classes: arr
              .filter((s) => s.isSuppressed)
              .map((s) => PROMPT_MAP[s.name] ?? s.name),
            profile: 'Mobile Control',
          };
          const r = await fetch(`${apiBase}/suppress/set`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          });
          setOnline(r.ok);
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          setError(null);
        } catch {
          setOnline(false);
          setError('⚠️ Could not reach suppression API');
        }
      }, 150); // short debounce
    },
    [apiBase, sounds]
  );

  // Initial sync + poll for external changes
  useEffect(() => {
    let cancelled = false;

    const fetchCurrent = async () => {
      try {
        const r = await fetch(`${apiBase}/suppress/current`, { cache: 'no-store' });
        if (cancelled) return;
        if (!r.ok) {
          setOnline(false);
          return;
        }
        const d = await r.json();
        setOnline(true);

        const apiClasses: string[] = Array.isArray(d?.classes) ? d.classes : [];

        setSounds((prev) =>
          prev.map((s) => {
            const prompt = PROMPT_MAP[s.name] ?? s.name;
            const fromApi = apiClasses.includes(prompt);
            return { ...s, isSuppressed: fromApi };
          })
        );
      } catch {
        if (!cancelled) setOnline(false);
      }
    };

    fetchCurrent();
    const id = setInterval(fetchCurrent, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [apiBase]);

  const onToggle = (id: string) => {
    setSounds((prev) => {
      const next = prev.map((s) => (s.id === id ? { ...s, isSuppressed: !s.isSuppressed } : s));
      pushState(next);
      return next;
    });
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>Noise Suppression</Text>
        <Text style={styles.subtitle}>Tap a card to mute/unmute a sound type</Text>
      </View>

      {/* API bar */}
      <View style={styles.apiRow}>
        <TextInput
          style={styles.apiInput}
          value={apiBase}
          onChangeText={setApiBase}
          autoCapitalize="none"
          autoCorrect={false}
          placeholder="http://<mac-ip>:8000"
        />
        <View style={styles.statusPill}>
          <View style={[styles.dot, { backgroundColor: online ? '#34C759' : '#FF3B30' }]} />
          <Text style={styles.statusText}>{online ? 'Online' : 'Offline'}</Text>
        </View>
      </View>

      {/* Cards */}
      <FlatList
        contentContainerStyle={{ paddingHorizontal: 20, paddingBottom: 20 }}
        data={sounds}
        keyExtractor={(item) => item.id}
        ItemSeparatorComponent={() => <View style={{ height: 12 }} />}
        renderItem={({ item }) => (
          <Pressable
            onPress={() => onToggle(item.id)}
            style={[styles.card, item.isSuppressed && styles.cardMuted]}
          >
            <View style={styles.cardRow}>
              <Ionicons
                name={item.isSuppressed ? 'volume-mute' : 'volume-high'}
                size={28}
                color={item.isSuppressed ? '#FF3B30' : '#0A84FF'}
              />
              <View style={{ flex: 1 }}>
                <Text style={[styles.cardTitle, item.isSuppressed && styles.cardMutedText]}>
                  {capitalize(item.name)}
                </Text>
                <Text style={styles.cardSubtitle}>
                  {item.isSuppressed ? 'Muted (will be dropped)' : 'Active'}
                </Text>
              </View>
              <Switch value={!!item.isSuppressed} onValueChange={() => onToggle(item.id)} />
            </View>
          </Pressable>
        )}
      />

      {/* Footer */}
      <View style={styles.footer}>
        <Text style={styles.footerText}>
          Muted: {classes.length ? classes.join(', ') : 'none'}
        </Text>
      </View>

      {error && (
        <View style={styles.errorBox}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}
    </SafeAreaView>
  );
}

function capitalize(s: string) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F8F9FA' },
  header: { paddingHorizontal: 20, paddingTop: 16, paddingBottom: 8 },
  title: { fontSize: 28, fontWeight: '800', color: '#1D1D1F', marginBottom: 4 },
  subtitle: { fontSize: 14, color: '#6B7280' },

  apiRow: {
    flexDirection: 'row',
    gap: 10,
    alignItems: 'center',
    paddingHorizontal: 20,
    marginTop: 6,
    marginBottom: 12,
  },
  apiInput: {
    flex: 1,
    height: 40,
    borderRadius: 8,
    paddingHorizontal: 12,
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#E5E7EB',
  },
  statusPill: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#F3F4F6',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
  },
  dot: { width: 8, height: 8, borderRadius: 4, marginRight: 6 },
  statusText: { fontSize: 12, color: '#374151' },

  card: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    paddingVertical: 16,
    paddingHorizontal: 16,
    shadowColor: '#000',
    shadowOpacity: 0.06,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 2 },
    elevation: 2,
  },
  cardMuted: {
    backgroundColor: '#FFF5F5',
    borderColor: '#FECACA',
    borderWidth: 1,
  },
  cardRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1D1D1F',
  },
  cardMutedText: {
    textDecorationLine: 'line-through',
    color: '#8E8E93',
  },
  cardSubtitle: {
    fontSize: 13,
    color: '#8E8E93',
    marginTop: 2,
  },
  footer: {
    paddingHorizontal: 20,
    paddingVertical: 12,
  },
  footerText: { color: '#6B7280', fontSize: 13 },

  errorBox: {
    backgroundColor: '#FFF5F5',
    borderWidth: 1,
    borderColor: '#FECACA',
    padding: 10,
    marginHorizontal: 20,
    marginBottom: 16,
    borderRadius: 10,
  },
  errorText: { color: '#FF3B30', textAlign: 'center' },
});
