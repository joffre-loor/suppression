import { useState } from 'react';
import { useAudioRecorder, RecordingPresets, requestRecordingPermissionsAsync, setAudioModeAsync } from 'expo-audio';

export function useSimpleRecorder(serverUrl: string) {
  const rec = useAudioRecorder(RecordingPresets.HIGH_QUALITY);
  const [lastUrl, setLastUrl] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);

  async function start() {
    const perm = await requestRecordingPermissionsAsync();
    if (!perm.granted) {
      throw new Error('Microphone permission not granted');
    }
    await setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });
    await rec.prepareToRecordAsync();
    rec.record();
    setIsRecording(true);
    console.log('[audio] Started 5-second recording');
  }

  async function stop() {
    try {
      await rec.stop();
      setIsRecording(false);
      
      const status = rec.getStatus();
      const url: string | undefined = (status as any)?.uri ?? (rec as any)?.uri ?? undefined;
      
      if (url) {
        console.log('[audio] Recording complete', { url });
        setLastUrl(url);
        
        // Upload to server
        const form = new FormData();
        form.append('file', { uri: url, name: 'recording.m4a', type: 'audio/mp4' } as any);
        form.append('duration', '5');
        fetch(`${serverUrl}/chunk`, { method: 'POST', body: form }).catch(() => {});
      }
    } catch (e) {
      console.log('[audio] Stop error:', e);
      setIsRecording(false);
    }
  }

  return { start, stop, lastUrl, isRecording };
}


