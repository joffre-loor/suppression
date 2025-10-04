# 🎤 Live Audio Streaming Setup Guide

This guide will help you set up live audio streaming between your React Native app and a FastAPI Python server.

## 📋 Prerequisites

- Node.js and npm installed
- Python 3.8+ installed
- iOS device or simulator
- EAS CLI installed (`npm install -g eas-cli`)

## 🚀 Quick Start

### 1. Start the Python Server

```bash
cd audio-server
python start_server.py
```

The server will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 2. Test the Server

In a new terminal:
```bash
cd audio-server
python test_api.py
```

### 3. Start the React Native App

```bash
npx expo start
```

### 4. Build for iOS (Required for Live Audio)

Since `react-native-live-audio-stream` requires native code, you need a custom development build:

```bash
eas build --platform ios --profile development
```

## 🔧 Configuration

### Server Configuration

The Python server is configured in `audio-server/main.py`:
- **Port**: 8000 (configurable)
- **CORS**: Enabled for all origins
- **Memory**: Keeps last 100 audio chunks
- **Logging**: Enabled for debugging

### Mobile App Configuration

Update the server URL in your app:
```typescript
// In app/(tabs)/live-audio.tsx
const [serverUrl, setServerUrl] = useState<string>('http://YOUR_IP:8000');
```

**Important**: Replace `localhost` with your computer's IP address when testing on a physical device.

## 📱 Usage

### 1. Recording Tab
- Tap "Start" to record 5-second audio chunks
- Tap "Play" to hear the last recording
- Audio is saved locally and can be uploaded to server

### 2. Live Stream Tab
- Tap "Start Stream" to begin live audio streaming
- Audio chunks are sent to the Python server in real-time
- Server connection status is displayed
- Chunk count shows received audio data

## 🔍 Monitoring

### Server Logs
The Python server logs all incoming audio chunks:
```
INFO: Received audio chunk 0 (32000 bytes)
INFO: Received audio chunk 1 (32000 bytes)
```

### Mobile App Logs
Check your React Native console for:
```
Server connection successful
Received audio chunk!
Audio chunk sent successfully: 0
```

## 🛠️ Troubleshooting

### Common Issues

1. **"Cannot scan QR code"**
   - You need a custom development build (not Expo Go)
   - Run: `eas build --platform ios --profile development`

2. **"Server not connected"**
   - Check if Python server is running
   - Verify IP address in mobile app
   - Test with: `curl http://YOUR_IP:8000/health`

3. **"Permission denied"**
   - iOS: Check Settings > Privacy > Microphone
   - Android: Grant microphone permission when prompted

4. **"Audio chunks not received"**
   - Check server logs for errors
   - Verify network connectivity
   - Test with `python test_api.py`

### Network Configuration

For physical device testing:
1. Find your computer's IP: `ipconfig` (Windows) or `ifconfig` (Mac/Linux)
2. Update server URL in mobile app
3. Ensure firewall allows port 8000
4. Test with browser: `http://YOUR_IP:8000/health`

## 🎯 Next Steps

### Enhance the Server
- Add authentication
- Implement audio processing (speech-to-text)
- Add WebSocket support for real-time communication
- Use Redis for chunk storage

### Enhance the Mobile App
- Add audio visualization
- Implement chunk buffering
- Add error recovery
- Support multiple audio formats

## 📚 API Reference

### Endpoints

- `POST /audio/chunk` - Send single audio chunk
- `GET /audio/chunks` - Get recent chunks
- `GET /audio/stream` - Stream chunks via SSE
- `POST /audio/process` - Process audio with AI
- `GET /stats` - Server statistics

### Audio Chunk Format

```json
{
  "audio_data": "base64_encoded_audio",
  "timestamp": "2024-01-01T00:00:00Z",
  "sample_rate": 16000,
  "channels": 1,
  "bits_per_sample": 16
}
```

## 🆘 Support

If you encounter issues:
1. Check server logs for errors
2. Verify network connectivity
3. Test with the provided test script
4. Check mobile app console for errors

