# YAMNet Real-time Audio Classification Server

FastAPI WebSocket server for real-time audio classification using YAMNet.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the server:**
   ```bash
   python main.py
   ```

3. **Connect React Native app:**
   - Update IP address in React Native files
   - Connect to `ws://your-ip:8000/ws`

## 📡 Endpoints

- **WebSocket:** `ws://localhost:8000/ws` - Real-time audio classification
- **Health:** `http://localhost:8000/health` - Server status
- **Docs:** `http://localhost:8000/docs` - API documentation

## 🎤 How it works

1. React Native captures live audio
2. Sends audio chunks via WebSocket
3. YAMNet classifies each chunk
4. Results sent back to React Native
5. Real-time classification display

## 📁 Files

- `main.py` - FastAPI WebSocket server
- `requirements.txt` - Python dependencies
- `realtime_yamnet.py` - Original YAMNet implementation
- `recording.m4a` - Test audio file
