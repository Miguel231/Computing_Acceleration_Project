import React, { useState, useEffect, useRef } from 'react';
import { Camera, Users, Activity, Settings, Shield, AlertTriangle, CheckCircle, Plus, Trash2 } from 'lucide-react';

//const API_URL = 'http://localhost:8000';
//const WS_URL = 'ws://localhost:8000/ws/client';

const API_URL = 'http://172.20.10.3:8000';
const WS_URL = 'ws://172.20.10.3:8000/ws/client';


export default function SecurityDashboard() {
  const [familyMembers, setFamilyMembers] = useState([]);
  const [events, setEvents] = useState([]);
  const [stats, setStats] = useState({ total_events: 0, family_access: 0, intruder_alerts: 0 });
  const [config, setConfig] = useState({ threshold: 0.40, enable_alerts: true });
  const [currentFrame, setCurrentFrame] = useState(null);
  const [activeTab, setActiveTab] = useState('live');
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    connectWebSocket();
    fetchInitialData();
    
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const connectWebSocket = () => {
    const ws = new WebSocket(WS_URL);
    
    ws.onopen = () => {
      console.log('✓ Connected to server');
      setIsConnected(true);
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'init') {
        setFamilyMembers(data.family_members);
        setConfig(data.config);
      } else if (data.type === 'new_event') {
        setEvents(prev => [data.event, ...prev].slice(0, 100));
        fetchStats();
      } else if (data.type === 'video_frame') {
        setCurrentFrame(data.frame);
      }
    };
    
    ws.onclose = () => {
      console.log('✗ Disconnected from server');
      setIsConnected(false);
      setTimeout(connectWebSocket, 3000);
    };
    
    wsRef.current = ws;
  };

  const fetchInitialData = async () => {
    try {
      const [membersRes, eventsRes, statsRes, configRes] = await Promise.all([
        fetch(`${API_URL}/api/family`),
        fetch(`${API_URL}/api/events?limit=50`),
        fetch(`${API_URL}/api/events/stats`),
        fetch(`${API_URL}/api/config`)
      ]);
      
      setFamilyMembers(await membersRes.json());
      setEvents(await eventsRes.json());
      setStats(await statsRes.json());
      setConfig(await configRes.json());
    } catch (err) {
      console.error('Error fetching data:', err);
    }
  };

  const fetchStats = async () => {
    const res = await fetch(`${API_URL}/api/events/stats`);
    setStats(await res.json());
  };

  const addFamilyMember = async (name, imageFile) => {
    const formData = new FormData();
    formData.append('name', name);
    formData.append('image', imageFile);
    
    const res = await fetch(`${API_URL}/api/family?name=${name}`, {
      method: 'POST',
      body: formData
    });
    
    const newMember = await res.json();
    setFamilyMembers(prev => [...prev, newMember]);
  };

  const deleteFamilyMember = async (memberId) => {
    await fetch(`${API_URL}/api/family/${memberId}`, { method: 'DELETE' });
    setFamilyMembers(prev => prev.filter(m => m.id !== memberId));
  };

  const updateConfig = async (newConfig) => {
    await fetch(`${API_URL}/api/config`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(newConfig)
    });
    setConfig(newConfig);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <header className="bg-slate-800/50 backdrop-blur-lg border-b border-slate-700">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Shield className="w-8 h-8 text-cyan-400" />
            <div>
              <h1 className="text-2xl font-bold text-white">Smart EdgeAI Security</h1>
              <p className="text-sm text-slate-400">Facial Recognition System</p>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${isConnected ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`}></div>
              <span className="text-sm font-medium">{isConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <StatCard 
            icon={Activity} 
            label="Total Events" 
            value={stats.total_events} 
            color="blue" 
          />
          <StatCard 
            icon={CheckCircle} 
            label="Family Access" 
            value={stats.family_access} 
            color="green" 
          />
          <StatCard 
            icon={AlertTriangle} 
            label="Intruder Alerts" 
            value={stats.intruder_alerts} 
            color="red" 
          />
        </div>

        <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl border border-slate-700 overflow-hidden">
          <div className="flex border-b border-slate-700">
            {/* <TabButton icon={Camera} label="Live" active={activeTab === 'live'} onClick={() => setActiveTab('live')} /> */}
            <TabButton icon={Users} label="Family" active={activeTab === 'family'} onClick={() => setActiveTab('family')} />
            <TabButton icon={Activity} label="Events" active={activeTab === 'events'} onClick={() => setActiveTab('events')} />
            <TabButton icon={Settings} label="Settings" active={activeTab === 'config'} onClick={() => setActiveTab('config')} />
          </div>

          <div className="p-6">
            {/*{activeTab === 'live' && <LiveView frame={currentFrame} />}*/}
            {activeTab === 'family' && <FamilyView members={familyMembers} onAdd={addFamilyMember} onDelete={deleteFamilyMember} />}
            {activeTab === 'events' && <EventsView events={events} />}
            {activeTab === 'config' && <ConfigView config={config} onUpdate={updateConfig} />}
          </div>
        </div>
      </div>
    </div>
  );
}

function StatCard({ icon: Icon, label, value, color }) {
  const colors = {
    blue: 'from-blue-500/20 to-blue-600/10 border-blue-500/30 text-blue-400',
    green: 'from-green-500/20 to-green-600/10 border-green-500/30 text-green-400',
    red: 'from-red-500/20 to-red-600/10 border-red-500/30 text-red-400'
  };
  
  return (
    <div className={`bg-gradient-to-br ${colors[color]} border rounded-xl p-6`}>
      <Icon className="w-8 h-8 mb-3" />
      <p className="text-slate-300 text-sm mb-1">{label}</p>
      <p className="text-3xl font-bold text-white">{value}</p>
    </div>
  );
}

function TabButton({ icon: Icon, label, active, onClick }) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-6 py-3 font-medium transition-colors ${
        active 
          ? 'bg-cyan-500/20 text-cyan-400 border-b-2 border-cyan-400' 
          : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/30'
      }`}
    >
      <Icon className="w-5 h-5" />
      {label}
    </button>
  );
}
/*
function LiveView() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const startCamera = async () => {
      try {
        // Pedir acceso a la cámara
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play(); // asegurar que el video se reproduzca
        }
      } catch (err) {
        console.error('Error al acceder a la cámara:', err);
        setError('No se pudo acceder a la cámara. Revisa permisos y HTTPS.');
      }
    };

    startCamera();

    return () => {
      // Detener cámara al desmontar
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const takePhoto = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const dataUrl = canvas.toDataURL('image/jpeg');
    setCapturedImage(dataUrl);
  };

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold text-white">Live View</h2>

      <div className="bg-slate-900 rounded-lg overflow-hidden aspect-video flex flex-col items-center justify-center">
        {error && <p className="text-red-500">{error}</p>}
        {!capturedImage && !error && (
          <video ref={videoRef} autoPlay muted className="w-full h-full object-contain rounded-lg" />
        )}
        {capturedImage && (
          <img src={capturedImage} alt="Captured" className="w-full h-full object-contain rounded-lg" />
        )}
      </div>

      <div className="flex gap-2">
        {!capturedImage && !error && (
          <button
            onClick={takePhoto}
            className="px-4 py-2 bg-cyan-500 hover:bg-cyan-600 text-white rounded-lg"
          >
            Take Photo
          </button>
        )}
        {capturedImage && (
          <button
            onClick={() => setCapturedImage(null)}
            className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg"
          >
            Retake
          </button>
        )}
      </div>

      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
}
*/

function FamilyView({ members, onAdd, onDelete }) {
  const [showAddForm, setShowAddForm] = useState(false);
  const [newName, setNewName] = useState('');
  const [newImage, setNewImage] = useState(null);

  const handleAdd = () => {
    if (newName && newImage) {
      onAdd(newName, newImage);
      setNewName('');
      setNewImage(null);
      setShowAddForm(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold text-white">Family Members</h2>
        <button
          onClick={() => setShowAddForm(!showAddForm)}
          className="flex items-center gap-2 px-4 py-2 bg-cyan-500 hover:bg-cyan-600 text-white rounded-lg transition-colors"
        >
          <Plus className="w-5 h-5" />
          Add Member
        </button>
      </div>

      {showAddForm && (
        <div className="bg-slate-900 rounded-lg p-6 space-y-4">
          <input
            type="text"
            placeholder="Name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            className="w-full px-4 py-2 bg-slate-800 text-white rounded-lg border border-slate-700 focus:border-cyan-500 focus:outline-none"
          />
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setNewImage(e.target.files[0])}
            className="w-full px-4 py-2 bg-slate-800 text-white rounded-lg border border-slate-700"
          />
          <button
            onClick={handleAdd}
            className="w-full py-2 bg-green-500 hover:bg-green-600 text-white rounded-lg transition-colors"
          >
            Save
          </button>
        </div>
      )}

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {members.map(member => (
          <div key={member.id} className="bg-slate-900 rounded-lg p-4 text-center group relative">
            <img src={member.image_path} alt={member.name} className="w-24 h-24 rounded-full mx-auto mb-3 object-cover" />
            <p className="text-white font-medium">{member.name}</p>
            <p className="text-xs text-slate-400 mt-1">{new Date(member.added_date).toLocaleDateString()}</p>
            <button
              onClick={() => onDelete(member.id)}
              className="absolute top-2 right-2 p-2 bg-red-500/80 hover:bg-red-500 text-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

function EventsView({ events }) {
  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold text-white">Event History</h2>
      <div className="space-y-2 max-h-[600px] overflow-y-auto">
        {events.map(event => (
          <div key={event.id} className={`bg-slate-900 rounded-lg p-4 flex items-center gap-4 border-l-4 ${event.is_family ? 'border-green-500' : 'border-red-500'}`}>
            {event.image_snapshot && (
              <img src={`data:image/jpeg;base64,${event.image_snapshot}`} alt="Snapshot" className="w-16 h-16 rounded object-cover" />
            )}
            <div className="flex-1">
              <p className="text-white font-medium">
                {event.is_family ? (
                  <><CheckCircle className="w-4 h-4 inline text-green-400 mr-2" />Access: {event.person_name}</>
                ) : (
                  <><AlertTriangle className="w-4 h-4 inline text-red-400 mr-2" />⚠️ Intruder Detected</>
                )}
              </p>
              <p className="text-slate-400 text-sm">{new Date(event.timestamp).toLocaleString()}</p>
              <p className="text-slate-500 text-xs">Confidence: {(event.confidence * 100).toFixed(1)}%</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ConfigView({ config, onUpdate }) {
  const [threshold, setThreshold] = useState(config.threshold);
  const [webhook, setWebhook] = useState(config.webhook_url || '');

  const handleSave = () => {
    onUpdate({ ...config, threshold, webhook_url: webhook });
  };

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold text-white">System Configuration</h2>
      
      <div className="bg-slate-900 rounded-lg p-6 space-y-4">
        <div>
          <label className="block text-white mb-2">Detection Threshold: {threshold.toFixed(2)}</label>
          <input
            type="range"
            min="0.1"
            max="0.8"
            step="0.05"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="w-full"
          />
          <p className="text-slate-400 text-sm mt-2">Lower value = more strict</p>
        </div>

        <div>
          <label className="block text-white mb-2">Webhook URL (optional)</label>
          <input
            type="text"
            placeholder="https://your-webhook.com/alerts"
            value={webhook}
            onChange={(e) => setWebhook(e.target.value)}
            className="w-full px-4 py-2 bg-slate-800 text-white rounded-lg border border-slate-700 focus:border-cyan-500 focus:outline-none"
          />
        </div>

        <button
          onClick={handleSave}
          className="w-full py-2 bg-cyan-500 hover:bg-cyan-600 text-white rounded-lg transition-colors"
        >
          Save Configuration
        </button>
      </div>
    </div>
  );
}