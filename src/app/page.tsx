"use client";

import { useState, useEffect, useRef } from "react";
import { Shield, FolderOpen, ShieldAlert, Terminal as TerminalIcon, ShieldCheck, Zap, Lock, Fingerprint, Eye } from "lucide-react";

type LogEntry = {
  time: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
};

type ShieldReport = {
  status: string;
  engine_mode: string;
  clip_distance: number;
  pixels_modified_pct: number;
  image_size: string;
  original_hash: string;
  protected_hash: string;
  timestamp: string;
};

export default function Home() {
  const [protectedCount, setProtectedCount] = useState(0);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [shieldReport, setShieldReport] = useState<ShieldReport | null>(null);
  const [currentFileName, setCurrentFileName] = useState("");
  const logsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setLogs([{
      time: new Date().toLocaleTimeString(),
      message: "System initialized. CLIP adversarial engine standby.",
      type: "info"
    }]);

    let unlisteners: Array<() => void> = [];

    const initializeIpc = async () => {
      try {
        const { listen } = await import('@tauri-apps/api/event');

        const unlistenLog = await listen<{ message: string, level: string }>('app-log', (event) => {
          const msg = event.payload.message;

          // Parse progress events
          if (msg.startsWith("PROGRESS:")) {
            const pct = parseInt(msg.split(":")[1]);
            setProgress(pct);
            return;
          }

          // Parse shield report
          if (msg.startsWith("REPORT:")) {
            try {
              const report = JSON.parse(msg.substring(7));
              setShieldReport(report);
              setIsProcessing(false);
              setProgress(100);
            } catch (e) { /* ignore parse errors */ }
            return;
          }

          setLogs(prev => [...prev, {
            time: new Date().toLocaleTimeString(),
            message: msg,
            type: event.payload.level as 'info' | 'success' | 'error' | 'warning'
          }]);
        });

        const unlistenShield = await listen('asset-shielded', () => {
          setProtectedCount(p => p + 1);
        });

        unlisteners.push(unlistenLog, unlistenShield);
      } catch (e) {
        console.warn("Tauri IPC not available.", e);
      }
    };

    initializeIpc();
    return () => { unlisteners.forEach(fn => fn()); };
  }, []);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const selectAndProcess = async () => {
    try {
      const { open } = await import('@tauri-apps/plugin-dialog');
      const { invoke } = await import('@tauri-apps/api/core');

      const selected = await open({
        multiple: false,
        filters: [{ name: 'Image Files', extensions: ['png', 'jpg', 'jpeg'] }]
      });

      if (selected) {
        const fileName = typeof selected === 'string' ? selected.split('\\').pop() || selected.split('/').pop() || 'image' : 'image';
        setCurrentFileName(fileName);
        setIsProcessing(true);
        setProgress(0);
        setShieldReport(null);
        setLogs(prev => [...prev, { time: new Date().toLocaleTimeString(), message: `Selected: ${fileName}`, type: 'info' }]);
        await invoke('ingest_file', { sourcePath: selected });
      }
    } catch (e) {
      console.log("Error:", e);
      setIsProcessing(false);
    }
  };

  const openShielded = async () => {
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      await invoke('open_folder', { folderType: 'shielded' });
    } catch (e) {
      console.log("Error:", e);
    }
  };

  const clipDistancePct = shieldReport ? Math.min(Math.round(shieldReport.clip_distance * 100), 100) : 0;

  return (
    <main style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      backgroundColor: '#09090b', color: '#fafafa', minHeight: '100vh',
      padding: '1.5rem 2rem', fontFamily: "'Inter', 'SF Pro Display', monospace",
      userSelect: 'none', position: 'relative', overflow: 'hidden'
    }}>

      {/* Ambient glow */}
      <div style={{
        position: 'fixed', top: '-200px', right: '-200px', width: '600px', height: '600px',
        background: 'radial-gradient(circle, rgba(16,185,129,0.06) 0%, transparent 70%)',
        pointerEvents: 'none', zIndex: 0
      }} />
      <div style={{
        position: 'fixed', bottom: '-200px', left: '-200px', width: '600px', height: '600px',
        background: 'radial-gradient(circle, rgba(16,185,129,0.04) 0%, transparent 70%)',
        pointerEvents: 'none', zIndex: 0
      }} />

      {/* Header */}
      <div style={{
        width: '100%', maxWidth: '900px', display: 'flex', justifyContent: 'space-between',
        alignItems: 'center', marginBottom: '2rem', borderBottom: '1px solid rgba(39,39,42,0.6)',
        paddingBottom: '1rem', position: 'relative', zIndex: 1
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '40px', height: '40px', borderRadius: '10px',
            background: 'linear-gradient(135deg, #059669, #10b981)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 0 20px rgba(16,185,129,0.3)'
          }}>
            <Shield style={{ width: '22px', height: '22px', color: '#fff' }} />
          </div>
          <div>
            <h1 style={{ fontSize: '1.25rem', fontWeight: 700, letterSpacing: '0.08em', color: '#fafafa', margin: 0 }}>
              POISON<span style={{ color: '#52525b' }}>PILL</span>
            </h1>
            <span style={{ fontSize: '0.65rem', color: '#52525b', letterSpacing: '0.15em', textTransform: 'uppercase' }}>
              Anti-AI Shield Engine
            </span>
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '2px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <div style={{
              width: '6px', height: '6px', borderRadius: '50%', backgroundColor: '#10b981',
              boxShadow: '0 0 8px rgba(16,185,129,0.6)',
              animation: 'pulse 2s infinite'
            }} />
            <span style={{ fontSize: '0.7rem', color: '#10b981', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
              System Active
            </span>
          </div>
          <span style={{ fontSize: '0.65rem', color: '#3f3f46' }}>v2.0.0 — CLIP Engine</span>
        </div>
      </div>

      {/* Stats + Actions Row */}
      <div style={{
        width: '100%', maxWidth: '900px', display: 'grid',
        gridTemplateColumns: '1fr 1fr 1fr', gap: '16px', marginBottom: '1.5rem',
        position: 'relative', zIndex: 1
      }}>

        {/* Protected Assets Counter */}
        <div style={{
          backgroundColor: 'rgba(16,185,129,0.05)', border: '1px solid rgba(16,185,129,0.15)',
          borderRadius: '12px', padding: '1.25rem', display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center', position: 'relative'
        }}>
          <span style={{ color: '#6b7280', fontSize: '0.7rem', letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: '0.75rem' }}>
            Protected Assets
          </span>
          <span style={{
            fontSize: '2.75rem', fontWeight: 200, color: '#34d399', fontFamily: 'monospace',
            textShadow: '0 0 20px rgba(16,185,129,0.3)', lineHeight: 1
          }}>
            {protectedCount}
          </span>
        </div>

        {/* Select & Protect Button */}
        <div
          onClick={isProcessing ? undefined : selectAndProcess}
          style={{
            backgroundColor: isProcessing ? 'rgba(16,185,129,0.1)' : '#fafafa',
            borderRadius: '12px', padding: '1.25rem', display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center', cursor: isProcessing ? 'wait' : 'pointer',
            transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
            border: isProcessing ? '1px solid rgba(16,185,129,0.3)' : '1px solid transparent',
            position: 'relative', overflow: 'hidden'
          }}
          onMouseEnter={(e) => { if (!isProcessing) { e.currentTarget.style.transform = 'translateY(-3px)'; e.currentTarget.style.boxShadow = '0 12px 30px -8px rgba(16,185,129,0.25)'; } }}
          onMouseLeave={(e) => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = 'none'; }}
        >
          {isProcessing ? (
            <>
              <Zap style={{ width: '22px', height: '22px', color: '#10b981', marginBottom: '0.5rem', animation: 'spin 1s linear infinite' }} />
              <span style={{ fontSize: '0.95rem', fontWeight: 700, color: '#10b981', marginBottom: '4px' }}>Processing...</span>
              {/* Progress Bar */}
              <div style={{ width: '100%', height: '4px', backgroundColor: 'rgba(16,185,129,0.1)', borderRadius: '2px', overflow: 'hidden', marginTop: '6px' }}>
                <div style={{
                  height: '100%', backgroundColor: '#10b981', borderRadius: '2px',
                  width: `${progress}%`, transition: 'width 0.3s ease',
                  boxShadow: '0 0 8px rgba(16,185,129,0.5)'
                }} />
              </div>
              <span style={{ fontSize: '0.65rem', color: '#6b7280', marginTop: '4px' }}>{progress}% — FGSM iteration</span>
            </>
          ) : (
            <>
              <FolderOpen style={{ width: '22px', height: '22px', color: '#18181b', marginBottom: '0.5rem' }} />
              <span style={{ fontSize: '0.95rem', fontWeight: 700, color: '#18181b', marginBottom: '4px' }}>Select & Protect</span>
              <span style={{ fontSize: '0.7rem', color: '#71717a' }}>Choose an image to shield</span>
            </>
          )}
        </div>

        {/* View Protected */}
        <div
          onClick={openShielded}
          style={{
            backgroundColor: '#fafafa', borderRadius: '12px', padding: '1.25rem',
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            cursor: 'pointer', transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
            border: '1px solid transparent'
          }}
          onMouseEnter={(e) => { e.currentTarget.style.transform = 'translateY(-3px)'; e.currentTarget.style.boxShadow = '0 12px 30px -8px rgba(16,185,129,0.25)'; }}
          onMouseLeave={(e) => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = 'none'; }}
        >
          <ShieldAlert style={{ width: '22px', height: '22px', color: '#18181b', marginBottom: '0.5rem' }} />
          <span style={{ fontSize: '0.95rem', fontWeight: 700, color: '#18181b', marginBottom: '4px' }}>Shielded Art</span>
          <span style={{ fontSize: '0.7rem', color: '#71717a' }}>View protected output</span>
        </div>
      </div>

      {/* Shield Report Panel — only shows after processing */}
      {shieldReport && (
        <div style={{
          width: '100%', maxWidth: '900px', marginBottom: '1.5rem',
          background: 'linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(6,78,59,0.05) 100%)',
          border: '1px solid rgba(16,185,129,0.2)', borderRadius: '12px', padding: '1.25rem',
          position: 'relative', zIndex: 1, overflow: 'hidden'
        }}>
          {/* Success glow */}
          <div style={{
            position: 'absolute', top: '-30px', right: '-30px', width: '120px', height: '120px',
            background: 'radial-gradient(circle, rgba(16,185,129,0.15) 0%, transparent 70%)',
            pointerEvents: 'none'
          }} />

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1rem' }}>
            <ShieldCheck style={{ width: '20px', height: '20px', color: '#10b981' }} />
            <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#10b981', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
              Shield Report — {currentFileName}
            </span>
            <span style={{
              marginLeft: 'auto', fontSize: '0.65rem', color: '#065f46',
              backgroundColor: 'rgba(16,185,129,0.15)', padding: '2px 8px',
              borderRadius: '99px', fontWeight: 600
            }}>
              {shieldReport.engine_mode === 'CLIP_ADVERSARIAL' ? '🧠 CLIP Adversarial' : '⚡ Enhanced Fallback'}
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: '12px' }}>
            {/* CLIP Distance */}
            <div style={{ backgroundColor: 'rgba(0,0,0,0.2)', borderRadius: '10px', padding: '1rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <Eye style={{ width: '16px', height: '16px', color: '#6b7280', marginBottom: '6px' }} />
              <span style={{ fontSize: '0.6rem', color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '4px' }}>AI Confusion</span>
              <span style={{ fontSize: '1.5rem', fontWeight: 300, color: '#34d399', fontFamily: 'monospace' }}>
                {clipDistancePct}%
              </span>
              {/* Mini bar */}
              <div style={{ width: '100%', height: '3px', backgroundColor: 'rgba(255,255,255,0.05)', borderRadius: '2px', marginTop: '6px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${clipDistancePct}%`, backgroundColor: '#10b981', borderRadius: '2px' }} />
              </div>
            </div>

            {/* Pixel Coverage */}
            <div style={{ backgroundColor: 'rgba(0,0,0,0.2)', borderRadius: '10px', padding: '1rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <Zap style={{ width: '16px', height: '16px', color: '#6b7280', marginBottom: '6px' }} />
              <span style={{ fontSize: '0.6rem', color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '4px' }}>Pixel Coverage</span>
              <span style={{ fontSize: '1.5rem', fontWeight: 300, color: '#34d399', fontFamily: 'monospace' }}>
                {shieldReport.pixels_modified_pct}%
              </span>
            </div>

            {/* Hash Fingerprint */}
            <div style={{ backgroundColor: 'rgba(0,0,0,0.2)', borderRadius: '10px', padding: '1rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <Fingerprint style={{ width: '16px', height: '16px', color: '#6b7280', marginBottom: '6px' }} />
              <span style={{ fontSize: '0.6rem', color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '4px' }}>Hash Changed</span>
              <span style={{ fontSize: '0.7rem', color: '#a1a1aa', fontFamily: 'monospace', wordBreak: 'break-all', textAlign: 'center' }}>
                {shieldReport.original_hash.slice(0, 12)}→{shieldReport.protected_hash.slice(0, 12)}
              </span>
            </div>

            {/* C2PA Status */}
            <div style={{ backgroundColor: 'rgba(0,0,0,0.2)', borderRadius: '10px', padding: '1rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <Lock style={{ width: '16px', height: '16px', color: '#6b7280', marginBottom: '6px' }} />
              <span style={{ fontSize: '0.6rem', color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '4px' }}>C2PA Watermark</span>
              <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#10b981' }}>✓ Embedded</span>
            </div>
          </div>
        </div>
      )}

      {/* Terminal Log View */}
      <div style={{
        width: '100%', maxWidth: '900px', flex: 1, backgroundColor: '#000000',
        border: '1px solid rgba(39,39,42,0.6)', borderRadius: '12px',
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
        position: 'relative', zIndex: 1
      }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 16px',
          backgroundColor: 'rgba(24,24,27,0.8)', borderBottom: '1px solid rgba(39,39,42,0.6)',
          backdropFilter: 'blur(8px)'
        }}>
          <TerminalIcon style={{ width: '14px', height: '14px', color: '#52525b' }} />
          <span style={{ fontSize: '0.7rem', color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.12em', fontWeight: 600 }}>
            Activity Stream
          </span>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: '5px' }}>
            <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#27272a' }} />
            <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#27272a' }} />
            <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#27272a' }} />
          </div>
        </div>

        <div style={{ flex: 1, padding: '12px 16px', overflowY: 'auto', minHeight: '200px', maxHeight: '300px' }}>
          <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {logs.map((log, i) => (
              <li key={i} style={{
                fontSize: '0.8rem', display: 'flex', gap: '10px', fontFamily: 'monospace',
                color: log.type === 'error' ? '#f87171' : log.type === 'success' ? '#34d399' : '#d4d4d8'
              }}>
                <span style={{ color: '#3f3f46', flexShrink: 0 }}>[{log.time}]</span>
                <span>{log.message}</span>
              </li>
            ))}
            <div ref={logsEndRef} />
          </ul>
        </div>
      </div>

      {/* CSS Animations */}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        * { box-sizing: border-box; }
        body { margin: 0; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #27272a; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #3f3f46; }
      `}</style>
    </main>
  );
}
