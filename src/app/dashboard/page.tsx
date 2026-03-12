'use client';

import { useEffect, useState, useMemo } from 'react';
import { createClient } from '@/lib/supabase-browser';
import { Shield, Upload, LogOut, Grid, Image as ImageIcon, Zap, CheckCircle, Clock, CreditCard, Box, LayoutGrid, Info, Activity } from 'lucide-react';
import { motion, AnimatePresence, useSpring } from 'framer-motion';

export const dynamic = 'force-dynamic';

// Apple-Standard Spring Configurations
const organicSpring = { type: "spring", stiffness: 300, damping: 30, mass: 1 };
const entranceSpring = { type: "spring", stiffness: 100, damping: 20, mass: 1 };

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.2
    }
  }
};

const item = {
  hidden: { opacity: 0, y: 30 },
  show: { 
    opacity: 1, 
    y: 0,
    transition: organicSpring
  }
};

export default function Dashboard() {
  const supabase = useMemo(() => createClient(), []);
  const [user, setUser] = useState<any>(null);
  const [profile, setProfile] = useState<any>(null);
  const [assets, setAssets] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        window.location.href = '/login';
        return;
      }
      setUser(user);

      const [profileRes, assetsRes] = await Promise.all([
        supabase.from('profiles').select('*').eq('id', user.id).single(),
        supabase.from('protected_images').select('*').eq('user_id', user.id).order('created_at', { ascending: false })
      ]);

      setProfile(profileRes.data);
      setAssets(assetsRes.data || []);
      setLoading(false);
    };

    fetchData();

    const channel = supabase
      .channel('schema-db-changes')
      .on('postgres_changes', 
        { event: '*', schema: 'public', table: 'protected_images', filter: `user_id=eq.${user?.id}` }, 
        (payload) => {
          setAssets(prev => {
            const index = prev.findIndex(a => a.id === (payload.new as any).id);
            if (index > -1) {
              const next = [...prev];
              next[index] = payload.new;
              return next;
            }
            return [payload.new, ...prev];
          });
        }
      )
      .subscribe();

    return () => { supabase.removeChannel(channel); };
  }, [user?.id]);

  const handleProtect = async () => {
    setIsUploading(true);
    try {
      const mockUrl = `https://picsum.photos/seed/${Math.random()}/800/600`;
      const res = await fetch('/api/protect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          imageUrl: mockUrl,
          fileName: `Masterpiece_${Date.now()}.png`
        })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error);

    } catch (err) {
      console.error('Handoff Failed:', err);
    } finally {
      setIsUploading(false);
    }
  };

  const handleLogout = async () => {
    await supabase.auth.signOut();
    window.location.href = '/login';
  };

  if (loading) return (
    <div className="min-h-screen bg-[#030304] flex items-center justify-center">
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={organicSpring}
        className="flex flex-col items-center gap-10"
      >
        <div className="w-20 h-20 glass-v2 rounded-[2.5rem] flex items-center justify-center border border-white/5 relative overflow-hidden">
          <Shield className="w-8 h-8 text-gold-500 z-10 animate-pulse" />
          <div className="absolute inset-0 glow-spectral opacity-20" />
        </div>
        <div className="flex flex-col items-center gap-4">
          <span className="text-white/40 font-serif text-2xl tracking-tight">Synchronizing Enclave</span>
          <div className="w-64 h-[2px] bg-white/[0.03] rounded-full overflow-hidden">
            <motion.div 
              initial={{ x: '-100%' }}
              animate={{ x: '100%' }}
              transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
              className="w-1/2 h-full bg-gold-500/50"
            />
          </div>
        </div>
      </motion.div>
    </div>
  );

  return (
    <div className="min-h-screen bg-[#030304] text-white font-sans selection:bg-gold-500/40 pb-40">
      <div className="fixed inset-0 pointer-events-none noise-fine z-50" />
      <div className="fixed inset-0 pointer-events-none glow-spectral opacity-30 z-0" />

      {/* Navigation: Airy & Material */}
      <nav className="glass-v2 sticky top-0 z-[60] px-12 py-8 flex items-center justify-between border-b border-white/[0.03] backdrop-blur-[60px]">
        <div className="flex items-center gap-6">
          <div className="w-12 h-12 glass-v2 rounded-2xl flex items-center justify-center border border-white/5">
            <Shield className="text-gold-500 w-6 h-6" />
          </div>
          <div className="flex flex-col">
            <span className="text-2xl font-serif leading-none tracking-tight">SANFA <span className="text-gold-500">CLOUD</span></span>
            <span className="text-[10px] uppercase tracking-[0.5em] text-white/20 font-bold mt-2">Sovereign Control</span>
          </div>
        </div>

        <div className="flex items-center gap-10">
          <div className="hidden lg:flex items-center gap-8 px-8 py-3 glass-v2 rounded-full border border-white/5">
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 bg-emerald-500 rounded-full shadow-[0_0_12px_rgba(16,185,129,0.3)] animate-pulse" />
              <span className="text-[11px] font-bold text-white/30 uppercase tracking-[0.3em]">{user?.email}</span>
            </div>
            <div className="w-px h-4 bg-white/5" />
            <div className="flex items-center gap-3 text-gold-500">
              <CreditCard className="w-4 h-4" />
              <span className="text-[11px] font-black uppercase tracking-[0.3em]">{profile?.credits_remaining || 0} PRC</span>
            </div>
          </div>
          <button 
            onClick={handleLogout}
            className="group flex items-center gap-4 text-white/20 hover:text-white transition-all bg-white/[0.03] px-8 py-3 rounded-full hover:bg-white/5 border border-white/5"
          >
            <span className="text-[11px] font-black uppercase tracking-[0.4em]">Exit</span>
            <LogOut className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
          </button>
        </div>
      </nav>

      <main className="max-w-[1400px] mx-auto px-12 py-20 relative z-10">
        {/* Bento Stats Grid: Spacing 1.5x */}
        <motion.div 
          variants={container}
          initial="hidden"
          animate="show"
          className="grid grid-cols-1 md:grid-cols-4 gap-10 mb-20"
        >
          <motion.div 
            variants={item} 
            whileHover={{ y: -8, scale: 1.01 }}
            className="md:col-span-2 glass-card-v2 rounded-[3.5rem] p-12 relative overflow-hidden group border-white/[0.03]"
          >
            <div className="absolute top-0 right-0 p-12 opacity-[0.03] group-hover:opacity-[0.08] transition-opacity duration-1000">
              <Box className="w-40 h-40 text-gold-500" />
            </div>
            <div className="relative z-10 h-full flex flex-col justify-between">
              <div>
                <p className="text-white/20 text-[11px] font-black uppercase tracking-[0.5em] mb-6">Total Protected Assets</p>
                <h3 className="text-[var(--font-size-2xl)] font-serif text-aura">{assets.length}</h3>
              </div>
              <div className="mt-12 flex items-center gap-4 text-[11px] font-black text-emerald-500/60 uppercase tracking-[0.4em]">
                <Activity className="w-4 h-4" />
                Active Surveillance enabled
              </div>
            </div>
          </motion.div>

          <motion.div 
            variants={item}
            whileHover={{ y: -8, scale: 1.01 }}
            className="glass-card-v2 rounded-[3.5rem] p-12 flex flex-col justify-between border-white/[0.03]"
          >
            <div className="w-14 h-14 glass-v2 rounded-2xl flex items-center justify-center border border-white/5">
              <LayoutGrid className="text-gold-500 w-6 h-6" />
            </div>
            <div>
              <p className="text-white/20 text-[11px] font-black uppercase tracking-[0.5em] mb-4">Tier</p>
              <h3 className="text-3xl font-serif text-white/90">{profile?.subscription_status?.toUpperCase() || 'STANDARD'}</h3>
            </div>
          </motion.div>

          <motion.div 
            variants={item}
            whileHover={{ y: -8, scale: 1.01 }}
            className="glass-card-v2 rounded-[3.5rem] p-12 flex flex-col justify-between bg-gold-500/[0.02] border-gold-500/10"
          >
            <div className="w-14 h-14 bg-gold-500/10 rounded-2xl flex items-center justify-center border border-gold-500/20">
              <CreditCard className="text-gold-500 w-6 h-6" />
            </div>
            <div>
              <p className="text-gold-500/40 text-[11px] font-black uppercase tracking-[0.5em] mb-4">Available PRC</p>
              <h3 className="text-3xl font-serif text-gold-500">{profile?.credits_remaining || 0}</h3>
            </div>
          </motion.div>
        </motion.div>

        {/* Primary Action Center: High-Fidelity Elevation */}
        <motion.div 
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ ...organicSpring, delay: 0.4 }}
          className="glass-card-v2 rounded-[80px] p-32 text-center relative overflow-hidden border-white/[0.02] shadow-[0_0_100px_rgba(0,0,0,0.5)]"
        >
          <div className="absolute inset-0 bg-gradient-to-tr from-gold-500/[0.03] via-transparent to-transparent opacity-50" />
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[1000px] h-[400px] bg-gold-500/[0.02] blur-[120px] rounded-full" />
          
          <div className="relative z-10 max-w-4xl mx-auto">
            <motion.div 
              animate={isUploading ? { scale: [1, 1.05, 1], rotate: [0, 5, -5, 0] } : {}}
              transition={{ repeat: Infinity, duration: 3, ease: "easeInOut" }}
              className="w-28 h-28 glass-v2 rounded-[3rem] flex items-center justify-center mx-auto mb-16 border border-white/10"
            >
              <Upload className="text-gold-500 w-12 h-12" />
            </motion.div>
            
            <h2 className="text-[var(--font-size-xl)] font-serif mb-10 text-aura">Protect Your Sovereign Genius.</h2>
            <p className="text-white/20 text-2xl leading-relaxed mb-16 tracking-tight text-balance">
              Initiate a high-resolution deposit into the SANFA Enclave. Our serverless GPU enclaves will execute the <span className="text-gold-500 italic">Spectral Scramble</span> protocol with mathematical precision.
            </p>
            
            <motion.button 
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              onClick={handleProtect}
              disabled={isUploading}
              className="bg-white text-black px-20 py-8 rounded-full text-xl font-black uppercase tracking-[0.4em] hover:bg-gold-400 transition-all shadow-[0_0_50px_rgba(255,255,255,0.05)] flex items-center gap-6 mx-auto disabled:opacity-30"
            >
              {isUploading ? 'Scrambling Kernels...' : 'Select & Secure'}
              <Zap className={`w-6 h-6 ${isUploading ? 'animate-bounce' : 'fill-current'}`} />
            </motion.button>
          </div>
        </motion.div>

        {/* Vault Section: 1.5x Spacing */}
        <section className="mt-40">
          <div className="flex items-center justify-between mb-20 border-b border-white/[0.03] pb-10 px-4">
            <h3 className="text-3xl font-serif flex items-center gap-6">
              <LayoutGrid className="text-gold-500 w-8 h-8" />
              Archives <span className="text-white/10 text-xs font-bold ml-4 tracking-[0.8em] uppercase">Secured History</span>
            </h3>
            <button className="text-[11px] font-black uppercase tracking-[0.5em] text-white/20 hover:text-white transition-all">Export Metadata</button>
          </div>
          
          <motion.div 
            variants={container}
            initial="hidden"
            animate="show"
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-12"
          >
            <AnimatePresence>
              {assets.map((asset) => (
                <motion.div 
                  key={asset.id} 
                  variants={item}
                  whileHover={{ y: -8 }}
                  transition={organicSpring}
                  className="glass-card-v2 rounded-[3rem] p-6 group cursor-pointer border-white/[0.02]"
                >
                  <div className="aspect-[3/4] bg-black/40 rounded-[2rem] mb-8 overflow-hidden relative">
                    {asset.status === 'completed' ? (
                      <motion.img 
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 0.6 }}
                        whileHover={{ opacity: 1, scale: 1.05 }}
                        transition={{ duration: 0.8 }}
                        src={asset.protected_url || asset.original_url} 
                        alt={asset.original_name} 
                        className="w-full h-full object-cover grayscale group-hover:grayscale-0 transition-all duration-1000" 
                      />
                    ) : (
                      <div className="w-full h-full flex flex-col items-center justify-center gap-10">
                        <div className="relative">
                          <Clock className="w-14 h-14 text-gold-500/10" />
                          <motion.div 
                            animate={{ rotate: 360 }}
                            transition={{ repeat: Infinity, duration: 3, ease: "linear" }}
                            className="absolute inset-0 border-[3px] border-t-gold-500/40 border-transparent rounded-full"
                          />
                        </div>
                        <span className="text-[10px] uppercase tracking-[0.5em] text-gold-500/20 font-black">Neural Poisoning...</span>
                      </div>
                    )}
                    <div className="absolute top-6 right-6">
                      <div className={`p-2 rounded-xl backdrop-blur-md border border-white/10 ${asset.status === 'completed' ? 'bg-emerald-500/20' : 'bg-gold-500/20'}`}>
                        {asset.status === 'completed' ? (
                          <CheckCircle className="w-5 h-5 text-emerald-400" />
                        ) : (
                          <Activity className="w-5 h-5 text-gold-500 animate-pulse" />
                        )}
                      </div>
                    </div>
                  </div>

                  <h4 className="text-lg font-serif mb-4 truncate px-2 text-white/80">{asset.original_name}</h4>
                  
                  <div className="flex items-center justify-between px-2">
                    <span className="text-[10px] text-white/10 uppercase tracking-[0.4em] font-black">
                      {new Date(asset.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                    </span>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-gold-500 rounded-full animate-pulse" />
                      <span className="text-[11px] text-gold-500 font-black tracking-tight">
                        {asset.clip_distance ? `${Math.round(asset.clip_distance * 100)}% DISTANCE` : 'SHIELDED'}
                      </span>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {assets.length === 0 && !isUploading && (
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="col-span-full py-40 glass-v2 border-dashed border-white/5 rounded-[5rem] flex flex-col items-center justify-center gap-10"
              >
                <div className="w-24 h-24 glass-v2 rounded-[2.5rem] flex items-center justify-center border border-white/5">
                  <ImageIcon className="w-10 h-10 text-white/10" />
                </div>
                <div className="text-center">
                  <span className="text-white/40 font-serif text-3xl block mb-4">Archives Vacant.</span>
                  <span className="text-white/10 font-bold tracking-[1em] uppercase text-[11px]">Upload to secure heritage</span>
                </div>
              </motion.div>
            )}
          </motion.div>
        </section>
      </main>

      {/* Persistent Status Enclave */}
      <div className="fixed bottom-12 right-12 z-[70]">
        <div className="glass-v2 px-8 py-4 rounded-3xl flex items-center gap-4 border-white/5 shadow-2xl">
          <div className="w-2 h-2 bg-emerald-500 rounded-full animate-ping" />
          <span className="text-[11px] font-black uppercase tracking-[0.6em] text-white/40">Secure Enclave v6.0</span>
        </div>
      </div>
    </div>
  );
}
