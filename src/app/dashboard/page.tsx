'use client';

import { useEffect, useState, useMemo } from 'react';
import { createClient } from '@/lib/supabase-browser';
import { Shield, Upload, LogOut, Grid, Image as ImageIcon, Zap, CheckCircle, Clock, CreditCard, Box, LayoutGrid, Info, Activity } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export const dynamic = 'force-dynamic';

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 }
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
      alert('System handoff failed. Check connectivity.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleLogout = async () => {
    await supabase.auth.signOut();
    window.location.href = '/login';
  };

  if (loading) return (
    <div className="min-h-screen bg-[#0a0a0b] flex items-center justify-center">
      <motion.div 
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="flex flex-col items-center gap-6"
      >
        <div className="w-16 h-16 bg-gold-500/10 rounded-2xl flex items-center justify-center border border-gold-500/20 shadow-2xl shadow-gold-500/10">
          <Shield className="w-8 h-8 text-gold-500 animate-pulse" />
        </div>
        <div className="flex flex-col items-center gap-2">
          <span className="text-white font-serif text-xl tracking-tight">Accessing Enclave</span>
          <div className="w-48 h-1 bg-white/5 rounded-full overflow-hidden">
            <motion.div 
              initial={{ x: '-100%' }}
              animate={{ x: '100%' }}
              transition={{ repeat: Infinity, duration: 1.5, ease: 'linear' }}
              className="w-1/2 h-full bg-gold-500 shadow-[0_0_10px_rgba(201,168,76,0.5)]"
            />
          </div>
        </div>
      </motion.div>
    </div>
  );

  return (
    <div className="min-h-screen bg-[#0a0a0b] text-zinc-100 font-sans selection:bg-gold-500/30">
      <div className="fixed inset-0 pointer-events-none opacity-[0.03] noise z-50" />

      {/* Navigation */}
      <nav className="glass sticky top-0 z-[60] px-8 py-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 bg-gold-500/20 rounded-xl flex items-center justify-center border border-gold-500/30">
            <Shield className="text-gold-500 w-6 h-6" />
          </div>
          <div className="flex flex-col">
            <span className="text-lg font-serif leading-none tracking-tight">SANFA <span className="text-gold-500">CLOUD</span></span>
            <span className="text-[10px] uppercase tracking-[0.3em] text-white/30 font-bold mt-1">Enclave Authority</span>
          </div>
        </div>

        <div className="flex items-center gap-8">
          <div className="hidden lg:flex items-center gap-6 px-5 py-2 bg-white/5 rounded-full border border-white/5">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-emerald-500 rounded-full shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
              <span className="text-[11px] font-bold text-white/40 uppercase tracking-widest">{user?.email}</span>
            </div>
            <div className="w-px h-3 bg-white/10" />
            <div className="flex items-center gap-2 text-gold-500">
              <CreditCard className="w-3.5 h-3.5" />
              <span className="text-[11px] font-bold uppercase tracking-widest">{profile?.credits_remaining || 0} PRC</span>
            </div>
          </div>
          <button 
            onClick={handleLogout}
            className="group flex items-center gap-2 text-white/30 hover:text-white transition-all"
          >
            <span className="text-[11px] font-bold uppercase tracking-widest hidden sm:block">Exit Terminal</span>
            <LogOut className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </button>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-8 py-12">
        {/* Bento Stats Grid */}
        <motion.div 
          variants={container}
          initial="hidden"
          animate="show"
          className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12"
        >
          <motion.div variants={item} className="md:col-span-2 glass-card rounded-3xl p-8 relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-8 opacity-10 group-hover:opacity-20 transition-opacity">
              <Box className="w-24 h-24 text-gold-500" />
            </div>
            <div className="relative z-10">
              <p className="text-white/40 text-[10px] font-bold uppercase tracking-[0.3em] mb-4">Total Protected Assets</p>
              <h3 className="text-[var(--font-size-2xl)] font-serif text-gradient">{assets.length}</h3>
              <div className="mt-6 flex items-center gap-2 text-[10px] font-bold text-emerald-500 uppercase tracking-widest">
                <Activity className="w-3 h-3" />
                +12% vs last session
              </div>
            </div>
          </motion.div>

          <motion.div variants={item} className="glass-card rounded-3xl p-8 group">
            <div className="flex items-center justify-between mb-8">
              <div className="w-10 h-10 bg-gold-500/10 rounded-xl flex items-center justify-center border border-gold-500/20">
                <LayoutGrid className="text-gold-500 w-5 h-5" />
              </div>
              <Info className="text-white/10 w-4 h-4" />
            </div>
            <p className="text-white/40 text-[10px] font-bold uppercase tracking-[0.3em] mb-2">Protocol Tier</p>
            <h3 className="text-xl font-serif text-white">{profile?.subscription_status?.toUpperCase() || 'STANDARD'}</h3>
          </motion.div>

          <motion.div variants={item} className="glass-card rounded-3xl p-8 group border-gold-500/20 bg-gold-500/5">
            <div className="flex items-center justify-between mb-8">
              <div className="w-10 h-10 bg-gold-500/20 rounded-xl flex items-center justify-center border border-gold-500/30">
                <CreditCard className="text-gold-500 w-5 h-5" />
              </div>
              <Zap className="text-gold-500 w-4 h-4 animate-pulse" />
            </div>
            <p className="text-gold-500/60 text-[10px] font-bold uppercase tracking-[0.3em] mb-2">Available Credits</p>
            <h3 className="text-xl font-serif text-gold-500">{profile?.credits_remaining || 0} PRC</h3>
          </motion.div>
        </motion.div>

        {/* Primary Action Center */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="glass-card rounded-[40px] p-16 text-center relative overflow-hidden group shadow-2xl shadow-black/50"
        >
          <div className="absolute inset-0 bg-gradient-to-br from-gold-500/10 via-transparent to-transparent opacity-50" />
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[300px] bg-gold-500/5 blur-[100px] rounded-full" />
          
          <div className="relative z-10">
            <motion.div 
              animate={isUploading ? { scale: [1, 1.1, 1], rotate: [0, 180, 360] } : {}}
              transition={{ repeat: Infinity, duration: 4 }}
              className="w-24 h-24 bg-gold-500/10 rounded-[32px] flex items-center justify-center mx-auto mb-10 border border-gold-500/20 shadow-xl shadow-gold-500/5"
            >
              <Upload className="text-gold-500 w-10 h-10" />
            </motion.div>
            
            <h2 className="text-[var(--font-size-xl)] font-serif mb-6 text-gradient">Protect Your Masterpiece.</h2>
            <p className="text-white/40 max-w-xl mx-auto mb-12 text-lg leading-relaxed">
              Inject your high-resolution artwork into the SANFA Enclave. 
              Our decoupled GPU workers will perform a <span className="text-gold-400 font-bold italic">Neural Collapse</span> protocol in silence.
            </p>
            
            <motion.button 
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleProtect}
              disabled={isUploading}
              className="bg-white text-black px-16 py-5 rounded-full text-lg font-bold hover:bg-gold-500 transition-all shadow-2xl hover:shadow-gold-500/30 flex items-center gap-4 mx-auto disabled:opacity-50"
            >
              {isUploading ? 'Initializing Neural Tunnel...' : 'Select & Scramble'}
              <Zap className={`w-5 h-5 ${isUploading ? 'animate-bounce' : 'fill-current'}`} />
            </motion.button>
          </div>
        </motion.div>

        {/* Asset Grid Section */}
        <section className="mt-24">
          <div className="flex items-center justify-between mb-12 border-b border-white/5 pb-6">
            <h3 className="text-2xl font-serif flex items-center gap-4">
              <LayoutGrid className="text-gold-500 w-6 h-6" />
              Vault Enclave <span className="text-white/20 text-sm font-sans font-bold ml-2 tracking-widest">LATEST DEPOSITS</span>
            </h3>
            <button className="text-[10px] font-bold uppercase tracking-[0.3em] text-white/30 hover:text-white transition-all">View All Archives</button>
          </div>
          
          <motion.div 
            variants={container}
            initial="hidden"
            animate="show"
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8"
          >
            <AnimatePresence>
              {assets.map((asset) => (
                <motion.div 
                  key={asset.id} 
                  variants={item}
                  layout
                  className="glass-card rounded-[2rem] p-5 group cursor-pointer"
                >
                  <div className="aspect-[4/5] bg-black/40 rounded-2xl mb-5 overflow-hidden relative">
                    {asset.status === 'completed' ? (
                      <motion.img 
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 0.7 }}
                        whileHover={{ opacity: 1 }}
                        src={asset.protected_url || asset.original_url} 
                        alt={asset.original_name} 
                        className="w-full h-full object-cover transition-opacity duration-500" 
                      />
                    ) : (
                      <div className="w-full h-full flex flex-col items-center justify-center gap-4">
                        <div className="relative">
                          <Clock className="w-10 h-10 text-gold-500/20" />
                          <motion.div 
                            animate={{ rotate: 360 }}
                            transition={{ repeat: Infinity, duration: 2, ease: 'linear' }}
                            className="absolute inset-0 border-2 border-t-gold-500 border-transparent rounded-full"
                          />
                        </div>
                        <span className="text-[10px] uppercase tracking-widest text-gold-500/40 font-bold">Neural Sync...</span>
                      </div>
                    )}
                    <div className="absolute top-4 right-4">
                      {asset.status === 'completed' ? (
                        <div className="bg-emerald-500 p-1.5 rounded-lg shadow-lg">
                          <CheckCircle className="w-4 h-4 text-white" />
                        </div>
                      ) : (
                        <div className="bg-gold-500 p-1.5 rounded-lg shadow-lg">
                          <Zap className="w-4 h-4 text-black animate-pulse" />
                        </div>
                      )}
                    </div>
                  </div>

                  <h4 className="text-sm font-sans font-bold truncate mb-2 px-1">{asset.original_name}</h4>
                  
                  <div className="flex items-center justify-between px-1">
                    <span className="text-[10px] text-white/30 uppercase tracking-widest font-bold">
                      {new Date(asset.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                    </span>
                    <div className="flex items-center gap-1">
                      <div className="w-1 h-1 bg-gold-500 rounded-full" />
                      <span className="text-[10px] text-gold-500 font-black tracking-tighter">
                        {asset.clip_distance ? `${Math.round(asset.clip_distance * 100)}% CONFUSION` : 'READY'}
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
                className="col-span-full py-32 bg-white/[0.02] border border-dashed border-white/5 rounded-[3rem] flex flex-col items-center justify-center gap-6"
              >
                <div className="w-20 h-20 bg-white/5 rounded-full flex items-center justify-center">
                  <ImageIcon className="w-8 h-8 text-white/10" />
                </div>
                <div className="text-center">
                  <span className="text-white font-serif text-xl block mb-2">The Enclave is Empty.</span>
                  <span className="text-white/20 font-sans tracking-[0.2em] uppercase text-[10px] font-bold">Secure your legacy today</span>
                </div>
              </motion.div>
            )}
          </motion.div>
        </section>
      </main>

      {/* Floating Support Info */}
      <div className="fixed bottom-8 right-8 z-[70]">
        <div className="glass px-6 py-3 rounded-2xl flex items-center gap-3 border-gold-500/20">
          <div className="w-2 h-2 bg-emerald-500 rounded-full animate-ping" />
          <span className="text-[10px] font-bold uppercase tracking-widest text-white/50">Cluster v5.2 Online</span>
        </div>
      </div>
    </div>
  );
}
