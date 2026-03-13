'use client';

import { useEffect, useState, useMemo } from 'react';
import { createClient } from '@/lib/supabase-browser';
import { Shield, Upload, LogOut, Grid, Image as ImageIcon, Zap, CheckCircle, Clock, CreditCard, Box, LayoutGrid, Info, Activity, Menu, Bell } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// Lemonade-Standard Spring Configurations
const lemonSpring = { type: "spring", stiffness: 400, damping: 30, mass: 1 } as any;

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.1
    }
  }
} as any;

const item = {
  hidden: { opacity: 0, scale: 0.95 },
  show: { 
    opacity: 1, 
    scale: 1,
    transition: lemonSpring
  }
} as any;

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
      console.error('Protection Failed:', err);
    } finally {
      setIsUploading(false);
    }
  };

  const handleLogout = async () => {
    await supabase.auth.signOut();
    window.location.href = '/login';
  };

  if (loading) return (
    <div className="min-h-screen bg-white flex items-center justify-center">
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={lemonSpring}
        className="flex flex-col items-center gap-6"
      >
        <div className="w-16 h-16 bg-lemon rounded-2xl flex items-center justify-center shadow-lg animate-pulse">
          <Shield className="w-8 h-8 text-white" />
        </div>
        <span className="text-gray-400 font-black text-xs uppercase tracking-[0.4em]">Synchronizing Enclave</span>
      </motion.div>
    </div>
  );

  return (
    <div className="min-h-screen bg-[#FAFAFA] text-[#111111] font-sans selection:bg-[#FF0066]/10 pb-20">
      
      {/* Dashboard Nav: Clean & Minimal */}
      <nav className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-100 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-lemon rounded-xl flex items-center justify-center">
              <Shield className="text-white w-5 h-5" />
            </div>
            <div className="flex flex-col">
              <span className="text-xl font-black tracking-tighter leading-none">SANFA</span>
              <span className="text-[10px] font-black uppercase text-lemon-400 tracking-[0.2em] mt-1">Dashboard</span>
            </div>
          </div>

          <div className="flex items-center gap-6">
            <div className="hidden md:flex items-center gap-4 px-4 py-2 bg-gray-50 rounded-full border border-gray-100">
               <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
               <span className="text-[10px] font-bold text-gray-500">{user?.email}</span>
            </div>
            <div className="flex items-center gap-3 text-lemon font-black text-xs uppercase tracking-widest pl-4 border-l">
               <Bell className="w-4 h-4 text-gray-400 cursor-pointer" />
               <button onClick={handleLogout} className="text-gray-400 hover:text-lemon transition-colors">
                  <LogOut className="w-4 h-4" />
               </button>
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-12">
        {/* Bento Stats */}
        <motion.div 
          variants={container}
          initial="hidden"
          animate="show"
          className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12"
        >
          <motion.div variants={item} className="md:col-span-2 lemon-card p-8 flex flex-col justify-between">
            <div>
              <p className="text-gray-400 text-[10px] font-black uppercase tracking-[0.4em] mb-4">Secured Heritage</p>
              <h3 className="text-4xl font-black">{assets.length}</h3>
            </div>
            <div className="flex items-center justify-between mt-8">
               <div className="flex items-center gap-2 text-[10px] font-black text-green-500 uppercase tracking-widest">
                  <Activity className="w-4 h-4" />
                  Protection Active
               </div>
               <span className="text-gray-300 text-[10px] font-bold">Updated just now</span>
            </div>
          </motion.div>

          <motion.div variants={item} className="lemon-card p-8 flex flex-col justify-between">
            <div className="w-10 h-10 bg-gray-50 rounded-xl flex items-center justify-center">
               <Grid className="text-lemon w-5 h-5" />
            </div>
            <div>
              <p className="text-gray-400 text-[10px] font-black uppercase tracking-[0.4em] mb-2">Current Tier</p>
              <h3 className="text-xl font-black">{profile?.subscription_status?.toUpperCase() || 'FREE'}</h3>
            </div>
          </motion.div>

          <motion.div variants={item} className="lemon-card p-8 flex flex-col justify-between bg-lemon text-white">
            <div className="w-10 h-10 bg-white/10 rounded-xl flex items-center justify-center">
               <CreditCard className="text-white w-5 h-5" />
            </div>
            <div>
              <p className="text-white/40 text-[10px] font-black uppercase tracking-[0.4em] mb-2">Available Credits</p>
              <h3 className="text-2xl font-black">{profile?.credits_remaining || 0} PRC</h3>
            </div>
          </motion.div>
        </motion.div>

        {/* Primary Action */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ ...lemonSpring, delay: 0.3 }}
          className="lemon-card p-12 md:p-20 text-center relative overflow-hidden mb-20"
        >
          <div className="absolute top-0 right-0 p-8 text-lemon/5">
             <Upload className="w-64 h-64" />
          </div>
          
          <div className="relative z-10 max-w-2xl mx-auto">
             <div className="w-20 h-20 bg-gray-50 rounded-3xl flex items-center justify-center mx-auto mb-10 shadow-sm">
                <Upload className="text-lemon w-8 h-8" />
             </div>
             <h2 className="text-3xl font-black mb-6">Upload your masterpiece.</h2>
             <p className="text-gray-500 text-lg mb-10">
                Protect your digital assets from neural network mining in under 90 seconds. 
                Our <strong>Spectral Shield</strong> ensures total mathematical sovereignty.
             </p>
             
             <button 
               onClick={handleProtect}
               disabled={isUploading}
               className="btn-lemon text-lg px-12 py-5 w-full sm:w-auto shadow-xl flex items-center justify-center gap-4 mx-auto disabled:opacity-50"
             >
               {isUploading ? 'Securing...' : 'Start New Protection'}
               <Zap className={`w-5 h-5 ${isUploading ? 'animate-bounce' : 'fill-current'}`} />
             </button>
          </div>
        </motion.div>

        {/* Vault History */}
        <section>
          <div className="flex items-center justify-between mb-10 border-b border-gray-100 pb-6">
            <h3 className="text-xl font-black flex items-center gap-3">
              <LayoutGrid className="text-lemon w-5 h-5" />
              Your Vault
            </h3>
            <span className="text-gray-400 font-bold text-[10px] uppercase tracking-[0.2em]">{assets.length} Assets Secured</span>
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
                  className="lemon-card p-4 group"
                >
                  <div className="aspect-square bg-gray-50 rounded-2xl mb-6 overflow-hidden relative">
                    {asset.status === 'completed' ? (
                      <img 
                        src={asset.protected_url || asset.original_url} 
                        alt={asset.original_name} 
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" 
                      />
                    ) : (
                      <div className="w-full h-full flex flex-col items-center justify-center gap-4">
                        <Clock className="w-10 h-10 text-gray-200 animate-spin" />
                        <span className="text-[8px] font-black uppercase tracking-[0.4em] text-gray-300">Poisoning Kernels</span>
                      </div>
                    )}
                    <div className="absolute top-4 right-4">
                      <div className={`p-2 rounded-xl shadow-lg border border-white ${asset.status === 'completed' ? 'bg-green-500' : 'bg-lemon'}`}>
                        {asset.status === 'completed' ? (
                          <CheckCircle className="w-4 h-4 text-white" />
                        ) : (
                          <Activity className="w-4 h-4 text-white animate-pulse" />
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="px-2">
                    <h4 className="text-sm font-black truncate mb-2">{asset.original_name}</h4>
                    <div className="flex items-center justify-between">
                       <span className="text-[10px] text-gray-400 font-bold">
                         {new Date(asset.created_at).toLocaleDateString()}
                       </span>
                       <span className="text-[10px] text-lemon font-black tracking-tighter">
                         {asset.status === 'completed' ? 'SECURED' : 'PENDING'}
                       </span>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {assets.length === 0 && !isUploading && (
              <div className="col-span-full py-32 border-2 border-dashed border-gray-100 rounded-[40px] flex flex-col items-center justify-center gap-6">
                <ImageIcon className="w-12 h-12 text-gray-100" />
                <p className="text-gray-300 font-black text-xs uppercase tracking-[0.4em]">Vault is currently empty</p>
              </div>
            )}
          </motion.div>
        </section>
      </main>
    </div>
  );
}

